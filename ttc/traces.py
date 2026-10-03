"""Readable timelines from a trial's saved agent event streams (Copilot CLI or Claude Code).

    ttc trace RESULTS_DIR [--agent 0] [--max-chars 2000] [--no-tool-output]

Prints, in order, each agent's prompt, the model's reasoning (assistant.reasoning), its messages,
every tool call (e.g. the shell command it ran) and the tool's result.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def _clip(text: str, n: int) -> str:
    text = text.rstrip()
    return text if n <= 0 or len(text) <= n else text[:n] + f" … [{len(text) - n} more chars]"


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines()) or "    "


def _claude_entries(e: dict, max_chars: int, tool_output: bool) -> list[tuple[int, str]]:
    """Claude Code stream-json: one content block per assistant event, tool results in user events."""
    t, ts = e.get("type"), (e.get("timestamp") or "")[11:19]
    content = (e.get("message") or {}).get("content")
    out: list[tuple[int, str]] = []
    if t == "user" and isinstance(content, str):
        out.append((0, f"{ts} USER\n{_indent(_clip(content, max_chars))}"))
    for b in content if isinstance(content, list) else []:
        bt = b.get("type")
        if bt == "thinking":
            out.append((1, f"{ts} THINKING\n{_indent(_clip(b.get('thinking', ''), max_chars))}"))
        elif bt == "text" and t == "assistant":
            out.append((2, f"{ts} ASSISTANT\n{_indent(_clip(b.get('text', ''), max_chars))}"))
        elif bt == "text" and t == "user":
            out.append((0, f"{ts} USER\n{_indent(_clip(b.get('text', ''), max_chars))}"))
        elif bt == "tool_use":
            inp = b.get("input") or {}
            shown = inp.get("command") or json.dumps(inp)
            sub = " (subagent)" if e.get("parent_tool_use_id") else ""
            out.append((2, f"{ts} TOOL CALL {b.get('name')}{sub}\n{_indent(_clip(shown, max_chars))}"))
        elif bt == "tool_result" and tool_output:
            c = b.get("content")
            text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
            out.append((3, f"{ts} TOOL RESULT ({'error' if b.get('is_error') else 'ok'})\n{_indent(_clip(text, max_chars))}"))
    if t == "result":
        out.append((3, f"{ts} RUN END ({'error' if e.get('is_error') else 'ok'}) turns={e.get('num_turns')} "
                       f"cost=${e.get('total_cost_usd')}: {_clip(str(e.get('result') or ''), 200)}"))
    return out


def _entries(e: dict, max_chars: int, tool_output: bool) -> list[tuple[int, str]]:
    """(rank, text) for one event; rank orders entries within a turn."""
    if "message" in e or e.get("type") == "result":  # Claude Code
        return _claude_entries(e, max_chars, tool_output)
    t, d, ts = e.get("type", ""), e.get("data") or {}, (e.get("timestamp") or "")[11:19]
    if t == "user.message":
        return [(0, f"{ts} USER\n{_indent(_clip(d.get('content', ''), max_chars))}")]
    if t == "assistant.reasoning":
        return [(1, f"{ts} THINKING\n{_indent(_clip(d.get('content', ''), max_chars))}")]
    if t == "assistant.message":
        out = []
        if (d.get("content") or "").strip():
            out.append((2, f"{ts} ASSISTANT\n{_indent(_clip(d['content'], max_chars))}"))
        for r in d.get("toolRequests") or []:
            args = r.get("arguments") or {}
            shown = args.get("command") or json.dumps(args)
            out.append((2, f"{ts} TOOL CALL {r.get('name')}\n{_indent(_clip(shown, max_chars))}"))
        return out
    if t == "tool.execution_complete" and tool_output:
        res = (d.get("result") or {}).get("content", "")
        return [(3, f"{ts} TOOL RESULT ({'ok' if d.get('success') else 'failed'})\n{_indent(_clip(res, max_chars))}")]
    if t == "session.error":
        return [(3, f"{ts} SESSION ERROR {json.dumps(d)[:max_chars]}")]
    return []


def render_agent(path: Path, max_chars: int = 2000, tool_output: bool = True) -> str:
    """Copilot emits a turn's reasoning after its message; within each turn, show reasoning first."""
    out: list[str] = []
    turn: list[tuple[int, int, str]] = []

    def flush():
        out.extend(text for _, _, text in sorted(turn))
        turn.clear()

    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                turn.append((3, len(turn), f"[cli] {line.rstrip()}"))
                continue
            if e.get("type") == "assistant.turn_start":
                flush()
            for rank, text in _entries(e, max_chars, tool_output):
                turn.append((rank, len(turn), text))
            if e.get("type") == "assistant.turn_end" or "message" in e or e.get("type") == "result":
                flush()  # Claude Code events are already in order
    flush()
    return "\n".join(out)


def render(results_dir: Path, agent: int | None = None, max_chars: int = 2000, tool_output: bool = True) -> str:
    traj = results_dir / "trajectories"
    files = sorted(traj.glob("agent-*.events.jsonl.gz"), key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    if agent is not None:
        files = [p for p in files if p.name == f"agent-{agent}.events.jsonl.gz"]
    if not files:
        raise SystemExit(f"no agent event streams under {traj}")
    parts = []
    for p in files:
        name = p.name.split(".")[0]
        parts.append(f"{'=' * 30} {name} {'=' * 30}\n{render_agent(p, max_chars, tool_output)}")
    return "\n\n".join(parts)
