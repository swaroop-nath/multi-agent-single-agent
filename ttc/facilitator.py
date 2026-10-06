"""Facilitator (`--facilitator`): a third Claude Code agent that only reminds team members to
follow the communication protocol they were given (Appendix A.2). It is not a teammate and not a
new protocol.

Every round (default every 15 minutes) it reads the shared folder and each agent's private scratch,
read-only, and returns one short reminder per active agent: share your progress the way the
protocol says (score log, findings, disconfirmations), and check the shared files for news from
your teammates. It may quote the protocol. It may not pass on any content (no teammate ideas, code,
scores), invent files or conventions, or give task advice. The harness delivers each reminder into
the agent's running Claude Code session as a user message; Claude Code shows it to the agent at its
next tool call ("The user sent a new message while you were working: ...").

Every round is logged to work/facilitator/rounds.jsonl (copied to trajectories/facilitator_rounds.jsonl.gz).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path

from . import prompts

log = logging.getLogger("ttc")

REMINDER_PREFIX = "[Team reminder from the harness] "
TOOLS = "Read,Glob,Grep"
SCHEMA = {
    "type": "object",
    "properties": {"reminders": {"type": "array", "items": {
        "type": "object",
        "properties": {"slot": {"type": "integer"}, "agent": {"type": "integer"}, "message": {"type": "string"}},
        "required": ["message"]}}},
    "required": ["reminders"],
}

SYSTEM = """\
You are the reminder service for a team of {n} AI agents working on the same task in parallel. You
are not a member of the team and you do not work on the task. Your only output is one short reminder
per agent per round; the harness delivers it to that agent as a user message.

Every agent received this communication protocol, word for word:

<protocol>
{protocol}
</protocol>

Your job is to make the agents actually follow the protocol's sharing and checking steps, exactly as
the protocol defines them. Each round, look at the shared folder ({shared}) and at each agent's
private scratch ({scratch}/work-<slot>), then remind each agent to:
- share its progress the way the protocol says: log every scored attempt in the score log in the
  protocol's format (rule 3), append findings with evidence to the findings file, and record failed
  or falsified ideas in the disconfirmations file (rule 4);
- check the shared files (slots/approaches, findings, disconfirmations, score log, coordination) for
  new information from its teammates (rules 4, 5 and 8).
You may quote or paraphrase any part of the protocol that fits what you observe (for example the
plateau rule, keeping a distinct approach, or the coordination file).

Rules:
1. You are a reminder, not a channel. Never pass on content: no teammate's ideas, methods, results,
   scores, code or file contents, and nothing from any agent's private scratch. You may name which
   shared files changed since your last round, so the agent knows where to look.
2. No new protocol: do not introduce new files, formats, schedules, roles or ways to divide the work.
   Use only the protocol's own files and steps.
3. No task help: no algorithmic or technical advice, and no opinions on whose approach is better.
4. Use an agent's private scratch, and the facts the harness gives you about its own submissions,
   only to judge what that agent has not yet shared (for example attempts missing from the score
   log, or results with no findings entry). Mention an agent's own numbers only to that agent.
5. Read only inside {task_root}. Never read {agents_dir}.
6. Write one reminder for every active agent, at most about 100 words, addressed to that agent and
   specific to what you observed this round. Identify each agent by its slot, as the shared files do;
   use the agent index only for an agent whose slot is not known yet. Do not ask the agent to reply to you; it should act in
   the shared workspace.
"""

ROUND = """\
Round {round}: {elapsed:.0f} min into the trial, {remaining:.0f} min left.

Active agents, by slot (inferred from the scratch paths each agent uses):
{agents}

Shared folder ({shared}), file sizes and minutes since last change:
{shared_listing}

Private scratch ({scratch}), per folder: file count and minutes since the newest change:
{scratch_listing}

Look at whatever you need (read-only), then return the reminders as JSON, one per active agent:
{{"reminders": [{{"slot": <slot>, "message": "<reminder>"}}, ...]}}
(for an agent whose slot is unknown, use {{"agent": <agent index>, "message": ...}} instead).
"""


def _listing(root: Path, now: float, recursive: bool) -> str:
    if not root.exists():
        return "  (does not exist yet)"
    lines = []
    for p in sorted(root.rglob("*") if recursive else root.iterdir()):
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(root)
        if p.is_dir():
            if recursive:
                continue
            files = [f for f in p.rglob("*") if f.is_file()]
            newest = max((f.stat().st_mtime for f in files), default=st.st_mtime)
            lines.append(f"  {rel}/  {len(files)} files, newest change {(now - newest) / 60:.0f} min ago")
        else:
            lines.append(f"  {rel}  {st.st_size} bytes, changed {(now - st.st_mtime) / 60:.0f} min ago")
    return "\n".join(lines) or "  (empty)"


class SlotTracker:
    """Infers which slot each agent claimed from the scratch paths in its tool calls (the protocol
    puts an agent's private scratch at <scratch>/work-<slot>). Reads only new stdout bytes."""

    PATTERN = re.compile(r"scratch/work-(\d+)")

    def __init__(self, stdouts: list[Path]):
        self.stdouts = stdouts
        self.offsets = [0] * len(stdouts)
        self.counts: list[dict[int, int]] = [{} for _ in stdouts]

    def update(self) -> list[int | None]:
        for i, path in enumerate(self.stdouts):
            if not path.exists():
                continue
            with open(path, "rb") as f:
                f.seek(self.offsets[i])
                chunk = f.read()
            cut = chunk.rfind(b"\n") + 1  # only whole lines
            self.offsets[i] += cut
            for line in chunk[:cut].splitlines():
                if b'"tool_use"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for b in (e.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        for m in self.PATTERN.finditer(json.dumps(b.get("input"))):
                            s = int(m.group(1))
                            self.counts[i][s] = self.counts[i].get(s, 0) + 1
        return [max(c, key=c.get) if c else None for c in self.counts]


def parse_reminders(out: str, k: int, slots: list[int | None] | None = None) -> dict[int, str]:
    """Reminders from a `claude -p --output-format json --json-schema` result, keyed by agent index.
    A reminder addressed to a slot goes to the agent that holds it; one addressed by agent index is
    accepted only for an agent whose slot is unknown."""
    slots = slots if slots is not None else [None] * k
    owner = {s: i for i, s in enumerate(slots) if s is not None}
    try:
        r = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}
    data = r.get("structured_output")
    if not isinstance(data, dict):
        try:
            data = json.loads(r.get("result") or "")
        except (json.JSONDecodeError, TypeError):
            return {}
    msgs: dict[int, str] = {}
    for item in data.get("reminders") or []:
        try:
            text = str(item["message"]).strip()
            if item.get("slot") is not None:
                i = owner.get(int(item["slot"]))
            else:
                i = int(item["agent"])
                i = i if 0 <= i < k and slots[i] is None else None
        except (KeyError, TypeError, ValueError):
            continue
        if i is not None and text:
            msgs[i] = text
    return msgs


class Facilitator:
    def __init__(self, trial, protocol_prompt: str):
        self.trial = trial
        self.st = trial.st
        self.work = trial.work
        self.dir = self.work / "facilitator"
        self.session_id = str(uuid.uuid4())
        self.rounds = 0
        self.sent = 0
        self.failures = 0
        self.task_root = self.work / "task"
        self.shared = self.task_root / "shared"
        self.scratch = self.task_root / "scratch"
        self.system = SYSTEM.format(n=trial.k, protocol=protocol_prompt.strip(), shared=self.shared,
                                    scratch=self.scratch, task_root=self.task_root,
                                    agents_dir=self.work / "agents")
        self.current_slots: list[int | None] = [None] * trial.k
        self.slots = SlotTracker([self.work / "agents" / str(i) / "stdout.jsonl" for i in range(trial.k)])

    def prepare(self) -> None:
        for sub in ("claude_config", "home", "tmp"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        (self.dir / "system_prompt.md").write_text(self.system)

    def _env(self) -> dict[str, str]:
        env = self.trial._agent_env(0)  # same PATH/locale; own home, temp and Claude config
        env.update({"HOME": str(self.dir / "home"), "TMPDIR": str(self.dir / "tmp"),
                    "CLAUDE_CONFIG_DIR": str(self.dir / "claude_config")})
        for key in [k for k in env if k.startswith(("ARC_", "SUBMIT_"))]:
            env.pop(key)  # no task credentials: the facilitator cannot act on the task
        return env

    def _argv(self, prompt: str) -> list[str]:
        st = self.st
        argv = [st.claude_binary, "-p", prompt]
        argv += ["--session-id", self.session_id, "--append-system-prompt", self.system] if self.rounds == 1 \
            else ["--resume", self.session_id]
        argv += ["--safe-mode", "--model", st.facilitator_model or st.model_alias, "--output-format", "json",
                 "--json-schema", json.dumps(SCHEMA), "--tools", TOOLS, "--permission-mode", "dontAsk",
                 "--settings", self.trial.cli._settings()]
        effort = st.facilitator_effort or st.reasoning_effort
        if effort:
            argv += ["--effort", effort]
        return argv

    def _round_prompt(self, now: float) -> str:
        slots = self.current_slots = self.slots.update()
        lines = []
        for i, s in sorted(enumerate(self.trial.task.sessions), key=lambda x: (slots[x[0]] is None, slots[x[0]] or 0)):
            if s.status != "active":
                continue
            who = f"slot {slots[i]}" if slots[i] is not None else f"agent {i} (slot not known yet)"
            own = self.trial.task.progress(i)
            lines.append(f"  {who}" + (f": its own submissions: {own}" if own else ""))
        return ROUND.format(round=self.rounds, elapsed=(now - self.trial.t_start) / 60,
                            remaining=max(0.0, self.trial.deadline - now) / 60, agents="\n".join(lines),
                            shared=self.shared, shared_listing=_listing(self.shared, now, recursive=True),
                            scratch=self.scratch, scratch_listing=_listing(self.scratch, now, recursive=False))

    async def round(self) -> None:
        self.rounds += 1
        now = time.time()
        prompt = self._round_prompt(now)
        rc, out = await self.trial.launcher.run(None, self._argv(prompt), self._env(), str(self.task_root),
                                                timeout=self.st.facilitator_timeout_seconds)
        with open(self.dir / "stdout.jsonl", "a") as f:
            f.write(out if out.endswith("\n") else out + "\n")
        msgs = parse_reminders(out, self.trial.k, self.current_slots) if rc == 0 else {}
        delivered = {}
        for i, text in sorted(msgs.items()):
            if self.trial.task.sessions[i].status == "active":
                delivered[i] = self.trial.send_message(i, REMINDER_PREFIX + text)
                self.sent += 1
        if not msgs:
            self.failures += 1
            log.warning("%s facilitator round %d produced no reminders (rc=%s): %s",
                        self.trial.label, self.rounds, rc, out[-300:])
        with open(self.dir / "rounds.jsonl", "a") as f:
            f.write(json.dumps({"round": self.rounds, "t": round(now - self.trial.t_start, 1), "rc": rc,
                                "slots": {str(i): s for i, s in enumerate(self.current_slots)},
                                "prompt": prompt, "reminders": {str(i): m for i, m in msgs.items()},
                                "delivery": {str(i): d for i, d in delivered.items()},
                                "seconds": round(time.time() - now, 1)}) + "\n")

    async def run(self) -> None:
        st = self.st
        next_t = time.time() + st.facilitator_first_seconds  # agents have just launched
        while True:
            await asyncio.sleep(max(0.0, next_t - time.time()))
            if self.trial.deadline - time.time() < st.facilitator_min_remaining_seconds:
                return
            if not any(s.status == "active" for s in self.trial.task.sessions):
                return
            try:
                await self.round()
            except Exception:  # noqa: BLE001 - a failed round must never end the trial
                self.failures += 1
                log.exception("%s facilitator round %d failed", self.trial.label, self.rounds)
            # a slow round never causes back-to-back rounds
            next_t = max(next_t + st.facilitator_interval_seconds,
                         time.time() + min(60.0, st.facilitator_interval_seconds / 2))

    def summary(self) -> dict:
        usage = self.trial.cli.usage(self.dir, self.dir / "stdout.jsonl")
        usage.pop("timeline", None)
        from .services import estimate_cost
        return {"rounds": self.rounds, "reminders_sent": self.sent, "failed_rounds": self.failures,
                "model": self.st.facilitator_model or self.st.model_alias,
                "effort": self.st.facilitator_effort or self.st.reasoning_effort,
                "interval_seconds": self.st.facilitator_interval_seconds,
                "usage": usage, "estimated_cost_usd": estimate_cost(usage, self.st)}

