"""The coding-agent CLI each agent runs: GitHub Copilot CLI (the paper's harness) or Claude Code.

Both get the same prompt, task tools, workspace and verifiers; only the agent program differs.

Copilot CLI (`--agent-cli copilot`): BYOK against the local proxy, offline otherwise.

Claude Code (`--agent-cli claude-code`), run headless with:
  --safe-mode               no personal customization: CLAUDE.md, skills, plugins, hooks, MCP servers,
                            custom commands/agents are all off (subscription login still works;
                            --bare would also do this but cannot use a subscription)
  CLAUDE_CONFIG_DIR         a fresh per-agent config dir, so nothing is read from ~/.claude; auto-memory
                            and CLAUDE.md loading are also switched off by environment variable
  --disallowedTools         no web access, no messaging other local Claude sessions, no
                            multi-agent workflows, no scheduling or worktrees
  auth                      `subscription`: a long-lived token from `claude setup-token`, given as
                            CLAUDE_CODE_OAUTH_TOKEN (talks to Anthropic directly; usage is read from
                            Claude Code's own transcripts). `api-key`: ANTHROPIC_API_KEY, routed through
                            the local proxy like Copilot (exact per-call accounting, retries).
"""

from __future__ import annotations

import glob
import json
import re
from datetime import datetime
from pathlib import Path

# Claude Code tools that would break the experiment or reach outside the trial
CLAUDE_DISALLOWED_TOOLS = (
    "WebSearch", "WebFetch",                      # closed book
    "SendMessage", "ListAgents",                  # would let separate trials talk to each other
    "Workflow",                                   # fans out many agents at once
    "CronCreate", "CronDelete", "CronList", "ScheduleWakeup",
    "EnterWorktree", "ExitWorktree", "ReportFindings",
    "RemoteTrigger",                              # starts cloud runs outside the trial
)
LIMIT_PATTERN = re.compile(r"hit your (session |weekly |usage )?limit|usage limit|limit reached|resets? at|· resets",
                           re.IGNORECASE)
OVERFLOW_PATTERN = re.compile(r"prompt is too long|context (window|length) (exceeded|limit)", re.IGNORECASE)
AUTH_PATTERN = re.compile(r"not logged in|invalid api key|authentication|unauthorized|oauth token", re.IGNORECASE)


class CopilotCLI:
    name = "copilot"

    def __init__(self, st):
        self.st = st

    def env(self, i: int, agent_dir: Path, proxy_base: str) -> dict[str, str]:
        st = self.st
        env = {
            "COPILOT_PROVIDER_BASE_URL": proxy_base,
            "COPILOT_PROVIDER_TYPE": st.provider,
            "COPILOT_PROVIDER_API_KEY": "ttc-local-proxy",  # the real key never reaches agents
            "COPILOT_MODEL": st.model_alias,
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": str(st.max_prompt_tokens),
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": str(st.max_output_tokens),
            "COPILOT_OFFLINE": "true",
            "COPILOT_AUTO_UPDATE": "false",
            "COPILOT_HOME": str(agent_dir / "copilot_home"),
        }
        if st.provider == "openai":
            env["COPILOT_PROVIDER_WIRE_API"] = "completions"
        return env

    def argv(self, agent_dir: Path, session_id: str, attempt: int, prompt: str) -> list[str]:
        st = self.st
        argv = [st.copilot_binary]
        if attempt == 0:
            argv += ["--session-id", session_id, "-p", prompt]
        else:
            argv += ["--resume", session_id, "-p", st.continue_prompt]
        argv += ["--allow-all", "--no-ask-user", "--no-auto-update", "--no-custom-instructions",
                 "--disable-builtin-mcps", "--output-format", "json",
                 "--log-dir", str(agent_dir / "logs"), "--log-level", st.copilot_log_level,
                 "--usage-output-file", str(agent_dir / f"usage-{attempt:03d}.json")]
        if st.reasoning_effort:
            argv += ["--reasoning-effort", st.reasoning_effort]
        return argv + list(st.copilot_extra_args)

    def version_argv(self) -> list[str]:
        return [self.st.copilot_binary, "--version"]

    def private_dirs(self) -> tuple[str, ...]:
        return ("copilot_home", "home", "tmp", "logs")

    def last_outcome(self, stdout: Path) -> str | None:
        return None


class ClaudeCodeCLI:
    name = "claude-code"

    def __init__(self, st):
        self.st = st

    @property
    def via_proxy(self) -> bool:
        return self.st.claude_auth == "api-key"

    def _settings(self) -> str:
        s: dict = {"cleanupPeriodDays": 3650}
        if self.via_proxy:
            s["apiKeyHelper"] = "echo ttc-local-proxy"  # the proxy swaps in the real key
        return json.dumps(s)

    def env(self, i: int, agent_dir: Path, proxy_base: str) -> dict[str, str]:
        env = {
            "CLAUDE_CONFIG_DIR": str(agent_dir / "claude_config"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_PROJECT_DIR_NAME": "session",  # transcripts in claude_config/projects/session/
            "DISABLE_AUTOUPDATER": "1",
        }
        if self.via_proxy:
            env["ANTHROPIC_BASE_URL"] = proxy_base
        elif self.st.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.st.oauth_token
        return env

    def argv(self, agent_dir: Path, session_id: str, attempt: int, prompt: str) -> list[str]:
        st = self.st
        argv = [st.claude_binary, "-p", prompt if attempt == 0 else st.continue_prompt]
        argv += ["--session-id", session_id] if attempt == 0 else ["--resume", session_id]
        argv += ["--safe-mode", "--model", st.model_alias, "--output-format", "stream-json", "--verbose",
                 "--dangerously-skip-permissions", "--settings", self._settings(),
                 "--disallowedTools", ",".join(CLAUDE_DISALLOWED_TOOLS)]
        if st.reasoning_effort:
            argv += ["--effort", st.reasoning_effort]
        return argv + list(st.claude_extra_args)

    def version_argv(self) -> list[str]:
        return [self.st.claude_binary, "--version"]

    def probe_argv(self) -> list[str]:
        """A tiny real request, used as the startup check for subscription auth."""
        return [self.st.claude_binary, "-p", "Reply with OK.", "--safe-mode", "--model", self.st.model_alias,
                "--max-turns", "1", "--output-format", "json", "--settings", self._settings(),
                "--tools", "", "--no-session-persistence"]

    def private_dirs(self) -> tuple[str, ...]:
        return ("claude_config", "home", "tmp", "logs")

    @staticmethod
    def _results(stdout: Path) -> list[dict]:
        out = []
        if stdout.exists():
            for line in stdout.read_text(errors="replace").splitlines():
                if line.startswith("{") and '"type":"result"' in line.replace(" ", ""):
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def last_outcome(self, stdout: Path) -> str | None:
        """Why the last Claude Code run ended: limit | overflow | auth | error | None (normal)."""
        results = self._results(stdout)
        if not results:
            return None
        r = results[-1]
        if not r.get("is_error"):
            return None
        text = json.dumps(r.get("result") or r.get("errors") or "")
        if LIMIT_PATTERN.search(text):
            return "limit"
        if OVERFLOW_PATTERN.search(text):
            return "overflow"
        if AUTH_PATTERN.search(text):
            return "auth"
        return "error"

    def usage(self, agent_dir: Path, stdout: Path) -> dict:
        """Per-agent token usage from Claude Code's own transcripts (final per-message usage,
        deduplicated by message id, subagents included) plus its reported cost."""
        seen: dict[str, tuple[float, dict]] = {}
        for f in glob.glob(str(agent_dir / "claude_config" / "projects" / "**" / "*.jsonl"), recursive=True):
            for line in Path(f).read_text(errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = r.get("message") if r.get("type") == "assistant" else None
                if not isinstance(m, dict) or not m.get("id") or not m.get("usage"):
                    continue
                try:
                    ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp()
                except (KeyError, ValueError):
                    ts = 0.0
                seen[m["id"]] = (ts, m["usage"])
        out = {"requests": len(seen), "ok": len(seen), "prompt_tokens": 0, "output_tokens": 0,
               "cached_tokens": 0, "cache_write_tokens": 0, "timeline": []}
        for ts, u in sorted(seen.values(), key=lambda x: x[0]):
            read = u.get("cache_read_input_tokens") or 0
            write = u.get("cache_creation_input_tokens") or 0
            out["prompt_tokens"] += (u.get("input_tokens") or 0) + read + write
            out["output_tokens"] += u.get("output_tokens") or 0
            out["cached_tokens"] += read
            out["cache_write_tokens"] += write
            out["timeline"].append((ts, u.get("output_tokens") or 0))
        results = self._results(stdout)
        out["failed"] = sum(1 for r in results if r.get("is_error"))
        out["reported_cost_usd"] = round(sum(r.get("total_cost_usd") or 0 for r in results), 4)
        return out


def make_cli(st):
    return ClaudeCodeCLI(st) if st.agent_cli == "claude-code" else CopilotCLI(st)
