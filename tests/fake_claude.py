#!/usr/bin/env python3
"""A stand-in for the `claude` binary, for testing the facilitator end to end without a model.

  claude --version                         prints a version
  claude -p ... --max-turns 1              (startup probe) prints a successful JSON result
  claude -p PROMPT ... --json-schema S     (facilitator round) prints reminders for agents 0 and 1
  claude -p --input-format stream-json ... (agent) reads user messages from stdin, works in its slot's
                                           scratch dir, records every message it receives, and keeps
                                           its turn going until killed

Each call writes one assistant message with usage into CLAUDE_CONFIG_DIR, like Claude Code's own
transcripts, so the harness counts a successful model call.
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path


def transcript(text: str) -> None:
    d = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "session"
    d.mkdir(parents=True, exist_ok=True)
    msg = {"type": "assistant", "timestamp": "2026-10-04T12:00:00Z",
           "message": {"id": f"msg_{uuid.uuid4().hex}", "role": "assistant", "content": [{"type": "text", "text": text}],
                       "usage": {"input_tokens": 10, "output_tokens": 5}}}
    with open(d / "fake.jsonl", "a") as f:
        f.write(json.dumps(msg) + "\n")


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def facilitator(prompt: str) -> None:
    transcript("reminders")
    reminders = [{"slot": int(s), "message": f"log your attempts (slot {s})"}
                 for s in re.findall(r"^  slot (\d+)", prompt, re.M)]
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.0,
                      "structured_output": {"reminders": reminders}}))


def agent() -> None:
    home = Path(os.environ["HOME"])
    agent_index = Path(os.environ["CLAUDE_CONFIG_DIR"]).parent.name
    received = home / "received.jsonl"
    msgs: list[str] = []

    def reader():
        for line in sys.stdin:
            try:
                content = json.loads(line)["message"]["content"]
            except (json.JSONDecodeError, KeyError):
                continue
            text = "".join(c.get("text", "") for c in content)
            msgs.append(text)
            with open(received, "a") as f:
                f.write(json.dumps({"t": time.time(), "text": text[:200]}) + "\n")

    threading.Thread(target=reader, daemon=True).start()
    emit({"type": "system", "subtype": "init"})
    transcript("working")
    scratch = Path.cwd() / "scratch" / f"work-{1 - int(agent_index)}"  # slots claimed in reverse order
    emit({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": f"mkdir -p {scratch}"}}]}})
    scratch.mkdir(parents=True, exist_ok=True)
    while True:  # one long turn, like the real agents
        time.sleep(0.5)


def main() -> None:
    argv = sys.argv[1:]
    if "--version" in argv:
        print("2.1.280 (Claude Code)")
    elif "--input-format" in argv:
        agent()
    elif "--json-schema" in argv:
        facilitator(argv[argv.index("-p") + 1])
    else:
        transcript("OK")
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "OK"}))


if __name__ == "__main__":
    main()
