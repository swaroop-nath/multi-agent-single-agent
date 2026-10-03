"""Scripted OpenAI-compatible stub endpoint for testing the harness without a GPU.

Each conversation runs the commands in --script, one shell tool call per model turn, then
gives a final text answer. Behaviors of slow or unusual servers can be switched on:

  --first-byte-delay S   hold every response for S seconds, then send it all at once
                         (streaming is not incremental; no keepalive bytes)
  --overflow-after N     after N chat calls, answer with HTTP 200, empty content and
                         finish_reason "length" (context overflow)
  --fail-every N         every Nth chat call returns HTTP 503
  --no-models            404 on /models (only chat completions is guaranteed)

Every request (method, path, body) is appended to --dump.

    python tests/mock_llm.py --port 8901 --script "arc info" "arc act 1" --dump /tmp/reqs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from aiohttp import web

SHELL_HINTS = ("bash", "shell", "powershell", "run_command", "exec")


def pick_shell_tool(tools):
    for t in tools or []:
        fn = t.get("function", t)
        if any(h in fn.get("name", "").lower() for h in SHELL_HINTS):
            return fn
    return None


def fill_args(fn, command):
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    args = {}
    cmd_key = next((k for k in ("command", "cmd", "script", "input") if k in props), None)
    for name in params.get("required", []):
        spec = props.get(name, {})
        if name == cmd_key:
            continue
        if "enum" in spec:
            args[name] = spec["enum"][0]
        elif spec.get("type") == "boolean":
            args[name] = False
        elif spec.get("type") in ("integer", "number"):
            args[name] = 60
        else:
            args[name] = "harness test step"
    args[cmd_key or "command"] = command
    return args


class Mock:
    def __init__(self, a):
        self.a = a
        self.dump = open(a.dump, "a") if a.dump else None
        self.calls = 0

    def _record(self, request, body):
        if self.dump:
            self.dump.write(json.dumps({"t": time.time(), "method": request.method, "path": request.path,
                                        "auth": request.headers.get("Authorization"), "body": body}) + "\n")
            self.dump.flush()

    def _decide(self, body):
        msgs = body.get("messages", [])
        fn = pick_shell_tool(body.get("tools"))
        done = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
        if fn is None or done >= len(self.a.script):
            return {"content": "Finished the scripted steps."}, []
        call = {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                "function": {"name": fn["name"], "arguments": json.dumps(fill_args(fn, self.a.script[done]))}}
        return {"content": None, "tool_calls": [call]}, [call]

    async def models(self, request):
        self._record(request, None)
        if self.a.no_models:
            raise web.HTTPNotFound()
        return web.json_response({"object": "list", "data": [{"id": self.a.model, "object": "model"}]})

    async def chat(self, request):
        body = await request.json()
        self._record(request, body)
        self.calls += 1
        if self.a.fail_every and self.calls % self.a.fail_every == 0:
            raise web.HTTPServiceUnavailable(text="stub: transient failure")
        if self.a.first_byte_delay:
            await asyncio.sleep(self.a.first_byte_delay)
        overflow = self.a.overflow_after is not None and self.calls > self.a.overflow_after
        if overflow:
            msg, calls, finish = {"content": ""}, [], "length"
        else:
            msg, calls = self._decide(body)
            finish = "tool_calls" if calls else "stop"
        usage = {"prompt_tokens": 100 + 10 * len(body.get("messages", [])), "completion_tokens": 7,
                 "total_tokens": 0}
        rid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        if not body.get("stream"):
            return web.json_response({"id": rid, "object": "chat.completion", "created": int(time.time()),
                                      "model": self.a.model, "usage": usage,
                                      "choices": [{"index": 0, "finish_reason": finish,
                                                   "message": {"role": "assistant", **msg}}]})

        def chunk(delta, finish_reason=None):
            return {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": self.a.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

        # Everything in one write: non-incremental streaming, all tool calls in a single delta.
        events = [chunk({"role": "assistant"})]
        if calls:
            events.append(chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]}))
        else:
            events.append(chunk({"content": msg["content"]}))
        events.append(chunk({}, finish))
        if (body.get("stream_options") or {}).get("include_usage"):
            events.append({"id": rid, "object": "chat.completion.chunk", "choices": [], "usage": usage})
        payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return web.Response(body=payload.encode(), headers={"Content-Type": "text/event-stream"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--model", default="mock-model")
    p.add_argument("--script", nargs="+", default=["arc info"])
    p.add_argument("--dump")
    p.add_argument("--first-byte-delay", type=float, default=0)
    p.add_argument("--overflow-after", type=int, default=None)
    p.add_argument("--fail-every", type=int, default=0)
    p.add_argument("--no-models", action="store_true")
    a = p.parse_args()
    m = Mock(a)
    app = web.Application()
    app.router.add_get("/v1/models", m.models)
    app.router.add_post("/v1/chat/completions", m.chat)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
