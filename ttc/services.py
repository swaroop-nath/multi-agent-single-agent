"""Local harness services, served from one aiohttp app on 127.0.0.1:

  task routes                     the task's own API (e.g. /arc/..., /poly/...), see ttc/tasks
  /llm/{agent_key}/...            model proxy in front of the configured provider

The proxy exists so that agents never hold the real API key, and so that every model call is
attributed to an agent. Two providers are supported:

  openai     OpenAI-compatible chat completions (e.g. vLLM, SGLang); upstream = OPENAI_BASE_URL
             exactly as given (path prefix included) + the request's tail
  anthropic  the Anthropic Messages API (/v1/messages); upstream = ANTHROPIC_BASE_URL

It also makes Copilot robust to slow or unusual servers:

* answers /models locally
* retries transport errors, 429, 5xx and 529 with exponential backoff, including Anthropic
  overload errors that arrive inside an HTTP 200 stream (responses are buffered, so a retry is
  invisible to Copilot)
* when a call takes long, sends SSE comment keepalives to Copilot (a server may send the whole
  stream at once after minutes, with no bytes before)
* openai: merges a streamed response into a few chunks before passing it on. Responses are
  buffered anyway, and a server that streams one chunk per token (SGLang) would otherwise hand
  Copilot tens of thousands of events at once, which its event loop cannot keep up with. The
  merged stream carries exactly the same text, reasoning, tool calls, finish reason and usage.
* openai: adds max_tokens when Copilot omits it (it always does), maps `developer` -> `system`
* detects context overflow (an error saying the prompt is too long, or an empty HTTP 200 with
  finish_reason "length") and repeated call failures, and reports them per agent
* records per-agent tokens, including cache reads and writes, for cost accounting
* optionally saves every raw model response (incl. thinking) for post-hoc analysis
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable

import aiohttp
from aiohttp import web

HOP_HEADERS = {"host", "content-length", "authorization", "connection", "accept-encoding",
               "transfer-encoding", "keep-alive", "x-api-key"}
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524, 529}
ANTHROPIC_VERSION = "2023-06-01"
# substrings of error bodies that mean "the prompt no longer fits the context window"
OVERFLOW_MARKERS = (b"prompt is too long", b"maximum context length", b"context_length_exceeded",
                    b"context length", b"too many tokens")


@dataclass
class AgentCalls:
    requests: int = 0
    ok: int = 0
    failed: int = 0  # failed after all retries (or a non-retryable error status)
    retries: int = 0
    context_overflows: int = 0
    truncated: int = 0  # stopped at the output limit, with some content
    prompt_tokens: int = 0  # all input tokens, cached or not
    output_tokens: int = 0
    reasoning_tokens: int = 0  # openai only; Anthropic counts thinking inside output_tokens
    cached_tokens: int = 0  # cache reads
    cache_write_tokens: int = 0  # anthropic cache creation
    consecutive_failures: int = 0
    reported_cost_usd: float | None = None  # Claude Code's own cost figure, when it runs the agent
    timeline: list = field(default_factory=list)  # (t, output_tokens) per completed call
    sampling_params: list = field(default_factory=list)  # distinct non-message params Copilot sent

    def summary(self) -> dict:
        d = dict(self.__dict__)
        d.pop("timeline")
        d.pop("consecutive_failures")
        return d


def normalize_usage(u: dict | None) -> dict:
    """OpenAI chat-completions, OpenAI Responses and Anthropic usage blocks -> one schema.

    prompt_tokens counts every input token (uncached + cache reads + cache writes)."""
    if not u:
        return {}
    if "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:  # anthropic
        read = u.get("cache_read_input_tokens") or 0
        write = u.get("cache_creation_input_tokens") or 0
        return {"prompt_tokens": (u.get("input_tokens") or 0) + read + write,
                "output_tokens": u.get("output_tokens") or 0, "reasoning_tokens": 0,
                "cached_tokens": read, "cache_write_tokens": write}
    out = {
        "prompt_tokens": u.get("prompt_tokens", u.get("input_tokens", 0)) or 0,
        "output_tokens": u.get("completion_tokens", u.get("output_tokens", 0)) or 0,
    }
    det = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    out["reasoning_tokens"] = det.get("reasoning_tokens") or u.get("reasoning_tokens") or 0  # SGLang: top level
    pdet = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    out["cached_tokens"] = pdet.get("cached_tokens", 0) or 0
    out["cache_write_tokens"] = 0
    return out


def estimate_cost(t: dict, settings) -> float | None:
    """USD from token totals and the per-million prices given on the command line."""
    prices = (settings.price_input, settings.price_output, settings.price_cache_read, settings.price_cache_write)
    if all(p is None for p in prices):
        return None
    pi, po, pr, pw = (p or 0.0 for p in prices)
    uncached = t.get("prompt_tokens", 0) - t.get("cached_tokens", 0) - t.get("cache_write_tokens", 0)
    return round((uncached * pi + t.get("output_tokens", 0) * po + t.get("cached_tokens", 0) * pr
                  + t.get("cache_write_tokens", 0) * pw) / 1e6, 4)


@dataclass
class Completion:
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    has_content: bool = False
    has_tool_calls: bool = False
    overflow_error: bool = False  # the server rejected the prompt as too long
    stream_error: str | None = None  # anthropic: an `error` event inside a 200 stream
    complete: bool = False  # anthropic: saw message_stop / a full message

    @property
    def is_overflow(self) -> bool:
        if self.overflow_error:
            return True
        return self.finish_reason == "length" and not self.has_content and not self.has_tool_calls

    @property
    def is_truncated(self) -> bool:
        return self.finish_reason in ("length", "max_tokens") and not self.is_overflow


def _sse_payloads(body: bytes):
    for line in body.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            data = line[5:].strip()
            if data and data != b"[DONE]":
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj


def _inspect_openai(objs) -> Completion:
    c = Completion()
    for obj in objs:
        if obj.get("usage"):
            c.usage = obj["usage"]
        for ch in obj.get("choices") or []:
            if ch.get("finish_reason"):
                c.finish_reason = ch["finish_reason"]
            part = ch.get("delta") or ch.get("message") or {}
            if any(isinstance(part.get(f), str) and part[f].strip() for f in ("content", "reasoning_content", "reasoning")):
                c.has_content = True
            if part.get("tool_calls"):
                c.has_tool_calls = True
    return c


def _inspect_anthropic(objs) -> Completion:
    c = Completion()
    usage: dict = {}
    for obj in objs:
        t = obj.get("type")
        if t == "message":  # non-streaming response
            usage.update(obj.get("usage") or {})
            c.finish_reason = obj.get("stop_reason")
            for b in obj.get("content") or []:
                if b.get("type") == "tool_use":
                    c.has_tool_calls = True
                elif (b.get("text") or b.get("thinking") or "").strip():
                    c.has_content = True
            c.complete = True
        elif t == "message_start":
            usage.update((obj.get("message") or {}).get("usage") or {})
        elif t == "content_block_start":
            b = obj.get("content_block") or {}
            if b.get("type") == "tool_use":
                c.has_tool_calls = True
            elif (b.get("text") or b.get("thinking") or "").strip():
                c.has_content = True
        elif t == "content_block_delta":
            d = obj.get("delta") or {}
            if d.get("type") == "input_json_delta":
                c.has_tool_calls = True
            elif (d.get("text") or d.get("thinking") or "").strip():
                c.has_content = True
        elif t == "message_delta":
            usage.update({k: v for k, v in (obj.get("usage") or {}).items() if v is not None})
            stop = (obj.get("delta") or {}).get("stop_reason")
            if stop:
                c.finish_reason = stop
        elif t == "message_stop":
            c.complete = True
        elif t == "error":
            c.stream_error = json.dumps(obj.get("error") or obj)[:500]
    c.usage = usage
    return c


def inspect_response(body: bytes, is_sse: bool, provider: str = "openai") -> Completion:
    if is_sse:
        objs = list(_sse_payloads(body))
    else:
        try:
            obj = json.loads(body)
            objs = [obj] if isinstance(obj, dict) else []
        except json.JSONDecodeError:
            objs = []
    return _inspect_anthropic(objs) if provider == "anthropic" else _inspect_openai(objs)


def coalesce_openai_sse(body: bytes) -> bytes | None:
    """Merge a complete chat-completions SSE stream into one delta per choice (plus the usage
    chunk), with the same content, reasoning, tool calls, finish reason and usage. Returns None
    when the stream has anything unexpected (non-JSON data, an error object), so the caller can
    pass the original through untouched."""
    first: dict | None = None
    choices: dict[int, dict] = {}  # index -> {"delta": {...}, "tool_calls": {i: call}, "finish_reason": ...}
    usage = None
    for line in body.split(b"\n"):
        line = line.strip()
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            continue  # event:/id:/retry: lines carry nothing Copilot needs here
        data = line[5:].strip()
        if data == b"[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "error" in obj:
            return None
        if first is None:
            first = obj
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            c = choices.setdefault(ch.get("index", 0), {"delta": {}, "tool_calls": {}, "finish_reason": None})
            if ch.get("finish_reason"):
                c["finish_reason"] = ch["finish_reason"]
            for k, v in (ch.get("delta") or {}).items():
                if k == "tool_calls":
                    for tc in v or []:
                        slot = c["tool_calls"].setdefault(tc.get("index", 0), {"index": tc.get("index", 0)})
                        for f in ("id", "type"):
                            if tc.get(f) and not slot.get(f):
                                slot[f] = tc[f]
                        fn = tc.get("function") or {}
                        sfn = slot.setdefault("function", {})
                        if fn.get("name") and not sfn.get("name"):
                            sfn["name"] = fn["name"]
                        if fn.get("arguments"):
                            sfn["arguments"] = sfn.get("arguments", "") + fn["arguments"]
                elif isinstance(v, str) and k != "role":
                    c["delta"][k] = c["delta"].get(k, "") + v
                elif v is not None and k not in c["delta"]:
                    c["delta"][k] = v
    if first is None:
        return None
    head = {k: first[k] for k in ("id", "object", "created", "model", "system_fingerprint") if k in first}
    out = []
    for idx in sorted(choices):
        c = choices[idx]
        delta = {"role": "assistant", **{k: v for k, v in c["delta"].items() if k != "role"}}
        if c["tool_calls"]:
            calls = [c["tool_calls"][i] for i in sorted(c["tool_calls"])]
            for call in calls:
                call.setdefault("function", {}).setdefault("arguments", "")
            delta["tool_calls"] = calls
        out.append({**head, "choices": [{"index": idx, "delta": delta, "finish_reason": None}]})
        out.append({**head, "choices": [{"index": idx, "delta": {}, "finish_reason": c["finish_reason"]}]})
    if usage is not None:
        out.append({**head, "choices": [], "usage": usage})
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in out) + "data: [DONE]\n\n").encode()


def is_overflow_error(status: int, body: bytes) -> bool:
    return status in (400, 413) and any(m in body.lower() for m in OVERFLOW_MARKERS)


@dataclass
class Upstream:
    status: int
    content_type: str
    body: bytes
    error: str | None = None


class Services:
    def __init__(self, settings, model_log_path, on_overflow: Callable[[str], None] | None = None,
                 on_model_failures: Callable[[str, int], None] | None = None, responses_path=None):
        self.s = settings
        self.responses_path = responses_path
        self._responses = None
        self.anthropic = getattr(settings, "provider", "openai") == "anthropic"
        self.model_log_path = model_log_path
        self.on_overflow = on_overflow or (lambda key: None)
        self.on_model_failures = on_model_failures or (lambda key, n: None)
        self.calls: dict[str, AgentCalls] = {}
        self._session: aiohttp.ClientSession | None = None
        self._runner: web.AppRunner | None = None
        self._log = None

    # ---- lifecycle ------------------------------------------------------------------------------
    async def start(self, add_routes=None) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.s.upstream_timeout_seconds, sock_connect=60))
        self._log = open(self.model_log_path, "a")
        if self.responses_path:
            self._responses = open(self.responses_path, "a")
        app = web.Application(client_max_size=256 * 1024 * 1024)
        if add_routes:
            add_routes(app)
        app.router.add_route("*", "/llm/{key}/{tail:.*}", self.llm_proxy)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.s.host, self.s.port).start()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        if self._session:
            await self._session.close()
        if self._log:
            self._log.close()
        if self._responses:
            self._responses.close()

    def register(self, agent_keys: list[str]) -> None:
        for key in agent_keys:
            self.calls[key] = AgentCalls()

    # ---- upstream ---------------------------------------------------------------------------------------
    def upstream_url(self, tail: str) -> str:
        return self.s.base_url.rstrip("/") + "/" + tail.lstrip("/")

    def upstream_headers(self, incoming: dict | None = None) -> dict:
        headers = {k: v for k, v in (incoming or {}).items() if k.lower() not in HOP_HEADERS}
        headers["Accept-Encoding"] = "identity"
        if self.anthropic:
            if self.s.api_key:
                headers["x-api-key"] = self.s.api_key
            if not any(k.lower() == "anthropic-version" for k in headers):
                headers["anthropic-version"] = ANTHROPIC_VERSION
        elif self.s.api_key:
            headers["Authorization"] = f"Bearer {self.s.api_key}"
        return headers

    def completion_path(self) -> str:
        return "v1/messages" if self.anthropic else "chat/completions"

    def is_completion(self, tail: str) -> bool:
        tail = tail.rstrip("/")
        if self.anthropic:
            return tail == "messages" or tail.endswith("/messages")
        return tail.endswith("chat/completions")

    async def probe(self) -> tuple[Upstream, int]:
        """A minimal 1-token completion, used as the startup reachability check."""
        body = {"model": self.s.model_alias, "max_tokens": 1,
                "messages": [{"role": "user", "content": "Reply with OK."}]}
        if not self.anthropic:
            body["stream"] = False
        return await self.fetch("POST", self.upstream_url(self.completion_path()), json.dumps(body).encode(),
                                {**self.upstream_headers(), "Content-Type": "application/json"})

    def _rewrite(self, key: str, payload: dict) -> dict:
        if not self.anthropic:
            for m in payload.get("messages") or []:
                if m.get("role") == "developer":
                    m["role"] = "system"
            if self.s.inject_max_tokens and "max_tokens" not in payload and "max_completion_tokens" not in payload:
                payload["max_tokens"] = self.s.max_output_tokens
            if payload.get("stream"):
                payload.setdefault("stream_options", {})["include_usage"] = True
        params = {k: v for k, v in payload.items() if k not in ("messages", "tools", "system")}
        calls = self.calls[key]
        if params not in calls.sampling_params and len(calls.sampling_params) < 20:
            calls.sampling_params.append(params)
        return payload

    def _stream_failed(self, up: Upstream) -> bool:
        """Anthropic can report overload as an `error` event inside a 200 stream."""
        if not self.anthropic or up.status != 200 or "text/event-stream" not in up.content_type:
            return False
        c = inspect_response(up.body, True, "anthropic")
        return c.stream_error is not None and not c.complete

    async def fetch(self, method: str, url: str, body: bytes | None, headers: dict) -> tuple[Upstream, int]:
        """Request with retries; returns (response, retries used). Never raises."""
        assert self._session is not None
        retries = 0
        while True:
            try:
                async with self._session.request(method, url, data=body, headers=headers) as r:
                    data = await r.read()
                    up = Upstream(r.status, r.headers.get("Content-Type", ""), data)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                up = Upstream(599, "text/plain", b"", error=f"{type(e).__name__}: {e}")
            retryable = up.status in RETRY_STATUS or self._stream_failed(up)
            if not retryable or retries >= self.s.model_retries:
                return up, retries
            retries += 1
            delay = min(self.s.model_backoff_max_seconds, self.s.model_backoff_seconds * 2 ** (retries - 1))
            await asyncio.sleep(delay)

    # ---- proxy ------------------------------------------------------------------------------------------------
    def _models_response(self) -> web.Response:
        if self.anthropic:
            return web.json_response({"data": [{"id": self.s.model_alias, "type": "model",
                                                "display_name": self.s.model_alias}],
                                      "has_more": False, "first_id": self.s.model_alias,
                                      "last_id": self.s.model_alias})
        return web.json_response({"object": "list", "data": [
            {"id": self.s.model_alias, "object": "model", "owned_by": "ttc"}]})

    def _stream_error_event(self, message: str, status: int) -> bytes:
        if self.anthropic:
            err = {"type": "error", "error": {"type": "api_error", "message": message}}
            return f"event: error\ndata: {json.dumps(err)}\n\n".encode()
        err = {"error": {"message": message, "type": "upstream_error", "code": status}}
        return f"data: {json.dumps(err)}\n\n".encode()

    async def llm_proxy(self, request: web.Request) -> web.StreamResponse:
        key = request.match_info["key"]
        if key not in self.calls:
            raise web.HTTPForbidden(text="unknown agent key")
        tail = request.match_info["tail"]
        if tail.rstrip("/").endswith("models"):
            return self._models_response()

        body = await request.read()
        payload = None
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
        is_chat = self.is_completion(tail) and isinstance(payload, dict)
        streaming = bool(is_chat and payload.get("stream"))
        if is_chat:
            payload = self._rewrite(key, payload)
            body = json.dumps(payload).encode()
        url = self.upstream_url(tail) + (f"?{request.query_string}" if request.query_string else "")
        headers = self.upstream_headers(dict(request.headers))
        t0 = time.time()

        task = asyncio.ensure_future(self.fetch(request.method, url, body or None, headers))
        resp: web.StreamResponse | None = None
        wait = self.s.sse_keepalive_after_seconds
        if streaming and wait > 0:
            done, _ = await asyncio.wait({task}, timeout=wait)
            if not done:  # slow call: start the SSE stream now and keep Copilot's connection alive
                resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                               "Cache-Control": "no-cache"})
                await resp.prepare(request)
                while not task.done():
                    try:
                        await resp.write(b": keepalive\n\n")
                    except (ConnectionResetError, RuntimeError):
                        break
                    await asyncio.wait({task}, timeout=self.s.sse_keepalive_interval_seconds)
        up, retries = await task

        ok = 200 <= up.status < 300 and not self._stream_failed(up)
        is_sse = "text/event-stream" in up.content_type
        provider = "anthropic" if self.anthropic else "openai"
        comp = inspect_response(up.body, is_sse, provider) if (ok and is_chat) else Completion()
        if is_chat and not ok and is_overflow_error(up.status, up.body):
            comp.overflow_error = True
        self._account(key, t0, tail, up, retries, comp, is_chat, ok)
        if is_chat and self._responses:
            self._responses.write(json.dumps({
                "t": round(time.time(), 3), "agent": key.rsplit(".", 1)[-1], "status": up.status,
                "retries": retries, "latency_s": round(time.time() - t0, 3),
                "content_type": up.content_type, "body": up.body.decode(errors="replace")}) + "\n")
            self._responses.flush()

        if resp is None:
            if ok and is_sse:
                return web.Response(status=up.status, body=self._for_client(up.body),
                                    headers={"Content-Type": up.content_type})
            if not up.body and up.status >= 400:
                return web.Response(status=502 if up.status == 599 else up.status,
                                    text=f"upstream error: {up.error or up.status}")
            return web.Response(status=up.status, body=up.body,
                                headers={"Content-Type": up.content_type or "application/octet-stream"})
        # headers already sent: deliver the body, or an in-stream error the client can see
        try:
            if ok and is_sse:
                await resp.write(self._for_client(up.body))
            elif ok and not self.anthropic:  # upstream ignored stream=true; wrap the JSON as one event
                await resp.write(b"data: " + up.body.strip() + b"\n\ndata: [DONE]\n\n")
            else:
                detail = up.error or up.body[:500].decode(errors="replace")
                await resp.write(self._stream_error_event(f"upstream error {up.status}: {detail}", up.status))
            await resp.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass
        return resp

    def _for_client(self, body: bytes) -> bytes:
        """The SSE body to hand the agent CLI: merged into a few chunks (openai), else as received."""
        if self.anthropic or not getattr(self.s, "coalesce_stream", True):
            return body
        merged = coalesce_openai_sse(body)
        return body if merged is None else merged

    def _account(self, key: str, t0: float, tail: str, up: Upstream, retries: int, comp: Completion,
                 is_chat: bool, ok: bool) -> None:
        a = self.calls.get(key)
        if a is None or not is_chat:
            return
        u = normalize_usage(comp.usage)
        a.requests += 1
        a.retries += retries
        if ok:
            a.ok += 1
            a.consecutive_failures = 0
            a.prompt_tokens += u.get("prompt_tokens", 0)
            a.output_tokens += u.get("output_tokens", 0)
            a.reasoning_tokens += u.get("reasoning_tokens", 0)
            a.cached_tokens += u.get("cached_tokens", 0)
            a.cache_write_tokens += u.get("cache_write_tokens", 0)
            a.timeline.append((time.time(), u.get("output_tokens", 0)))
            if comp.is_overflow:
                a.context_overflows += 1
            elif comp.is_truncated:
                a.truncated += 1
        elif comp.is_overflow:
            a.context_overflows += 1
        else:
            a.failed += 1
            a.consecutive_failures += 1
        if self._log:
            self._log.write(json.dumps({
                "t": round(time.time(), 3), "agent": key.rsplit(".", 1)[-1], "path": tail,
                "status": up.status, "error": up.error, "retries": retries,
                "latency_s": round(time.time() - t0, 3), "finish_reason": comp.finish_reason,
                "overflow": comp.is_overflow, **u}) + "\n")
            self._log.flush()
        if comp.is_overflow:
            self.on_overflow(key)
        elif not ok:
            self.on_model_failures(key, a.consecutive_failures)
