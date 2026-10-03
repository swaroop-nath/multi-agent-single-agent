"""Local harness services, served from one aiohttp app on 127.0.0.1:

  task routes                     the task's own API (e.g. /arc/..., /poly/...), see ttc/tasks
  /llm/{agent_key}/...            OpenAI-compatible proxy in front of OPENAI_BASE_URL

The proxy exists so that agents never hold the real API key, and so that every model call is
attributed to an agent. It also makes Copilot robust to slow or unusual OpenAI-compatible servers:

* forwards to OPENAI_BASE_URL exactly as given (path prefix included) + the request's tail
* answers /models locally (the endpoint only guarantees chat completions)
* retries transport errors, 429 and 5xx with exponential backoff
* when a call takes long, sends SSE comment keepalives to Copilot (the endpoint may send the
  whole stream at once after minutes, with no bytes before)
* adds max_tokens when Copilot omits it (it always does), maps `developer` -> `system`
* detects context overflow (HTTP 200, empty content, finish_reason "length") and repeated
  call failures, and reports them per agent
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


@dataclass
class AgentCalls:
    requests: int = 0
    ok: int = 0
    failed: int = 0  # failed after all retries (or a non-retryable error status)
    retries: int = 0
    context_overflows: int = 0
    truncated: int = 0  # finish_reason "length" with some content
    prompt_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    consecutive_failures: int = 0
    timeline: list = field(default_factory=list)  # (t, output_tokens) per completed call
    sampling_params: list = field(default_factory=list)  # distinct non-message params Copilot sent

    def summary(self) -> dict:
        d = dict(self.__dict__)
        d.pop("timeline")
        d.pop("consecutive_failures")
        return d


def normalize_usage(u: dict | None) -> dict:
    """Chat-completions and Responses usage blocks -> one schema."""
    if not u:
        return {}
    out = {
        "prompt_tokens": u.get("prompt_tokens", u.get("input_tokens", 0)) or 0,
        "output_tokens": u.get("completion_tokens", u.get("output_tokens", 0)) or 0,
    }
    det = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    out["reasoning_tokens"] = det.get("reasoning_tokens", 0) or 0
    pdet = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    out["cached_tokens"] = pdet.get("cached_tokens", 0) or 0
    return out


@dataclass
class Completion:
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    has_content: bool = False
    has_tool_calls: bool = False

    @property
    def is_overflow(self) -> bool:
        return self.finish_reason == "length" and not self.has_content and not self.has_tool_calls


def inspect_response(body: bytes, is_sse: bool) -> Completion:
    c = Completion()

    def take(obj: dict) -> None:
        if obj.get("usage"):
            c.usage = obj["usage"]
        for ch in obj.get("choices") or []:
            if ch.get("finish_reason"):
                c.finish_reason = ch["finish_reason"]
            part = ch.get("delta") or ch.get("message") or {}
            if (part.get("content") or "").strip() or (part.get("reasoning_content") or "").strip():
                c.has_content = True
            if part.get("tool_calls"):
                c.has_tool_calls = True

    if is_sse:
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
                        take(obj)
    else:
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                take(obj)
        except json.JSONDecodeError:
            pass
    return c


@dataclass
class Upstream:
    status: int
    content_type: str
    body: bytes
    error: str | None = None


class Services:
    def __init__(self, settings, model_log_path, on_overflow: Callable[[str], None] | None = None,
                 on_model_failures: Callable[[str, int], None] | None = None):
        self.s = settings
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

    def register(self, agent_keys: list[str]) -> None:
        for key in agent_keys:
            self.calls[key] = AgentCalls()

    # ---- model proxy ----------------------------------------------------------------------------------
    def upstream_url(self, tail: str) -> str:
        return self.s.base_url.rstrip("/") + "/" + tail.lstrip("/")

    def upstream_headers(self, incoming: dict | None = None) -> dict:
        headers = {k: v for k, v in (incoming or {}).items() if k.lower() not in HOP_HEADERS}
        headers["Accept-Encoding"] = "identity"
        if self.s.api_key:
            headers["Authorization"] = f"Bearer {self.s.api_key}"
        return headers

    def _rewrite(self, key: str, payload: dict) -> dict:
        for m in payload.get("messages") or []:
            if m.get("role") == "developer":
                m["role"] = "system"
        if self.s.inject_max_tokens and "max_tokens" not in payload and "max_completion_tokens" not in payload:
            payload["max_tokens"] = self.s.max_output_tokens
        if payload.get("stream"):
            payload.setdefault("stream_options", {})["include_usage"] = True
        params = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
        calls = self.calls[key]
        if params not in calls.sampling_params and len(calls.sampling_params) < 20:
            calls.sampling_params.append(params)
        return payload

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
            if up.status not in RETRY_STATUS or retries >= self.s.model_retries:
                return up, retries
            retries += 1
            delay = min(self.s.model_backoff_max_seconds, self.s.model_backoff_seconds * 2 ** (retries - 1))
            await asyncio.sleep(delay)

    async def llm_proxy(self, request: web.Request) -> web.StreamResponse:
        key = request.match_info["key"]
        if key not in self.calls:
            raise web.HTTPForbidden(text="unknown agent key")
        tail = request.match_info["tail"]
        if tail.rstrip("/").endswith("models"):
            return web.json_response({"object": "list", "data": [
                {"id": self.s.model_alias, "object": "model", "owned_by": "ttc"}]})

        body = await request.read()
        payload = None
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
        is_chat = tail.endswith("chat/completions") and isinstance(payload, dict)
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

        ok = 200 <= up.status < 300
        is_sse = "text/event-stream" in up.content_type
        comp = inspect_response(up.body, is_sse) if (ok and is_chat) else Completion()
        self._account(key, t0, tail, up, retries, comp, is_chat)

        if resp is None:
            if not ok and not up.body:
                return web.Response(status=502 if up.status == 599 else up.status,
                                    text=f"upstream error: {up.error or up.status}")
            return web.Response(status=up.status, body=up.body,
                                headers={"Content-Type": up.content_type or "application/octet-stream"})
        # headers already sent: deliver the body, or an in-stream error the client can see
        try:
            if ok and is_sse:
                await resp.write(up.body)
            elif ok:  # upstream ignored stream=true; wrap the JSON completion as one SSE event
                await resp.write(b"data: " + up.body.strip() + b"\n\ndata: [DONE]\n\n")
            else:
                err = {"error": {"message": f"upstream error {up.status}: {up.error or up.body[:500].decode(errors='replace')}",
                                 "type": "upstream_error", "code": up.status}}
                await resp.write(f"data: {json.dumps(err)}\n\n".encode())
            await resp.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass
        return resp

    def _account(self, key: str, t0: float, tail: str, up: Upstream, retries: int, comp: Completion,
                 is_chat: bool) -> None:
        a = self.calls.get(key)
        if a is None or not is_chat:
            return
        ok = 200 <= up.status < 300
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
            a.timeline.append((time.time(), u.get("output_tokens", 0)))
            if comp.is_overflow:
                a.context_overflows += 1
            elif comp.finish_reason == "length":
                a.truncated += 1
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
        if ok and comp.is_overflow:
            self.on_overflow(key)
        elif not ok:
            self.on_model_failures(key, a.consecutive_failures)
