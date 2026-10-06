import json

from ttc.services import inspect_response, normalize_usage


def sse(*events):
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def chunk(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def test_overflow_is_empty_length_response():
    body = sse(chunk({"role": "assistant"}), chunk({"content": ""}, "length"),
               {"choices": [], "usage": {"prompt_tokens": 130000, "completion_tokens": 0}})
    c = inspect_response(body, is_sse=True)
    assert c.is_overflow and c.usage["prompt_tokens"] == 130000


def test_truncated_output_is_not_overflow():
    c = inspect_response(sse(chunk({"content": "partial answer"}, "length")), is_sse=True)
    assert c.finish_reason == "length" and not c.is_overflow


def test_single_delta_tool_calls():
    call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    c = inspect_response(sse(chunk({"tool_calls": [call]}), chunk({}, "tool_calls")), is_sse=True)
    assert c.has_tool_calls and not c.is_overflow


def test_non_streaming_json_and_keepalive_comments_ignored():
    body = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": None}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 0}}).encode()
    assert inspect_response(body, is_sse=False).is_overflow
    assert not inspect_response(b": keepalive\n\n" + sse(chunk({"content": "hi"}, "stop")), is_sse=True).is_overflow


def test_normalize_usage_both_apis():
    chat = {"prompt_tokens": 10, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 4}}
    assert normalize_usage(chat) == {"prompt_tokens": 10, "output_tokens": 3, "reasoning_tokens": 2, "cached_tokens": 4,
                                     "cache_write_tokens": 0}
    resp = {"input_tokens": 7, "output_tokens": 1}
    assert normalize_usage(resp)["prompt_tokens"] == 7 and normalize_usage(None) == {}


# ---- Anthropic Messages API -------------------------------------------------------------------------

from types import SimpleNamespace

from ttc.services import estimate_cost, is_overflow_error


def anth_sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_anthropic_stream_with_tool_use_and_cache_usage():
    body = anth_sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 3, "cache_creation_input_tokens": 200,
                                                        "cache_read_input_tokens": 5000, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "bash",
                                                                      "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}},
        {"type": "message_stop"})
    c = inspect_response(body, is_sse=True, provider="anthropic")
    assert c.has_tool_calls and c.complete and c.finish_reason == "tool_use" and not c.is_overflow
    assert normalize_usage(c.usage) == {"prompt_tokens": 5203, "output_tokens": 42, "reasoning_tokens": 0,
                                        "cached_tokens": 5000, "cache_write_tokens": 200}


def test_anthropic_in_stream_error_and_truncation():
    err = inspect_response(anth_sse({"type": "error", "error": {"type": "overloaded_error"}}), True, "anthropic")
    assert err.stream_error and not err.complete
    trunc = inspect_response(anth_sse(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"}), True, "anthropic")
    assert trunc.is_truncated and not trunc.is_overflow


def test_overflow_errors_from_real_servers():
    assert is_overflow_error(400, b'{"error":{"message":"prompt is too long: 210000 tokens > 200000 maximum"}}')
    assert is_overflow_error(400, b"This model's maximum context length is 131072 tokens")
    assert not is_overflow_error(400, b'{"error":{"message":"invalid tool schema"}}')
    assert not is_overflow_error(500, b"prompt is too long")


def test_cost_matches_a_real_sonnet_4_6_run():
    # A collaborator's worker: Sonnet 4.6 prices, 1-hour cache writes ($6/M); reported $18.33.
    prices = SimpleNamespace(price_input=3, price_output=15, price_cache_read=0.3, price_cache_write=6)
    totals = {"prompt_tokens": 8821 + 24170009 + 620790, "output_tokens": 489042,
              "cached_tokens": 24170009, "cache_write_tokens": 620790}
    assert abs(estimate_cost(totals, prices) - 18.33) < 0.01
    none = SimpleNamespace(price_input=None, price_output=None, price_cache_read=None, price_cache_write=None)
    assert estimate_cost(totals, none) is None


# ---- merging token-by-token streams (SGLang sends one chunk per token) ------------------------------

from ttc.services import coalesce_openai_sse


def _per_token_stream():
    head = {"id": "r1", "object": "chat.completion.chunk", "created": 1, "model": "glm"}
    events = [{**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}]
    for tok in ["Let", " me", " think", "."] * 500:
        events.append({**head, "choices": [{"index": 0, "delta": {"reasoning_content": tok}, "finish_reason": None}]})
    for tok in ["I'll", " look"]:
        events.append({**head, "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]})
    calls = [(0, "c1", "bash", ['{"cmd"', ': "ls"}']), (1, "c2", "bash", ['{"cmd": ', '"which g++"}'])]
    for i, cid, name, parts in calls:
        events.append({**head, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "id": cid, "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": None}]})
        for p in parts:
            events.append({**head, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": i, "function": {"arguments": p}}]}, "finish_reason": None}]})
    events.append({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    events.append({**head, "choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 2010,
                                                    "reasoning_tokens": 2000}})
    return sse(*events)


def _rebuild(body):
    """What a client reassembles from a stream: text per field, tool calls, finish, usage."""
    from ttc.services import _sse_payloads
    text, calls, finish, usage = {}, {}, None, None
    for obj in _sse_payloads(body):
        usage = obj.get("usage") or usage
        for ch in obj.get("choices") or []:
            finish = ch.get("finish_reason") or finish
            for k, v in (ch.get("delta") or {}).items():
                if k == "tool_calls":
                    for tc in v:
                        c = calls.setdefault(tc["index"], {"id": None, "name": None, "args": ""})
                        c["id"] = c["id"] or tc.get("id")
                        c["name"] = c["name"] or (tc.get("function") or {}).get("name")
                        c["args"] += (tc.get("function") or {}).get("arguments") or ""
                elif isinstance(v, str) and k != "role":
                    text[k] = text.get(k, "") + v
    return text, calls, finish, usage


def test_per_token_stream_is_merged_without_changing_it():
    body = _per_token_stream()
    merged = coalesce_openai_sse(body)
    assert merged.count(b"data: ") <= 4 < body.count(b"data: ")
    assert _rebuild(merged) == _rebuild(body)
    text, calls, finish, usage = _rebuild(merged)
    assert text["reasoning_content"].startswith("Let me think.") and text["content"] == "I'll look"
    assert calls[1] == {"id": "c2", "name": "bash", "args": '{"cmd": "which g++"}'} and finish == "tool_calls"
    first = json.loads(merged.split(b"\n\n")[0][6:])
    assert first["id"] == "r1" and first["model"] == "glm" and first["choices"][0]["delta"]["role"] == "assistant"
    assert merged.endswith(b"data: [DONE]\n\n")
    c = inspect_response(merged, is_sse=True)
    assert c.has_content and c.has_tool_calls and normalize_usage(c.usage)["reasoning_tokens"] == 2000


def test_unusual_streams_are_passed_through():
    assert coalesce_openai_sse(b"data: not json\n\ndata: [DONE]\n\n") is None
    assert coalesce_openai_sse(sse({"error": {"message": "boom"}})) is None
    assert coalesce_openai_sse(b": keepalive\n\n") is None
    merged = coalesce_openai_sse(b": keepalive\n\n" + sse(chunk({"content": "a"}), chunk({"content": "b"}, "stop")))
    assert _rebuild(merged)[0] == {"content": "ab"} and _rebuild(merged)[2] == "stop"


def test_reasoning_only_length_stop_is_truncation_not_overflow():
    """A model that spends its whole output budget thinking (finish_reason=length) must not be
    mistaken for a context overflow, which would end the agent."""
    body = sse(chunk({"reasoning": "hmm " * 10}), chunk({}, "length"))
    c = inspect_response(coalesce_openai_sse(body), is_sse=True)
    assert c.is_truncated and not c.is_overflow


def test_proxy_forwards_merged_stream(tmp_path):
    """Through the real proxy: a per-token upstream stream reaches the client as a few chunks,
    on both the fast path and the slow path (keepalives sent first)."""
    import asyncio

    import aiohttp
    from aiohttp import web

    from ttc.services import Services
    from ttc.settings import TrialSettings

    body = _per_token_stream()

    async def main():
        async def completions(request):
            await asyncio.sleep(float(request.query.get("delay", "0")))
            return web.Response(body=body, content_type="text/event-stream")
        up = web.Application()
        up.router.add_post("/v1/chat/completions", completions)
        runner = web.AppRunner(up)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 8793).start()
        st = TrialSettings(mode="solo", k=1, trial=0, game="ls20", port=8794, base_url="http://127.0.0.1:8793/v1",
                           sse_keepalive_after_seconds=0.2, sse_keepalive_interval_seconds=0.1)
        svc = Services(st, tmp_path / "calls.jsonl")
        await svc.start()
        svc.register(["a.0"])
        out = []
        async with aiohttp.ClientSession() as s:
            for delay in ("0", "0.6"):
                async with s.post(f"http://127.0.0.1:8794/llm/a.0/chat/completions?delay={delay}",
                                  json={"model": "m", "stream": True, "messages": []}) as r:
                    out.append(await r.read())
        await svc.stop()
        await runner.cleanup()
        return out, svc.calls["a.0"]

    (fast, slow), calls = asyncio.run(main())
    assert fast.count(b"data: ") <= 4 and _rebuild(fast) == _rebuild(body)
    assert slow.startswith(b": keepalive") and _rebuild(slow) == _rebuild(body)
    assert calls.ok == 2 and calls.reasoning_tokens == 4000
