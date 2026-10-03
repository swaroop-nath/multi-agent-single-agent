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
