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
    assert normalize_usage(chat) == {"prompt_tokens": 10, "output_tokens": 3, "reasoning_tokens": 2, "cached_tokens": 4}
    resp = {"input_tokens": 7, "output_tokens": 1}
    assert normalize_usage(resp)["prompt_tokens"] == 7 and normalize_usage(None) == {}
