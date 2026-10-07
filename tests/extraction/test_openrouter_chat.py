"""Streaming chat client: SSE parsing, pre-stream retries, safe errors."""

import json

import httpx
import pytest

from src.extraction import (
    LLMConfigurationError,
    LLMResponseError,
    LLMTransportError,
    OpenRouterChatClient,
)


API_KEY = "sk-or-SECRET-KEY"


def _sse(*events: object, done: bool = True) -> bytes:
    lines = [": OPENROUTER PROCESSING", ""]
    for event in events:
        lines += [f"data: {json.dumps(event)}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def _delta(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _client(handler, **kwargs) -> OpenRouterChatClient:
    return OpenRouterChatClient(
        api_key=API_KEY,
        model="test/model",
        http_transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        **kwargs,
    )


def test_stream_yields_deltas_and_sends_streaming_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, content=_sse(_delta("Hello "), _delta("world"), {"choices": []}))

    stream = _client(handler).open(
        system_prompt="SYS", messages=[{"role": "user", "content": "hi"}]
    )

    assert list(stream) == ["Hello ", "world"]
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert seen["body"]["stream"] is True
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    assert "response_format" not in seen["body"]
    assert "temperature" not in seen["body"]  # model default


def test_transient_status_is_retried_before_first_token():
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, content=_sse(_delta("ok")) if status == 200 else b"")

    stream = _client(handler, max_retries=1).open(system_prompt="S", messages=[])

    assert list(stream) == ["ok"]


def test_non_transient_status_raises_without_leaking_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {API_KEY}")

    with pytest.raises(LLMTransportError) as excinfo:
        _client(handler).open(system_prompt="S", messages=[])

    assert "401" in str(excinfo.value)
    assert API_KEY not in str(excinfo.value)


def test_transport_failure_after_retries_raises_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(LLMTransportError):
        _client(handler, max_retries=2).open(system_prompt="S", messages=[])


def test_mid_stream_provider_error_raises_response_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_sse(_delta("partial"), {"error": {"message": API_KEY}}, done=False)
        )

    stream = iter(_client(handler).open(system_prompt="S", messages=[]))

    assert next(stream) == "partial"
    with pytest.raises(LLMResponseError) as excinfo:
        next(stream)
    assert API_KEY not in str(excinfo.value)


def test_from_env_requires_key_and_model():
    with pytest.raises(LLMConfigurationError):
        OpenRouterChatClient.from_env(environ={"OPENROUTER_MODEL": "m"})
    with pytest.raises(LLMConfigurationError):
        OpenRouterChatClient.from_env(environ={"OPENROUTER_API_KEY": "k"})
