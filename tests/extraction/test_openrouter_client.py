"""Tests for the synchronous OpenRouter structured-output adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from src.extraction.openrouter_client import (
    LLMConfigurationError,
    LLMResponseError,
    LLMTransportError,
    OpenRouterLLMClient,
)
from src.extraction.pipeline import GroundedAnalysisError, analyze_financials


API_KEY = "test-secret-api-key"
MODEL = "test/provider-model"


def qualitative_output(**overrides: object) -> dict[str, object]:
    output: dict[str, object] = {
        "key_positive_developments": [],
        "key_risks": [],
        "management_outlook": {
            "summary": "Insufficient narrative evidence.",
            "sentiment": "unknown",
            "source_ids": [],
        },
        "executive_summary": "No narrative evidence was supplied for analysis.",
    }
    output.update(overrides)
    return output


def completion_response(
    output: object | None = None,
    *,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        qualitative_output() if output is None else output
                    )
                }
            }
        ]
    }
    if usage is not None:
        body["usage"] = usage
    return body


def mock_http_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **overrides: Any,
) -> OpenRouterLLMClient:
    parameters: dict[str, Any] = {
        "api_key": API_KEY,
        "model": MODEL,
        "http_client": mock_http_client(handler),
        "sleep": lambda _: None,
    }
    parameters.update(overrides)
    return OpenRouterLLMClient(**parameters)


def successful_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=completion_response(), request=request)


def test_valid_environment_configuration() -> None:
    client = OpenRouterLLMClient.from_env(
        environ={
            "OPENROUTER_API_KEY": API_KEY,
            "OPENROUTER_MODEL": MODEL,
            "OPENROUTER_BASE_URL": "https://router.example/v1/",
            "OPENROUTER_TIMEOUT_SECONDS": "12.5",
            "OPENROUTER_MAX_RETRIES": "1",
        },
        http_client=mock_http_client(successful_handler),
        sleep=lambda _: None,
    )

    assert client.model == MODEL
    assert client.base_url == "https://router.example/v1"
    assert client.timeout_seconds == 12.5
    assert client.max_retries == 1


def test_missing_api_key_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterLLMClient.from_env(environ={"OPENROUTER_MODEL": MODEL})


def test_missing_model_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENROUTER_MODEL"):
        OpenRouterLLMClient.from_env(environ={"OPENROUTER_API_KEY": API_KEY})


def test_request_uses_chat_completions_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return successful_handler(request)

    client = make_client(handler)
    client.generate_structured(system_prompt="system", user_prompt="user")

    assert str(requests[0].url) == (
        "https://openrouter.ai/api/v1/chat/completions"
    )


def test_request_has_safe_required_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return successful_handler(request)

    client = make_client(handler)
    client.generate_structured(system_prompt="system", user_prompt="user")

    assert requests[0].headers["Authorization"] == f"Bearer {API_KEY}"
    assert requests[0].headers["Content-Type"] == "application/json"
    assert requests[0].headers["Accept"] == "application/json"


def test_request_contains_system_and_user_messages() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return successful_handler(request)

    client = make_client(handler)
    client.generate_structured(
        system_prompt="grounded system",
        user_prompt="structured user",
    )

    assert payloads[0]["messages"] == [
        {"role": "system", "content": "grounded system"},
        {"role": "user", "content": "structured user"},
    ]
    assert payloads[0]["temperature"] == 0.0
    assert payloads[0]["stream"] is False


def test_request_uses_pydantic_json_schema() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return successful_handler(request)

    client = make_client(handler)
    client.generate_structured(system_prompt="system", user_prompt="user")

    response_format = payloads[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "financial_metrics" not in json.dumps(schema)
    assert "key_positive_developments" in schema["properties"]


def test_request_requires_provider_parameter_support() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return successful_handler(request)

    client = make_client(handler)
    client.generate_structured(system_prompt="system", user_prompt="user")

    assert payloads[0]["provider"] == {"require_parameters": True}


def test_valid_response_content_is_parsed() -> None:
    client = make_client(successful_handler)

    output = client.generate_structured(system_prompt="system", user_prompt="user")

    assert output == qualitative_output()


def test_usage_and_cost_are_preserved() -> None:
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "cost": 0.0042,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=completion_response(usage=usage),
            request=request,
        )

    client = make_client(handler)
    client.generate_structured(system_prompt="system", user_prompt="user")

    assert client.last_usage is not None
    assert client.last_usage.prompt_tokens == 120
    assert client.last_usage.completion_tokens == 30
    assert client.last_usage.total_tokens == 150
    assert client.last_usage.cost == 0.0042


def test_invalid_content_json_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": "not-json"}}]}
        return httpx.Response(200, json=body, request=request)

    client = make_client(handler)

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        client.generate_structured(system_prompt="system", user_prompt="user")


def test_empty_choices_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []}, request=request)

    client = make_client(handler)

    with pytest.raises(LLMResponseError, match="does not contain choices"):
        client.generate_structured(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_content_is_rejected(content: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": content}}]}
        return httpx.Response(200, json=body, request=request)

    client = make_client(handler)

    with pytest.raises(LLMResponseError, match="content is empty"):
        client.generate_structured(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_non_transient_client_errors_are_not_retried(status_code: int) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, json={"error": "denied"}, request=request)

    client = make_client(handler, max_retries=5)

    with pytest.raises(LLMTransportError, match=f"HTTP {status_code}"):
        client.generate_structured(system_prompt="system", user_prompt="user")
    assert call_count == 1


def test_redirect_is_not_treated_as_a_completion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"Location": "/other"}, request=request)

    client = make_client(handler)

    with pytest.raises(LLMTransportError, match="HTTP 307"):
        client.generate_structured(system_prompt="system", user_prompt="user")


def test_429_is_retried_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, request=request)
        return successful_handler(request)

    client = make_client(handler, max_retries=1)

    assert client.generate_structured(system_prompt="system", user_prompt="user")
    assert call_count == 2


def test_500_is_retried_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(500, request=request)
        return successful_handler(request)

    client = make_client(handler, max_retries=1)

    assert client.generate_structured(system_prompt="system", user_prompt="user")
    assert call_count == 2


def test_max_retries_are_bounded() -> None:
    call_count = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, request=request)

    client = make_client(
        handler,
        max_retries=2,
        sleep=sleeps.append,
    )

    with pytest.raises(LLMTransportError, match="after 3 attempt"):
        client.generate_structured(system_prompt="system", user_prompt="user")
    assert call_count == 3
    assert sleeps == [0.25, 0.5]


def test_timeout_is_explicit_and_retried() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    client = make_client(handler, timeout_seconds=7.5, max_retries=1)

    with pytest.raises(LLMTransportError, match="timed out after 2 attempt"):
        client.generate_structured(system_prompt="system", user_prompt="user")
    assert len(requests) == 2
    assert requests[0].extensions["timeout"]["read"] == 7.5


def test_api_key_never_appears_in_error_messages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": API_KEY}, request=request)

    client = make_client(handler)

    with pytest.raises(LLMTransportError) as caught:
        client.generate_structured(system_prompt="system", user_prompt="user")
    assert API_KEY not in str(caught.value)


def test_d6a_still_validates_structured_output_locally() -> None:
    invalid = qualitative_output(
        management_outlook={
            "summary": "Invalid remote output.",
            "sentiment": "optimistic",
            "source_ids": [],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=completion_response(invalid),
            request=request,
        )

    client = make_client(handler)

    with pytest.raises(GroundedAnalysisError, match="invalid LLM output schema"):
        analyze_financials(
            company="Apple Inc.",
            ticker="AAPL",
            period="Q3 2026",
            filing_type="10-Q",
            financial_metrics=[],
            retrieval_results=[],
            llm_client=client,
        )


def test_successful_mock_transport_never_uses_real_network() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return successful_handler(request)

    client = make_client(handler)

    assert client.generate_structured(system_prompt="system", user_prompt="user")
    assert calls == 1
