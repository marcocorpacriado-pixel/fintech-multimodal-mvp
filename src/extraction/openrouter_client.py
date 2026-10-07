"""OpenRouter implementation of the provider-agnostic ``LLMClient`` protocol."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import httpx

from .pipeline import qualitative_analysis_json_schema


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_TOTAL_DEADLINE_SECONDS = 90.0
MAX_TOTAL_DEADLINE_SECONDS = 600.0
DEFAULT_MAX_RETRIES = 2
MAX_ALLOWED_RETRIES = 5
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class LLMProviderError(RuntimeError):
    """Base error for provider configuration, transport, and response failures."""


class LLMConfigurationError(LLMProviderError):
    """Raised when required OpenRouter configuration is missing or invalid."""


class LLMTransportError(LLMProviderError):
    """Raised when an OpenRouter HTTP request cannot complete successfully."""


class LLMTotalDeadlineError(LLMTransportError):
    """Raised when one complete generation exhausts its wall-clock budget."""

    reason_code = "TOTAL_DEADLINE_EXCEEDED"


class LLMResponseError(LLMProviderError):
    """Raised when OpenRouter returns an unusable completion response."""


@dataclass(frozen=True)
class LLMUsage:
    """Token and optional cost metadata from the latest successful response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class OpenRouterLLMClient:
    """Synchronous OpenRouter client using strict JSON Schema responses."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        total_deadline_seconds: float = DEFAULT_TOTAL_DEADLINE_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None] | None] = asyncio.sleep,
    ) -> None:
        self._api_key = _required_secret(api_key, "OPENROUTER_API_KEY")
        self.model = _required_text(model, "OPENROUTER_MODEL")
        self.base_url = _validate_base_url(base_url)
        self.timeout_seconds = _validate_timeout(timeout_seconds)
        self.total_deadline_seconds = _validate_total_deadline(
            total_deadline_seconds
        )
        self.max_retries = _validate_retries(max_retries)
        self._http_transport = http_transport
        self._sleep = sleep
        self.last_usage: LLMUsage | None = None

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None] | None] = asyncio.sleep,
    ) -> Self:
        """Build a client from explicit OpenRouter environment variables."""

        values = os.environ if environ is None else environ
        api_key = values.get("OPENROUTER_API_KEY", "")
        model = values.get("OPENROUTER_MODEL", "")
        base_url = values.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        timeout_seconds = _parse_float_setting(
            values.get("OPENROUTER_TIMEOUT_SECONDS"),
            name="OPENROUTER_TIMEOUT_SECONDS",
            default=DEFAULT_TIMEOUT_SECONDS,
        )
        total_deadline_seconds = _parse_float_setting(
            values.get("OPENROUTER_TOTAL_DEADLINE_SECONDS"),
            name="OPENROUTER_TOTAL_DEADLINE_SECONDS",
            default=DEFAULT_TOTAL_DEADLINE_SECONDS,
        )
        max_retries = _parse_int_setting(
            values.get("OPENROUTER_MAX_RETRIES"),
            name="OPENROUTER_MAX_RETRIES",
            default=DEFAULT_MAX_RETRIES,
        )
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            total_deadline_seconds=total_deadline_seconds,
            max_retries=max_retries,
            http_transport=http_transport,
            sleep=sleep,
        )

    @property
    def endpoint_url(self) -> str:
        """Return the non-streaming chat completions endpoint."""

        return f"{self.base_url}/chat/completions"

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Mapping[str, Any]:
        """Request, parse, and return one structured qualitative response."""

        payload = self._request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.last_usage = None
        response = self._run_generation_request(payload)
        body = _response_json(response)
        self.last_usage = _parse_usage(body.get("usage"))
        content = _extract_content(body)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise LLMResponseError(
                "OpenRouter message content is not valid JSON"
            ) from error
        if not isinstance(parsed, Mapping):
            raise LLMResponseError(
                "OpenRouter structured content must decode to a JSON object"
            )
        return dict(parsed)

    def close(self) -> None:
        """Retained for context-manager compatibility; requests self-close."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_financial_analysis",
                    "strict": True,
                    "schema": qualitative_analysis_json_schema(),
                },
            },
            "provider": {"require_parameters": True},
        }

    def _run_generation_request(self, payload: dict[str, Any]) -> httpx.Response:
        """Run one cancellable generation from synchronous application code."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:  # The public adapter is intentionally synchronous.
            raise LLMTransportError(
                "OpenRouter synchronous client cannot run inside an event loop"
            )
        return asyncio.run(self._post_with_retries(payload))

    async def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        """Bound all HTTP attempts and backoff by one generation deadline."""

        try:
            async with asyncio.timeout(self.total_deadline_seconds):
                return await self._post_attempts(payload)
        except TimeoutError as error:
            raise LLMTotalDeadlineError(
                "OpenRouter generation exceeded its total deadline"
            ) from error

    async def _post_attempts(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        attempts = self.max_retries + 1
        async with httpx.AsyncClient(transport=self._http_transport) as client:
            for attempt in range(attempts):
                try:
                    response = await client.post(
                        self.endpoint_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                except httpx.TimeoutException as error:
                    if attempt < self.max_retries:
                        await self._backoff(attempt)
                        continue
                    raise LLMTransportError(
                        f"OpenRouter request timed out after {attempts} attempt(s)"
                    ) from error
                except httpx.RequestError as error:
                    if attempt < self.max_retries:
                        await self._backoff(attempt)
                        continue
                    raise LLMTransportError(
                        f"OpenRouter transport failed after {attempts} attempt(s)"
                    ) from error

                if 200 <= response.status_code < 300:
                    return response
                if (
                    response.status_code in TRANSIENT_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    await self._backoff(attempt)
                    continue
                if response.status_code in TRANSIENT_STATUS_CODES:
                    raise LLMTransportError(
                        "OpenRouter returned transient HTTP "
                        f"{response.status_code} after {attempts} attempt(s)"
                    )
                raise LLMTransportError(
                    f"OpenRouter request failed with HTTP {response.status_code}"
                )
        raise AssertionError("retry loop exhausted without returning or raising")

    async def _backoff(self, attempt: int) -> None:
        result = self._sleep(0.25 * (2**attempt))
        if inspect.isawaitable(result):
            await result


def _required_secret(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigurationError(f"{name} is required")
    return value.strip()


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigurationError(f"{name} is required")
    return value.strip()


def _validate_base_url(value: str) -> str:
    base_url = _required_text(value, "OPENROUTER_BASE_URL").rstrip("/")
    try:
        parsed = httpx.URL(base_url)
    except httpx.InvalidURL as error:
        raise LLMConfigurationError(
            "OPENROUTER_BASE_URL must be an HTTP(S) URL"
        ) from error
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise LLMConfigurationError("OPENROUTER_BASE_URL must be an HTTP(S) URL")
    return base_url


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMConfigurationError("OPENROUTER_TIMEOUT_SECONDS must be numeric")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise LLMConfigurationError(
            "OPENROUTER_TIMEOUT_SECONDS must be finite and greater than zero"
        )
    return timeout


def _validate_total_deadline(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMConfigurationError(
            "OPENROUTER_TOTAL_DEADLINE_SECONDS must be numeric"
        )
    deadline = float(value)
    if not math.isfinite(deadline) or deadline <= 0:
        raise LLMConfigurationError(
            "OPENROUTER_TOTAL_DEADLINE_SECONDS must be finite and greater than zero"
        )
    if deadline > MAX_TOTAL_DEADLINE_SECONDS:
        raise LLMConfigurationError(
            "OPENROUTER_TOTAL_DEADLINE_SECONDS must not exceed "
            f"{MAX_TOTAL_DEADLINE_SECONDS:g}"
        )
    return deadline


def _validate_retries(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMConfigurationError("OPENROUTER_MAX_RETRIES must be an integer")
    if not 0 <= value <= MAX_ALLOWED_RETRIES:
        raise LLMConfigurationError(
            f"OPENROUTER_MAX_RETRIES must be between 0 and {MAX_ALLOWED_RETRIES}"
        )
    return value


def _parse_float_setting(
    value: str | None,
    *,
    name: str,
    default: float,
) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise LLMConfigurationError(f"{name} must be numeric") from error


def _parse_int_setting(
    value: str | None,
    *,
    name: str,
    default: int,
) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise LLMConfigurationError(f"{name} must be an integer") from error


def _response_json(response: httpx.Response) -> Mapping[str, Any]:
    try:
        body = response.json()
    except ValueError as error:
        raise LLMResponseError("OpenRouter response body is not valid JSON") from error
    if not isinstance(body, Mapping):
        raise LLMResponseError("OpenRouter response body must be a JSON object")
    return body


def _extract_content(body: Mapping[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("OpenRouter response does not contain choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise LLMResponseError("OpenRouter first choice is invalid")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise LLMResponseError("OpenRouter first choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("OpenRouter message content is empty")
    return content


def _parse_usage(value: Any) -> LLMUsage | None:
    if not isinstance(value, Mapping):
        return None
    return LLMUsage(
        prompt_tokens=_optional_nonnegative_int(value.get("prompt_tokens")),
        completion_tokens=_optional_nonnegative_int(
            value.get("completion_tokens")
        ),
        total_tokens=_optional_nonnegative_int(value.get("total_tokens")),
        cost=_optional_nonnegative_float(value.get("cost")),
    )


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_TOTAL_DEADLINE_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMResponseError",
    "LLMTotalDeadlineError",
    "LLMTransportError",
    "LLMUsage",
    "OpenRouterLLMClient",
]
