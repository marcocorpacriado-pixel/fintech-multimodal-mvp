"""Streaming free-text chat over OpenRouter.

Unlike :class:`OpenRouterLLMClient` (strict JSON schema, one-shot), this client
streams plain-text tokens for the conversational layer.  It reuses the same
environment variables, validators and error types, so ``map_integration_error``
classifies its failures exactly like the analysis client.

``open()`` performs the HTTP request and checks the status *before* returning
the token iterator.  Callers can therefore map configuration and provider
errors to a clean error response before committing to a streamed 200.
Retries only happen before the first token is received.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Self

import httpx

from .openrouter_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    TRANSIENT_STATUS_CODES,
    LLMResponseError,
    LLMTransportError,
    _parse_float_setting,
    _parse_int_setting,
    _required_secret,
    _required_text,
    _validate_base_url,
    _validate_retries,
    _validate_timeout,
)


DEFAULT_CHAT_MAX_TOKENS = 1024


class ChatStream:
    """Iterator of text deltas bound to one open HTTP response."""

    def __init__(self, client: httpx.Client, response: httpx.Response) -> None:
        self._client = client
        self._response = response

    def __iter__(self) -> Iterator[str]:
        try:
            for line in self._response.iter_lines():
                # SSE comments (": OPENROUTER PROCESSING") keep the connection alive.
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                delta = _parse_delta(data)
                if delta:
                    yield delta
        except httpx.TimeoutException as error:
            raise LLMTransportError("OpenRouter stream timed out") from error
        except httpx.HTTPError as error:
            raise LLMTransportError("OpenRouter stream was interrupted") from error
        finally:
            self.close()

    def close(self) -> None:
        self._response.close()
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class OpenRouterChatClient:
    """Synchronous OpenRouter client that streams chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
        http_transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = _required_secret(api_key, "OPENROUTER_API_KEY")
        self.model = _required_text(model, "OPENROUTER_MODEL")
        self.base_url = _validate_base_url(base_url)
        self.timeout_seconds = _validate_timeout(timeout_seconds)
        self.max_retries = _validate_retries(max_retries)
        self.max_tokens = max_tokens
        self._http_transport = http_transport
        self._sleep = sleep

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        http_transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Self:
        """Build a client from the same variables as the analysis client."""

        values = os.environ if environ is None else environ
        return cls(
            api_key=values.get("OPENROUTER_API_KEY", ""),
            model=values.get("OPENROUTER_MODEL", ""),
            base_url=values.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            timeout_seconds=_parse_float_setting(
                values.get("OPENROUTER_TIMEOUT_SECONDS"),
                name="OPENROUTER_TIMEOUT_SECONDS",
                default=DEFAULT_TIMEOUT_SECONDS,
            ),
            max_retries=_parse_int_setting(
                values.get("OPENROUTER_MAX_RETRIES"),
                name="OPENROUTER_MAX_RETRIES",
                default=DEFAULT_MAX_RETRIES,
            ),
            http_transport=http_transport,
            sleep=sleep,
        )

    @property
    def endpoint_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def open(
        self,
        *,
        system_prompt: str,
        messages: Sequence[Mapping[str, str]],
    ) -> ChatStream:
        """Start a streamed completion; raises before any token on failure."""

        payload = self._request_payload(system_prompt=system_prompt, messages=messages)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        attempts = self.max_retries + 1
        client = httpx.Client(transport=self._http_transport, timeout=self.timeout_seconds)
        try:
            for attempt in range(attempts):
                request = client.build_request(
                    "POST", self.endpoint_url, headers=headers, json=payload
                )
                try:
                    response = client.send(request, stream=True)
                except httpx.TimeoutException as error:
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue
                    raise LLMTransportError(
                        f"OpenRouter request timed out after {attempts} attempt(s)"
                    ) from error
                except httpx.RequestError as error:
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue
                    raise LLMTransportError(
                        f"OpenRouter transport failed after {attempts} attempt(s)"
                    ) from error

                if 200 <= response.status_code < 300:
                    return ChatStream(client, response)
                response.close()
                if (
                    response.status_code in TRANSIENT_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    self._backoff(attempt)
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
        except BaseException:
            client.close()
            raise

    def _request_payload(
        self,
        *,
        system_prompt: str,
        messages: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *({"role": m["role"], "content": m["content"]} for m in messages),
            ],
            "max_tokens": self.max_tokens,
            "stream": True,
            # No temperature on purpose: the provider/model default is used.
        }

    def _backoff(self, attempt: int) -> None:
        self._sleep(0.25 * (2**attempt))


def _parse_delta(data: str) -> str:
    try:
        event = json.loads(data)
    except json.JSONDecodeError as error:
        raise LLMResponseError("OpenRouter stream event is not valid JSON") from error
    if not isinstance(event, Mapping):
        raise LLMResponseError("OpenRouter stream event must be a JSON object")
    if event.get("error") is not None:
        # Mid-stream provider failure; the body is never surfaced to clients.
        raise LLMResponseError("OpenRouter reported an error mid-stream")
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") if isinstance(choice, Mapping) else None
    content = delta.get("content") if isinstance(delta, Mapping) else None
    return content if isinstance(content, str) else ""
