"""HTTP access to the FastAPI backend for the Dash presentation.

The UI talks to FastAPI over HTTP only. Nothing here imports extraction,
integration, SEC, XBRL, LLM, or audio implementation modules.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import httpx

from src.visualization.presentation import sort_filings

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_VOICES = ["af_heart"]
CHAT_METRICS_SEPARATOR = "\x1e"  # API trailer: answer text | JSON metrics


def ttl_cache(seconds: float, *, cacheable: Callable[[Any], bool] = lambda _: True):
    """Cache results per positional args for ``seconds``; skip uncacheable ones."""

    def decorator(function):
        store: dict[tuple, tuple[float, Any]] = {}

        @wraps(function)
        def wrapper(*args):
            hit = store.get(args)
            if hit is not None and time.monotonic() - hit[0] < seconds:
                return hit[1]
            value = function(*args)
            if cacheable(value):
                store[args] = (time.monotonic(), value)
            return value

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def safe_error(
    code: str = "UNKNOWN_ERROR",
    *,
    message: str = "",
    retryable: bool = True,
    status_code: int | None = None,
) -> dict[str, Any]:
    """The small, safe error shape the UI is allowed to display."""

    return {
        "code": code,
        "message": message,
        "retryable": retryable,
        "status_code": status_code,
    }


def parse_api_error(response: httpx.Response) -> dict[str, Any]:
    """Convert an API failure into the small safe shape needed by the UI."""

    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return safe_error(
            detail["code"],
            message=str(detail.get("message") or ""),
            retryable=bool(detail.get("retryable")),
            status_code=response.status_code,
        )
    return safe_error(
        "INPUT_ERROR" if response.status_code == 422 else "UNKNOWN_ERROR",
        retryable=response.status_code >= 500,
        status_code=response.status_code,
    )


def api_request(
    method: str,
    path: str,
    **kwargs: Any,
) -> tuple[httpx.Response | None, dict[str, Any] | None]:
    """Make one backend request and return data or a safe presentation error."""

    try:
        response = httpx.request(method, f"{API_URL}{path}", **kwargs)
    except httpx.HTTPError:
        return None, safe_error("API_UNAVAILABLE")
    if response.is_success:
        return response, None
    return None, parse_api_error(response)


@ttl_cache(60, cacheable=lambda voices: voices is not DEFAULT_VOICES)
def fetch_voices(provider: str = "local") -> list[str]:
    try:
        response = httpx.request(
            "GET",
            f"{API_URL}/api/v1/audio/voices",
            params={"provider": provider},
            timeout=30,
        )
        return response.json()["voices"] if response.is_success else DEFAULT_VOICES
    except (httpx.HTTPError, ValueError, KeyError):
        return DEFAULT_VOICES


@ttl_cache(60, cacheable=lambda result: result[1] is None)
def fetch_filings(
    ticker: str,
    filing_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fetch lightweight filing metadata; this never runs XBRL or an LLM."""

    response, error = api_request(
        "GET",
        f"/api/v1/filings/{ticker}",
        params={"filing_type": filing_type, "limit": 10},
        timeout=45,
    )
    if response is None:
        return [], error
    try:
        payload = response.json()
    except ValueError:
        return [], safe_error(status_code=response.status_code)
    if not isinstance(payload, list):
        return [], safe_error(status_code=response.status_code)
    return sort_filings(payload), None


@ttl_cache(300, cacheable=bool)
def fetch_pricing() -> dict[str, Any]:
    response, _ = api_request("GET", "/api/v1/pricing", timeout=10)
    try:
        return response.json() if response is not None else {}
    except ValueError:
        return {}


@dataclass
class ChatAnswer:
    """One finished chat turn: text, timings/usage metrics, or a safe error."""

    text: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


def chat_answer(payload: dict[str, Any]) -> ChatAnswer:
    """Read the whole streaming chat response and return it as one answer.

    The JSON metrics trailer after ``CHAT_METRICS_SEPARATOR`` is never part of
    the text; it is merged with the client-side timings into ``metrics``.
    """

    started = time.perf_counter()
    first_token: float | None = None
    text: list[str] = []
    trailer: list[str] = []
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/api/v1/chat",
            json=payload,
            timeout=httpx.Timeout(120, connect=10),
        ) as response:
            if not response.is_success:
                response.read()
                return ChatAnswer(error=parse_api_error(response))
            for chunk in response.iter_text():
                if trailer:
                    trailer.append(chunk)
                    continue
                piece, separator, rest = chunk.partition(CHAT_METRICS_SEPARATOR)
                if separator:
                    trailer.append(rest)
                if piece:
                    first_token = first_token or time.perf_counter() - started
                    text.append(piece)
    except httpx.HTTPError:
        return ChatAnswer(error=safe_error("API_UNAVAILABLE"))
    try:
        server = json.loads("".join(trailer))["metrics"] if trailer else {}
    except (ValueError, KeyError, TypeError):
        server = {}
    return ChatAnswer(
        text="".join(text),
        metrics={
            **server,
            "client_ttft_ms": None if first_token is None else first_token * 1000,
            "client_total_ms": (time.perf_counter() - started) * 1000,
        },
    )


def transcribe_audio(
    wav: bytes,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float]:
    """Send a recorded question to the STT endpoint: (body, error, latency_ms)."""

    started = time.perf_counter()
    response, error = api_request(
        "POST",
        "/api/v1/audio/transcribe",
        content=wav,
        headers={"Content-Type": "audio/wav"},
        timeout=60,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if response is None:
        return None, error, latency_ms
    try:
        return response.json(), None, latency_ms
    except ValueError:
        return None, safe_error(), latency_ms
