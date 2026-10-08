"""Published unit prices used to cost each inference shown in the UI.

The LLM cost is NOT estimated here: OpenRouter returns the exact charged amount
(``usage.cost``) for every call, streamed or not. These constants cover the
providers that do not report a per-request cost. Prices checked 2026-10-08;
update them here if the providers change their rates.
"""

from __future__ import annotations

import os

# Groq Orpheus TTS (canopylabs/orpheus-v1-english): $22 per 1M characters.
# https://console.groq.com/docs/text-to-speech/orpheus
GROQ_TTS_USD_PER_MILLION_CHARS = 22.0

# Groq Whisper (whisper-large-v3-turbo): $0.04 per audio hour, 10 s minimum.
# https://console.groq.com/docs/speech-to-text
GROQ_STT_USD_PER_HOUR = 0.04
GROQ_STT_MIN_BILLED_SECONDS = 10.0

# Cloud Run, Tier 1, instance-based billing, per second.
# https://cloud.google.com/run/pricing
CLOUD_RUN_USD_PER_VCPU_SECOND = 0.000018
CLOUD_RUN_USD_PER_GIB_SECOND = 0.000002


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def cloud_run_vcpu() -> float:
    """vCPUs of the service (Terraform passes CLOUD_RUN_VCPU)."""

    return _env_float("CLOUD_RUN_VCPU", 4.0)


def cloud_run_memory_gib() -> float:
    return _env_float("CLOUD_RUN_MEMORY_GIB", 4.0)


def cloud_run_usd_per_second() -> float:
    return (
        cloud_run_vcpu() * CLOUD_RUN_USD_PER_VCPU_SECOND
        + cloud_run_memory_gib() * CLOUD_RUN_USD_PER_GIB_SECOND
    )


def cloud_run_hourly_cost() -> float:
    """Cost of one instance running for an hour."""

    return cloud_run_usd_per_second() * 3600


def groq_tts_cost(characters: int) -> float:
    return characters * GROQ_TTS_USD_PER_MILLION_CHARS / 1_000_000


def groq_stt_cost(audio_seconds: float) -> float:
    billed = max(audio_seconds, GROQ_STT_MIN_BILLED_SECONDS)
    return billed / 3600 * GROQ_STT_USD_PER_HOUR


def cpu_time_cost(seconds: float) -> float:
    """Estimated cost of on-instance compute (Kokoro): busy seconds × instance price."""

    return seconds * cloud_run_usd_per_second()


__all__ = [
    "GROQ_STT_MIN_BILLED_SECONDS",
    "GROQ_STT_USD_PER_HOUR",
    "GROQ_TTS_USD_PER_MILLION_CHARS",
    "cloud_run_hourly_cost",
    "cloud_run_memory_gib",
    "cloud_run_usd_per_second",
    "cloud_run_vcpu",
    "cpu_time_cost",
    "groq_stt_cost",
    "groq_tts_cost",
]
