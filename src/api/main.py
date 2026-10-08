"""FastAPI orchestration layer between Dani's pipeline, Cristian's audio and the UI.

This module only orchestrates and serializes. Financial values come from the
``AnalysisHandoff`` contract unchanged; no metric is recalculated here.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.audio import (
    DEFAULT_GROQ_VOICE,
    GROQ_VOICES,
    detect_speech_language,
    list_voices,
    normalize_for_speech,
    synthesize_groq,
)
from src.api.inference_costs import (
    GROQ_STT_USD_PER_HOUR,
    GROQ_TTS_USD_PER_MILLION_CHARS,
    cloud_run_hourly_cost,
    cloud_run_memory_gib,
    cloud_run_vcpu,
    cpu_time_cost,
    groq_stt_cost,
    groq_tts_cost,
)
from src.audio.stt import MAX_FILE_SIZE_MB, transcribe
from src.audio.tts import DEFAULT_VOICE_BY_LANG, language_for_voice, synthesize
from src.extraction import (
    AnalysisPipelineResult,
    OpenRouterChatClient,
    OpenRouterLLMClient,
    SECInputError,
    discover_sec_filings,
    normalize_sec_ticker,
    prepare_sec_analysis_inputs,
    run_analysis_pipeline,
)
from src.extraction.schemas import FilingType
from src.integration import (
    CHAT_SYSTEM_PROMPT,
    AnalysisHandoff,
    AnalysisMode,
    ChatRequest,
    InferenceMetricsDTO,
    IntegrationError,
    build_analysis_handoff,
    build_chat_context,
    diagnose_integration_failure,
    map_integration_error,
)

load_dotenv()
logger = logging.getLogger(__name__)

DEMO_FIXTURE_PATH = Path(__file__).with_name("demo_fixture.json")
DEFAULT_VOICE = "af_heart"
MAX_TRANSCRIBE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# Fixed, non-revealing marker appended when the provider fails mid-stream:
# the 200 status is already committed, so the error cannot be a JSON body.
STREAM_INTERRUPTED_MARKER = "\n\n[stream interrupted]"
# ASCII Record Separator: never produced by the model, so the UI can split the
# answer from the JSON metrics trailer sent after the last token.
CHAT_METRICS_SEPARATOR = "\x1e"
_ERROR_STATUS = {
    "INPUT_ERROR": 422,
    "FILING_NOT_FOUND": 404,
    "SEC_INGESTION_ERROR": 503,
    "VERIFICATION_ERROR": 422,
    "GROUNDING_ERROR": 422,
    "LLM_PROVIDER_ERROR": 503,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalysisRequest(StrictModel):
    ticker: str
    filing_date: date | None = None
    filing_type: FilingType = "10-Q"
    mode: AnalysisMode = "demo"

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        try:
            return normalize_sec_ticker(value)
        except SECInputError as error:
            raise ValueError("ticker has invalid SEC syntax") from error

    @model_validator(mode="after")
    def require_real_filing_date(self):
        if self.mode == "real" and self.filing_date is None:
            raise ValueError("filing_date is required in real mode")
        return self


class FilingMetadataResponse(StrictModel):
    ticker: str
    company: str
    filing_date: date
    report_date: date
    form: FilingType
    accession: str


TTSProvider = Literal["local", "groq"]


class AudioSummaryRequest(StrictModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = DEFAULT_VOICE
    # "local" = Kokoro on this container (EN/ES); "groq" = Orpheus on Groq
    # (faster, English only).
    provider: TTSProvider = "local"
    # Kokoro language; when omitted it is derived from the voice prefix so a
    # Spanish voice (ef_*) is not pronounced with English phonemes.
    language: str | None = None


class TranscriptionResponse(StrictModel):
    text: str
    language: str
    latency_ms: float | None = None
    audio_seconds: float | None = None
    cost_usd: float | None = None


class HealthResponse(StrictModel):
    status: Literal["ok"]


class VoicesResponse(StrictModel):
    voices: list[str]


class ErrorResponse(StrictModel):
    detail: IntegrationError


_ERROR_RESPONSES = {
    status: {"model": ErrorResponse} for status in (404, 422, 500, 503)
}

app = FastAPI(title="Fintech Multimodal API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(error: IntegrationError) -> JSONResponse:
    return JSONResponse(
        status_code=_ERROR_STATUS.get(error.code, 500),
        content={"detail": error.model_dump()},
    )


def _run_demo_analysis() -> AnalysisHandoff:
    """Validated synthetic fixture, always labelled as demo."""

    result = AnalysisPipelineResult.model_validate_json(
        DEMO_FIXTURE_PATH.read_text(encoding="utf-8")
    )
    return build_analysis_handoff(result, analysis_mode="demo", provider="fixture")


class _UsageMeter:
    """LLM client proxy that adds up latency, tokens and cost of every call.

    The pipeline may call the model twice (initial + one repair); the analysis
    cost is the sum. Usage comes from OpenRouter's response (``last_usage``).
    """

    def __init__(self, client: OpenRouterLLMClient) -> None:
        self._client = client
        self.model = client.model
        self.calls = 0
        self.llm_seconds = 0.0
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.cost_usd: float | None = None

    def generate_structured(self, *, system_prompt: str, user_prompt: str):
        started = time.perf_counter()
        try:
            return self._client.generate_structured(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
        finally:
            self.calls += 1
            self.llm_seconds += time.perf_counter() - started
            usage = self._client.last_usage
            if usage is not None:
                self.prompt_tokens = _add(self.prompt_tokens, usage.prompt_tokens)
                self.completion_tokens = _add(
                    self.completion_tokens, usage.completion_tokens
                )
                self.cost_usd = _add(self.cost_usd, usage.cost)


def _add(total, value):
    if value is None:
        return total
    return value if total is None else total + value


def _run_real_analysis(
    request: AnalysisRequest,
    *,
    request_id: str = "not-provided",
) -> AnalysisHandoff:
    """SEC ingestion -> grounded pipeline -> handoff. No fallback to demo."""

    if request.filing_date is None:  # guarded by AnalysisRequest validation
        raise AssertionError("real analysis requires filing_date")

    started = time.perf_counter()
    inputs = prepare_sec_analysis_inputs(
        ticker=request.ticker,
        filing_date=request.filing_date,
        form=request.filing_type,
    )
    sec_seconds = time.perf_counter() - started
    try:
        with OpenRouterLLMClient.from_env() as client:
            llm = _UsageMeter(client)
            result = run_analysis_pipeline(
                filing_path=inputs.filing_path,
                company=inputs.company,
                ticker=inputs.ticker,
                period=inputs.report_period,
                filing_type=inputs.filing_type,
                current_xbrl_filing=inputs.current_filing,
                previous_xbrl_filing=inputs.previous_filing,
                llm_client=llm,
            )
            logger.info(
                "real analysis completed request_id=%s accession=%s "
                "generation_attempts=%d repair_used=%s first_failure=%s",
                request_id,
                inputs.current_accession,
                result.generation_attempts,
                result.repair_used,
                result.first_failure_category or "none",
            )
            handoff = build_analysis_handoff(
                result,
                analysis_mode="real",
                provider="openrouter",
                model=llm.model,
                filing_date=inputs.filing_date,
            )
            metrics = InferenceMetricsDTO(
                total_ms=(time.perf_counter() - started) * 1000,
                sec_ingestion_ms=sec_seconds * 1000,
                llm_ms=llm.llm_seconds * 1000,
                llm_calls=llm.calls,
                prompt_tokens=llm.prompt_tokens,
                completion_tokens=llm.completion_tokens,
                cost_usd=llm.cost_usd,
            )
            return handoff.model_copy(
                update={
                    "pipeline_metadata": handoff.pipeline_metadata.model_copy(
                        update={"metrics": metrics}
                    )
                }
            )
    except Exception as error:
        diagnostic = diagnose_integration_failure(error)
        logger.warning(
            "real analysis rejected request_id=%s accession=%s category=%s "
            "reason=%s verification_issues=%s generation_attempts=%s "
            "first_failure=%s",
            request_id,
            inputs.current_accession,
            diagnostic.category,
            diagnostic.reason_code,
            ",".join(diagnostic.verification_issue_codes) or "none",
            diagnostic.generation_attempts or "n/a",
            diagnostic.first_failure_category or "none",
        )
        raise


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/api/v1/filings/{ticker}",
    response_model=list[FilingMetadataResponse],
    responses=_ERROR_RESPONSES,
)
def filings(
    ticker: str,
    filing_type: FilingType = "10-Q",
    limit: int = 10,
):
    """Discover recent filing metadata without XBRL or LLM execution."""

    try:
        values = discover_sec_filings(
            ticker=ticker,
            form=filing_type,
            limit=limit,
        )
    except Exception as error:
        logger.exception("filing discovery failed")
        return _error_response(map_integration_error(error))
    return [
        FilingMetadataResponse(
            ticker=value.ticker,
            company=value.company,
            filing_date=value.filing_date,
            report_date=value.report_date,
            form=value.form,
            accession=value.accession,
        ).model_dump(mode="json")
        for value in values
    ]


@app.get("/api/v1/pricing")
def pricing() -> dict:
    """Unit prices behind the Performance tab (the LLM cost comes per call)."""

    return {
        "cloud_run": {
            "vcpu": cloud_run_vcpu(),
            "memory_gib": cloud_run_memory_gib(),
            "usd_per_hour": cloud_run_hourly_cost(),
        },
        "groq_tts_usd_per_million_chars": GROQ_TTS_USD_PER_MILLION_CHARS,
        "groq_stt_usd_per_hour": GROQ_STT_USD_PER_HOUR,
        "llm": "OpenRouter reports the exact charged cost of every call",
    }


@app.get("/api/v1/audio/voices", response_model=VoicesResponse)
def voices(provider: TTSProvider = "local") -> dict:
    if provider == "groq":
        return {"voices": GROQ_VOICES}
    # First call loads (and may download) the Kokoro model.
    try:
        return {"voices": list_voices()}
    except Exception:
        logger.exception("list_voices failed; returning default voice")
        return {"voices": [DEFAULT_VOICE]}


@app.post(
    "/api/v1/analysis",
    response_model=AnalysisHandoff,
    responses=_ERROR_RESPONSES,
)
def analysis(request: AnalysisRequest):
    request_id = uuid.uuid4().hex
    try:
        if request.mode == "real":
            handoff = _run_real_analysis(request, request_id=request_id)
        else:
            handoff = _run_demo_analysis()
    except Exception as error:
        diagnostic = diagnose_integration_failure(error)
        logger.warning(
            "analysis failed request_id=%s mode=%s category=%s reason=%s",
            request_id,
            request.mode,
            diagnostic.category,
            diagnostic.reason_code,
        )
        return _error_response(map_integration_error(error))
    return handoff.model_dump(mode="json")


@app.post(
    "/api/v1/audio/summary",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}, **_ERROR_RESPONSES},
)
def audio_summary(request: AudioSummaryRequest):
    # Sync def: FastAPI runs it in the threadpool, so TTS never blocks the loop.
    provider, voice, language = _route_speech(request)
    fallback = ""
    started = time.perf_counter()
    try:
        text = normalize_for_speech(
            request.text, "es" if language.startswith("es") else "en"
        )
        if provider == "groq":
            try:
                result = synthesize_groq(text, voice=voice)
            except ValueError:
                raise
            except Exception as error:
                # Groq's free tier caps Orpheus at a few thousand characters
                # per day (429). Read it locally rather than failing the click.
                logger.warning("groq tts unavailable, using kokoro: %s", type(error).__name__)
                provider, voice, fallback = "local", DEFAULT_VOICE_BY_LANG["en-us"], "groq_unavailable"
                started = time.perf_counter()
                result = synthesize(text=text, voice=voice, language=language)
        else:
            result = synthesize(text=text, voice=voice, language=language)
    except (ValueError, AssertionError):  # Kokoro asserts on unknown voices
        return _error_response(
            IntegrationError(
                code="INPUT_ERROR",
                message="The text or voice is not supported for synthesis.",
                retryable=False,
            )
        )
    except Exception:
        logger.exception("synthesize failed provider=%s", request.provider)
        return JSONResponse(
            status_code=503,
            content={
                "detail": IntegrationError(
                    code="UNKNOWN_ERROR",
                    message="Audio synthesis is currently unavailable.",
                    retryable=True,
                ).model_dump()
            },
        )
    seconds = time.perf_counter() - started
    # Groq bills per character; Kokoro runs on this instance, so its cost is
    # the instance price for the seconds it was busy (an estimate).
    cost = groq_tts_cost(len(text)) if provider == "groq" else cpu_time_cost(seconds)
    audio_seconds = getattr(result, "duration", None)
    return Response(
        content=result.audio_bytes,
        media_type="audio/wav",
        # Engine/voice actually used plus metrics for the Performance tab.
        headers={
            "X-TTS-Provider": provider,
            "X-TTS-Voice": voice or "",
            "X-TTS-Duration-Ms": f"{seconds * 1000:.1f}",
            "X-TTS-Audio-Seconds": "" if audio_seconds is None else f"{audio_seconds:.2f}",
            "X-TTS-Chars": str(len(text)),
            "X-TTS-Cost-USD": f"{cost:.8f}",
            "X-TTS-Fallback": fallback,
        },
    )


def _route_speech(request: AudioSummaryRequest) -> tuple[str, str | None, str]:
    """Pick engine, voice and Kokoro language from the text's language.

    Groq Orpheus is English-only, so Spanish text always goes to Kokoro. A
    voice from the other language is replaced by that language's default.
    """

    voice_language = "es" if language_for_voice(request.voice).startswith("es") else "en"
    detected = detect_speech_language(request.text, default=voice_language)
    language = request.language or ("es" if detected == "es" else "en-us")
    spanish = language.startswith("es")
    if request.provider == "groq" and not spanish:
        return "groq", request.voice or DEFAULT_GROQ_VOICE, language
    voice = request.voice
    if voice is None or voice in GROQ_VOICES or (
        language_for_voice(voice).startswith("es") != spanish
    ):
        voice = DEFAULT_VOICE_BY_LANG["es" if spanish else "en-us"]
    return "local", voice, language


@app.post(
    "/api/v1/audio/transcribe",
    response_model=TranscriptionResponse,
    responses=_ERROR_RESPONSES,
)
def audio_transcribe(
    audio: bytes = Body(..., media_type="audio/wav"),
):
    """Speech-to-text for chat questions. Raw audio body, no multipart."""

    if not audio or len(audio) > MAX_TRANSCRIBE_BYTES:
        return _error_response(
            IntegrationError(
                code="INPUT_ERROR",
                message="The audio is empty or exceeds the supported size.",
                retryable=False,
            )
        )
    started = time.perf_counter()
    try:
        result = transcribe(audio)
    except ValueError:
        return _error_response(
            IntegrationError(
                code="INPUT_ERROR",
                message="The audio could not be transcribed.",
                retryable=False,
            )
        )
    except Exception:
        logger.exception("transcribe failed")
        return JSONResponse(
            status_code=503,
            content={
                "detail": IntegrationError(
                    code="UNKNOWN_ERROR",
                    message="Speech recognition is currently unavailable.",
                    retryable=True,
                ).model_dump()
            },
        )
    audio_seconds = getattr(result, "duration", None) or None
    return {
        "text": result.text,
        "language": result.language,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "audio_seconds": audio_seconds,
        "cost_usd": groq_stt_cost(audio_seconds or 0.0),
    }


def _chat_metrics(stream: object, model: str | None) -> dict:
    """Server-side timings plus OpenRouter's reported usage for one chat turn."""

    usage = getattr(stream, "usage", None)

    def millis(name: str) -> float | None:
        value = getattr(stream, name, None)
        return None if value is None else round(value * 1000, 1)

    return {
        "model": model,
        "ttft_ms": millis("first_token_s"),
        "total_ms": millis("total_s"),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "cost_usd": getattr(usage, "cost", None),
    }


@app.post(
    "/api/v1/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/plain": {}}}, **_ERROR_RESPONSES},
)
def chat(request: ChatRequest):
    """Stream an answer grounded only in the verified handoff sent by the client."""

    request_id = uuid.uuid4().hex
    if not request.handoff.verification.valid:
        return _error_response(
            IntegrationError(
                code="INPUT_ERROR",
                message="Chat is unavailable for analyses that failed verification.",
                retryable=False,
            )
        )
    system_prompt = (
        f"{CHAT_SYSTEM_PROMPT}\n{build_chat_context(request.handoff)}"
    )
    try:
        client = OpenRouterChatClient.from_env()
        stream = client.open(
            system_prompt=system_prompt,
            messages=[message.model_dump() for message in request.messages],
        )
    except Exception as error:
        diagnostic = diagnose_integration_failure(error)
        logger.warning(
            "chat failed request_id=%s category=%s reason=%s",
            request_id,
            diagnostic.category,
            diagnostic.reason_code,
        )
        return _error_response(map_integration_error(error))

    def tokens() -> Iterator[str]:
        try:
            yield from stream
            yield CHAT_METRICS_SEPARATOR + json.dumps(
                {"metrics": _chat_metrics(stream, getattr(client, "model", None))}
            )
        except Exception as error:
            diagnostic = diagnose_integration_failure(error)
            logger.warning(
                "chat stream interrupted request_id=%s category=%s reason=%s",
                request_id,
                diagnostic.category,
                diagnostic.reason_code,
            )
            yield STREAM_INTERRUPTED_MARKER

    return StreamingResponse(
        tokens(),
        media_type="text/plain; charset=utf-8",
        # Disable proxy buffering so tokens reach the browser as they arrive.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
