"""FastAPI orchestration layer between Dani's pipeline, Cristian's audio and the UI.

This module only orchestrates and serializes. Financial values come from the
``AnalysisHandoff`` contract unchanged; no metric is recalculated here.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from src.audio import list_voices
from src.audio.tts import synthesize
from src.extraction import (
    AnalysisPipelineResult,
    OpenRouterLLMClient,
    PipelineInputError,
    prepare_sec_analysis_inputs,
    run_analysis_pipeline,
)
from src.extraction.schemas import FilingType
from src.integration import (
    AnalysisHandoff,
    AnalysisMode,
    IntegrationError,
    build_analysis_handoff,
    map_integration_error,
)

load_dotenv()
logger = logging.getLogger(__name__)

DEMO_FIXTURE_PATH = Path(__file__).with_name("demo_fixture.json")
DEFAULT_VOICE = "af_heart"
_ERROR_STATUS = {
    "INPUT_ERROR": 422,
    "VERIFICATION_ERROR": 422,
    "GROUNDING_ERROR": 422,
    "LLM_PROVIDER_ERROR": 503,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalysisRequest(StrictModel):
    ticker: str = Field(min_length=1, max_length=10)
    # Real mode: SEC filing date (YYYY-MM-DD), the selector Dani's ingestion supports.
    period: str = Field(min_length=1, max_length=32)
    filing_type: FilingType = "10-Q"
    mode: AnalysisMode = "demo"


class AudioSummaryRequest(StrictModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = DEFAULT_VOICE


class HealthResponse(StrictModel):
    status: Literal["ok"]


class VoicesResponse(StrictModel):
    voices: list[str]


class ErrorResponse(StrictModel):
    detail: IntegrationError


_ERROR_RESPONSES = {
    status: {"model": ErrorResponse} for status in (422, 500, 503)
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


def _run_real_analysis(request: AnalysisRequest) -> AnalysisHandoff:
    """SEC ingestion -> grounded pipeline -> handoff. No fallback to demo."""

    try:
        filing_date = date.fromisoformat(request.period)
    except ValueError as error:
        raise PipelineInputError("period must be an ISO date") from error

    inputs = prepare_sec_analysis_inputs(
        ticker=request.ticker,
        filing_date=filing_date,
        form=request.filing_type,
    )
    with OpenRouterLLMClient.from_env() as llm:
        result = run_analysis_pipeline(
            filing_path=inputs.filing_path,
            company=inputs.company,
            ticker=inputs.ticker,
            period=inputs.period,
            filing_type=inputs.filing_type,
            current_xbrl_filing=inputs.current_filing,
            previous_xbrl_filing=inputs.previous_filing,
            llm_client=llm,
        )
        return build_analysis_handoff(
            result, analysis_mode="real", provider="openrouter", model=llm.model
        )


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/audio/voices", response_model=VoicesResponse)
def voices() -> dict:
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
    try:
        if request.mode == "real":
            handoff = _run_real_analysis(request)
        else:
            handoff = _run_demo_analysis()
    except Exception as error:
        logger.exception("analysis failed (mode=%s)", request.mode)
        return _error_response(map_integration_error(error))
    return handoff.model_dump(mode="json")


@app.post(
    "/api/v1/audio/summary",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}, **_ERROR_RESPONSES},
)
def audio_summary(request: AudioSummaryRequest):
    # Sync def: FastAPI runs it in the threadpool, so Kokoro never blocks the loop.
    try:
        result = synthesize(text=request.text, voice=request.voice)
    except (ValueError, AssertionError):  # Kokoro asserts on unknown voices
        return _error_response(
            IntegrationError(
                code="INPUT_ERROR",
                message="The text or voice is not supported for synthesis.",
                retryable=False,
            )
        )
    except Exception:
        logger.exception("synthesize failed")
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
    return Response(content=result.audio_bytes, media_type="audio/wav")
