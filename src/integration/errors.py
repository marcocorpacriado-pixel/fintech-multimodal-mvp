"""Safe mapping from domain exceptions to API-facing error contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.extraction.analysis_verifier import AnalysisVerificationError
from src.extraction.document_loader import (
    DocumentDecodingError,
    EmptyDocumentError,
    UnsupportedDocumentFormatError,
)
from src.extraction.financial_analyzer import (
    FinancialMetricSelectionError,
    XBRLNormalizationError,
)
from src.extraction.openrouter_client import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTotalDeadlineError,
    LLMTransportError,
)
from src.extraction.pipeline import (
    GroundedAnalysisError,
    ModelOutputRejectedError,
    PipelineAnalysisError,
    PipelineInputError,
    PipelineVerificationError,
)
from src.extraction.sec_ingestion import (
    SECFilingNotFoundError,
    SECIdentityError,
    SECIngestionError,
    SECInputError,
    SECNarrativeExtractionError,
    SECPreviousFilingNotFoundError,
    SECServiceError,
    SECXBRLUnavailableError,
)


IntegrationErrorCode = Literal[
    "INPUT_ERROR",
    "FILING_NOT_FOUND",
    "SEC_INGESTION_ERROR",
    "ANALYSIS_ERROR",
    "LLM_PROVIDER_ERROR",
    "GROUNDING_ERROR",
    "VERIFICATION_ERROR",
    "UNKNOWN_ERROR",
]


class IntegrationError(BaseModel):
    """Sanitized error payload safe to return from an API endpoint."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    code: IntegrationErrorCode
    message: str = Field(min_length=1)
    retryable: bool


class IntegrationDiagnostic(BaseModel):
    """Small internal-only failure description safe for structured logs.

    It deliberately excludes exception messages, prompts, evidence text and
    provider responses.  API consumers continue to receive ``IntegrationError``.
    """

    model_config = ConfigDict(extra="forbid")

    category: IntegrationErrorCode
    reason_code: str = Field(min_length=1)
    verification_issue_codes: list[str] = Field(default_factory=list)
    generation_attempts: int | None = None
    first_failure_category: str | None = None


_FILING_NOT_FOUND_ERRORS = (
    SECFilingNotFoundError,
    SECPreviousFilingNotFoundError,
)

_DETERMINISTIC_SEC_ERRORS = (
    SECIdentityError,
    SECNarrativeExtractionError,
    SECXBRLUnavailableError,
)

_INPUT_ERRORS = (
    PipelineInputError,
    UnsupportedDocumentFormatError,
    DocumentDecodingError,
    EmptyDocumentError,
    FileNotFoundError,
    IsADirectoryError,
)

_ANALYSIS_ERRORS = (
    PipelineAnalysisError,
    FinancialMetricSelectionError,
    XBRLNormalizationError,
)


def map_integration_error(error: BaseException) -> IntegrationError:
    """Map a domain failure to a stable category without leaking internals.

    The exception chain is inspected because D7 deliberately wraps provider
    and grounding failures with pipeline context.  Public messages are fixed
    by category, so provider bodies, prompts, credentials, and stack traces
    can never be reflected to the client.
    """

    chain = _exception_chain(error)

    if _contains(chain, PipelineVerificationError, AnalysisVerificationError):
        return IntegrationError(
            code="VERIFICATION_ERROR",
            message="Analysis failed deterministic verification.",
            # The invalid result remains blocked, but a new provider generation
            # may comply. Retry is always explicit; there is no automatic fallback.
            retryable=True,
        )
    grounding = next(
        (item for item in chain if isinstance(item, GroundedAnalysisError)),
        None,
    )
    if grounding is not None:
        return IntegrationError(
            code="GROUNDING_ERROR",
            message="Generated analysis failed evidence grounding.",
            # Only a rejected model generation can differ on a new attempt;
            # empty retrieval and pre-generation input problems are deterministic.
            retryable=(
                isinstance(grounding, ModelOutputRejectedError)
                and grounding.reason_code != "EMPTY_RETRIEVAL"
            ),
        )
    if _contains(chain, LLMConfigurationError):
        return IntegrationError(
            code="LLM_PROVIDER_ERROR",
            message="The language-model provider is not configured correctly.",
            retryable=False,
        )
    if _contains(chain, LLMTransportError, LLMResponseError, LLMProviderError):
        return IntegrationError(
            code="LLM_PROVIDER_ERROR",
            message="The language-model provider could not complete the request.",
            retryable=True,
        )
    if _contains(chain, SECInputError):
        return IntegrationError(
            code="INPUT_ERROR",
            message="The SEC request contains invalid input.",
            retryable=False,
        )
    if _contains(chain, *_FILING_NOT_FOUND_ERRORS):
        return IntegrationError(
            code="FILING_NOT_FOUND",
            message="The requested or comparable SEC filing was not found.",
            retryable=False,
        )
    if _contains(chain, *_DETERMINISTIC_SEC_ERRORS):
        return IntegrationError(
            code="SEC_INGESTION_ERROR",
            message="SEC inputs could not be prepared for analysis.",
            retryable=False,
        )
    if _contains(chain, SECServiceError, SECIngestionError):
        return IntegrationError(
            code="SEC_INGESTION_ERROR",
            message="The SEC data service could not complete the request.",
            retryable=True,
        )
    if _contains(chain, *_INPUT_ERRORS):
        return IntegrationError(
            code="INPUT_ERROR",
            message="The analysis request contains invalid or unavailable input.",
            retryable=False,
        )
    if _contains(chain, *_ANALYSIS_ERRORS):
        return IntegrationError(
            code="ANALYSIS_ERROR",
            message="Financial analysis could not be completed.",
            retryable=False,
        )
    return IntegrationError(
        code="UNKNOWN_ERROR",
        message="An unexpected integration error occurred.",
        retryable=False,
    )


def diagnose_integration_failure(error: BaseException) -> IntegrationDiagnostic:
    """Classify a failure for safe internal logs without leaking source text."""

    chain = _exception_chain(error)
    mapped = map_integration_error(error)
    attempts = next(
        (item.attempts for item in chain if getattr(item, "attempts", ())),
        (),
    )
    trace = {
        "generation_attempts": len(attempts) or None,
        "first_failure_category": (
            attempts[0].outcome if len(attempts) > 1 else None
        ),
    }
    verification = next(
        (
            item
            for item in chain
            if isinstance(item, (PipelineVerificationError, AnalysisVerificationError))
        ),
        None,
    )
    if verification is not None:
        issue_codes = sorted({issue.code for issue in verification.report.issues})
        return IntegrationDiagnostic(
            category=mapped.code,
            reason_code="DETERMINISTIC_VERIFICATION_REJECTED",
            verification_issue_codes=issue_codes,
            **trace,
        )

    grounding = next(
        (item for item in chain if isinstance(item, GroundedAnalysisError)),
        None,
    )
    if grounding is not None:
        return IntegrationDiagnostic(
            category=mapped.code,
            reason_code=_grounding_reason_code(grounding),
            **trace,
        )

    if _contains(chain, LLMTotalDeadlineError):
        return IntegrationDiagnostic(
            category=mapped.code,
            reason_code=LLMTotalDeadlineError.reason_code,
            **trace,
        )

    return IntegrationDiagnostic(
        category=mapped.code,
        reason_code={
            "INPUT_ERROR": "INVALID_INPUT",
            "FILING_NOT_FOUND": "FILING_NOT_FOUND",
            "SEC_INGESTION_ERROR": "SEC_INGESTION_FAILED",
            "ANALYSIS_ERROR": "ANALYSIS_FAILED",
            "LLM_PROVIDER_ERROR": "LLM_PROVIDER_FAILED",
            "UNKNOWN_ERROR": "UNEXPECTED_FAILURE",
        }.get(mapped.code, "UNCLASSIFIED_FAILURE"),
        **trace,
    )


def _grounding_reason_code(error: GroundedAnalysisError) -> str:
    """Stable, non-sensitive reason code for a grounding failure."""

    if isinstance(error, ModelOutputRejectedError):
        return error.reason_code
    return "GROUNDING_REJECTED"


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _contains(
    chain: tuple[BaseException, ...],
    *error_types: type[BaseException],
) -> bool:
    return any(isinstance(error, error_types) for error in chain)


__all__ = [
    "IntegrationDiagnostic",
    "IntegrationError",
    "IntegrationErrorCode",
    "diagnose_integration_failure",
    "map_integration_error",
]
