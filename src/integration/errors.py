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
    LLMTransportError,
)
from src.extraction.pipeline import (
    GroundedAnalysisError,
    PipelineAnalysisError,
    PipelineInputError,
    PipelineVerificationError,
)
from src.extraction.sec_ingestion import (
    SECFilingNotFoundError,
    SECIdentityError,
    SECIngestionError,
    SECNarrativeExtractionError,
    SECPreviousFilingNotFoundError,
    SECXBRLUnavailableError,
)


IntegrationErrorCode = Literal[
    "INPUT_ERROR",
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


_DETERMINISTIC_SEC_ERRORS = (
    SECIdentityError,
    SECFilingNotFoundError,
    SECPreviousFilingNotFoundError,
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
            retryable=False,
        )
    if _contains(chain, GroundedAnalysisError):
        return IntegrationError(
            code="GROUNDING_ERROR",
            message="Generated analysis failed evidence grounding.",
            retryable=False,
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
    if _contains(chain, *_DETERMINISTIC_SEC_ERRORS):
        return IntegrationError(
            code="SEC_INGESTION_ERROR",
            message="SEC inputs could not be prepared for analysis.",
            retryable=False,
        )
    if _contains(chain, SECIngestionError):
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
    "IntegrationError",
    "IntegrationErrorCode",
    "map_integration_error",
]
