"""Stable data contracts for financial extraction and analysis.

The models in this module form the serialization boundary between the
extraction layer and downstream consumers such as visualization and audio.
They intentionally contain validation only; orchestration, retrieval, and
model calls belong in their respective modules.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FilingType = Literal["10-K", "10-Q", "10-K/A", "10-Q/A"]
SourceType = Literal["filing", "xbrl", "earnings_call", "unknown"]
Sentiment = Literal["positive", "neutral", "negative", "mixed", "unknown"]


class ExtractionSchema(BaseModel):
    """Common validation policy for extraction-layer contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class FinancialMetric(ExtractionSchema):
    """A comparable financial metric for the current and prior periods."""

    name: str = Field(min_length=1, description="Human-readable metric name.")
    current_value: float | None = Field(default=None, allow_inf_nan=False)
    previous_value: float | None = Field(default=None, allow_inf_nan=False)
    change_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Percentage change expressed in percentage points.",
    )
    unit: str | None = Field(
        default=None,
        min_length=1,
        description="Unit or currency associated with the metric.",
    )


class Evidence(ExtractionSchema):
    """A qualitative finding and the source evidence supporting it."""

    finding: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_section: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    source_type: SourceType = "unknown"


class ManagementOutlook(ExtractionSchema):
    """Management's forward-looking view and its overall sentiment."""

    summary: str = Field(min_length=1)
    sentiment: Sentiment = "unknown"


class FinancialAnalysisResult(ExtractionSchema):
    """Primary result returned by the financial analysis pipeline."""

    company: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    period: str = Field(min_length=1)
    filing_type: FilingType
    financial_metrics: list[FinancialMetric] = Field(default_factory=list)
    key_positive_developments: list[Evidence] = Field(default_factory=list)
    key_risks: list[Evidence] = Field(default_factory=list)
    management_outlook: ManagementOutlook
    executive_summary: str = Field(min_length=1)


class LoadedDocumentMetadata(ExtractionSchema):
    """Deterministic file metadata retained for document traceability."""

    file_name: str = Field(min_length=1)
    file_extension: str = Field(min_length=1, pattern=r"^\.")
    encoding: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LoadedDocument(ExtractionSchema):
    """A normalized source document ready for the chunking stage.

    Unlike the other contracts, this model preserves leading and trailing
    whitespace in ``text``. Whitespace is inspected only to reject documents
    without narrative content.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        validate_assignment=True,
    )

    ticker: str = Field(min_length=1)
    filing_type: FilingType
    period: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: SourceType
    metadata: LoadedDocumentMetadata

    @field_validator("ticker", "period", "source_id")
    @classmethod
    def normalize_identifier_text(cls, value: str) -> str:
        """Trim identifying values while rejecting whitespace-only strings."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("value must contain non-whitespace characters")
        return normalized

    @field_validator("text")
    @classmethod
    def validate_narrative_text(cls, value: str) -> str:
        """Reject blank documents without altering their narrative text."""

        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class DocumentChunk(ExtractionSchema):
    """A traceable text fragment shared by chunking and retrieval stages."""

    chunk_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    filing_type: FilingType
    period: str = Field(min_length=1)
    section: str | None = Field(default=None, min_length=1)
    text: str = Field(min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_character_range(self) -> Self:
        """Ensure a complete character range is ordered when both ends exist."""

        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class RetrievalResult(ExtractionSchema):
    """A ranked lexical retrieval hit with its source chunk."""

    chunk: DocumentChunk
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)


__all__ = [
    "DocumentChunk",
    "Evidence",
    "FilingType",
    "FinancialAnalysisResult",
    "FinancialMetric",
    "LoadedDocument",
    "LoadedDocumentMetadata",
    "ManagementOutlook",
    "RetrievalResult",
    "Sentiment",
    "SourceType",
]
