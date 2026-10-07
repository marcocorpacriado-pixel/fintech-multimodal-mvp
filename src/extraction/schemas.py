"""Stable data contracts for financial extraction and analysis.

The models in this module form the serialization boundary between the
extraction layer and downstream consumers such as visualization and audio.
They intentionally contain validation only; orchestration, retrieval, and
model calls belong in their respective modules.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FilingType = Literal["10-K", "10-Q", "10-K/A", "10-Q/A"]
SourceType = Literal["filing", "xbrl", "earnings_call", "unknown"]
Sentiment = Literal["positive", "neutral", "negative", "mixed", "unknown"]
XBRLPeriodType = Literal["instant", "duration"]
ComparisonType = Literal["QoQ", "YoY", "YoY_YTD"]
VerificationSeverity = Literal["error", "warning"]
VerificationCode = Literal[
    "METRICS_MISMATCH",
    "UNSUPPORTED_NUMBER",
    "INVALID_SOURCE_ID",
    "EVIDENCE_NOT_IN_SOURCE",
    "SECTION_MISMATCH",
    "UNGROUNDED_CLAIM",
    "SOURCE_TICKER_MISMATCH",
    "INVESTMENT_RECOMMENDATION",
    "SUMMARY_TOO_SHORT",
    "SUMMARY_TOO_LONG",
    "OUTLOOK_WITHOUT_EVIDENCE",
    "COMPANY_MISMATCH",
    "TICKER_MISMATCH",
    "PERIOD_MISMATCH",
]


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
    comparison_type: ComparisonType | None = None
    current_period: str | None = Field(default=None, min_length=1)
    previous_period: str | None = Field(default=None, min_length=1)
    source_ids: list[str] = Field(default_factory=list)


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
    source_ids: list[str] = Field(default_factory=list)


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


class NormalizedXBRLFact(ExtractionSchema):
    """Canonical long-form representation of one numeric XBRL fact."""

    ticker: str = Field(min_length=1)
    form: FilingType
    accession: str = Field(min_length=1)
    filing_date: date
    fact_id: str | None = Field(default=None, min_length=1)
    context_ref: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    standard_concept: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)
    numeric_value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    period_type: XBRLPeriodType
    period_start: date | None = None
    period_end: date | None = None
    period_instant: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = Field(default=None, min_length=1)
    dimensions: dict[str, str] = Field(default_factory=dict)
    statement_type: str | None = Field(default=None, min_length=1)
    amended: bool = False

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject blank axes or members and keep deterministic key ordering."""

        normalized: dict[str, str] = {}
        for axis, member in value.items():
            clean_axis = axis.strip()
            clean_member = member.strip()
            if not clean_axis or not clean_member:
                raise ValueError("dimension axes and members must be non-empty")
            normalized[clean_axis] = clean_member
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def validate_period_fields(self) -> Self:
        """Enforce mutually exclusive instant and duration period metadata."""

        if self.period_type == "instant":
            if self.period_instant is None:
                raise ValueError("period_instant is required for instant facts")
            if self.period_start is not None or self.period_end is not None:
                raise ValueError(
                    "period_start and period_end must be None for instant facts"
                )
            return self

        if self.period_start is None or self.period_end is None:
            raise ValueError(
                "period_start and period_end are required for duration facts"
            )
        if self.period_instant is not None:
            raise ValueError("period_instant must be None for duration facts")
        if self.period_start > self.period_end:
            raise ValueError("period_start must be before or equal to period_end")
        return self


class VerificationIssue(ExtractionSchema):
    """One stable, machine-readable deterministic verification finding."""

    code: VerificationCode
    severity: VerificationSeverity
    message: str = Field(min_length=1)
    field: str = Field(min_length=1)
    source_id: str | None = Field(default=None, min_length=1)


class VerificationReport(ExtractionSchema):
    """Aggregate verification outcome; warnings do not invalidate analysis."""

    valid: bool
    issues: list[VerificationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        expected = not any(issue.severity == "error" for issue in self.issues)
        if self.valid != expected:
            raise ValueError("valid must equal the absence of error issues")
        return self


class AnalysisPipelineResult(ExtractionSchema):
    """Auditable output of the end-to-end extraction and analysis pipeline."""

    analysis: FinancialAnalysisResult
    verification: VerificationReport
    queries: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )
    retrieval_count: int = Field(ge=0)
    retrieved_source_ids: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )
    generation_attempts: int = Field(default=1, ge=1, le=2)
    repair_used: bool = False
    first_failure_category: Literal["GROUNDING_ERROR", "VERIFICATION_ERROR"] | None = None

    @model_validator(mode="after")
    def validate_generation_metadata(self) -> Self:
        """Repair metadata must agree: a repair means exactly two attempts."""

        if self.repair_used != (self.generation_attempts > 1):
            raise ValueError("repair_used must match generation_attempts > 1")
        if self.repair_used != (self.first_failure_category is not None):
            raise ValueError("first_failure_category requires a repair attempt")
        return self

    @model_validator(mode="after")
    def validate_retrieval_metadata(self) -> Self:
        """Keep the compact retrieval audit metadata internally consistent."""

        if self.retrieval_count != len(self.retrieved_source_ids):
            raise ValueError(
                "retrieval_count must equal the number of retrieved_source_ids"
            )
        if len(set(self.retrieved_source_ids)) != len(self.retrieved_source_ids):
            raise ValueError("retrieved_source_ids must be unique")
        return self


__all__ = [
    "AnalysisPipelineResult",
    "ComparisonType",
    "DocumentChunk",
    "Evidence",
    "FilingType",
    "FinancialAnalysisResult",
    "FinancialMetric",
    "LoadedDocument",
    "LoadedDocumentMetadata",
    "ManagementOutlook",
    "NormalizedXBRLFact",
    "RetrievalResult",
    "Sentiment",
    "SourceType",
    "VerificationCode",
    "VerificationIssue",
    "VerificationReport",
    "VerificationSeverity",
    "XBRLPeriodType",
]
