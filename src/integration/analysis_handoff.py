"""Stable serialization boundary for analysis, UI, and audio consumers.

The DTOs in this module deliberately copy pipeline output without deriving or
recalculating financial values.  Downstream API and presentation layers can
therefore evolve without depending on extraction-layer implementation details.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.extraction.schemas import (
    AnalysisPipelineResult,
    ComparisonType,
    FilingType,
    Sentiment,
    SourceType,
    VerificationSeverity,
)


AnalysisMode = Literal["real", "demo"]


class IntegrationSchema(BaseModel):
    """Validation policy for objects crossing the integration boundary."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class AnalysisMetricDTO(IntegrationSchema):
    """Display-ready copy of one canonical D5C metric."""

    name: str = Field(min_length=1)
    current_value: float | None = Field(default=None, allow_inf_nan=False)
    previous_value: float | None = Field(default=None, allow_inf_nan=False)
    change_pct: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = Field(default=None, min_length=1)
    comparison_type: ComparisonType | None = None
    current_period: str | None = Field(default=None, min_length=1)
    previous_period: str | None = Field(default=None, min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class AnalysisEvidenceDTO(IntegrationSchema):
    """Grounded qualitative claim exposed to API and presentation layers."""

    finding: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_section: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    source_type: SourceType


class TokenAttributionDTO(IntegrationSchema):
    """Word-level FinBERT attribution for the outlook rationale sentence."""

    token: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class ManagementOutlookDTO(IntegrationSchema):
    """Verified management outlook handoff."""

    summary: str = Field(min_length=1)
    sentiment: Sentiment
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = Field(default=None, min_length=1)
    rationale_sentence: str | None = Field(default=None, min_length=1)
    rationale_score: float | None = Field(default=None, ge=0.0, le=1.0)
    token_attributions: list[TokenAttributionDTO] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class VerificationIssueDTO(IntegrationSchema):
    """Stable issue representation suitable for an API response."""

    code: str = Field(min_length=1)
    severity: VerificationSeverity
    message: str = Field(min_length=1)
    field: str = Field(min_length=1)
    source_id: str | None = Field(default=None, min_length=1)


class VerificationDTO(IntegrationSchema):
    """Deterministic verification state accompanying the analysis."""

    valid: bool
    issues: list[VerificationIssueDTO] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        expected = not any(issue.severity == "error" for issue in self.issues)
        if self.valid != expected:
            raise ValueError("valid must equal the absence of error issues")
        return self


class InferenceMetricsDTO(IntegrationSchema):
    """Measured latency and provider cost of one analysis run.

    Filled by the API for real analyses (``None`` for the demo fixture). The
    cost is the amount OpenRouter reports as charged, not an estimate.
    """

    total_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sec_ingestion_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    llm_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class PipelineMetadataDTO(IntegrationSchema):
    """Compact execution metadata, excluding prompts and source documents."""

    analysis_mode: AnalysisMode
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    filing_date: date | None = None
    effective_queries: list[str] = Field(default_factory=list)
    retrieval_count: int = Field(ge=0)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    generation_attempts: int = Field(default=1, ge=1, le=2)
    repair_used: bool = False
    first_failure_category: (
        Literal["GROUNDING_ERROR", "VERIFICATION_ERROR"] | None
    ) = None
    metrics: InferenceMetricsDTO | None = None

    @model_validator(mode="after")
    def validate_retrieval_metadata(self) -> Self:
        if self.retrieval_count != len(self.retrieved_source_ids):
            raise ValueError(
                "retrieval_count must equal the number of retrieved_source_ids"
            )
        if len(set(self.retrieved_source_ids)) != len(self.retrieved_source_ids):
            raise ValueError("retrieved_source_ids must be unique")
        return self


class AnalysisHandoff(IntegrationSchema):
    """Stable, JSON-serializable result consumed by Marco's API and UI."""

    company: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    period: str = Field(min_length=1)
    filing_type: FilingType
    financial_metrics: list[AnalysisMetricDTO] = Field(default_factory=list)
    positives: list[AnalysisEvidenceDTO] = Field(default_factory=list)
    risks: list[AnalysisEvidenceDTO] = Field(default_factory=list)
    management_outlook: ManagementOutlookDTO
    executive_summary: str = Field(min_length=1)
    verification: VerificationDTO
    pipeline_metadata: PipelineMetadataDTO


class TTSInput(IntegrationSchema):
    """Minimal verified-analysis payload for Cristian's ``synthesize`` API."""

    text: str = Field(min_length=1)
    company: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    period: str = Field(min_length=1)
    filing_type: FilingType
    analysis_mode: AnalysisMode
    verification_valid: bool


def build_analysis_handoff(
    result: AnalysisPipelineResult,
    *,
    analysis_mode: AnalysisMode,
    provider: str | None = None,
    model: str | None = None,
    filing_date: date | None = None,
) -> AnalysisHandoff:
    """Copy one pipeline result into the stable downstream DTO.

    ``analysis_mode`` is mandatory so demo fixtures cannot be represented as a
    real provider execution by omission.  Every financial value is copied
    directly from D5C output; this function performs no arithmetic.
    """

    analysis = result.analysis
    return AnalysisHandoff(
        company=analysis.company,
        ticker=analysis.ticker,
        period=analysis.period,
        filing_type=analysis.filing_type,
        financial_metrics=[
            AnalysisMetricDTO.model_validate(metric.model_dump(mode="python"))
            for metric in analysis.financial_metrics
        ],
        positives=[
            AnalysisEvidenceDTO.model_validate(item.model_dump(mode="python"))
            for item in analysis.key_positive_developments
        ],
        risks=[
            AnalysisEvidenceDTO.model_validate(item.model_dump(mode="python"))
            for item in analysis.key_risks
        ],
        management_outlook=ManagementOutlookDTO.model_validate(
            analysis.management_outlook.model_dump(mode="python")
        ),
        executive_summary=analysis.executive_summary,
        verification=VerificationDTO(
            valid=result.verification.valid,
            issues=[
                VerificationIssueDTO.model_validate(issue.model_dump(mode="python"))
                for issue in result.verification.issues
            ],
        ),
        pipeline_metadata=PipelineMetadataDTO(
            analysis_mode=analysis_mode,
            provider=provider,
            model=model,
            filing_date=filing_date,
            effective_queries=list(result.queries),
            retrieval_count=result.retrieval_count,
            retrieved_source_ids=list(result.retrieved_source_ids),
            generation_attempts=result.generation_attempts,
            repair_used=result.repair_used,
            first_failure_category=result.first_failure_category,
        ),
    )


def build_tts_input(
    result: AnalysisPipelineResult,
    *,
    analysis_mode: AnalysisMode,
) -> TTSInput:
    """Build the minimal payload whose ``text`` feeds ``synthesize(text)``."""

    analysis = result.analysis
    return TTSInput(
        text=analysis.executive_summary,
        company=analysis.company,
        ticker=analysis.ticker,
        period=analysis.period,
        filing_type=analysis.filing_type,
        analysis_mode=analysis_mode,
        verification_valid=result.verification.valid,
    )


__all__ = [
    "AnalysisEvidenceDTO",
    "AnalysisHandoff",
    "AnalysisMetricDTO",
    "AnalysisMode",
    "InferenceMetricsDTO",
    "ManagementOutlookDTO",
    "TokenAttributionDTO",
    "PipelineMetadataDTO",
    "TTSInput",
    "VerificationDTO",
    "VerificationIssueDTO",
    "build_analysis_handoff",
    "build_tts_input",
]
