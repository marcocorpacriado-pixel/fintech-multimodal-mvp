"""Offline contract tests for the D8C analysis handoff boundary."""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.extraction.openrouter_client import (
    LLMResponseError,
    LLMTotalDeadlineError,
    LLMTransportError,
)
from src.extraction.pipeline import (
    GenerationAttempt,
    GroundedAnalysisError,
    ModelOutputProblem,
    ModelOutputRejectedError,
    PipelineAnalysisError,
    PipelineInputError,
    PipelineVerificationError,
)
from src.extraction.schemas import (
    AnalysisPipelineResult,
    Evidence,
    FinancialAnalysisResult,
    FinancialMetric,
    ManagementOutlook,
    VerificationIssue,
    VerificationReport,
)
from src.extraction.sec_ingestion import (
    SECFilingNotFoundError,
    SECIngestionError,
    SECInputError,
    SECServiceError,
)
from src.integration import (
    AnalysisHandoff,
    PipelineMetadataDTO,
    VerificationDTO,
    VerificationIssueDTO,
    build_analysis_handoff,
    build_tts_input,
    diagnose_integration_failure,
    map_integration_error,
)


def _metric(
    name: str,
    current: float,
    previous: float,
    change: float,
    comparison: str,
    index: int,
) -> FinancialMetric:
    return FinancialMetric(
        name=name,
        current_value=current,
        previous_value=previous,
        change_pct=change,
        unit="usdPerShare" if name == "Diluted EPS" else "usd",
        comparison_type=comparison,
        current_period=f"current-{index}",
        previous_period=f"previous-{index}",
        source_ids=[f"xbrl:current:{index}", f"xbrl:previous:{index}"],
    )


@pytest.fixture
def pipeline_result() -> AnalysisPipelineResult:
    metrics = [
        _metric("Revenue", 109_417_000_000.0, 111_184_000_000.0, -1.589, "QoQ", 1),
        _metric("Net Income", 29_789_000_000.0, 29_578_000_000.0, 0.713, "QoQ", 2),
        _metric("Diluted EPS", 2.02, 2.01, 0.498, "QoQ", 3),
        _metric(
            "Cash and Cash Equivalents",
            39_544_000_000.0,
            45_572_000_000.0,
            -13.227,
            "QoQ",
            4,
        ),
        _metric("Total Debt", 84_344_000_000.0, 84_711_000_000.0, -0.433, "QoQ", 5),
        _metric(
            "Operating Cash Flow",
            116_996_000_000.0,
            81_754_000_000.0,
            43.107,
            "YoY_YTD",
            6,
        ),
        _metric(
            "Capital Expenditures",
            6_799_000_000.0,
            9_473_000_000.0,
            -28.228,
            "YoY_YTD",
            7,
        ),
    ]
    analysis = FinancialAnalysisResult(
        company="Apple Inc.",
        ticker="AAPL",
        period="2026-06-27",
        filing_type="10-Q",
        financial_metrics=metrics,
        key_positive_developments=[
            Evidence(
                finding="Operating performance remained resilient.",
                evidence="Net sales increased in the period.",
                source_section="PART_I_ITEM_2",
                source_id="chunk:positive",
                source_type="filing",
            )
        ],
        key_risks=[
            Evidence(
                finding="Liquidity requires monitoring.",
                evidence="Cash and cash equivalents decreased.",
                source_section="PART_I_ITEM_1",
                source_id="chunk:risk",
                source_type="filing",
            )
        ],
        management_outlook=ManagementOutlook(
            summary="Management did not provide sufficiently grounded guidance.",
            sentiment="unknown",
            source_ids=[],
        ),
        executive_summary="Apple reported a mixed but resilient quarter.",
    )
    warning = VerificationIssue(
        code="SUMMARY_TOO_SHORT",
        severity="warning",
        message="Executive summary is shorter than the preferred range.",
        field="executive_summary",
    )
    return AnalysisPipelineResult(
        analysis=analysis,
        verification=VerificationReport(valid=True, issues=[warning]),
        queries=["revenue performance", "liquidity risk"],
        retrieval_count=2,
        retrieved_source_ids=["chunk:positive", "chunk:risk"],
    )


@pytest.fixture
def handoff(pipeline_result: AnalysisPipelineResult) -> AnalysisHandoff:
    return build_analysis_handoff(
        pipeline_result,
        analysis_mode="demo",
        provider="fixture",
        model="deterministic-fake",
        filing_date=date(2026, 7, 31),
    )


def test_valid_pipeline_result_builds_serializable_handoff(
    handoff: AnalysisHandoff,
) -> None:
    payload = json.loads(handoff.model_dump_json())

    assert payload["company"] == "Apple Inc."
    assert payload["pipeline_metadata"]["analysis_mode"] == "demo"
    assert payload["pipeline_metadata"]["filing_date"] == "2026-07-31"


def test_all_seven_metrics_preserve_exact_values(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    source = [item.model_dump() for item in pipeline_result.analysis.financial_metrics]
    target = [item.model_dump() for item in handoff.financial_metrics]

    assert len(target) == 7
    assert target == source


def test_comparison_types_are_preserved(handoff: AnalysisHandoff) -> None:
    assert [metric.comparison_type for metric in handoff.financial_metrics] == [
        "QoQ",
        "QoQ",
        "QoQ",
        "QoQ",
        "QoQ",
        "YoY_YTD",
        "YoY_YTD",
    ]


def test_metric_source_ids_are_preserved(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    assert [item.source_ids for item in handoff.financial_metrics] == [
        item.source_ids for item in pipeline_result.analysis.financial_metrics
    ]


def test_positives_are_preserved(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    expected = pipeline_result.analysis.key_positive_developments[0].model_dump()
    assert handoff.positives[0].model_dump() == expected


def test_risks_are_preserved(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    expected = pipeline_result.analysis.key_risks[0].model_dump()
    assert handoff.risks[0].model_dump() == expected


def test_management_outlook_is_preserved(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    expected = pipeline_result.analysis.management_outlook.model_dump()
    assert handoff.management_outlook.model_dump() == expected


def test_executive_summary_is_string_equivalent(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    assert handoff.executive_summary == pipeline_result.analysis.executive_summary


def test_verification_and_warning_are_preserved(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    assert handoff.verification.valid is True
    assert handoff.verification.issues[0].model_dump() == (
        pipeline_result.verification.issues[0].model_dump()
    )


def test_pipeline_retrieval_metadata_is_preserved(
    pipeline_result: AnalysisPipelineResult,
    handoff: AnalysisHandoff,
) -> None:
    assert handoff.pipeline_metadata.effective_queries == pipeline_result.queries
    assert handoff.pipeline_metadata.retrieval_count == pipeline_result.retrieval_count
    assert (
        handoff.pipeline_metadata.retrieved_source_ids
        == pipeline_result.retrieved_source_ids
    )


def test_default_generation_metadata_reports_single_attempt(
    handoff: AnalysisHandoff,
) -> None:
    meta = handoff.pipeline_metadata
    assert (meta.generation_attempts, meta.repair_used) == (1, False)
    assert meta.first_failure_category is None


def test_repair_metadata_is_carried_to_the_handoff(
    pipeline_result: AnalysisPipelineResult,
) -> None:
    repaired = pipeline_result.model_copy(
        update={
            "generation_attempts": 2,
            "repair_used": True,
            "first_failure_category": "GROUNDING_ERROR",
        }
    )

    meta = build_analysis_handoff(repaired, analysis_mode="real").pipeline_metadata

    assert (meta.generation_attempts, meta.repair_used) == (2, True)
    assert meta.first_failure_category == "GROUNDING_ERROR"
    assert "raw" not in meta.model_dump_json()


@pytest.mark.parametrize(
    "fields",
    [
        {"generation_attempts": 2, "repair_used": False},
        {"generation_attempts": 1, "repair_used": True},
        {"generation_attempts": 2, "repair_used": True},  # missing first failure
        {"generation_attempts": 3, "repair_used": True,
         "first_failure_category": "GROUNDING_ERROR"},
    ],
)
def test_inconsistent_repair_metadata_is_rejected(
    pipeline_result: AnalysisPipelineResult,
    fields: dict[str, object],
) -> None:
    payload = pipeline_result.model_dump()
    payload.update(fields)

    with pytest.raises(ValueError):
        AnalysisPipelineResult.model_validate(payload)


def test_handoff_round_trips_through_json(handoff: AnalysisHandoff) -> None:
    restored = AnalysisHandoff.model_validate_json(handoff.model_dump_json())
    assert restored == handoff


def test_integration_metadata_rejects_inconsistent_retrieval_count() -> None:
    with pytest.raises(ValueError, match="retrieval_count"):
        PipelineMetadataDTO(
            analysis_mode="demo",
            effective_queries=[],
            retrieval_count=1,
            retrieved_source_ids=[],
        )


def test_integration_verification_rejects_inconsistent_valid_flag() -> None:
    with pytest.raises(ValueError, match="absence of error issues"):
        VerificationDTO(
            valid=True,
            issues=[
                VerificationIssueDTO(
                    code="METRICS_MISMATCH",
                    severity="error",
                    message="Metrics changed.",
                    field="financial_metrics",
                )
            ],
        )


def test_handoff_does_not_recalculate_change_percentage(
    pipeline_result: AnalysisPipelineResult,
) -> None:
    pipeline_result.analysis.financial_metrics[0].change_pct = 987.654321
    handoff = build_analysis_handoff(pipeline_result, analysis_mode="demo")

    assert handoff.financial_metrics[0].change_pct == 987.654321


def test_analysis_mode_is_explicit_and_provider_metadata_is_optional(
    pipeline_result: AnalysisPipelineResult,
) -> None:
    with pytest.raises(TypeError):
        build_analysis_handoff(pipeline_result)  # type: ignore[call-arg]

    handoff = build_analysis_handoff(pipeline_result, analysis_mode="real")
    assert handoff.pipeline_metadata.analysis_mode == "real"
    assert handoff.pipeline_metadata.provider is None
    assert handoff.pipeline_metadata.model is None


def test_empty_findings_and_unknown_outlook_serialize(
    pipeline_result: AnalysisPipelineResult,
) -> None:
    pipeline_result.analysis.key_positive_developments = []
    pipeline_result.analysis.key_risks = []
    handoff = build_analysis_handoff(pipeline_result, analysis_mode="demo")

    assert handoff.positives == []
    assert handoff.risks == []
    assert json.loads(handoff.model_dump_json())["management_outlook"]["sentiment"] == (
        "unknown"
    )


def test_tts_input_is_exact_executive_summary(
    pipeline_result: AnalysisPipelineResult,
) -> None:
    tts = build_tts_input(pipeline_result, analysis_mode="demo")
    assert tts.text == pipeline_result.analysis.executive_summary


def test_tts_metadata_is_minimal_and_correct(
    pipeline_result: AnalysisPipelineResult,
) -> None:
    tts = build_tts_input(pipeline_result, analysis_mode="real")
    assert tts.model_dump() == {
        "text": "Apple reported a mixed but resilient quarter.",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "period": "2026-06-27",
        "filing_type": "10-Q",
        "analysis_mode": "real",
        "verification_valid": True,
    }


def _rejection(code: str, message: str = "rejected") -> ModelOutputRejectedError:
    return ModelOutputRejectedError([ModelOutputProblem("output", code, message)])


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (PipelineInputError("bad path"), "INPUT_ERROR", False),
        (SECInputError("bad ticker"), "INPUT_ERROR", False),
        (SECFilingNotFoundError("missing"), "FILING_NOT_FOUND", False),
        (SECServiceError("network"), "SEC_INGESTION_ERROR", True),
        (SECIngestionError("network"), "SEC_INGESTION_ERROR", True),
        (_rejection("INVALID_EVIDENCE_ID"), "GROUNDING_ERROR", True),
        (GroundedAnalysisError("pre-generation input"), "GROUNDING_ERROR", False),
        (LLMTransportError("timeout"), "LLM_PROVIDER_ERROR", True),
        (PipelineAnalysisError("analysis"), "ANALYSIS_ERROR", False),
        (ValueError("unexpected"), "UNKNOWN_ERROR", False),
    ],
)
def test_error_mapping_categories(
    error: BaseException,
    code: str,
    retryable: bool,
) -> None:
    mapped = map_integration_error(error)
    assert mapped.code == code
    assert mapped.retryable is retryable


def test_wrapped_provider_error_is_detected() -> None:
    provider_error = LLMResponseError("empty response")
    pipeline_error = PipelineAnalysisError("analysis failed")
    pipeline_error.__cause__ = provider_error

    mapped = map_integration_error(pipeline_error)
    assert mapped.code == "LLM_PROVIDER_ERROR"
    assert mapped.retryable is True


def test_total_deadline_maps_to_safe_retryable_provider_error() -> None:
    secret = "sk-or-secret-provider-body"
    error = LLMTotalDeadlineError(secret)

    mapped = map_integration_error(error)
    diagnostic = diagnose_integration_failure(error)

    assert mapped.code == "LLM_PROVIDER_ERROR"
    assert mapped.retryable is True
    assert secret not in mapped.model_dump_json()
    assert diagnostic.category == "LLM_PROVIDER_ERROR"
    assert diagnostic.reason_code == "TOTAL_DEADLINE_EXCEEDED"
    assert secret not in diagnostic.model_dump_json()


def test_verification_error_mapping() -> None:
    report = VerificationReport(
        valid=False,
        issues=[
            VerificationIssue(
                code="METRICS_MISMATCH",
                severity="error",
                message="Metrics changed.",
                field="financial_metrics",
            )
        ],
    )
    mapped = map_integration_error(PipelineVerificationError(report))
    assert mapped.code == "VERIFICATION_ERROR"
    assert mapped.retryable is True


def test_empty_retrieval_grounding_failure_is_not_retryable() -> None:
    mapped = map_integration_error(_rejection("EMPTY_RETRIEVAL"))

    assert mapped.code == "GROUNDING_ERROR"
    assert mapped.retryable is False


def test_internal_diagnostic_exposes_verifier_codes_without_issue_text() -> None:
    secret = "private evidence sk-or-secret"
    report = VerificationReport(
        valid=False,
        issues=[
            VerificationIssue(
                code="UNSUPPORTED_NUMBER",
                severity="error",
                message=secret,
                field="key_positive_developments[1].finding",
            )
        ],
    )

    diagnostic = diagnose_integration_failure(PipelineVerificationError(report))
    payload = diagnostic.model_dump_json()

    assert diagnostic.category == "VERIFICATION_ERROR"
    assert diagnostic.reason_code == "DETERMINISTIC_VERIFICATION_REJECTED"
    assert diagnostic.verification_issue_codes == ["UNSUPPORTED_NUMBER"]
    assert secret not in payload


@pytest.mark.parametrize(
    "code",
    [
        "INVALID_EVIDENCE_ID",
        "INVALID_OUTLOOK_EVIDENCE_ID",
        "OUTLOOK_WITHOUT_EVIDENCE",
        "EMPTY_RETRIEVAL",
        "INVALID_OUTPUT_SCHEMA",
    ],
)
def test_internal_grounding_diagnostic_uses_stable_safe_reason_codes(
    code: str,
) -> None:
    diagnostic = diagnose_integration_failure(
        _rejection(code, "private secret model text")
    )

    assert diagnostic.category == "GROUNDING_ERROR"
    assert diagnostic.reason_code == code
    assert "secret" not in diagnostic.model_dump_json()


def test_untyped_grounding_failure_has_generic_diagnostic() -> None:
    diagnostic = diagnose_integration_failure(GroundedAnalysisError("secret"))

    assert diagnostic.reason_code == "GROUNDING_REJECTED"
    assert "secret" not in diagnostic.model_dump_json()


def test_diagnostic_reports_repair_trace_without_content() -> None:
    error = PipelineAnalysisError("boom")
    error.attempts = (
        GenerationAttempt(1, "VERIFICATION_ERROR", ("UNSUPPORTED_NUMBER",)),
        GenerationAttempt(2, "GROUNDING_ERROR", ("INVALID_EVIDENCE_ID",)),
    )
    error.__cause__ = _rejection("INVALID_EVIDENCE_ID", "private secret")

    diagnostic = diagnose_integration_failure(error)

    assert diagnostic.generation_attempts == 2
    assert diagnostic.first_failure_category == "VERIFICATION_ERROR"
    assert diagnostic.reason_code == "INVALID_EVIDENCE_ID"
    assert "secret" not in diagnostic.model_dump_json()


def test_serialized_errors_never_reflect_secrets_or_internal_messages() -> None:
    secret = "sk-or-v1-super-secret-token"
    error = LLMResponseError(
        f"Authorization: Bearer {secret}; prompt=complete filing contents"
    )

    payload = map_integration_error(error).model_dump_json()
    assert secret not in payload
    assert "Authorization" not in payload
    assert "complete filing contents" not in payload

