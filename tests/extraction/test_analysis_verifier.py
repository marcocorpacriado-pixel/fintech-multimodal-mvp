"""Tests for deterministic verification of grounded financial analysis."""

from __future__ import annotations

import json
import socket

import pytest

from src.extraction.analysis_verifier import (
    AnalysisVerificationError,
    assert_verified_analysis,
    verify_analysis,
)
from src.extraction.schemas import (
    DocumentChunk,
    Evidence,
    FinancialAnalysisResult,
    FinancialMetric,
    ManagementOutlook,
    RetrievalResult,
)


POSITIVE_TEXT = "Services demand remained resilient during the quarter."
RISK_TEXT = "Currency volatility may adversely affect reported results."
OUTLOOK_TEXT = "Management expects demand to remain stable next quarter."


def revenue_metric(**overrides: object) -> FinancialMetric:
    payload: dict[str, object] = {
        "name": "Revenue",
        "current_value": 109_417_000_000.0,
        "previous_value": 111_184_000_000.0,
        "change_pct": -1.5892574471146927,
        "unit": "usd",
        "comparison_type": "QoQ",
        "current_period": "2026-03-29/2026-06-27",
        "previous_period": "2025-12-28/2026-03-28",
        "source_ids": ["xbrl:current:revenue", "xbrl:previous:revenue"],
    }
    payload.update(overrides)
    return FinancialMetric(**payload)


def eps_metric() -> FinancialMetric:
    return FinancialMetric(
        name="Diluted EPS",
        current_value=2.02,
        previous_value=2.01,
        change_pct=0.4975124378109568,
        unit="usdPerShare",
        comparison_type="QoQ",
        current_period="2026-03-29/2026-06-27",
        previous_period="2025-12-28/2026-03-28",
        source_ids=["xbrl:current:eps", "xbrl:previous:eps"],
    )


def canonical_metrics() -> list[FinancialMetric]:
    return [revenue_metric(), eps_metric()]


def retrieval_result(
    chunk_id: str,
    section: str,
    text: str,
    rank: int,
    *,
    ticker: str = "AAPL",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            ticker=ticker,
            filing_type="10-Q",
            period="Q3 2026",
            section=section,
            text=text,
            source_id="filing:aapl:q3-2026",
            start_char=0,
            end_char=len(text),
        ),
        score=10.0 / rank,
        rank=rank,
    )


def retrieval_results() -> list[RetrievalResult]:
    return [
        retrieval_result("chunk-positive", "ITEM_7", POSITIVE_TEXT, 1),
        retrieval_result("chunk-risk", "ITEM_1A", RISK_TEXT, 2),
        retrieval_result("chunk-outlook", "ITEM_7", OUTLOOK_TEXT, 3),
    ]


def summary_with(claim: str) -> str:
    return (
        "Apple's Q3 2026 filing is assessed using deterministic financial data "
        "and narrative evidence retrieved from the SEC report. "
        f"{claim} "
        "Services demand remained resilient and supported operating momentum, "
        "while currency volatility continued to represent an important disclosed "
        "risk. Management expects demand to remain stable next quarter. The "
        "assessment preserves the reported comparison periods, uses only cited "
        "filing passages for qualitative conclusions, and avoids assumptions "
        "about information that was not supplied. The discussion is intended as "
        "a concise factual summary for downstream reporting and contains no "
        "investment advice or external market knowledge."
    )


def valid_analysis(
    *,
    metrics: list[FinancialMetric] | None = None,
    summary: str | None = None,
    positives: list[Evidence] | None = None,
    risks: list[Evidence] | None = None,
    outlook: ManagementOutlook | None = None,
    company: str = "Apple Inc.",
    ticker: str = "AAPL",
    period: str = "Q3 2026",
) -> FinancialAnalysisResult:
    return FinancialAnalysisResult(
        company=company,
        ticker=ticker,
        period=period,
        filing_type="10-Q",
        financial_metrics=canonical_metrics() if metrics is None else metrics,
        key_positive_developments=(
            [
                Evidence(
                    finding="Services demand supported operating momentum.",
                    evidence=POSITIVE_TEXT,
                    source_section="ITEM_7",
                    source_id="chunk-positive",
                    source_type="filing",
                )
            ]
            if positives is None
            else positives
        ),
        key_risks=(
            [
                Evidence(
                    finding="Currency volatility remains a material risk.",
                    evidence=RISK_TEXT,
                    source_section="ITEM_1A",
                    source_id="chunk-risk",
                    source_type="filing",
                )
            ]
            if risks is None
            else risks
        ),
        management_outlook=(
            ManagementOutlook(
                summary="Management expects stable demand.",
                sentiment="neutral",
                source_ids=["chunk-outlook"],
            )
            if outlook is None
            else outlook
        ),
        executive_summary=(
            summary_with(
                "Revenue declined 1.59%, ending near $109.4 billion, while "
                "diluted EPS was 2.02."
            )
            if summary is None
            else summary
        ),
    )


def verify(
    analysis: FinancialAnalysisResult | None = None,
    *,
    metrics: list[FinancialMetric] | None = None,
    retrieval: list[RetrievalResult] | None = None,
):
    return verify_analysis(
        valid_analysis() if analysis is None else analysis,
        canonical_metrics=canonical_metrics() if metrics is None else metrics,
        retrieval_results=retrieval_results() if retrieval is None else retrieval,
        expected_company="Apple Inc.",
        expected_ticker="AAPL",
        expected_period="Q3 2026",
    )


def codes(report) -> list[str]:
    return [issue.code for issue in report.issues]


def test_completely_valid_analysis() -> None:
    report = verify()

    assert report.valid is True
    assert report.issues == []


def test_exact_metrics_are_accepted() -> None:
    report = verify()

    assert "METRICS_MISMATCH" not in codes(report)


def test_modified_metric_is_rejected() -> None:
    modified = [revenue_metric(change_pct=12.0), eps_metric()]
    analysis = valid_analysis(metrics=modified)

    report = verify(analysis)

    assert "METRICS_MISMATCH" in codes(report)
    assert report.valid is False


def test_metric_order_is_part_of_canonical_equality() -> None:
    analysis = valid_analysis(metrics=list(reversed(canonical_metrics())))

    report = verify(analysis)

    assert "METRICS_MISMATCH" in codes(report)


def test_exact_percentage_is_reconciled() -> None:
    analysis = valid_analysis(
        summary=summary_with("Revenue declined 1.5892574471146927%.")
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_rounded_percentage_is_reconciled() -> None:
    analysis = valid_analysis(summary=summary_with("Revenue declined 1.59%."))

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_invented_percentage_is_rejected() -> None:
    analysis = valid_analysis(summary=summary_with("Revenue increased 12%."))

    report = verify(analysis)

    assert "UNSUPPORTED_NUMBER" in codes(report)


def test_wrong_direction_is_rejected_even_with_same_magnitude() -> None:
    analysis = valid_analysis(summary=summary_with("Revenue increased 1.59%."))

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis))


def test_billion_normalization_accepts_human_rounding() -> None:
    analysis = valid_analysis(
        summary=summary_with("Revenue was approximately $109.4 billion.")
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_million_normalization_is_reconciled() -> None:
    analysis = valid_analysis(
        summary=summary_with("Revenue was $109,417 million.")
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_eps_plain_decimal_is_reconciled() -> None:
    analysis = valid_analysis(summary=summary_with("Diluted EPS was 2.02."))

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_invented_financial_value_is_rejected() -> None:
    analysis = valid_analysis(summary=summary_with("Revenue was $999 billion."))

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis))


def test_year_is_not_treated_as_financial_number() -> None:
    analysis = valid_analysis(summary=summary_with("The reporting year was 2026."))

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_fiscal_quarter_is_not_treated_as_financial_number() -> None:
    analysis = valid_analysis(summary=summary_with("The filing covers Q3."))

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_valid_source_id_is_accepted() -> None:
    assert "INVALID_SOURCE_ID" not in codes(verify())


def test_invalid_source_id_is_rejected() -> None:
    finding = Evidence(
        finding="Unsupported citation.",
        evidence=POSITIVE_TEXT,
        source_section="ITEM_7",
        source_id="invented",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    assert "INVALID_SOURCE_ID" in codes(verify(analysis))


def test_evidence_present_in_chunk_is_accepted() -> None:
    assert "EVIDENCE_NOT_IN_SOURCE" not in codes(verify())


def test_invented_evidence_is_rejected() -> None:
    finding = Evidence(
        finding="Invented evidence.",
        evidence="This sentence does not exist in the source.",
        source_section="ITEM_7",
        source_id="chunk-positive",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    assert "EVIDENCE_NOT_IN_SOURCE" in codes(verify(analysis))


def test_section_mismatch_is_rejected() -> None:
    finding = Evidence(
        finding="Wrong section.",
        evidence=POSITIVE_TEXT,
        source_section="ITEM_1A",
        source_id="chunk-positive",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    assert "SECTION_MISMATCH" in codes(verify(analysis))


def test_claim_without_source_id_is_ungrounded() -> None:
    finding = Evidence(
        finding="No citation.",
        evidence=POSITIVE_TEXT,
        source_section="ITEM_7",
        source_id=None,
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    assert "UNGROUNDED_CLAIM" in codes(verify(analysis))


def test_unknown_outlook_does_not_require_citation() -> None:
    analysis = valid_analysis(
        outlook=ManagementOutlook(
            summary="Insufficient evidence for management outlook.",
            sentiment="unknown",
            source_ids=[],
        )
    )

    assert "OUTLOOK_WITHOUT_EVIDENCE" not in codes(verify(analysis))


def test_known_outlook_without_citation_is_rejected() -> None:
    analysis = valid_analysis(
        outlook=ManagementOutlook(
            summary="Management expects stable demand.",
            sentiment="positive",
            source_ids=[],
        )
    )

    assert "OUTLOOK_WITHOUT_EVIDENCE" in codes(verify(analysis))


@pytest.mark.parametrize(
    "recommendation",
    [
        "Investors should buy the stock.",
        "Investors should sell the stock.",
        "The report sets a target price of $250.",
    ],
)
def test_investment_recommendations_are_rejected(recommendation: str) -> None:
    analysis = valid_analysis(summary=summary_with(recommendation))

    assert "INVESTMENT_RECOMMENDATION" in codes(verify(analysis))


def test_sales_is_not_confused_with_sell() -> None:
    analysis = valid_analysis(summary=summary_with("International sales were stable."))

    assert "INVESTMENT_RECOMMENDATION" not in codes(verify(analysis))


def test_short_summary_produces_warning_but_remains_valid() -> None:
    analysis = valid_analysis(summary="Apple reported stable operating performance.")

    report = verify(analysis)

    assert "SUMMARY_TOO_SHORT" in codes(report)
    assert report.valid is True
    assert next(
        issue for issue in report.issues if issue.code == "SUMMARY_TOO_SHORT"
    ).severity == "warning"


def test_normal_summary_length_is_accepted() -> None:
    report = verify()

    assert "SUMMARY_TOO_SHORT" not in codes(report)
    assert "SUMMARY_TOO_LONG" not in codes(report)


def test_slightly_long_summary_is_warning_only() -> None:
    analysis = valid_analysis(summary=" ".join(["context"] * 210))

    report = verify(analysis)

    issue = next(issue for issue in report.issues if issue.code == "SUMMARY_TOO_LONG")
    assert issue.severity == "warning"
    assert report.valid is True


def test_very_long_summary_is_error() -> None:
    analysis = valid_analysis(summary=" ".join(["context"] * 221))

    report = verify(analysis)

    issue = next(issue for issue in report.issues if issue.code == "SUMMARY_TOO_LONG")
    assert issue.severity == "error"
    assert report.valid is False


def test_multiple_issues_are_reported_together() -> None:
    modified = [revenue_metric(change_pct=12.0), eps_metric()]
    bad_finding = Evidence(
        finding="Investors should buy the stock.",
        evidence="Invented evidence.",
        source_section="ITEM_1A",
        source_id="missing",
        source_type="filing",
    )
    analysis = valid_analysis(
        metrics=modified,
        positives=[bad_finding],
        summary=summary_with("Revenue increased 12%."),
    )

    report = verify(analysis)

    assert {"METRICS_MISMATCH", "INVALID_SOURCE_ID", "UNSUPPORTED_NUMBER"} <= set(
        codes(report)
    )
    assert "INVESTMENT_RECOMMENDATION" in codes(report)


def test_report_valid_reflects_errors_not_warnings() -> None:
    warning_report = verify(valid_analysis(summary="Brief neutral summary."))
    error_report = verify(
        valid_analysis(summary=summary_with("Revenue increased 12%."))
    )

    assert warning_report.valid is True
    assert error_report.valid is False


def test_assert_helper_returns_valid_report() -> None:
    report = assert_verified_analysis(
        valid_analysis(),
        canonical_metrics=canonical_metrics(),
        retrieval_results=retrieval_results(),
        expected_company="Apple Inc.",
        expected_ticker="AAPL",
        expected_period="Q3 2026",
    )

    assert report.valid is True


def test_assert_helper_raises_with_report_on_error() -> None:
    analysis = valid_analysis(summary=summary_with("Revenue increased 12%."))

    with pytest.raises(AnalysisVerificationError) as caught:
        assert_verified_analysis(
            analysis,
            canonical_metrics=canonical_metrics(),
            retrieval_results=retrieval_results(),
        )
    assert caught.value.report.valid is False
    assert "UNSUPPORTED_NUMBER" in codes(caught.value.report)


def test_expected_metadata_mismatches_are_reported() -> None:
    analysis = valid_analysis(company="Other Corp.", ticker="MSFT", period="FY 2025")

    report = verify(analysis)

    assert {"COMPANY_MISMATCH", "TICKER_MISMATCH", "PERIOD_MISMATCH"} <= set(
        codes(report)
    )


def test_cited_chunk_ticker_must_match_analysis() -> None:
    retrieval = retrieval_results()
    retrieval[0] = retrieval_result(
        "chunk-positive",
        "ITEM_7",
        POSITIVE_TEXT,
        1,
        ticker="MSFT",
    )

    assert "SOURCE_TICKER_MISMATCH" in codes(verify(retrieval=retrieval))


def test_future_source_type_is_not_hardcoded_to_filing() -> None:
    finding = Evidence(
        finding="Services demand supported operating momentum.",
        evidence=POSITIVE_TEXT,
        source_section="ITEM_7",
        source_id="chunk-positive",
        source_type="earnings_call",
    )
    analysis = valid_analysis(positives=[finding])

    assert verify(analysis).valid is True


def test_report_serializes_deterministically() -> None:
    first = verify()
    second = verify()

    assert first.model_dump() == second.model_dump()
    assert json.loads(first.model_dump_json()) == first.model_dump()


def test_verifier_performs_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*_: object, **__: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    assert verify().valid is True
