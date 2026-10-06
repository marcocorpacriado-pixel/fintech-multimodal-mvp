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
REAL_AAPL_SUMMARY = (
    "Apple Inc. reported Q3 2026 results for the period ended June 27, 2026. "
    "Revenue declined 1.6% quarter over quarter to $109.4 billion, while net "
    "income rose 0.7% to $29.8 billion and diluted EPS increased 0.5% to "
    "$2.02. Cash and cash equivalents fell 13.2% to $39.5 billion, and total "
    "debt was roughly flat at $84.3 billion. Year-to-date operating cash flow "
    "rose 43.1% to $117.0 billion, while capital expenditures declined 28.2% "
    "to $6.8 billion."
)
LATEST_OPENROUTER_SUMMARY = (
    "Apple Inc. reported its fiscal third quarter 2026 results for the period "
    "ending June 27, 2026. Revenue declined 1.6% sequentially to $109.4 "
    "billion, while net income and diluted EPS edged slightly higher. "
    "Year-to-date operating cash flow surged 43% to $117 billion, and capital "
    "expenditures fell 28%. The company saw strong year-over-year sales growth "
    "across all geographic segments, with total net sales up 16% from the "
    "prior year. Gross margin improved to 50.1%. However, management highlighted "
    "intensifying supply constraints for key components and ongoing tariff "
    "uncertainties, which are expected to materially impact future results. "
    "The outlook is negative due to these headwinds."
)
SANITIZED_OPENROUTER_SUMMARY = (
    "Apple Inc. reported its fiscal third quarter 2026 results for the period "
    "ending June 27, 2026. Revenue declined 1.6% sequentially to $109.4 "
    "billion, while net income and diluted EPS edged slightly higher. "
    "Year-to-date operating cash flow surged 43% to $117 billion, and capital "
    "expenditures fell 28%. The company also reported broad geographic sales "
    "growth and improved gross margin. However, management highlighted "
    "intensifying supply constraints for key components and ongoing tariff "
    "uncertainties, which are expected to materially impact future results. "
    "The outlook is negative due to these headwinds."
)
GROSS_MARGIN_EVIDENCE = "Total gross margin percentage 50.1 46.5"
SALES_GROWTH_EVIDENCE = "Total net sales $ 109,417 94,036 16%"


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


def real_aapl_metrics() -> list[FinancialMetric]:
    common_duration = {
        "comparison_type": "QoQ",
        "current_period": "2026-03-29/2026-06-27",
        "previous_period": "2025-12-28/2026-03-28",
        "unit": "usd",
    }
    return [
        revenue_metric(),
        FinancialMetric(
            name="Net Income",
            current_value=29_789_000_000.0,
            previous_value=29_578_000_000.0,
            change_pct=0.71336804381635,
            source_ids=["xbrl:current:net-income", "xbrl:previous:net-income"],
            **common_duration,
        ),
        eps_metric(),
        FinancialMetric(
            name="Cash and Cash Equivalents",
            current_value=39_544_000_000.0,
            previous_value=45_572_000_000.0,
            change_pct=-13.227420345826385,
            unit="usd",
            comparison_type="QoQ",
            current_period="2026-06-27",
            previous_period="2026-03-28",
            source_ids=["xbrl:current:cash", "xbrl:previous:cash"],
        ),
        FinancialMetric(
            name="Total Debt",
            current_value=84_344_000_000.0,
            previous_value=84_711_000_000.0,
            change_pct=-0.433237714110328,
            unit="usd",
            comparison_type="QoQ",
            current_period="2026-06-27",
            previous_period="2026-03-28",
            source_ids=["xbrl:current:debt", "xbrl:previous:debt"],
        ),
        FinancialMetric(
            name="Operating Cash Flow",
            current_value=116_996_000_000.0,
            previous_value=81_754_000_000.0,
            change_pct=43.10737089316731,
            unit="usd",
            comparison_type="YoY_YTD",
            current_period="2025-09-28/2026-06-27",
            previous_period="2024-09-29/2025-06-28",
            source_ids=["xbrl:current:ocf", "xbrl:previous:ocf"],
        ),
        FinancialMetric(
            name="Capital Expenditures",
            current_value=6_799_000_000.0,
            previous_value=9_473_000_000.0,
            change_pct=-28.227594215137756,
            unit="usd",
            comparison_type="YoY_YTD",
            current_period="2025-09-28/2026-06-27",
            previous_period="2024-09-29/2025-06-28",
            source_ids=["xbrl:current:capex", "xbrl:previous:capex"],
        ),
    ]


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


def real_numeric_findings() -> list[Evidence]:
    return [
        Evidence(
            finding="Gross margin improved to 50.1% from 46.5%.",
            evidence=GROSS_MARGIN_EVIDENCE,
            source_section="ITEM_2",
            source_id="chunk-gross-margin",
            source_type="filing",
        ),
        Evidence(
            finding="Total net sales increased 16% year over year.",
            evidence=SALES_GROWTH_EVIDENCE,
            source_section="ITEM_2",
            source_id="chunk-sales-growth",
            source_type="filing",
        ),
    ]


def retrieval_with_real_numeric_evidence() -> list[RetrievalResult]:
    return [
        *retrieval_results(),
        retrieval_result(
            "chunk-gross-margin",
            "ITEM_2",
            GROSS_MARGIN_EVIDENCE,
            4,
        ),
        retrieval_result(
            "chunk-sales-growth",
            "ITEM_2",
            SALES_GROWTH_EVIDENCE,
            5,
        ),
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


@pytest.mark.parametrize("displayed", ["43%", "43.1%"])
def test_percentage_rounding_uses_displayed_precision(displayed: str) -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with(f"Operating cash flow rose {displayed}."),
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis, metrics=metrics))


def test_percentage_outside_displayed_precision_is_rejected() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Operating cash flow rose 42%."),
    )

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis, metrics=metrics))


def test_negative_percentage_uses_direction_and_displayed_precision() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Capital expenditures fell 28%."),
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis, metrics=metrics))


def test_wrong_direction_is_rejected_with_precision_aware_rounding() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Capital expenditures rose 28%."),
    )

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis, metrics=metrics))


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


@pytest.mark.parametrize(
    "date_text",
    [
        "June 27, 2026",
        "Jun 27, 2026",
        "27 June 2026",
        "June 27 2026",
    ],
)
def test_textual_date_is_not_treated_as_financial_number(date_text: str) -> None:
    analysis = valid_analysis(
        summary=summary_with(f"The reporting period ended {date_text}.")
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


@pytest.mark.parametrize(
    "month",
    [
        "January",
        "Jan",
        "February",
        "Feb",
        "March",
        "Mar",
        "April",
        "Apr",
        "May",
        "June",
        "Jun",
        "July",
        "Jul",
        "August",
        "Aug",
        "September",
        "Sep",
        "October",
        "Oct",
        "November",
        "Nov",
        "December",
        "Dec",
    ],
)
def test_textual_dates_support_every_english_month(month: str) -> None:
    analysis = valid_analysis(
        summary=summary_with(f"The reporting period ended {month} 15, 2026.")
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_iso_date_remains_nonfinancial() -> None:
    analysis = valid_analysis(
        summary=summary_with("The reporting period ended 2026-06-27.")
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_item_reference_remains_nonfinancial() -> None:
    analysis = valid_analysis(summary=summary_with("Item 7 discusses revenue."))

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_financial_percentage_matching_a_day_number_is_still_checked() -> None:
    metrics = [revenue_metric(change_pct=27.0), eps_metric()]
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Revenue increased 27%."),
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis, metrics=metrics))


def test_financial_percentage_matching_a_day_number_is_still_rejected() -> None:
    analysis = valid_analysis(summary=summary_with("Revenue increased 27%."))

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis))


def test_currency_value_matching_a_day_number_is_still_checked() -> None:
    metrics = [
        revenue_metric(
            current_value=27_000_000_000.0,
            previous_value=26_000_000_000.0,
            change_pct=None,
        ),
        eps_metric(),
    ]
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Revenue was $27 billion."),
    )

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis, metrics=metrics))


def test_real_openrouter_summary_does_not_flag_date_day() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(metrics=metrics, summary=REAL_AAPL_SUMMARY)

    report = verify(analysis, metrics=metrics)

    assert report.valid is True
    assert not any(issue.code == "UNSUPPORTED_NUMBER" for issue in report.issues)


def test_summary_rejects_noncanonical_revenue_percentage() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Revenue increased 16% year over year."),
    )

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis, metrics=metrics))


def test_summary_rejects_noncanonical_gross_margin_percentage() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=summary_with("Gross margin improved to 50.1%."),
    )

    assert "UNSUPPORTED_NUMBER" in codes(verify(analysis, metrics=metrics))


@pytest.mark.parametrize("percentage", ["50.1%", "46.5%"])
def test_finding_accepts_percentage_from_its_own_evidence(
    percentage: str,
) -> None:
    finding = Evidence(
        finding=f"Gross margin was {percentage}.",
        evidence=GROSS_MARGIN_EVIDENCE,
        source_section="ITEM_2",
        source_id="chunk-gross-margin",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    report = verify(
        analysis,
        retrieval=[
            *retrieval_results(),
            retrieval_result(
                "chunk-gross-margin",
                "ITEM_2",
                GROSS_MARGIN_EVIDENCE,
                4,
            ),
        ],
    )

    assert "UNSUPPORTED_NUMBER" not in codes(report)


def test_finding_rejects_percentage_absent_from_its_evidence() -> None:
    finding = Evidence(
        finding="Gross margin improved to 55%.",
        evidence=GROSS_MARGIN_EVIDENCE,
        source_section="ITEM_2",
        source_id="chunk-gross-margin",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    report = verify(
        analysis,
        retrieval=[
            *retrieval_results(),
            retrieval_result(
                "chunk-gross-margin",
                "ITEM_2",
                GROSS_MARGIN_EVIDENCE,
                4,
            ),
        ],
    )

    assert "UNSUPPORTED_NUMBER" in codes(report)


def test_finding_accepts_revenue_percentage_from_its_own_evidence() -> None:
    finding = Evidence(
        finding="Total net sales increased 16% year over year.",
        evidence=SALES_GROWTH_EVIDENCE,
        source_section="ITEM_2",
        source_id="chunk-sales-growth",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    report = verify(
        analysis,
        retrieval=[
            *retrieval_results(),
            retrieval_result(
                "chunk-sales-growth",
                "ITEM_2",
                SALES_GROWTH_EVIDENCE,
                4,
            ),
        ],
    )

    assert "UNSUPPORTED_NUMBER" not in codes(report)


def test_finding_cannot_borrow_number_from_another_retrieved_chunk() -> None:
    own_evidence = "Gross margin improved during the quarter."
    finding = Evidence(
        finding="Gross margin improved to 50.1%.",
        evidence=own_evidence,
        source_section="ITEM_2",
        source_id="chunk-own-evidence",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    report = verify(
        analysis,
        retrieval=[
            *retrieval_results(),
            retrieval_result(
                "chunk-own-evidence",
                "ITEM_2",
                own_evidence,
                4,
            ),
            retrieval_result(
                "chunk-other-evidence",
                "ITEM_2",
                GROSS_MARGIN_EVIDENCE,
                5,
            ),
        ],
    )

    assert "UNSUPPORTED_NUMBER" in codes(report)


def test_finding_accepts_canonical_number_without_evidence_number() -> None:
    finding = Evidence(
        finding="Revenue declined 1.6%.",
        evidence=POSITIVE_TEXT,
        source_section="ITEM_7",
        source_id="chunk-positive",
        source_type="filing",
    )
    analysis = valid_analysis(positives=[finding])

    assert "UNSUPPORTED_NUMBER" not in codes(verify(analysis))


def test_latest_openrouter_output_has_only_noncanonical_summary_issues() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=LATEST_OPENROUTER_SUMMARY,
        positives=real_numeric_findings(),
    )

    report = verify(
        analysis,
        metrics=metrics,
        retrieval=retrieval_with_real_numeric_evidence(),
    )
    unsupported = [
        issue for issue in report.issues if issue.code == "UNSUPPORTED_NUMBER"
    ]

    assert report.valid is False
    assert len(unsupported) == 2
    assert {issue.field for issue in unsupported} == {"executive_summary"}
    assert any("'16%'" in issue.message for issue in unsupported)
    assert any("'50.1%'" in issue.message for issue in unsupported)


def test_sanitized_openrouter_output_is_valid() -> None:
    metrics = real_aapl_metrics()
    analysis = valid_analysis(
        metrics=metrics,
        summary=SANITIZED_OPENROUTER_SUMMARY,
        positives=real_numeric_findings(),
    )

    report = verify(
        analysis,
        metrics=metrics,
        retrieval=retrieval_with_real_numeric_evidence(),
    )

    assert report.valid is True
    assert report.issues == []


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
