"""Tests for extraction-layer data contracts."""

import json

import pytest
from pydantic import ValidationError

from src.extraction.schemas import (
    DocumentChunk,
    Evidence,
    FinancialAnalysisResult,
    FinancialMetric,
    ManagementOutlook,
)


def make_analysis_result() -> FinancialAnalysisResult:
    """Build a representative valid analysis result for schema tests."""

    return FinancialAnalysisResult(
        company="Example Financial Corp.",
        ticker="EFC",
        period="FY 2025",
        filing_type="10-K",
        financial_metrics=[
            FinancialMetric(
                name="Revenue",
                current_value=125.0,
                previous_value=100.0,
                change_pct=25.0,
                unit="USD millions",
            )
        ],
        key_positive_developments=[
            Evidence(
                finding="Revenue increased year over year.",
                evidence="Revenue rose to $125 million from $100 million.",
                source_section="Item 7. Management's Discussion and Analysis",
                source_id="filing-001",
                source_type="filing",
            )
        ],
        key_risks=[],
        management_outlook=ManagementOutlook(
            summary="Management expects stable demand.",
            sentiment="neutral",
        ),
        executive_summary="Revenue grew while the demand outlook remained stable.",
    )


def test_financial_analysis_result_can_be_built() -> None:
    result = make_analysis_result()

    assert result.ticker == "EFC"
    assert result.financial_metrics[0].change_pct == 25.0
    assert result.management_outlook.sentiment == "neutral"


def test_financial_analysis_result_serializes_to_dict_and_json() -> None:
    result = make_analysis_result()

    dumped = result.model_dump()
    json_dumped = json.loads(result.model_dump_json())

    assert dumped["filing_type"] == "10-K"
    assert dumped["key_positive_developments"][0]["source_type"] == "filing"
    assert json_dumped == dumped


@pytest.mark.parametrize("filing_type", ["8-K", "20-F", "10K", ""])
def test_financial_analysis_result_rejects_invalid_filing_type(
    filing_type: str,
) -> None:
    payload = make_analysis_result().model_dump()
    payload["filing_type"] = filing_type

    with pytest.raises(ValidationError):
        FinancialAnalysisResult.model_validate(payload)


@pytest.mark.parametrize("sentiment", ["optimistic", "bearish", "", "POSITIVE"])
def test_management_outlook_rejects_invalid_sentiment(sentiment: str) -> None:
    with pytest.raises(ValidationError):
        ManagementOutlook(summary="Forward guidance is unchanged.", sentiment=sentiment)


def test_document_chunk_can_be_built() -> None:
    chunk = DocumentChunk(
        chunk_id="AAPL-2025-10K-item7-0001",
        ticker="AAPL",
        filing_type="10-K",
        period="FY 2025",
        section="Item 7",
        text="Net sales increased during the fiscal year.",
        source_id="0000320193-25-000079",
        start_char=100,
        end_char=145,
    )

    assert chunk.section == "Item 7"
    assert chunk.start_char == 100
    assert chunk.end_char == 145


def test_financial_analysis_result_lists_default_to_independent_empty_lists() -> None:
    common = {
        "company": "Example Financial Corp.",
        "ticker": "EFC",
        "period": "Q1 2026",
        "filing_type": "10-Q",
        "management_outlook": {"summary": "No guidance was provided."},
        "executive_summary": "No material change was reported.",
    }

    first = FinancialAnalysisResult(**common)
    second = FinancialAnalysisResult(**common)
    first.financial_metrics.append(FinancialMetric(name="Cash", current_value=1.0))

    assert second.financial_metrics == []
    assert second.key_positive_developments == []
    assert second.key_risks == []


def test_evidence_accepts_filing_source_type() -> None:
    evidence = Evidence(
        finding="Margins expanded.",
        evidence="Gross margin increased by two percentage points.",
        source_type="filing",
    )

    assert evidence.source_type == "filing"
    assert evidence.source_section is None
    assert evidence.source_id is None
