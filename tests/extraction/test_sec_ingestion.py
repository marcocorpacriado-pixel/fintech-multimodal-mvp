"""Offline tests for the minimal edgartools SEC ingestion adapter."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.extraction.document_loader import load_filing
from src.extraction.pipeline import run_analysis_pipeline
from src.extraction.sec_ingestion import (
    SECIdentityError,
    SECIngestionError,
    SECFilingNotFoundError,
    SECNarrativeExtractionError,
    SECPreviousFilingNotFoundError,
    SECXBRLUnavailableError,
    prepare_sec_analysis_inputs,
)


CURRENT_ACCESSION = "0000320193-26-000020"
PREVIOUS_ACCESSION = "0000320193-26-000013"
REVENUE = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
NARRATIVE = """PART I

ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue growth reflected durable services demand and operating performance.

Liquidity remained strong and management expects margins to remain resilient.

PART II

ITEM 1A. RISK FACTORS

Supply chain disruption and currency volatility remain material risks.
"""

SUMMARY = (
    "Apple's quarterly filing combines deterministic financial metrics with "
    "the narrative evidence retrieved from the filing. Services demand "
    "supported operating performance, while liquidity remained strong and "
    "management described resilient margins. Supply chain disruption and "
    "currency volatility remain relevant business risks. The analysis uses "
    "only the supplied filing evidence and source XBRL facts. It preserves "
    "the canonical calculations, comparison periods, units, and citations for "
    "downstream reporting. Qualitative conclusions remain limited to the "
    "materialized narrative, and the outlook stays unknown where the evidence "
    "does not provide a separately grounded management statement."
)


@dataclass
class FakeFacts:
    dataframe: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        return self.dataframe.copy()


@dataclass
class FakeXBRL:
    facts: FakeFacts


@dataclass
class FakeFiling:
    accession_no: str
    form: str
    filing_date: date
    report_date: date
    narrative: str = NARRATIVE
    xbrl_value: FakeXBRL | None = None
    cik: int = 320193
    company: str = "Apple Inc."
    ticker: str = "AAPL"
    text_calls: int = 0
    xbrl_calls: int = 0

    @property
    def accession_number(self) -> str:
        return self.accession_no

    def text(self) -> str:
        self.text_calls += 1
        return self.narrative

    def xbrl(self) -> FakeXBRL | None:
        self.xbrl_calls += 1
        return self.xbrl_value


@dataclass
class FakeCompany:
    filings: list[FakeFiling]
    name: str = "Apple Inc."
    tickers: tuple[str, ...] = ("AAPL",)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get_filings(self, **kwargs: Any) -> list[FakeFiling]:
        self.calls.append(dict(kwargs))
        selected = list(self.filings)
        accession = kwargs.get("accession_number")
        if accession is not None:
            selected = [
                filing
                for filing in selected
                if filing.accession_no == accession
            ]
        requested_form = kwargs.get("form")
        if requested_form is not None:
            include_amendments = kwargs.get("amendments", True)
            selected = [
                filing
                for filing in selected
                if filing.form == requested_form
                or (
                    include_amendments
                    and filing.form == f"{requested_form}/A"
                )
            ]
        requested_date = kwargs.get("filing_date")
        if requested_date is not None:
            selected = [
                filing
                for filing in selected
                if filing.filing_date.isoformat() == requested_date
            ]
        return selected


@dataclass
class FakeCompanyFactory:
    company: FakeCompany
    calls: list[str] = field(default_factory=list)

    def __call__(self, ticker: str) -> FakeCompany:
        self.calls.append(ticker)
        return self.company


@dataclass
class FakeLLMClient:
    calls: int = 0

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        self.calls += 1
        return {
            "key_positive_developments": [],
            "key_risks": [],
            "management_outlook": {
                "summary": "Insufficient separately cited outlook evidence.",
                "sentiment": "unknown",
                "source_ids": [],
            },
            "executive_summary": SUMMARY,
        }


def source_row(
    *,
    value: float,
    fact_id: str,
    period_start: str,
    period_end: str,
    fiscal_period: str,
) -> dict[str, object]:
    return {
        "fact_key": f"{REVENUE}_{fact_id}",
        "fact_id": fact_id,
        "concept": REVENUE,
        "context_ref": f"context-{fact_id}",
        "value": str(value),
        "unit_ref": "usd",
        "numeric_value": value,
        "period_type": "duration",
        "period_start": period_start,
        "period_end": period_end,
        "period_instant": np.nan,
        "fiscal_year": 2026,
        "fiscal_period": fiscal_period,
        "label": "Revenue",
        "statement_type": "IncomeStatement",
        "dimension": np.nan,
        "member": np.nan,
    }


def xbrl_for(
    *,
    value: float,
    fact_id: str,
    period_start: str,
    period_end: str,
    fiscal_period: str,
) -> FakeXBRL:
    return FakeXBRL(
        FakeFacts(
            pd.DataFrame(
                [
                    source_row(
                        value=value,
                        fact_id=fact_id,
                        period_start=period_start,
                        period_end=period_end,
                        fiscal_period=fiscal_period,
                    )
                ]
            )
        )
    )


def current_filing(**overrides: object) -> FakeFiling:
    payload: dict[str, object] = {
        "accession_no": CURRENT_ACCESSION,
        "form": "10-Q",
        "filing_date": date(2026, 7, 31),
        "report_date": date(2026, 6, 27),
        "xbrl_value": xbrl_for(
            value=120.0,
            fact_id="revenue-current",
            period_start="2026-03-29",
            period_end="2026-06-27",
            fiscal_period="Q3",
        ),
    }
    payload.update(overrides)
    return FakeFiling(**payload)


def previous_filing(**overrides: object) -> FakeFiling:
    payload: dict[str, object] = {
        "accession_no": PREVIOUS_ACCESSION,
        "form": "10-Q",
        "filing_date": date(2026, 5, 1),
        "report_date": date(2026, 3, 28),
        "xbrl_value": xbrl_for(
            value=100.0,
            fact_id="revenue-previous",
            period_start="2025-12-28",
            period_end="2026-03-28",
            fiscal_period="Q2",
        ),
    }
    payload.update(overrides)
    return FakeFiling(**payload)


def company_with(
    *extra_filings: FakeFiling,
    current: FakeFiling | None = None,
    previous: FakeFiling | None = None,
) -> FakeCompany:
    return FakeCompany(
        [
            current or current_filing(),
            previous or previous_filing(),
            *extra_filings,
        ]
    )


def prepare(
    tmp_path: Path,
    *,
    company: FakeCompany | None = None,
    ticker: str = "AAPL",
    accession: str = CURRENT_ACCESSION,
    filing_date: date | None = None,
    form: str = "10-Q",
):
    selected_company = company or company_with()
    factory = FakeCompanyFactory(selected_company)
    result = prepare_sec_analysis_inputs(
        ticker=ticker,
        accession=accession,
        filing_date=filing_date,
        form=form,
        output_dir=tmp_path / "nested" / "sec",
        company_factory=factory,
    )
    return result, selected_company, factory


def test_target_accession_is_found_exactly(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path)

    assert result.current_accession == CURRENT_ACCESSION


def test_unknown_accession_is_rejected(tmp_path: Path) -> None:
    company = company_with()

    with pytest.raises(SECFilingNotFoundError, match="was not found"):
        prepare(tmp_path, company=company, accession="0000320193-26-999999")


def test_ticker_is_normalized_and_validated(tmp_path: Path) -> None:
    result, _, factory = prepare(tmp_path, ticker="  aapl ")

    assert result.ticker == "AAPL"
    assert factory.calls == ["AAPL"]


def test_selected_filing_has_requested_form(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path, form="10-Q")

    assert result.filing_type == "10-Q"
    assert result.current_filing.form == "10-Q"


def test_current_filing_object_is_preserved(tmp_path: Path) -> None:
    current = current_filing()
    company = company_with(current=current)

    result, _, _ = prepare(tmp_path, company=company)

    assert result.current_filing is current


def test_previous_fiscal_ten_q_is_selected(tmp_path: Path) -> None:
    previous = previous_filing()
    company = company_with(previous=previous)

    result, _, _ = prepare(tmp_path, company=company)

    assert result.previous_filing is previous
    assert result.previous_accession == PREVIOUS_ACCESSION


def test_intermediate_eight_k_is_not_selected(tmp_path: Path) -> None:
    eight_k = FakeFiling(
        accession_no="0000320193-26-000017",
        form="8-K",
        filing_date=date(2026, 6, 1),
        report_date=date(2026, 5, 30),
        xbrl_value=FakeXBRL(FakeFacts(pd.DataFrame())),
    )

    result, _, _ = prepare(tmp_path, company=company_with(eight_k))

    assert result.previous_accession == PREVIOUS_ACCESSION


def test_latest_amendment_wins_for_previous_fiscal_period(tmp_path: Path) -> None:
    original = previous_filing()
    older_amendment = previous_filing(
        accession_no="0000320193-26-000014",
        form="10-Q/A",
        filing_date=date(2026, 5, 5),
    )
    latest_amendment = previous_filing(
        accession_no="0000320193-26-000015",
        form="10-Q/A",
        filing_date=date(2026, 5, 6),
    )
    company = company_with(
        older_amendment,
        latest_amendment,
        previous=original,
    )

    result, _, _ = prepare(tmp_path, company=company)

    assert result.previous_filing is latest_amendment


def test_missing_previous_comparable_filing_is_rejected(tmp_path: Path) -> None:
    company = FakeCompany([current_filing()])

    with pytest.raises(SECPreviousFilingNotFoundError, match="no previous"):
        prepare(tmp_path, company=company)


def test_narrative_txt_is_created(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path)

    assert result.filing_path.is_file()
    assert result.filing_path.suffix == ".txt"


def test_narrative_filename_is_deterministic(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path)

    assert result.filing_path.name == (
        "AAPL_10-Q_2026-07-31_0000320193-26-000020.txt"
    )


def test_narrative_is_nonempty_and_loadable_by_d7_loader(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path)
    loaded = load_filing(
        result.filing_path,
        ticker=result.ticker,
        filing_type=result.filing_type,
        period=result.period,
    )

    assert loaded.text == NARRATIVE
    assert loaded.text.strip()


def test_existing_different_file_is_not_overwritten(tmp_path: Path) -> None:
    result, company, _ = prepare(tmp_path)
    result.filing_path.write_text("different content", encoding="utf-8")

    with pytest.raises(SECNarrativeExtractionError, match="refusing to overwrite"):
        prepare(tmp_path, company=company)

    assert result.filing_path.read_text(encoding="utf-8") == "different content"


def test_nested_output_directory_is_created(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path)

    assert result.filing_path.parent == tmp_path / "nested" / "sec"
    assert result.filing_path.parent.is_dir()


def test_runtime_metadata_is_coherent(tmp_path: Path) -> None:
    result, _, _ = prepare(tmp_path)

    assert result.company == "Apple Inc."
    assert result.period == "2026-06-27"
    assert result.current_accession == result.current_filing.accession_no
    assert result.previous_accession == result.previous_filing.accession_no


@pytest.mark.parametrize(
    ("filing_date", "form", "message"),
    [
        (date(2026, 7, 30), "10-Q", "does not match requested date"),
        (None, "10-K", "does not match requested form"),
    ],
)
def test_target_date_or_form_mismatch_is_rejected(
    tmp_path: Path,
    filing_date: date | None,
    form: str,
    message: str,
) -> None:
    with pytest.raises(SECIngestionError, match=message):
        prepare(tmp_path, filing_date=filing_date, form=form)


def test_current_filing_without_xbrl_is_rejected(tmp_path: Path) -> None:
    company = company_with(current=current_filing(xbrl_value=None))

    with pytest.raises(SECXBRLUnavailableError, match="current filing"):
        prepare(tmp_path, company=company)


def test_previous_filing_without_xbrl_is_rejected(tmp_path: Path) -> None:
    company = company_with(previous=previous_filing(xbrl_value=None))

    with pytest.raises(SECXBRLUnavailableError, match="previous filing"):
        prepare(tmp_path, company=company)


def test_adapter_result_runs_through_d7(tmp_path: Path) -> None:
    prepared, _, _ = prepare(tmp_path)
    client = FakeLLMClient()

    result = run_analysis_pipeline(
        filing_path=prepared.filing_path,
        company=prepared.company,
        ticker=prepared.ticker,
        period=prepared.period,
        filing_type=prepared.filing_type,
        current_xbrl_filing=prepared.current_filing,
        previous_xbrl_filing=prepared.previous_filing,
        llm_client=client,
    )

    revenue = next(
        metric
        for metric in result.analysis.financial_metrics
        if metric.name == "Revenue"
    )
    assert revenue.current_value == 120.0
    assert revenue.previous_value == 100.0
    assert result.verification.valid is True
    assert client.calls == 1


def test_offline_provider_makes_no_network_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in unit tests")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    result, company, _ = prepare(tmp_path)

    assert result.current_accession == CURRENT_ACCESSION
    assert company.calls
    assert all(call["trigger_full_load"] is False for call in company.calls)


def test_default_provider_requires_edgar_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    with pytest.raises(SECIdentityError, match="EDGAR_IDENTITY"):
        prepare_sec_analysis_inputs(
            ticker="AAPL",
            accession=CURRENT_ACCESSION,
            output_dir=tmp_path,
        )
