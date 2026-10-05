"""Tests for canonical XBRL fact normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.extraction.financial_analyzer import normalize_filing_facts
from src.extraction.schemas import NormalizedXBRLFact


def instant_payload(**overrides: object) -> dict[str, object]:
    """Build a valid normalized instant fact payload."""

    payload: dict[str, object] = {
        "ticker": "AAPL",
        "form": "10-Q",
        "accession": "0000320193-26-000020",
        "filing_date": date(2026, 7, 31),
        "fact_id": "f-1",
        "context_ref": "c-21",
        "source_id": "xbrl:0000320193-26-000020:f-1",
        "concept": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "label": "Cash and cash equivalents",
        "numeric_value": 39_544_000_000.0,
        "unit": "usd",
        "period_type": "instant",
        "period_instant": date(2026, 6, 27),
        "dimensions": {},
        "statement_type": "BalanceSheet",
        "amended": False,
    }
    payload.update(overrides)
    return payload


def duration_payload(**overrides: object) -> dict[str, object]:
    """Build a valid normalized duration fact payload."""

    payload = instant_payload(
        concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        label="Revenue",
        numeric_value=109_417_000_000.0,
        period_type="duration",
        period_start=date(2026, 3, 29),
        period_end=date(2026, 6, 27),
        period_instant=None,
        fiscal_year=2026,
        fiscal_period="Q3",
        statement_type="IncomeStatement",
    )
    payload.update(overrides)
    return payload


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
    dataframe: pd.DataFrame
    form: str = "10-Q"
    filing_date: date = date(2026, 7, 31)
    accession_number: str = "0000320193-26-000020"

    def xbrl(self) -> FakeXBRL:
        return FakeXBRL(FakeFacts(self.dataframe))


def source_row(**overrides: object) -> dict[str, object]:
    """Build one edgartools-shaped numeric source row."""

    row: dict[str, object] = {
        "fact_key": "us-gaap_Revenue_c-1",
        "fact_id": "f-1",
        "concept": "us-gaap:Revenue",
        "context_ref": "c-1",
        "value": "1000",
        "unit_ref": "usd",
        "numeric_value": 1000.0,
        "period_type": "duration",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "period_instant": np.nan,
        "fiscal_year": 2026,
        "fiscal_period": "Q1",
        "label": "Revenue",
        "statement_type": "IncomeStatement",
        "dimension": np.nan,
        "member": np.nan,
    }
    row.update(overrides)
    return row


def test_valid_instant_fact() -> None:
    fact = NormalizedXBRLFact(**instant_payload())

    assert fact.period_instant == date(2026, 6, 27)
    assert fact.period_start is None
    assert fact.period_end is None


def test_valid_duration_fact() -> None:
    fact = NormalizedXBRLFact(**duration_payload())

    assert fact.period_start == date(2026, 3, 29)
    assert fact.period_end == date(2026, 6, 27)
    assert fact.period_instant is None


@pytest.mark.parametrize(
    "payload",
    [
        instant_payload(period_instant=None),
        instant_payload(period_start=date(2026, 1, 1)),
        duration_payload(period_start=None),
        duration_payload(period_instant=date(2026, 6, 27)),
        duration_payload(
            period_start=date(2026, 7, 1),
            period_end=date(2026, 6, 27),
        ),
    ],
)
def test_invalid_period_dates(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NormalizedXBRLFact(**payload)


@pytest.mark.parametrize("numeric_value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numeric_value_is_rejected(numeric_value: float) -> None:
    with pytest.raises(ValidationError):
        NormalizedXBRLFact(**instant_payload(numeric_value=numeric_value))


def test_empty_unit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NormalizedXBRLFact(**instant_payload(unit="   "))


@pytest.mark.parametrize("unit", ["usd", "usdPerShare"])
def test_source_units_are_preserved(unit: str) -> None:
    assert NormalizedXBRLFact(**duration_payload(unit=unit)).unit == unit


def test_dimensions_are_preserved_and_sorted() -> None:
    fact = NormalizedXBRLFact(
        **duration_payload(
            dimensions={
                "srt:ProductOrServiceAxis": "aapl:IPhoneMember",
                "srt:ConsolidationItemsAxis": "us-gaap:OperatingSegmentsMember",
            }
        )
    )

    assert list(fact.dimensions) == [
        "srt:ConsolidationItemsAxis",
        "srt:ProductOrServiceAxis",
    ]


def test_consolidated_fact_has_empty_dimensions() -> None:
    assert NormalizedXBRLFact(**duration_payload()).dimensions == {}


def test_fiscal_period_is_preserved() -> None:
    fact = NormalizedXBRLFact(**duration_payload(fiscal_year=2026, fiscal_period="Q3"))

    assert fact.fiscal_year == 2026
    assert fact.fiscal_period == "Q3"


def test_amendment_metadata_is_preserved() -> None:
    dataframe = pd.DataFrame(
        [
            source_row(),
            source_row(
                concept="dei:AmendmentFlag",
                value="true",
                numeric_value=np.nan,
                unit_ref=np.nan,
            ),
        ]
    )

    facts = normalize_filing_facts(FakeFiling(dataframe), ticker="AAPL")

    assert len(facts) == 1
    assert facts[0].amended is True


def test_fact_serializes_to_json() -> None:
    fact = NormalizedXBRLFact(**duration_payload())

    assert json.loads(fact.model_dump_json()) == fact.model_dump(mode="json")


def test_normalization_is_deterministic() -> None:
    filing = FakeFiling(pd.DataFrame([source_row()]))

    first = normalize_filing_facts(filing, ticker="AAPL")
    second = normalize_filing_facts(filing, ticker="AAPL")

    assert [fact.model_dump() for fact in first] == [fact.model_dump() for fact in second]


def test_non_numeric_facts_are_ignored() -> None:
    dataframe = pd.DataFrame(
        [
            source_row(numeric_value=np.nan, value="narrative", unit_ref=np.nan),
            source_row(fact_id="f-2", numeric_value=2000.0),
        ]
    )

    facts = normalize_filing_facts(FakeFiling(dataframe), ticker="AAPL")

    assert [fact.fact_id for fact in facts] == ["f-2"]


def test_empty_input_returns_empty_list() -> None:
    assert normalize_filing_facts(FakeFiling(pd.DataFrame()), ticker="AAPL") == []


def test_duplicate_source_rows_are_preserved() -> None:
    row = source_row()
    facts = normalize_filing_facts(
        FakeFiling(pd.DataFrame([row, row])),
        ticker="AAPL",
    )

    assert len(facts) == 2
    assert facts[0].source_id == facts[1].source_id


def test_normalizer_preserves_dynamic_dimensions() -> None:
    dataframe = pd.DataFrame(
        [
            source_row(
                dimension="srt:ProductOrServiceAxis",
                member="aapl:IPhoneMember",
                dim_srt_ProductOrServiceAxis="aapl:IPhoneMember",
            )
        ]
    )

    fact = normalize_filing_facts(FakeFiling(dataframe), ticker="AAPL")[0]

    assert fact.dimensions == {
        "srt:ProductOrServiceAxis": "aapl:IPhoneMember"
    }


def test_normalizer_preserves_usd_per_share() -> None:
    dataframe = pd.DataFrame(
        [
            source_row(
                concept="us-gaap:EarningsPerShareDiluted",
                unit_ref="usdPerShare",
                numeric_value=2.02,
            )
        ]
    )

    fact = normalize_filing_facts(FakeFiling(dataframe), ticker="AAPL")[0]

    assert fact.unit == "usdPerShare"


def test_normalizer_does_not_convert_pandas_nulls_to_text() -> None:
    dataframe = pd.DataFrame(
        [
            source_row(
                fact_id=np.nan,
                fact_key="fallback-key",
                label=np.nan,
                fiscal_period=np.nan,
                statement_type=np.nan,
            )
        ]
    )

    fact = normalize_filing_facts(FakeFiling(dataframe), ticker="AAPL")[0]

    assert fact.fact_id is None
    assert fact.label is None
    assert fact.fiscal_period is None
    assert fact.statement_type is None
    assert fact.source_id.endswith(":fallback-key")


def test_standard_concept_is_only_preserved_when_supplied() -> None:
    dataframe = pd.DataFrame([source_row(standard_concept="Revenue")])

    fact = normalize_filing_facts(FakeFiling(dataframe), ticker="AAPL")[0]

    assert fact.standard_concept == "Revenue"
