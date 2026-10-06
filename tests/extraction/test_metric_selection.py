"""Unit tests for deterministic XBRL financial metric selection."""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.extraction.financial_analyzer import (
    FinancialMetricSelectionError,
    build_financial_metrics,
    classify_duration,
)
from src.extraction.schemas import FinancialMetric, NormalizedXBRLFact


CURRENT_ACCESSION = "0000320193-26-000020"
PREVIOUS_ACCESSION = "0000320193-26-000013"

REVENUE = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
NET_INCOME = "us-gaap:NetIncomeLoss"
EPS = "us-gaap:EarningsPerShareDiluted"
CASH = "us-gaap:CashAndCashEquivalentsAtCarryingValue"
COMMERCIAL_PAPER = "us-gaap:CommercialPaper"
DEBT_CURRENT = "us-gaap:LongTermDebtCurrent"
DEBT_NONCURRENT = "us-gaap:LongTermDebtNoncurrent"
OCF = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
CAPEX = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"


def make_fact(
    concept: str,
    value: float,
    *,
    unit: str = "usd",
    period_type: str = "duration",
    period_start: date | None = date(2026, 3, 29),
    period_end: date | None = date(2026, 6, 27),
    period_instant: date | None = None,
    fiscal_year: int | None = 2026,
    fiscal_period: str | None = "Q3",
    statement_type: str = "IncomeStatement",
    ticker: str = "AAPL",
    accession: str = CURRENT_ACCESSION,
    filing_date: date = date(2026, 7, 31),
    dimensions: dict[str, str] | None = None,
    amended: bool = False,
    source_id: str | None = None,
) -> NormalizedXBRLFact:
    """Create a small valid fact with explicit temporal metadata."""

    terminal = period_instant or period_end
    identifier = source_id or f"xbrl:{accession}:{concept}:{terminal}:{value}"
    form = "10-Q/A" if amended else "10-Q"
    return NormalizedXBRLFact(
        ticker=ticker,
        form=form,
        accession=accession,
        filing_date=filing_date,
        fact_id=None,
        context_ref=f"context:{identifier}",
        source_id=identifier,
        concept=concept,
        numeric_value=value,
        unit=unit,
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        period_instant=period_instant,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        dimensions=dimensions or {},
        statement_type=statement_type,
        amended=amended,
    )


def previous_quarter(concept: str, value: float, **overrides: object) -> NormalizedXBRLFact:
    payload: dict[str, object] = {
        "accession": PREVIOUS_ACCESSION,
        "filing_date": date(2026, 5, 1),
        "period_start": date(2025, 12, 28),
        "period_end": date(2026, 3, 28),
        "fiscal_year": 2026,
        "fiscal_period": "Q2",
    }
    payload.update(overrides)
    return make_fact(concept, value, **payload)


def instant_fact(concept: str, value: float, *, current: bool = True, **overrides: object) -> NormalizedXBRLFact:
    payload: dict[str, object] = {
        "period_type": "instant",
        "period_start": None,
        "period_end": None,
        "period_instant": date(2026, 6, 27) if current else date(2026, 3, 28),
        "fiscal_period": None,
        "statement_type": "BalanceSheet",
    }
    if not current:
        payload.update(
            accession=PREVIOUS_ACCESSION,
            filing_date=date(2026, 5, 1),
        )
    payload.update(overrides)
    return make_fact(concept, value, **payload)


def metric_named(metrics: list[FinancialMetric], name: str) -> FinancialMetric:
    return next(metric for metric in metrics if metric.name == name)


def test_revenue_uses_quarter_only_context() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(REVENUE, 120.0)],
            [previous_quarter(REVENUE, 100.0)],
        ),
        "Revenue",
    )

    assert metric.current_value == 120.0
    assert metric.previous_value == 100.0
    assert metric.comparison_type == "QoQ"
    assert metric.change_pct == pytest.approx(20.0)


def test_net_income_uses_quarter_only_context() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(NET_INCOME, 30.0)],
            [previous_quarter(NET_INCOME, 25.0)],
        ),
        "Net Income",
    )

    assert metric.current_value == 30.0
    assert metric.comparison_type == "QoQ"


def test_eps_uses_direct_quarter_only_context() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(EPS, 2.02, unit="usdPerShare")],
            [previous_quarter(EPS, 2.01, unit="usdPerShare")],
        ),
        "Diluted EPS",
    )

    assert metric.current_value == 2.02
    assert metric.previous_value == 2.01
    assert metric.comparison_type == "QoQ"


def test_eps_is_never_derived_from_ytd() -> None:
    current = make_fact(
        EPS,
        6.88,
        unit="usdPerShare",
        period_start=date(2025, 9, 28),
        fiscal_period=None,
    )
    previous = make_fact(
        EPS,
        4.85,
        unit="usdPerShare",
        accession=PREVIOUS_ACCESSION,
        filing_date=date(2026, 5, 1),
        period_start=date(2025, 9, 28),
        period_end=date(2026, 3, 28),
        fiscal_period=None,
    )

    metric = metric_named(build_financial_metrics([current], [previous]), "Diluted EPS")

    assert metric.current_value is None
    assert metric.previous_value is None
    assert metric.change_pct is None


def test_cash_compares_consecutive_instants() -> None:
    metric = metric_named(
        build_financial_metrics(
            [instant_fact(CASH, 90.0)],
            [instant_fact(CASH, 100.0, current=False)],
        ),
        "Cash and Cash Equivalents",
    )

    assert metric.current_value == 90.0
    assert metric.previous_value == 100.0
    assert metric.comparison_type == "QoQ"
    assert metric.change_pct == pytest.approx(-10.0)


def test_instant_metric_does_not_use_historical_value_from_current_filing() -> None:
    current = instant_fact(
        CASH,
        90.0,
        period_instant=date(2026, 3, 28),
    )
    embedded_historical = instant_fact(
        CASH,
        100.0,
        period_instant=date(2025, 12, 27),
        fiscal_year=2025,
    )

    metric = metric_named(
        build_financial_metrics([current, embedded_historical], []),
        "Cash and Cash Equivalents",
    )

    assert metric.current_value == 90.0
    assert metric.previous_value is None
    assert metric.comparison_type is None


def test_total_debt_sums_all_required_components() -> None:
    current = [
        instant_fact(COMMERCIAL_PAPER, 10.0),
        instant_fact(DEBT_CURRENT, 20.0),
        instant_fact(DEBT_NONCURRENT, 70.0),
    ]
    previous = [
        instant_fact(COMMERCIAL_PAPER, 5.0, current=False),
        instant_fact(DEBT_CURRENT, 15.0, current=False),
        instant_fact(DEBT_NONCURRENT, 80.0, current=False),
    ]

    metric = metric_named(build_financial_metrics(current, previous), "Total Debt")

    assert metric.current_value == 100.0
    assert metric.previous_value == 100.0
    assert len(metric.source_ids) == 6


def test_operating_cash_flow_uses_comparable_yoy_ytd() -> None:
    current = make_fact(
        OCF,
        117.0,
        period_start=date(2025, 9, 28),
        fiscal_period=None,
        statement_type="CashFlowStatement",
    )
    prior_year = make_fact(
        OCF,
        82.0,
        period_start=date(2024, 9, 29),
        period_end=date(2025, 6, 28),
        fiscal_year=2025,
        fiscal_period=None,
        statement_type="CashFlowStatement",
    )
    previous_filing_ytd = make_fact(
        OCF,
        83.0,
        accession=PREVIOUS_ACCESSION,
        filing_date=date(2026, 5, 1),
        period_start=date(2025, 9, 28),
        period_end=date(2026, 3, 28),
        fiscal_period=None,
        statement_type="CashFlowStatement",
    )

    metric = metric_named(
        build_financial_metrics([current, prior_year], [previous_filing_ytd]),
        "Operating Cash Flow",
    )

    assert metric.previous_value == 82.0
    assert metric.comparison_type == "YoY_YTD"


def test_capex_uses_comparable_yoy_ytd() -> None:
    current = make_fact(
        CAPEX,
        6.8,
        period_start=date(2025, 9, 28),
        fiscal_period=None,
        statement_type="CashFlowStatement",
    )
    prior_year = make_fact(
        CAPEX,
        9.5,
        period_start=date(2024, 9, 29),
        period_end=date(2025, 6, 28),
        fiscal_year=2025,
        fiscal_period=None,
        statement_type="CashFlowStatement",
    )

    metric = metric_named(
        build_financial_metrics([current, prior_year], []),
        "Capital Expenditures",
    )

    assert metric.previous_value == 9.5
    assert metric.comparison_type == "YoY_YTD"


def test_dimensioned_breakdown_is_discarded() -> None:
    consolidated = make_fact(REVENUE, 100.0, source_id="consolidated")
    product = make_fact(
        REVENUE,
        999.0,
        dimensions={"srt:ProductOrServiceAxis": "aapl:IPhoneMember"},
        source_id="product",
    )

    metric = metric_named(
        build_financial_metrics([product, consolidated], []),
        "Revenue",
    )

    assert metric.current_value == 100.0
    assert metric.source_ids == ["consolidated"]


def test_incorrect_unit_is_discarded() -> None:
    metric = metric_named(
        build_financial_metrics([make_fact(REVENUE, 100.0, unit="shares")], []),
        "Revenue",
    )

    assert metric.current_value is None


def test_incompatible_periods_are_not_compared() -> None:
    incompatible = make_fact(
        REVENUE,
        200.0,
        accession=PREVIOUS_ACCESSION,
        filing_date=date(2026, 5, 1),
        period_start=date(2025, 9, 28),
        period_end=date(2026, 3, 28),
        fiscal_period=None,
    )

    metric = metric_named(
        build_financial_metrics([make_fact(REVENUE, 120.0)], [incompatible]),
        "Revenue",
    )

    assert metric.previous_value is None
    assert metric.comparison_type is None
    assert metric.change_pct is None


def test_previous_zero_has_no_percentage_change() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(REVENUE, 10.0)],
            [previous_quarter(REVENUE, 0.0)],
        ),
        "Revenue",
    )

    assert metric.previous_value == 0.0
    assert metric.change_pct is None


def test_missing_current_preserves_available_previous_value() -> None:
    metric = metric_named(
        build_financial_metrics([], [previous_quarter(REVENUE, 100.0)]),
        "Revenue",
    )

    assert metric.current_value is None
    assert metric.previous_value == 100.0
    assert metric.comparison_type is None


def test_missing_previous_does_not_force_comparison() -> None:
    metric = metric_named(
        build_financial_metrics([make_fact(REVENUE, 100.0)], []),
        "Revenue",
    )

    assert metric.current_value == 100.0
    assert metric.previous_value is None
    assert metric.change_pct is None


def test_quarter_is_preferred_over_ytd_at_same_period_end() -> None:
    quarter = make_fact(REVENUE, 100.0, source_id="quarter")
    ytd = make_fact(
        REVENUE,
        300.0,
        period_start=date(2025, 9, 28),
        fiscal_period=None,
        source_id="ytd",
    )

    metric = metric_named(build_financial_metrics([ytd, quarter], []), "Revenue")

    assert metric.current_value == 100.0
    assert metric.source_ids == ["quarter"]


def test_equivalent_candidate_tie_break_is_deterministic() -> None:
    first = make_fact(REVENUE, 100.0, source_id="z-source")
    second = make_fact(REVENUE, 100.0, source_id="a-source")

    forward = metric_named(build_financial_metrics([first, second], []), "Revenue")
    reverse = metric_named(build_financial_metrics([second, first], []), "Revenue")

    assert forward.model_dump() == reverse.model_dump()
    assert forward.source_ids == ["a-source"]


def test_amendment_replaces_original_cohort_for_same_period() -> None:
    original = make_fact(REVENUE, 100.0, accession="original")
    amendment = make_fact(
        REVENUE,
        110.0,
        accession="amendment",
        filing_date=date(2026, 8, 5),
        amended=True,
    )

    metric = metric_named(
        build_financial_metrics([original, amendment], []),
        "Revenue",
    )

    assert metric.current_value == 110.0
    assert metric.source_ids[0].startswith("xbrl:amendment:")


@pytest.mark.parametrize(
    ("period_start", "period_end", "expected"),
    [
        (date(2026, 1, 1), date(2026, 3, 31), "quarter"),
        (date(2026, 1, 1), date(2026, 6, 30), "half_year_ytd"),
        (date(2026, 1, 1), date(2026, 9, 30), "nine_month_ytd"),
        (date(2026, 1, 1), date(2026, 12, 31), "annual"),
        (date(2026, 1, 1), date(2026, 2, 1), "other"),
    ],
)
def test_duration_classification(
    period_start: date,
    period_end: date,
    expected: str,
) -> None:
    assert classify_duration(period_start, period_end) == expected


def test_duration_classification_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="period_start"):
        classify_duration(date(2026, 2, 1), date(2026, 1, 1))


def test_different_tickers_are_rejected() -> None:
    with pytest.raises(FinancialMetricSelectionError, match="one ticker"):
        build_financial_metrics(
            [make_fact(REVENUE, 100.0, ticker="AAPL")],
            [previous_quarter(REVENUE, 90.0, ticker="MSFT")],
        )


def test_positive_percentage_change() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(REVENUE, 150.0)],
            [previous_quarter(REVENUE, 100.0)],
        ),
        "Revenue",
    )

    assert metric.change_pct == pytest.approx(50.0)


def test_negative_percentage_change() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(REVENUE, 75.0)],
            [previous_quarter(REVENUE, 100.0)],
        ),
        "Revenue",
    )

    assert metric.change_pct == pytest.approx(-25.0)


def test_negative_values_keep_financial_sign() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(NET_INCOME, -50.0)],
            [previous_quarter(NET_INCOME, -100.0)],
        ),
        "Net Income",
    )

    assert metric.change_pct == pytest.approx(50.0)


def test_incomplete_total_debt_is_unavailable() -> None:
    incomplete = [
        instant_fact(COMMERCIAL_PAPER, 10.0),
        instant_fact(DEBT_CURRENT, 20.0),
    ]

    metric = metric_named(build_financial_metrics(incomplete, []), "Total Debt")

    assert metric.current_value is None
    assert metric.source_ids == []


def test_financial_metric_serialization_includes_traceability() -> None:
    metric = metric_named(
        build_financial_metrics(
            [make_fact(REVENUE, 120.0)],
            [previous_quarter(REVENUE, 100.0)],
        ),
        "Revenue",
    )

    dumped = json.loads(metric.model_dump_json())

    assert dumped["comparison_type"] == "QoQ"
    assert dumped["current_period"] == "2026-03-29/2026-06-27"
    assert len(dumped["source_ids"]) == 2


def test_metric_order_is_stable() -> None:
    names = [metric.name for metric in build_financial_metrics([], [])]

    assert names == [
        "Revenue",
        "Net Income",
        "Diluted EPS",
        "Cash and Cash Equivalents",
        "Total Debt",
        "Operating Cash Flow",
        "Capital Expenditures",
    ]


def test_statement_type_must_match_registry() -> None:
    wrong_statement = make_fact(
        REVENUE,
        100.0,
        statement_type="BalanceSheet",
    )

    metric = metric_named(
        build_financial_metrics([wrong_statement], []),
        "Revenue",
    )

    assert metric.current_value is None
