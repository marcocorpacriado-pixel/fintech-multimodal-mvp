"""Normalize source XBRL facts and select comparable financial metrics.

Normalization preserves source fidelity. Metric selection is a separate,
deterministic layer that applies an explicit registry and temporal rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

import pandas as pd

from .schemas import (
    ComparisonType,
    FilingType,
    FinancialMetric,
    NormalizedXBRLFact,
    XBRLPeriodType,
)


class XBRLNormalizationError(ValueError):
    """Raised when a numeric source fact has invalid required metadata."""


class FinancialMetricSelectionError(ValueError):
    """Raised when fact collections cannot be analyzed safely together."""


DurationClass = Literal[
    "quarter",
    "half_year_ytd",
    "nine_month_ytd",
    "annual",
    "other",
]
MetricPolicy = Literal[
    "quarter_preferred",
    "quarter_required",
    "instant",
    "ytd_preferred",
    "debt_components",
]


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable selection rules for one supported financial metric."""

    key: str
    name: str
    concepts: tuple[str, ...]
    unit: str
    period_type: XBRLPeriodType
    statement_types: tuple[str, ...]
    policy: MetricPolicy
    component_concepts: tuple[tuple[str, ...], ...] = ()


METRIC_REGISTRY: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        key="revenue",
        name="Revenue",
        concepts=(
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        ),
        unit="usd",
        period_type="duration",
        statement_types=("IncomeStatement",),
        policy="quarter_preferred",
    ),
    MetricDefinition(
        key="net_income",
        name="Net Income",
        concepts=("us-gaap:NetIncomeLoss",),
        unit="usd",
        period_type="duration",
        statement_types=("IncomeStatement",),
        policy="quarter_preferred",
    ),
    MetricDefinition(
        key="diluted_eps",
        name="Diluted EPS",
        concepts=("us-gaap:EarningsPerShareDiluted",),
        unit="usdPerShare",
        period_type="duration",
        statement_types=("IncomeStatement",),
        policy="quarter_required",
    ),
    MetricDefinition(
        key="cash_and_cash_equivalents",
        name="Cash and Cash Equivalents",
        concepts=("us-gaap:CashAndCashEquivalentsAtCarryingValue",),
        unit="usd",
        period_type="instant",
        statement_types=("BalanceSheet",),
        policy="instant",
    ),
    MetricDefinition(
        key="total_debt",
        name="Total Debt",
        concepts=(),
        unit="usd",
        period_type="instant",
        statement_types=("BalanceSheet",),
        policy="debt_components",
        component_concepts=(
            ("us-gaap:CommercialPaper",),
            ("us-gaap:LongTermDebtCurrent",),
            ("us-gaap:LongTermDebtNoncurrent",),
        ),
    ),
    MetricDefinition(
        key="operating_cash_flow",
        name="Operating Cash Flow",
        concepts=("us-gaap:NetCashProvidedByUsedInOperatingActivities",),
        unit="usd",
        period_type="duration",
        statement_types=("CashFlowStatement",),
        policy="ytd_preferred",
    ),
    MetricDefinition(
        key="capital_expenditures",
        name="Capital Expenditures",
        concepts=("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",),
        unit="usd",
        period_type="duration",
        statement_types=("CashFlowStatement",),
        policy="ytd_preferred",
    ),
)


@dataclass(frozen=True)
class _SelectedValue:
    value: float
    unit: str
    period_type: XBRLPeriodType
    period_start: date | None
    period_end: date | None
    period_instant: date | None
    fiscal_year: int | None
    fiscal_period: str | None
    source_ids: tuple[str, ...]
    selection_rank: int = 0

    @property
    def terminal_date(self) -> date:
        terminal = self.period_instant or self.period_end
        if terminal is None:  # guarded by NormalizedXBRLFact invariants
            raise FinancialMetricSelectionError("selected value has no period date")
        return terminal


class _FactsLike(Protocol):
    def to_dataframe(self) -> pd.DataFrame:
        """Return source facts in edgartools long-form layout."""


class _XBRLLike(Protocol):
    facts: _FactsLike


class _FilingLike(Protocol):
    form: str
    filing_date: date
    accession_number: str

    def xbrl(self) -> _XBRLLike | None:
        """Return the filing's parsed XBRL representation."""


def normalize_filing_facts(
    filing: _FilingLike,
    *,
    ticker: str,
) -> list[NormalizedXBRLFact]:
    """Normalize usable numeric facts from one edgartools filing.

    Facts remain in source order and are never deduplicated. Non-numeric facts,
    non-finite numeric values, and numeric metadata without ``unit_ref`` are
    excluded because they cannot satisfy the quantitative fact contract.

    Args:
        filing: An edgartools filing or structurally compatible test double.
        ticker: Explicit ticker retained as source metadata.

    Returns:
        Canonical long-form numeric XBRL facts.

    Raises:
        XBRLNormalizationError: If XBRL is unavailable or a usable numeric fact
            has incomplete or inconsistent period metadata.
    """

    xbrl = filing.xbrl()
    if xbrl is None:
        raise XBRLNormalizationError("filing does not contain XBRL data")

    facts = xbrl.facts.to_dataframe()
    if facts.empty:
        return []

    form = filing.form
    accession = _filing_accession(filing)
    filing_date = filing.filing_date
    amended = _is_amended(form, facts)
    dimension_columns = [
        column for column in facts.columns if column.startswith("dim_")
    ]

    normalized: list[NormalizedXBRLFact] = []
    for row_number, row in enumerate(facts.to_dict(orient="records")):
        numeric_value = _finite_number(row.get("numeric_value"))
        unit = _optional_text(row.get("unit_ref"))
        if numeric_value is None or unit is None:
            continue

        try:
            period_type = _required_period_type(row.get("period_type"))
            dimensions = _extract_dimensions(row, dimension_columns)
            fact_id = _optional_text(row.get("fact_id"))
            context_ref = _required_text(row.get("context_ref"), "context_ref")
            concept = _required_text(row.get("concept"), "concept")
            source_id = _build_fact_source_id(
                accession=accession,
                fact_id=fact_id,
                fact_key=_optional_text(row.get("fact_key")),
                row_number=row_number,
                concept=concept,
                context_ref=context_ref,
                unit=unit,
                numeric_value=numeric_value,
                dimensions=dimensions,
            )
            normalized.append(
                NormalizedXBRLFact(
                    ticker=ticker,
                    form=form,
                    accession=accession,
                    filing_date=filing_date,
                    fact_id=fact_id,
                    context_ref=context_ref,
                    source_id=source_id,
                    concept=concept,
                    standard_concept=_optional_text(row.get("standard_concept")),
                    label=_optional_text(row.get("label")),
                    numeric_value=numeric_value,
                    unit=unit,
                    period_type=period_type,
                    period_start=_optional_date(row.get("period_start")),
                    period_end=_optional_date(row.get("period_end")),
                    period_instant=_optional_date(row.get("period_instant")),
                    fiscal_year=_optional_int(row.get("fiscal_year")),
                    fiscal_period=_optional_text(row.get("fiscal_period")),
                    dimensions=dimensions,
                    statement_type=_optional_text(row.get("statement_type")),
                    amended=amended,
                )
            )
        except (TypeError, ValueError) as error:
            raise XBRLNormalizationError(
                f"invalid numeric XBRL fact at row {row_number}: {error}"
            ) from error
    return normalized


def _filing_accession(filing: _FilingLike) -> str:
    """Read the stable accession name used by edgartools versions in scope."""

    accession = _optional_text(getattr(filing, "accession_number", None))
    if accession is None:
        accession = _optional_text(getattr(filing, "accession_no", None))
    if accession is None:
        raise XBRLNormalizationError("filing accession is missing")
    return accession


def _is_amended(form: str, facts: pd.DataFrame) -> bool:
    """Preserve amendment status from either form or the DEI source fact."""

    if form.endswith("/A"):
        return True
    if "concept" not in facts.columns or "value" not in facts.columns:
        return False

    flags = facts.loc[facts["concept"].eq("dei:AmendmentFlag"), "value"]
    return any(str(value).strip().casefold() in {"true", "1"} for value in flags)


def _extract_dimensions(
    row: Mapping[str, Any],
    dimension_columns: list[str],
) -> dict[str, str]:
    """Preserve every populated axis/member pair from edgartools facts."""

    dimensions: dict[str, str] = {}
    for column in dimension_columns:
        member = _optional_text(row.get(column))
        if member is not None:
            dimensions[_dimension_axis_from_column(column)] = member

    axis = _optional_text(row.get("dimension"))
    member = _optional_text(row.get("member"))
    if axis is not None and member is not None:
        dimensions.setdefault(axis, member)
    return dict(sorted(dimensions.items()))


def _dimension_axis_from_column(column: str) -> str:
    """Restore the QName separator used in a dynamic ``dim_`` column."""

    axis = column.removeprefix("dim_")
    prefix, separator, local_name = axis.partition("_")
    return f"{prefix}:{local_name}" if separator else axis


def _build_fact_source_id(
    *,
    accession: str,
    fact_id: str | None,
    fact_key: str | None,
    row_number: int,
    concept: str,
    context_ref: str,
    unit: str,
    numeric_value: float,
    dimensions: dict[str, str],
) -> str:
    """Build a deterministic fact identifier, preferring edgartools IDs."""

    source_fact_id = fact_id or fact_key
    if source_fact_id is not None:
        return f"xbrl:{accession}:{source_fact_id}"

    identity = {
        "accession": accession,
        "concept": concept,
        "context_ref": context_ref,
        "dimensions": dimensions,
        "numeric_value": numeric_value,
        "row_number": row_number,
        "unit": unit,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"xbrl:{accession}:{digest}"


def _finite_number(value: Any) -> float | None:
    """Return a finite float or mark the source value as unusable."""

    if _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _required_period_type(value: Any) -> XBRLPeriodType:
    """Validate the two period types supported by XBRL financial facts."""

    period_type = _required_text(value, "period_type")
    if period_type not in {"instant", "duration"}:
        raise ValueError(f"unsupported period_type: {period_type!r}")
    return period_type  # type: ignore[return-value]


def _required_text(value: Any, field_name: str) -> str:
    """Return non-empty source text or raise a contextual error."""

    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is missing")
    return text


def _optional_text(value: Any) -> str | None:
    """Normalize nullable dataframe text without converting NaN to text."""

    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_date(value: Any) -> date | None:
    """Normalize nullable date-like source values."""

    if _is_missing(value):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _optional_int(value: Any) -> int | None:
    """Normalize nullable integral dataframe values."""

    if _is_missing(value):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"expected an integer value, got {value!r}")
    return int(numeric)


def _is_missing(value: Any) -> bool:
    """Handle Python, NumPy, and pandas scalar missing values."""

    if value is None:
        return True
    missing = pd.isna(value)
    try:
        return bool(missing)
    except (TypeError, ValueError):
        # ``pd.isna`` returns an array for non-scalar inputs; source dataframe
        # cells should be scalar, so an array is not treated as a missing cell.
        return False


def classify_duration(period_start: date, period_end: date) -> DurationClass:
    """Classify an inclusive XBRL duration using reporting-calendar ranges.

    The ranges tolerate 52/53-week calendars and modest calendar variation:
    80-100 days for a quarter, 170-200 for half-year YTD, 250-290 for
    nine-month YTD, and 340-380 for an annual period.
    """

    if period_start > period_end:
        raise ValueError("period_start must be before or equal to period_end")
    duration_days = (period_end - period_start).days + 1
    if 80 <= duration_days <= 100:
        return "quarter"
    if 170 <= duration_days <= 200:
        return "half_year_ytd"
    if 250 <= duration_days <= 290:
        return "nine_month_ytd"
    if 340 <= duration_days <= 380:
        return "annual"
    return "other"


def build_financial_metrics(
    current_facts: Sequence[NormalizedXBRLFact],
    previous_facts: Sequence[NormalizedXBRLFact],
) -> list[FinancialMetric]:
    """Build the seven supported metrics from current and previous filings.

    Each input may contain duplicate presentation facts and comparative facts
    embedded in a filing. One accession is selected per input before metric
    selection, preferring the latest covered period and then an amendment.
    Comparatives embedded in the current filing may be used when they are the
    only semantically compatible prior period, notably for YTD cash flows.

    Missing or incompatible facts produce ``None`` rather than a fabricated
    value. The result order follows ``METRIC_REGISTRY``.
    """

    current_source = tuple(current_facts)
    previous_source = tuple(previous_facts)
    _validate_single_ticker(current_source, previous_source)
    current_cohort = _select_filing_cohort(current_source)
    previous_cohort = _select_filing_cohort(previous_source)

    metrics: list[FinancialMetric] = []
    for definition in METRIC_REGISTRY:
        if definition.policy == "debt_components":
            current_options = _debt_values(current_cohort, definition)
            previous_options = _debt_values(previous_cohort, definition)
        else:
            current_options = _scalar_values(current_cohort, definition)
            previous_options = _scalar_values(previous_cohort, definition)

        current = _latest_value(current_options, definition)
        previous: _SelectedValue | None = None
        comparison_type: ComparisonType | None = None
        if current is not None:
            comparison_candidates = previous_options
            if definition.period_type == "duration":
                comparison_candidates = (*previous_options, *current_options)
            previous, comparison_type = _select_comparison(
                current,
                definition,
                comparison_candidates,
            )
        elif previous_options:
            previous = _latest_value(previous_options, definition)

        change_pct = _percentage_change(current, previous)
        source_ids = _unique_source_ids(current, previous)
        metrics.append(
            FinancialMetric(
                name=definition.name,
                current_value=current.value if current else None,
                previous_value=previous.value if previous else None,
                change_pct=change_pct,
                unit=definition.unit,
                comparison_type=comparison_type,
                current_period=_format_period(current),
                previous_period=_format_period(previous),
                source_ids=source_ids,
            )
        )
    return metrics


def _validate_single_ticker(
    current_facts: Sequence[NormalizedXBRLFact],
    previous_facts: Sequence[NormalizedXBRLFact],
) -> None:
    tickers = {fact.ticker for fact in (*current_facts, *previous_facts)}
    if len(tickers) > 1:
        raise FinancialMetricSelectionError(
            "current and previous facts must belong to one ticker"
        )


def _select_filing_cohort(
    facts: Sequence[NormalizedXBRLFact],
) -> tuple[NormalizedXBRLFact, ...]:
    """Choose one accession without mixing original and amended filings."""

    by_accession: dict[str, list[NormalizedXBRLFact]] = {}
    for fact in facts:
        by_accession.setdefault(fact.accession, []).append(fact)
    if not by_accession:
        return ()

    def cohort_key(item: tuple[str, list[NormalizedXBRLFact]]) -> tuple[Any, ...]:
        accession, cohort = item
        covered_through = max(_fact_terminal_date(fact) for fact in cohort)
        amended = any(fact.amended for fact in cohort)
        filing_date = max(fact.filing_date for fact in cohort)
        return covered_through, amended, filing_date, accession

    _, selected = max(by_accession.items(), key=cohort_key)
    return tuple(selected)


def _scalar_values(
    cohort: Sequence[NormalizedXBRLFact],
    definition: MetricDefinition,
) -> tuple[_SelectedValue, ...]:
    facts = _eligible_facts(cohort, definition, definition.concepts)
    return tuple(
        _value_from_fact(fact, definition.concepts.index(fact.concept))
        for fact in facts
    )


def _debt_values(
    cohort: Sequence[NormalizedXBRLFact],
    definition: MetricDefinition,
) -> tuple[_SelectedValue, ...]:
    """Build conservative debt totals only when every component is present."""

    component_facts = [
        _eligible_facts(cohort, definition, alternatives)
        for alternatives in definition.component_concepts
    ]
    if not component_facts or any(not facts for facts in component_facts):
        return ()

    shared_dates = set.intersection(
        *[
            {fact.period_instant for fact in facts if fact.period_instant is not None}
            for facts in component_facts
        ]
    )
    values: list[_SelectedValue] = []
    for instant in sorted(shared_dates):
        selected = [
            min(
                (fact for fact in facts if fact.period_instant == instant),
                key=lambda fact: _fact_tie_key(fact, alternatives),
            )
            for facts, alternatives in zip(
                component_facts,
                definition.component_concepts,
                strict=True,
            )
        ]
        values.append(
            _SelectedValue(
                value=sum(fact.numeric_value for fact in selected),
                unit=definition.unit,
                period_type="instant",
                period_start=None,
                period_end=None,
                period_instant=instant,
                fiscal_year=max(
                    (fact.fiscal_year for fact in selected if fact.fiscal_year),
                    default=None,
                ),
                fiscal_period=None,
                source_ids=tuple(fact.source_id for fact in selected),
                selection_rank=0,
            )
        )
    return tuple(values)


def _eligible_facts(
    cohort: Sequence[NormalizedXBRLFact],
    definition: MetricDefinition,
    concepts: tuple[str, ...],
) -> list[NormalizedXBRLFact]:
    return [
        fact
        for fact in cohort
        if fact.concept in concepts
        and fact.unit == definition.unit
        and not fact.dimensions
        and fact.period_type == definition.period_type
        and fact.statement_type in definition.statement_types
    ]


def _value_from_fact(
    fact: NormalizedXBRLFact,
    selection_rank: int,
) -> _SelectedValue:
    return _SelectedValue(
        value=fact.numeric_value,
        unit=fact.unit,
        period_type=fact.period_type,
        period_start=fact.period_start,
        period_end=fact.period_end,
        period_instant=fact.period_instant,
        fiscal_year=fact.fiscal_year,
        fiscal_period=fact.fiscal_period,
        source_ids=(fact.source_id,),
        selection_rank=selection_rank,
    )


def _latest_value(
    values: Sequence[_SelectedValue],
    definition: MetricDefinition,
) -> _SelectedValue | None:
    if not values:
        return None
    latest_date = max(value.terminal_date for value in values)
    candidates = [value for value in values if value.terminal_date == latest_date]

    if definition.policy == "quarter_required":
        candidates = [value for value in candidates if _duration_class(value) == "quarter"]
        if not candidates:
            return None
    elif definition.policy == "quarter_preferred":
        quarter = [value for value in candidates if _duration_class(value) == "quarter"]
        if quarter:
            candidates = quarter
    elif definition.policy == "ytd_preferred":
        longest = max(_duration_days(value) for value in candidates)
        candidates = [value for value in candidates if _duration_days(value) == longest]

    return min(candidates, key=_value_tie_key)


def _select_comparison(
    current: _SelectedValue,
    definition: MetricDefinition,
    candidates: Sequence[_SelectedValue],
) -> tuple[_SelectedValue | None, ComparisonType | None]:
    compatible: list[tuple[_SelectedValue, ComparisonType]] = []
    for candidate in candidates:
        comparison_type = _comparison_type(current, candidate, definition)
        if comparison_type is not None:
            compatible.append((candidate, comparison_type))
    if not compatible:
        return None, None

    priority = {"YoY": 0, "YoY_YTD": 1}

    def candidate_key(
        item: tuple[_SelectedValue, ComparisonType],
    ) -> tuple[Any, ...]:
        candidate, comparison_type = item
        day_gap = (current.terminal_date - candidate.terminal_date).days
        return (
            priority[comparison_type],
            abs(day_gap - 365),
            -candidate.terminal_date.toordinal(),
            _value_tie_key(candidate),
        )

    previous, comparison_type = min(compatible, key=candidate_key)
    return previous, comparison_type


def _comparison_type(
    current: _SelectedValue,
    previous: _SelectedValue,
    definition: MetricDefinition,
) -> ComparisonType | None:
    if current.unit != previous.unit or current.period_type != previous.period_type:
        return None
    if previous.terminal_date >= current.terminal_date:
        return None

    # Strictly year-over-year: sequential quarters are seasonally biased.
    day_gap = (current.terminal_date - previous.terminal_date).days
    if current.period_type == "instant":
        return "YoY" if 340 <= day_gap <= 380 else None

    current_class = _duration_class(current)
    previous_class = _duration_class(previous)
    if current_class == "other" or current_class != previous_class:
        return None

    if definition.policy == "ytd_preferred":
        if _is_year_over_year(current, previous):
            return "YoY" if current_class == "annual" else "YoY_YTD"
        return None

    if current_class == "quarter":
        return "YoY" if _is_year_over_year(current, previous) else None

    if _is_year_over_year(current, previous):
        return "YoY" if current_class == "annual" else "YoY_YTD"
    return None


def _is_year_over_year(
    current: _SelectedValue,
    previous: _SelectedValue,
) -> bool:
    day_gap = (current.terminal_date - previous.terminal_date).days
    if not 340 <= day_gap <= 380:
        return False
    if current.fiscal_year is not None and previous.fiscal_year is not None:
        if current.fiscal_year != previous.fiscal_year + 1:
            return False
    if current.fiscal_period and previous.fiscal_period:
        return current.fiscal_period == previous.fiscal_period
    return True


def _duration_class(value: _SelectedValue) -> DurationClass:
    if value.period_start is None or value.period_end is None:
        raise FinancialMetricSelectionError("duration fact is missing dates")
    return classify_duration(value.period_start, value.period_end)


def _duration_days(value: _SelectedValue) -> int:
    if value.period_start is None or value.period_end is None:
        raise FinancialMetricSelectionError("duration fact is missing dates")
    return (value.period_end - value.period_start).days + 1


def _fact_terminal_date(fact: NormalizedXBRLFact) -> date:
    terminal = fact.period_instant or fact.period_end
    if terminal is None:  # guarded by NormalizedXBRLFact invariants
        raise FinancialMetricSelectionError("fact has no terminal period date")
    return terminal


def _fact_tie_key(
    fact: NormalizedXBRLFact,
    concepts: tuple[str, ...],
) -> tuple[Any, ...]:
    return (
        concepts.index(fact.concept),
        -(fact.fiscal_year or 0),
        fact.source_id,
        fact.context_ref,
    )


def _value_tie_key(value: _SelectedValue) -> tuple[Any, ...]:
    return (
        value.selection_rank,
        -(value.fiscal_year or 0),
        value.source_ids,
        value.value,
    )


def _percentage_change(
    current: _SelectedValue | None,
    previous: _SelectedValue | None,
) -> float | None:
    if current is None or previous is None or previous.value == 0:
        return None
    change = ((current.value - previous.value) / abs(previous.value)) * 100
    return change if math.isfinite(change) else None


def _format_period(value: _SelectedValue | None) -> str | None:
    if value is None:
        return None
    if value.period_type == "instant":
        return value.period_instant.isoformat() if value.period_instant else None
    if value.period_start is None or value.period_end is None:
        return None
    return f"{value.period_start.isoformat()}/{value.period_end.isoformat()}"


def _unique_source_ids(
    current: _SelectedValue | None,
    previous: _SelectedValue | None,
) -> list[str]:
    ordered = (
        *((current.source_ids if current else ())),
        *((previous.source_ids if previous else ())),
    )
    return list(dict.fromkeys(ordered))


__all__ = [
    "FinancialMetricSelectionError",
    "METRIC_REGISTRY",
    "MetricDefinition",
    "XBRLNormalizationError",
    "build_financial_metrics",
    "classify_duration",
    "normalize_filing_facts",
]
