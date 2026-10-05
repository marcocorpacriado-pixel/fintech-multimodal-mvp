"""Normalize source XBRL facts before financial metric selection.

This phase deliberately preserves source facts without mapping concepts,
choosing preferred contexts, comparing periods, or calculating metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

import pandas as pd

from .schemas import FilingType, NormalizedXBRLFact, XBRLPeriodType


class XBRLNormalizationError(ValueError):
    """Raised when a numeric source fact has invalid required metadata."""


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


__all__ = ["XBRLNormalizationError", "normalize_filing_facts"]
