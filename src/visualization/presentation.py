"""Pure presentation helpers shared by the Streamlit dashboard and its tests.

These helpers format API data only. They never query SEC, call an LLM, or
recalculate financial values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping, Sequence


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,14}$")
_SNAPSHOT_METRICS = (
    "Revenue",
    "Net Income",
    "Diluted EPS",
    "Operating Cash Flow",
)
_SECTION_LABELS = {
    "PART_I_ITEM_1": "Item 1 · Financial Statements",
    "PART_I_ITEM_2": "Item 2 · Management Discussion & Analysis",
    "PART_I_ITEM_3": "Item 3 · Market Risk Disclosures",
    "PART_I_ITEM_4": "Item 4 · Controls and Procedures",
    "PART_II_ITEM_1": "Item 1 · Legal Proceedings",
    "PART_II_ITEM_1A": "Item 1A · Risk Factors",
    "PART_II_ITEM_2": "Item 2 · Unregistered Sales and Use of Proceeds",
    "ITEM_1": "Item 1 · Business",
    "ITEM_1A": "Item 1A · Risk Factors",
    "ITEM_7": "Item 7 · Management Discussion & Analysis",
    "ITEM_7A": "Item 7A · Market Risk Disclosures",
    "ITEM_8": "Item 8 · Financial Statements",
    "UNSECTIONED": "Unsectioned filing content",
}


@dataclass(frozen=True, slots=True)
class ErrorPresentation:
    """Safe, user-facing copy for one integration error category."""

    title: str
    guidance: str


ERROR_PRESENTATIONS: dict[str, ErrorPresentation] = {
    "INPUT_ERROR": ErrorPresentation(
        "Check the analysis configuration.",
        "Review the ticker and selected filing parameters before trying again.",
    ),
    "FILING_NOT_FOUND": ErrorPresentation(
        "The requested filing was not found.",
        "Choose another filing available for this company and form.",
    ),
    "SEC_INGESTION_ERROR": ErrorPresentation(
        "SEC data is temporarily unavailable.",
        "The filing could not be prepared. Retry later if the service is unavailable.",
    ),
    "LLM_PROVIDER_ERROR": ErrorPresentation(
        "The AI provider did not return a usable response.",
        "No unverified analysis was shown. You can retry the same configuration.",
    ),
    "GROUNDING_ERROR": ErrorPresentation(
        "The generated analysis did not pass evidence checks.",
        "Qualitative claims must be supported by an exact, contiguous filing excerpt, "
        "so this result was blocked rather than shown as valid.",
    ),
    "VERIFICATION_ERROR": ErrorPresentation(
        "The generated analysis did not pass deterministic verification.",
        "The result was blocked because its metrics, citations, or narrative rules "
        "could not be verified.",
    ),
    "UNKNOWN_ERROR": ErrorPresentation(
        "The analysis could not be completed.",
        "No partial result was published. Review the configuration or try again later.",
    ),
    "API_UNAVAILABLE": ErrorPresentation(
        "The analysis service is unavailable.",
        "Confirm that the FastAPI service is running and try again.",
    ),
}


def normalize_ticker_for_ui(value: str) -> str | None:
    """Return the normalized SEC ticker, or ``None`` for invalid syntax."""

    normalized = value.strip().upper()
    return normalized if _TICKER_RE.fullmatch(normalized) else None


def format_display_date(value: str | date | None) -> str:
    """Format an ISO date for compact institutional UI copy."""

    if value in (None, ""):
        return "Not available"
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return parsed.strftime("%b %d, %Y")


def format_metric_period(value: str | None) -> str:
    """Humanize one instant or duration without inferring fiscal quarters."""

    if not value:
        return "Not available"
    if "/" not in value:
        return format_display_date(value)
    start_text, end_text = value.split("/", 1)
    try:
        start = date.fromisoformat(start_text)
        end = date.fromisoformat(end_text)
    except ValueError:
        return value
    start_label = start.strftime("%b %d").replace(" 0", " ")
    end_label = end.strftime("%b %d").replace(" 0", " ")
    if start.year == end.year:
        return f"{start_label}–{end_label}, {end.year}"
    return f"{start_label}, {start.year}–{end_label}, {end.year}"


def filing_option_label(filing: Mapping[str, Any]) -> str:
    """Build a human-readable filing selector label without inventing fiscal quarters."""

    report = format_display_date(filing.get("report_date"))
    filed = format_display_date(filing.get("filing_date"))
    form = str(filing.get("form") or "Filing")
    return f"Report {report} · Filed {filed} · {form}"


def sort_filings(filings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return newest filings first without mutating the API payload."""

    return sorted(
        (dict(filing) for filing in filings),
        key=lambda filing: (
            str(filing.get("filing_date") or ""),
            str(filing.get("report_date") or ""),
            str(filing.get("accession") or ""),
        ),
        reverse=True,
    )


def error_presentation(code: str) -> ErrorPresentation:
    """Return stable safe copy for an API error code."""

    return ERROR_PRESENTATIONS.get(code, ERROR_PRESENTATIONS["UNKNOWN_ERROR"])


def verification_label(
    verification: Mapping[str, Any],
) -> Literal["VERIFIED", "VERIFIED WITH WARNINGS", "FAILED VERIFICATION"]:
    """Classify the handoff verification state without changing its semantics."""

    if not verification.get("valid", False):
        return "FAILED VERIFICATION"
    issues = verification.get("issues") or []
    if any(issue.get("severity") == "warning" for issue in issues):
        return "VERIFIED WITH WARNINGS"
    return "VERIFIED"


def select_executive_metrics(
    metrics: Sequence[Mapping[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Select a fixed canonical snapshot without scoring or recomputation."""

    by_name = {str(metric.get("name")): dict(metric) for metric in metrics}
    selected = [by_name[name] for name in _SNAPSHOT_METRICS if name in by_name]
    if len(selected) < limit:
        selected_names = {metric["name"] for metric in selected}
        selected.extend(
            dict(metric)
            for metric in metrics
            if metric.get("name") not in selected_names
        )
    return selected[:limit]


def human_source_label(filing_type: str, source_section: str | None) -> str:
    """Present a canonical SEC section before its technical identifier."""

    section = source_section or "UNSECTIONED"
    readable = _SECTION_LABELS.get(section, section.replace("_", " ").title())
    return f"{filing_type} · {readable}"


__all__ = [
    "ErrorPresentation",
    "error_presentation",
    "filing_option_label",
    "format_display_date",
    "format_metric_period",
    "human_source_label",
    "normalize_ticker_for_ui",
    "select_executive_metrics",
    "sort_filings",
    "verification_label",
]
