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


__all__ = [
    "ErrorPresentation",
    "error_presentation",
    "filing_option_label",
    "format_display_date",
    "normalize_ticker_for_ui",
    "sort_filings",
    "verification_label",
]
