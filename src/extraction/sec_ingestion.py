"""Minimal SEC ingestion adapter for the end-to-end analysis pipeline.

The adapter selects one explicit target filing and the immediately preceding
comparable fiscal filing, verifies that both expose source XBRL, and
materializes edgartools' narrative text as a deterministic UTF-8 file. It is
intentionally not a bulk SEC downloader.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from edgar import Company
from edgar.core import is_probably_html
from edgar.documents import HTMLParser, ParserConfig
from edgar.httprequests import IdentityNotSetException
from edgar.richtools import rich_to_text

from .schemas import FilingType


DEFAULT_SEC_INGESTION_DIR = Path("data/processed/sec_ingestion")
SEC_IDENTITY_ENV_VAR = "EDGAR_IDENTITY"

_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,14}$")
_SUPPORTED_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})


class SECIngestionError(RuntimeError):
    """Base error for controlled SEC ingestion failures."""


class SECInputError(SECIngestionError):
    """Raised before network access when a SEC selector is invalid."""


class SECServiceError(SECIngestionError):
    """Raised when the SEC/edgartools service cannot complete a request."""


class SECIdentityError(SECIngestionError):
    """Raised when production SEC access lacks a User-Agent identity."""


class SECFilingNotFoundError(SECIngestionError):
    """Raised when the explicitly requested target filing cannot be found."""


class SECPreviousFilingNotFoundError(SECIngestionError):
    """Raised when no preceding filing of the same base form exists."""


class SECNarrativeExtractionError(SECIngestionError):
    """Raised when narrative text cannot be extracted or safely materialized."""


class SECXBRLUnavailableError(SECIngestionError):
    """Raised when a selected filing has no source XBRL object."""


class SECFilingLike(Protocol):
    """Structural subset of edgartools ``EntityFiling`` used by the adapter."""

    cik: int
    company: str
    form: str
    filing_date: date | str
    report_date: date | str
    accession_no: str

    def text(self) -> str:
        """Return normalized narrative text from the primary filing document."""

    def xbrl(self) -> Any | None:
        """Return the source XBRL object when present."""


class SECCompanyLike(Protocol):
    """Structural subset of edgartools ``Company`` used by the adapter."""

    name: str
    tickers: Iterable[str]

    def get_filings(self, **kwargs: Any) -> Iterable[SECFilingLike] | None:
        """Return filings matching edgartools-compatible keyword filters."""


SECCompanyFactory = Callable[[str], SECCompanyLike]


@dataclass(frozen=True, slots=True)
class SECAnalysisInputs:
    """Runtime inputs ready to be passed directly into D7.

    Filing objects remain runtime values rather than Pydantic fields because
    edgartools objects are stateful and intentionally not JSON serializable.
    """

    ticker: str
    company: str
    filing_date: date
    report_period: str
    filing_type: FilingType
    current_filing: SECFilingLike
    previous_filing: SECFilingLike
    filing_path: Path
    current_accession: str
    previous_accession: str

    @property
    def period(self) -> str:
        """Backward-compatible alias for the financial report period."""

        return self.report_period


@dataclass(frozen=True, slots=True)
class SECFilingMetadata:
    """Lightweight filing metadata for discovery without document analysis."""

    ticker: str
    company: str
    filing_date: date
    report_date: date
    form: FilingType
    accession: str


def prepare_sec_analysis_inputs(
    *,
    ticker: str,
    accession: str | None = None,
    filing_date: date | None = None,
    form: FilingType = "10-Q",
    output_dir: str | Path | None = None,
    company_factory: SECCompanyFactory = Company,
) -> SECAnalysisInputs:
    """Prepare one target SEC filing and its previous comparable filing.

    At least one exact target selector is required. When both ``accession`` and
    ``filing_date`` are supplied, both must match the selected filing. The
    default edgartools provider requires ``EDGAR_IDENTITY`` in the environment;
    injected providers are intended for deterministic offline testing.

    Previous filing policy:
        Select the closest earlier ``report_date`` among filings of the same
        base form. For duplicate filings covering that fiscal period, prefer an
        amendment, then the latest filing date, then accession number.
    """

    normalized_ticker = normalize_sec_ticker(ticker)
    normalized_accession = _normalize_accession(accession)
    _validate_target_selector(normalized_accession, filing_date)
    _validate_form(form)
    _validate_identity_for_provider(company_factory)

    try:
        company = company_factory(normalized_ticker)
        _validate_company_ticker(company, normalized_ticker)
        current = _select_current_filing(
            company,
            accession=normalized_accession,
            filing_date=filing_date,
            requested_form=form,
        )
        comparable_filings = _company_filings(
            company,
            form=_base_form(form),
            amendments=True,
            trigger_full_load=False,
        )
    except SECIngestionError:
        raise
    except IdentityNotSetException as error:
        raise _identity_error() from error
    except Exception as error:
        raise SECServiceError(f"SEC filing lookup failed: {error}") from error

    _validate_current_filing(
        current,
        ticker=normalized_ticker,
        requested_accession=normalized_accession,
        requested_date=filing_date,
        requested_form=form,
    )
    previous = _select_previous_filing(current, comparable_filings)
    _validate_same_company(current, previous)
    _require_xbrl(current, label="current filing")
    _require_xbrl(previous, label="previous filing")

    narrative = _extract_narrative(current)
    destination = _materialize_narrative(
        narrative,
        output_dir=(
            DEFAULT_SEC_INGESTION_DIR
            if output_dir is None
            else Path(output_dir)
        ),
        ticker=normalized_ticker,
        form=_filing_form(current),
        filing_date=_filing_date(current),
        accession=_filing_accession(current),
    )

    return SECAnalysisInputs(
        ticker=normalized_ticker,
        company=_company_name(company, current),
        filing_date=_filing_date(current),
        report_period=_report_date(current).isoformat(),
        filing_type=_filing_form(current),
        current_filing=current,
        previous_filing=previous,
        filing_path=destination,
        current_accession=_filing_accession(current),
        previous_accession=_filing_accession(previous),
    )


def discover_sec_filings(
    *,
    ticker: str,
    form: FilingType = "10-Q",
    limit: int = 10,
    company_factory: SECCompanyFactory = Company,
) -> list[SECFilingMetadata]:
    """Return recent comparable filing metadata without loading XBRL or text.

    One deterministic filing is returned per report date. When an original
    and amendment cover the same report period, the amendment and then the
    latest filing date/accession win, matching analysis ingestion policy.
    """

    normalized_ticker = normalize_sec_ticker(ticker)
    _validate_form(form)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise SECInputError("limit must be a positive integer")
    _validate_identity_for_provider(company_factory)

    try:
        company = company_factory(normalized_ticker)
        _validate_company_ticker(company, normalized_ticker)
        candidates = _company_filings(
            company,
            form=_base_form(form),
            amendments=True,
            trigger_full_load=False,
        )
    except SECIngestionError:
        raise
    except IdentityNotSetException as error:
        raise _identity_error() from error
    except Exception as error:
        raise SECServiceError(f"SEC filing discovery failed: {error}") from error

    by_report_date: dict[date, SECFilingLike] = {}
    for filing in candidates:
        if _base_form(_filing_form(filing)) != _base_form(form):
            continue
        report_date = _report_date(filing)
        existing = by_report_date.get(report_date)
        if existing is None or _filing_tie_key(filing) > _filing_tie_key(existing):
            by_report_date[report_date] = filing

    selected = sorted(
        by_report_date.values(),
        key=lambda filing: (
            _report_date(filing),
            _filing_date(filing),
            _filing_accession(filing),
        ),
        reverse=True,
    )[:limit]
    company_name = _company_name(company, selected[0]) if selected else getattr(
        company, "name", normalized_ticker
    )
    return [
        SECFilingMetadata(
            ticker=normalized_ticker,
            company=str(company_name).strip(),
            filing_date=_filing_date(filing),
            report_date=_report_date(filing),
            form=_filing_form(filing),
            accession=_filing_accession(filing),
        )
        for filing in selected
    ]


def normalize_sec_ticker(ticker: str) -> str:
    """Normalize and validate the ticker syntax accepted by SEC ingestion."""

    if not isinstance(ticker, str):
        raise SECInputError("ticker must be a string")
    normalized = ticker.strip().upper()
    if not _TICKER_RE.fullmatch(normalized):
        raise SECInputError(f"invalid ticker: {ticker!r}")
    return normalized


def _normalize_accession(accession: str | None) -> str | None:
    if accession is None:
        return None
    normalized = accession.strip()
    if not _ACCESSION_RE.fullmatch(normalized):
        raise SECInputError(f"invalid accession: {accession!r}")
    return normalized


def _validate_target_selector(
    accession: str | None,
    filing_date: date | None,
) -> None:
    if accession is None and filing_date is None:
        raise SECInputError("accession or filing_date is required")
    if filing_date is not None and not isinstance(filing_date, date):
        raise SECInputError("filing_date must be a date")


def _validate_form(form: str) -> None:
    if form not in _SUPPORTED_FORMS:
        raise SECInputError(f"unsupported SEC form: {form!r}")


def _validate_identity_for_provider(company_factory: SECCompanyFactory) -> None:
    if company_factory is Company and not os.environ.get(SEC_IDENTITY_ENV_VAR):
        raise _identity_error()


def _identity_error() -> SECIdentityError:
    return SECIdentityError(
        "edgartools requires EDGAR_IDENTITY in the format "
        "'Name email@domain.com' for the SEC User-Agent"
    )


def _validate_company_ticker(company: SECCompanyLike, ticker: str) -> None:
    raw_tickers = getattr(company, "tickers", ()) or ()
    known = {str(value).strip().upper() for value in raw_tickers}
    if known and ticker not in known:
        raise SECInputError(
            f"SEC company tickers {sorted(known)!r} do not include {ticker!r}"
        )


def _select_current_filing(
    company: SECCompanyLike,
    *,
    accession: str | None,
    filing_date: date | None,
    requested_form: FilingType,
) -> SECFilingLike:
    if accession is not None:
        candidates = _company_filings(
            company,
            accession_number=accession,
            trigger_full_load=False,
        )
        exact = [
            filing
            for filing in candidates
            if _filing_accession(filing) == accession
        ]
        if not exact:
            raise SECFilingNotFoundError(
                f"SEC filing accession {accession!r} was not found"
            )
        return max(exact, key=_filing_tie_key)

    candidates = _company_filings(
        company,
        form=_base_form(requested_form),
        amendments=True,
        filing_date=filing_date.isoformat() if filing_date else None,
        trigger_full_load=False,
    )
    exact_date = [
        filing for filing in candidates if _filing_date(filing) == filing_date
    ]
    if not exact_date:
        raise SECFilingNotFoundError(
            f"no {requested_form} filing found on {filing_date}"
        )
    return max(exact_date, key=_filing_tie_key)


def _company_filings(
    company: SECCompanyLike,
    **filters: Any,
) -> tuple[SECFilingLike, ...]:
    result = company.get_filings(**filters)
    if result is None:
        return ()
    try:
        return tuple(result)
    except TypeError as error:
        raise SECServiceError(
            "edgartools returned a non-iterable filings collection"
        ) from error


def _validate_current_filing(
    filing: SECFilingLike,
    *,
    ticker: str,
    requested_accession: str | None,
    requested_date: date | None,
    requested_form: FilingType,
) -> None:
    actual_accession = _filing_accession(filing)
    if requested_accession is not None and actual_accession != requested_accession:
        raise SECIngestionError(
            f"selected accession {actual_accession!r} does not match "
            f"requested accession {requested_accession!r}"
        )
    actual_date = _filing_date(filing)
    if requested_date is not None and actual_date != requested_date:
        raise SECIngestionError(
            f"selected filing date {actual_date} does not match requested "
            f"date {requested_date}"
        )
    actual_form = _filing_form(filing)
    if requested_form.endswith("/A"):
        compatible = actual_form == requested_form
    else:
        compatible = _base_form(actual_form) == requested_form
    if not compatible:
        raise SECIngestionError(
            f"selected filing form {actual_form!r} does not match requested "
            f"form {requested_form!r}"
        )

    filing_ticker = getattr(filing, "ticker", None)
    if filing_ticker is not None and str(filing_ticker).strip().upper() != ticker:
        raise SECIngestionError(
            f"selected filing ticker {filing_ticker!r} does not match {ticker!r}"
        )


def _select_previous_filing(
    current: SECFilingLike,
    filings: Iterable[SECFilingLike],
) -> SECFilingLike:
    current_report_date = _report_date(current)
    base_form = _base_form(_filing_form(current))
    eligible = [
        filing
        for filing in filings
        if _base_form(_filing_form(filing)) == base_form
        and _report_date(filing) < current_report_date
    ]
    if not eligible:
        raise SECPreviousFilingNotFoundError(
            f"no previous {base_form} filing exists before report date "
            f"{current_report_date}"
        )

    previous_report_date = max(_report_date(filing) for filing in eligible)
    same_period = [
        filing
        for filing in eligible
        if _report_date(filing) == previous_report_date
    ]
    return max(same_period, key=_filing_tie_key)


def _filing_tie_key(filing: SECFilingLike) -> tuple[bool, date, str]:
    return (
        _filing_form(filing).endswith("/A"),
        _filing_date(filing),
        _filing_accession(filing),
    )


def _validate_same_company(
    current: SECFilingLike,
    previous: SECFilingLike,
) -> None:
    current_cik = getattr(current, "cik", None)
    previous_cik = getattr(previous, "cik", None)
    if current_cik is not None and previous_cik is not None:
        if int(current_cik) != int(previous_cik):
            raise SECIngestionError(
                "current and previous filings belong to different CIKs"
            )


def _require_xbrl(filing: SECFilingLike, *, label: str) -> None:
    try:
        xbrl = filing.xbrl()
    except IdentityNotSetException as error:
        raise _identity_error() from error
    except Exception as error:
        raise SECServiceError(
            f"could not load XBRL for {label} "
            f"{_filing_accession(filing)!r}: {error}"
        ) from error
    if xbrl is None:
        raise SECXBRLUnavailableError(
            f"{label} {_filing_accession(filing)!r} has no XBRL data"
        )


def _extract_narrative(filing: SECFilingLike) -> str:
    primary_error: Exception | None = None
    try:
        narrative = filing.text()
        if isinstance(narrative, str) and narrative.strip():
            return _normalize_narrative_text(narrative)
        primary_error = ValueError("filing.text() returned empty narrative text")
    except Exception as error:
        primary_error = error

    try:
        fallback = _extract_document_narrative(filing)
        return _normalize_narrative_text(fallback)
    except Exception as fallback_error:
        if isinstance(primary_error, IdentityNotSetException) or isinstance(
            fallback_error, IdentityNotSetException
        ):
            raise _identity_error() from fallback_error
        primary_detail = (
            f"{type(primary_error).__name__}: {primary_error}"
            if primary_error is not None
            else "unknown primary extraction failure"
        )
        raise SECNarrativeExtractionError(
            f"could not extract narrative for {_filing_accession(filing)!r}; "
            f"filing.text failed ({primary_detail}); SGML document fallback "
            f"failed ({type(fallback_error).__name__}: {fallback_error})"
        ) from fallback_error


def _extract_document_narrative(filing: SECFilingLike) -> str:
    """Render the SGML primary attachment when ``EntityFiling.text`` fails."""

    document = getattr(filing, "document", None)
    if document is None:
        raise ValueError("filing.document is unavailable")
    content = getattr(document, "content", None)
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            content = content.decode("cp1252")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("primary SGML attachment has no textual content")

    if not is_probably_html(content):
        return content

    parser = HTMLParser(ParserConfig(form=_filing_form(filing)))
    parsed_document = parser.parse(content)
    if parsed_document.is_empty:
        raise ValueError("primary SGML attachment produced an empty document")
    rendered = rich_to_text(parsed_document, width=500)
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("primary SGML attachment rendered as empty text")
    return rendered


def _normalize_narrative_text(text: str) -> str:
    """Normalize transport whitespace without collapsing paragraph structure."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        raise ValueError("narrative contains only whitespace")
    return normalized + "\n"


def _materialize_narrative(
    narrative: str,
    *,
    output_dir: Path,
    ticker: str,
    form: FilingType,
    filing_date: date,
    accession: str,
) -> Path:
    if output_dir.exists() and not output_dir.is_dir():
        raise SECNarrativeExtractionError(
            f"SEC ingestion output path is not a directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_form = form.replace("/", "-")
    filename = (
        f"{ticker}_{safe_form}_{filing_date.isoformat()}_{accession}.txt"
    )
    path = output_dir / filename

    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(narrative)
    except FileExistsError:
        if not path.is_file():
            raise SECNarrativeExtractionError(
                f"refusing to replace non-file destination: {path}"
            ) from None
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise SECNarrativeExtractionError(
                f"could not validate existing narrative file {path}: {error}"
            ) from error
        if existing != narrative:
            raise SECNarrativeExtractionError(
                f"refusing to overwrite different narrative content: {path}"
            ) from None
    except OSError as error:
        raise SECNarrativeExtractionError(
            f"could not materialize narrative at {path}: {error}"
        ) from error
    return path


def _company_name(
    company: SECCompanyLike,
    filing: SECFilingLike,
) -> str:
    for value in (getattr(company, "name", None), getattr(filing, "company", None)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise SECIngestionError("SEC company name is missing")


def _filing_accession(filing: SECFilingLike) -> str:
    value = getattr(filing, "accession_no", None)
    if value is None:
        value = getattr(filing, "accession_number", None)
    if not isinstance(value, str) or not _ACCESSION_RE.fullmatch(value.strip()):
        raise SECIngestionError("selected filing has an invalid accession")
    return value.strip()


def _filing_form(filing: SECFilingLike) -> FilingType:
    value = getattr(filing, "form", None)
    if value not in _SUPPORTED_FORMS:
        raise SECIngestionError(f"selected filing has unsupported form {value!r}")
    return value


def _filing_date(filing: SECFilingLike) -> date:
    return _coerce_date(getattr(filing, "filing_date", None), "filing_date")


def _report_date(filing: SECFilingLike) -> date:
    return _coerce_date(getattr(filing, "report_date", None), "report_date")


def _coerce_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise SECIngestionError(
                f"selected filing has invalid {field}: {value!r}"
            ) from error
    raise SECIngestionError(f"selected filing is missing {field}")


def _base_form(form: str) -> str:
    return form.removesuffix("/A")


__all__ = [
    "DEFAULT_SEC_INGESTION_DIR",
    "SECAnalysisInputs",
    "SECCompanyFactory",
    "SECFilingMetadata",
    "SECFilingNotFoundError",
    "SECIdentityError",
    "SECIngestionError",
    "SECInputError",
    "SECNarrativeExtractionError",
    "SECPreviousFilingNotFoundError",
    "SECServiceError",
    "SECXBRLUnavailableError",
    "SEC_IDENTITY_ENV_VAR",
    "discover_sec_filings",
    "normalize_sec_ticker",
    "prepare_sec_analysis_inputs",
]
