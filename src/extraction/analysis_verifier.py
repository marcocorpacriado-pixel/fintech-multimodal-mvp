"""Deterministic verification for grounded financial analysis results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .schemas import (
    Evidence,
    FinancialAnalysisResult,
    FinancialMetric,
    RetrievalResult,
    VerificationCode,
    VerificationIssue,
    VerificationReport,
)


# Half-unit tolerance for one displayed decimal. Percentage matching scales
# this value by the actual precision of each mention rather than applying it
# as a global threshold.
PERCENTAGE_TOLERANCE_POINTS = 0.05
SUMMARY_MIN_WORDS = 80
SUMMARY_PREFERRED_MAX_WORDS = 200
SUMMARY_HARD_MAX_WORDS = 220


class AnalysisVerificationError(ValueError):
    """Raised by strict mode when deterministic verification finds errors."""

    def __init__(self, report: VerificationReport) -> None:
        self.report = report
        error_codes = [
            issue.code for issue in report.issues if issue.severity == "error"
        ]
        super().__init__(
            "analysis verification failed: " + ", ".join(error_codes)
        )


@dataclass(frozen=True)
class _NumericMention:
    raw: str
    value: float
    is_percentage: bool
    rounding_tolerance: float
    has_currency: bool
    has_magnitude: bool
    explicit_sign: str | None
    start: int
    end: int


@dataclass(frozen=True)
class _TextTarget:
    text: str
    field: str
    source_id: str | None = None
    grounding_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AliasHit:
    metric: FinancialMetric
    start: int
    end: int


_NUMBER_PATTERN = re.compile(
    r"""
    (?<![\w\-\u2013\u2014])
    (?P<currency>\$|USD\s*)?
    (?P<sign>[+-])?
    (?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    (?:\s*(?P<magnitude>billion|million|thousand|[KMB]))?
    (?:\s*(?P<percent>%|percent(?:age)?(?:\s+points?)?))?
    (?![\w\-\u2013\u2014])
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DIRECTION_PATTERN = re.compile(
    r"\b(?P<negative>declin(?:e|ed|ing)|decreas(?:e|ed|ing)|fell|fallen|"
    r"drop(?:ped|ping)?|down|contract(?:ed|ing)?)\b|"
    r"\b(?P<positive>increas(?:e|ed|ing)|grew|grown|rose|risen|up|"
    r"improv(?:e|ed|ing))\b",
    re.IGNORECASE,
)

_MONTH_NAME_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept(?:ember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_TEXTUAL_DATE_PATTERN = re.compile(
    rf"""
    \b(?:
        {_MONTH_NAME_PATTERN}\s+(?:0?[1-9]|[12]\d|3[01])
        (?:st|nd|rd|th)?(?:,\s*|\s+)(?:19|20)\d{{2}}
        |
        (?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+
        {_MONTH_NAME_PATTERN},?\s+(?:19|20)\d{{2}}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ISO_DATE_PATTERN = re.compile(
    r"(?<!\d)(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)"
)

_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:strong\s+)?(?:buy|sell|hold)\b|"
    r"\b(?:overweight|underweight)\b|"
    r"\b(?:price\s+target|target\s+price)\b|"
    r"\bshould\s+(?:invest|buy|sell|hold)\b|"
    r"\brecommend(?:s|ed|ing|ation)?\s+(?:to\s+)?(?:buy|sell|hold|buying|selling)\b",
    re.IGNORECASE,
)

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales"),
    "net income": ("net income", "net loss", "profit"),
    "diluted eps": ("diluted eps", "eps", "earnings per share"),
    "cash and cash equivalents": ("cash and cash equivalents", "cash"),
    "total debt": ("total debt", "debt"),
    "operating cash flow": (
        "operating cash flow",
        "cash flow from operations",
        "ocf",
    ),
    "capital expenditures": (
        "capital expenditures",
        "capital expenditure",
        "capital spending",
        "capex",
    ),
}


def verify_analysis(
    analysis: FinancialAnalysisResult,
    *,
    canonical_metrics: Sequence[FinancialMetric],
    retrieval_results: Sequence[RetrievalResult],
    expected_company: str | None = None,
    expected_ticker: str | None = None,
    expected_period: str | None = None,
) -> VerificationReport:
    """Return all deterministic errors and warnings without changing analysis."""

    metrics = tuple(canonical_metrics)
    retrieval = tuple(retrieval_results)
    issues: list[VerificationIssue] = []
    _verify_expected_metadata(
        analysis,
        issues,
        expected_company=expected_company,
        expected_ticker=expected_ticker,
        expected_period=expected_period,
    )
    _verify_metrics(analysis, metrics, issues)
    catalog = _build_source_catalog(retrieval, issues)
    _verify_findings(analysis, catalog, issues)
    _verify_outlook(analysis, catalog, issues)
    _verify_narrative_numbers(analysis, metrics, catalog, issues)
    _verify_recommendations(analysis, issues)
    _verify_summary_length(analysis.executive_summary, issues)
    valid = not any(issue.severity == "error" for issue in issues)
    return VerificationReport(valid=valid, issues=issues)


def assert_verified_analysis(
    analysis: FinancialAnalysisResult,
    *,
    canonical_metrics: Sequence[FinancialMetric],
    retrieval_results: Sequence[RetrievalResult],
    expected_company: str | None = None,
    expected_ticker: str | None = None,
    expected_period: str | None = None,
) -> VerificationReport:
    """Return the report or raise when at least one error is present."""

    report = verify_analysis(
        analysis,
        canonical_metrics=canonical_metrics,
        retrieval_results=retrieval_results,
        expected_company=expected_company,
        expected_ticker=expected_ticker,
        expected_period=expected_period,
    )
    if not report.valid:
        raise AnalysisVerificationError(report)
    return report


def _verify_expected_metadata(
    analysis: FinancialAnalysisResult,
    issues: list[VerificationIssue],
    *,
    expected_company: str | None,
    expected_ticker: str | None,
    expected_period: str | None,
) -> None:
    expected_fields = (
        ("company", analysis.company, expected_company, "COMPANY_MISMATCH"),
        ("ticker", analysis.ticker, expected_ticker, "TICKER_MISMATCH"),
        ("period", analysis.period, expected_period, "PERIOD_MISMATCH"),
    )
    for field, actual, expected, code in expected_fields:
        if expected is not None and _normalized_identity(actual) != (
            _normalized_identity(expected)
        ):
            issues.append(
                _issue(
                    code=code,  # type: ignore[arg-type]
                    message=f"{field} does not match the expected value",
                    field=field,
                )
            )


def _verify_metrics(
    analysis: FinancialAnalysisResult,
    canonical_metrics: Sequence[FinancialMetric],
    issues: list[VerificationIssue],
) -> None:
    actual = [metric.model_dump(mode="json") for metric in analysis.financial_metrics]
    expected = [metric.model_dump(mode="json") for metric in canonical_metrics]
    if actual != expected:
        issues.append(
            _issue(
                code="METRICS_MISMATCH",
                message=(
                    "financial_metrics differ from canonical metrics in content "
                    "or order"
                ),
                field="financial_metrics",
            )
        )


def _build_source_catalog(
    retrieval_results: Sequence[RetrievalResult],
    issues: list[VerificationIssue],
) -> dict[str, RetrievalResult]:
    catalog: dict[str, RetrievalResult] = {}
    for index, result in enumerate(retrieval_results):
        source_id = result.chunk.chunk_id
        if source_id in catalog:
            issues.append(
                _issue(
                    code="INVALID_SOURCE_ID",
                    message=f"duplicate retrieval chunk_id {source_id!r}",
                    field=f"retrieval_results[{index}].chunk.chunk_id",
                    source_id=source_id,
                )
            )
            continue
        catalog[source_id] = result
    return catalog


def _verify_findings(
    analysis: FinancialAnalysisResult,
    catalog: dict[str, RetrievalResult],
    issues: list[VerificationIssue],
) -> None:
    collections = (
        ("key_positive_developments", analysis.key_positive_developments),
        ("key_risks", analysis.key_risks),
    )
    for collection_name, findings in collections:
        for index, finding in enumerate(findings):
            _verify_evidence(
                finding,
                field=f"{collection_name}[{index}]",
                analysis_ticker=analysis.ticker,
                catalog=catalog,
                issues=issues,
            )


def _verify_evidence(
    evidence: Evidence,
    *,
    field: str,
    analysis_ticker: str,
    catalog: dict[str, RetrievalResult],
    issues: list[VerificationIssue],
) -> None:
    if not evidence.source_id or not evidence.source_section or not evidence.evidence:
        issues.append(
            _issue(
                code="UNGROUNDED_CLAIM",
                message="finding requires evidence, source_id, and source_section",
                field=field,
                source_id=evidence.source_id,
            )
        )
        return

    result = catalog.get(evidence.source_id)
    if result is None:
        issues.append(
            _issue(
                code="INVALID_SOURCE_ID",
                message=f"source_id {evidence.source_id!r} was not retrieved",
                field=f"{field}.source_id",
                source_id=evidence.source_id,
            )
        )
        return

    chunk = result.chunk
    if _normalized_identity(chunk.ticker) != _normalized_identity(analysis_ticker):
        issues.append(
            _issue(
                code="SOURCE_TICKER_MISMATCH",
                message="cited chunk ticker differs from analysis ticker",
                field=f"{field}.source_id",
                source_id=evidence.source_id,
            )
        )
    expected_section = chunk.section or "UNSECTIONED"
    if evidence.source_section != expected_section:
        issues.append(
            _issue(
                code="SECTION_MISMATCH",
                message=(
                    f"source section {evidence.source_section!r} does not match "
                    f"{expected_section!r}"
                ),
                field=f"{field}.source_section",
                source_id=evidence.source_id,
            )
        )
    if _normalized_text(evidence.evidence) not in _normalized_text(chunk.text):
        issues.append(
            _issue(
                code="EVIDENCE_NOT_IN_SOURCE",
                message="evidence text is not present in the cited chunk",
                field=f"{field}.evidence",
                source_id=evidence.source_id,
            )
        )


def _verify_outlook(
    analysis: FinancialAnalysisResult,
    catalog: dict[str, RetrievalResult],
    issues: list[VerificationIssue],
) -> None:
    outlook = analysis.management_outlook
    if outlook.sentiment != "unknown" and not outlook.source_ids:
        issues.append(
            _issue(
                code="OUTLOOK_WITHOUT_EVIDENCE",
                message="known management outlook requires at least one source_id",
                field="management_outlook.source_ids",
            )
        )
    for index, source_id in enumerate(outlook.source_ids):
        result = catalog.get(source_id)
        field = f"management_outlook.source_ids[{index}]"
        if result is None:
            issues.append(
                _issue(
                    code="INVALID_SOURCE_ID",
                    message=f"outlook source_id {source_id!r} was not retrieved",
                    field=field,
                    source_id=source_id,
                )
            )
        elif _normalized_identity(result.chunk.ticker) != _normalized_identity(
            analysis.ticker
        ):
            issues.append(
                _issue(
                    code="SOURCE_TICKER_MISMATCH",
                    message="outlook chunk ticker differs from analysis ticker",
                    field=field,
                    source_id=source_id,
                )
            )


def _verify_narrative_numbers(
    analysis: FinancialAnalysisResult,
    metrics: Sequence[FinancialMetric],
    catalog: dict[str, RetrievalResult],
    issues: list[VerificationIssue],
) -> None:
    targets = _narrative_targets(analysis, catalog)
    for target in targets:
        date_spans = _nonfinancial_date_spans(target.text)
        for mention in _extract_numeric_mentions(target.text):
            metric = _closest_metric(target.text, mention, metrics)
            if _is_nonfinancial_number(
                target.text,
                mention,
                date_spans=date_spans,
            ):
                continue
            if not _has_financial_signal(mention) and metric is None:
                continue
            if _mention_matches_metrics(mention, target.text, metrics, metric):
                continue
            if _mention_matches_grounding(
                mention,
                target.grounding_texts,
            ):
                continue
            context = f" for {metric.name}" if metric is not None else ""
            issues.append(
                _issue(
                    code="UNSUPPORTED_NUMBER",
                    message=(
                        f"financial number {mention.raw!r}{context} cannot be "
                        "reconciled with canonical metrics or cited evidence"
                    ),
                    field=target.field,
                    source_id=target.source_id,
                )
            )


def _narrative_targets(
    analysis: FinancialAnalysisResult,
    catalog: dict[str, RetrievalResult],
) -> list[_TextTarget]:
    outlook_sources = tuple(
        catalog[source_id].chunk.text
        for source_id in analysis.management_outlook.source_ids
        if source_id in catalog
    )
    targets = [
        _TextTarget(
            text=analysis.executive_summary,
            field="executive_summary",
        ),
        _TextTarget(
            text=analysis.management_outlook.summary,
            field="management_outlook.summary",
            grounding_texts=outlook_sources,
        ),
    ]
    for collection_name, findings in (
        ("key_positive_developments", analysis.key_positive_developments),
        ("key_risks", analysis.key_risks),
    ):
        for index, finding in enumerate(findings):
            targets.append(
                _TextTarget(
                    text=finding.finding,
                    field=f"{collection_name}[{index}].finding",
                    source_id=finding.source_id,
                    grounding_texts=(finding.evidence,),
                )
            )
    return targets


def _extract_numeric_mentions(text: str) -> list[_NumericMention]:
    mentions: list[_NumericMention] = []
    scales = {
        None: 1.0,
        "k": 1_000.0,
        "thousand": 1_000.0,
        "m": 1_000_000.0,
        "million": 1_000_000.0,
        "b": 1_000_000_000.0,
        "billion": 1_000_000_000.0,
    }
    for match in _NUMBER_PATTERN.finditer(text):
        number_text = match.group("number").replace(",", "")
        decimals = len(number_text.partition(".")[2])
        magnitude = match.group("magnitude")
        scale = scales[magnitude.casefold() if magnitude else None]
        sign = match.group("sign")
        value = float(number_text) * scale
        if sign == "-":
            value = -value
        is_percentage = match.group("percent") is not None
        rounding_tolerance = (
            PERCENTAGE_TOLERANCE_POINTS * (10 ** (1 - decimals))
            if is_percentage
            else 0.5 * (10**-decimals) * scale
        )
        mentions.append(
            _NumericMention(
                raw=match.group(0).strip(),
                value=value,
                is_percentage=is_percentage,
                rounding_tolerance=rounding_tolerance,
                has_currency=match.group("currency") is not None,
                has_magnitude=magnitude is not None,
                explicit_sign=sign,
                start=match.start(),
                end=match.end(),
            )
        )
    return mentions


def _closest_metric(
    text: str,
    mention: _NumericMention,
    metrics: Sequence[FinancialMetric],
) -> FinancialMetric | None:
    lowered = text.casefold()
    hits: list[_AliasHit] = []
    for metric in metrics:
        aliases = _METRIC_ALIASES.get(
            metric.name.casefold(),
            (metric.name.casefold(),),
        )
        for alias in aliases:
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            for match in pattern.finditer(lowered):
                if _span_distance(match.start(), match.end(), mention) <= 90:
                    hits.append(
                        _AliasHit(
                            metric=metric,
                            start=match.start(),
                            end=match.end(),
                        )
                    )
    longest_hits = [
        hit
        for hit in hits
        if not any(
            other.start <= hit.start
            and other.end >= hit.end
            and (other.end - other.start) > (hit.end - hit.start)
            for other in hits
        )
    ]
    if not longest_hits:
        return None
    closest = min(
        longest_hits,
        key=lambda hit: (
            0 if hit.end <= mention.start else 1,
            _span_distance(hit.start, hit.end, mention),
            -(hit.end - hit.start),
            hit.metric.name,
        ),
    )
    return closest.metric


def _span_distance(start: int, end: int, mention: _NumericMention) -> float:
    return abs(((start + end) / 2) - ((mention.start + mention.end) / 2))


def _mention_matches_metrics(
    mention: _NumericMention,
    text: str,
    metrics: Sequence[FinancialMetric],
    contextual_metric: FinancialMetric | None,
) -> bool:
    candidates = (contextual_metric,) if contextual_metric is not None else metrics
    if mention.is_percentage:
        observed = _signed_percentage(mention, text)
        return any(
            metric.change_pct is not None
            and _within_rounding_precision(
                observed,
                metric.change_pct,
                mention.rounding_tolerance,
            )
            for metric in candidates
        )
    return any(
        value is not None and _value_matches(mention, value)
        for metric in candidates
        for value in (metric.current_value, metric.previous_value)
    )


def _mention_matches_grounding(
    mention: _NumericMention,
    grounding_texts: Iterable[str],
) -> bool:
    for text in grounding_texts:
        date_spans = _nonfinancial_date_spans(text)
        for grounded in _extract_numeric_mentions(text):
            if _is_nonfinancial_number(
                text,
                grounded,
                date_spans=date_spans,
            ):
                continue
            if mention.is_percentage:
                if not (
                    grounded.is_percentage
                    or _has_percentage_context(text, grounded)
                ):
                    continue
                if _same_grounded_value(
                    abs(mention.value),
                    abs(grounded.value),
                ):
                    return True
            elif grounded.is_percentage:
                continue
            elif _same_grounded_value(mention.value, grounded.value):
                return True
    return False


def _within_rounding_precision(
    observed: float,
    canonical: float,
    rounding_tolerance: float,
) -> bool:
    """Match a canonical value at the precision displayed by the model."""

    floating_epsilon = max(1e-12, abs(canonical) * 1e-12)
    return abs(observed - canonical) <= rounding_tolerance + floating_epsilon


def _same_grounded_value(first: float, second: float) -> bool:
    """Compare cited values without allowing human-rounding substitutions."""

    floating_epsilon = max(1e-9, abs(first) * 1e-12, abs(second) * 1e-12)
    return abs(first - second) <= floating_epsilon


def _has_percentage_context(text: str, mention: _NumericMention) -> bool:
    """Recognize percentages whose symbol was lost in a rendered table."""

    local = text[max(0, mention.start - 40) : min(len(text), mention.end + 20)]
    return bool(re.search(r"\bpercent(?:age)?\b", local, re.IGNORECASE))


def _signed_percentage(mention: _NumericMention, text: str) -> float:
    if mention.explicit_sign == "-":
        return -abs(mention.value)
    if mention.explicit_sign == "+":
        return abs(mention.value)
    direction = _nearest_direction(text, mention)
    if direction == "negative":
        return -abs(mention.value)
    return abs(mention.value)


def _nearest_direction(
    text: str,
    mention: _NumericMention,
) -> Literal["positive", "negative"] | None:
    start = max(0, mention.start - 60)
    end = min(len(text), mention.end + 35)
    local = text[start:end]
    matches: list[tuple[float, Literal["positive", "negative"]]] = []
    mention_center = ((mention.start + mention.end) / 2) - start
    for match in _DIRECTION_PATTERN.finditer(local):
        direction: Literal["positive", "negative"] = (
            "negative" if match.group("negative") else "positive"
        )
        distance = abs(((match.start() + match.end()) / 2) - mention_center)
        matches.append((distance, direction))
    return min(matches, default=(0, None), key=lambda item: item[0])[1]


def _value_matches(mention: _NumericMention, canonical: float) -> bool:
    floating_epsilon = max(1e-9, abs(canonical) * 1e-12)
    return abs(mention.value - canonical) <= (
        mention.rounding_tolerance + floating_epsilon
    )


def _has_financial_signal(mention: _NumericMention) -> bool:
    return (
        mention.is_percentage or mention.has_currency or mention.has_magnitude
    )


def _nonfinancial_date_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return complete textual and ISO date spans for numeric exclusion."""

    matches = (
        *(_TEXTUAL_DATE_PATTERN.finditer(text)),
        *(_ISO_DATE_PATTERN.finditer(text)),
    )
    return tuple(sorted((match.start(), match.end()) for match in matches))


def _is_nonfinancial_number(
    text: str,
    mention: _NumericMention,
    *,
    date_spans: Sequence[tuple[int, int]] = (),
) -> bool:
    if any(
        start <= mention.start and mention.end <= end
        for start, end in date_spans
    ):
        return True
    plain_integer = (
        not _has_financial_signal(mention)
        and mention.value.is_integer()
        and mention.explicit_sign is None
    )
    if plain_integer and 1900 <= mention.value <= 2100:
        return True
    prefix = text[max(0, mention.start - 18) : mention.start]
    return bool(re.search(r"\b(?:item|part|quarter|q)\s*$", prefix, re.I))


def _verify_recommendations(
    analysis: FinancialAnalysisResult,
    issues: list[VerificationIssue],
) -> None:
    targets = [
        ("executive_summary", analysis.executive_summary, None),
        (
            "management_outlook.summary",
            analysis.management_outlook.summary,
            None,
        ),
    ]
    for collection_name, findings in (
        ("key_positive_developments", analysis.key_positive_developments),
        ("key_risks", analysis.key_risks),
    ):
        targets.extend(
            (
                f"{collection_name}[{index}].finding",
                finding.finding,
                finding.source_id,
            )
            for index, finding in enumerate(findings)
        )
    for field, text, source_id in targets:
        match = _RECOMMENDATION_PATTERN.search(text)
        if match:
            issues.append(
                _issue(
                    code="INVESTMENT_RECOMMENDATION",
                    message=(
                        f"generated text contains prohibited investment language "
                        f"{match.group(0)!r}"
                    ),
                    field=field,
                    source_id=source_id,
                )
            )


def _verify_summary_length(
    summary: str,
    issues: list[VerificationIssue],
) -> None:
    word_count = len(re.findall(r"\b[\w'’\-]+\b", summary, re.UNICODE))
    if word_count == 0:
        issues.append(
            _issue(
                code="SUMMARY_TOO_SHORT",
                severity="error",
                message="executive summary is empty",
                field="executive_summary",
            )
        )
    elif word_count < SUMMARY_MIN_WORDS:
        issues.append(
            _issue(
                code="SUMMARY_TOO_SHORT",
                severity="warning",
                message=(
                    f"executive summary has {word_count} words; recommended "
                    f"minimum is {SUMMARY_MIN_WORDS}"
                ),
                field="executive_summary",
            )
        )
    elif word_count > SUMMARY_HARD_MAX_WORDS:
        issues.append(
            _issue(
                code="SUMMARY_TOO_LONG",
                severity="error",
                message=(
                    f"executive summary has {word_count} words; hard maximum is "
                    f"{SUMMARY_HARD_MAX_WORDS}"
                ),
                field="executive_summary",
            )
        )
    elif word_count > SUMMARY_PREFERRED_MAX_WORDS:
        issues.append(
            _issue(
                code="SUMMARY_TOO_LONG",
                severity="warning",
                message=(
                    f"executive summary has {word_count} words; preferred maximum "
                    f"is {SUMMARY_PREFERRED_MAX_WORDS}"
                ),
                field="executive_summary",
            )
        )


def _issue(
    *,
    code: VerificationCode,
    message: str,
    field: str,
    severity: Literal["error", "warning"] = "error",
    source_id: str | None = None,
) -> VerificationIssue:
    return VerificationIssue(
        code=code,
        severity=severity,
        message=message,
        field=field,
        source_id=source_id,
    )


def _normalized_text(value: str) -> str:
    """Normalize rendering whitespace while preserving source case/content."""

    return " ".join(value.split())


def _normalized_identity(value: str) -> str:
    return _normalized_text(value)


__all__ = [
    "AnalysisVerificationError",
    "PERCENTAGE_TOLERANCE_POINTS",
    "SUMMARY_HARD_MAX_WORDS",
    "SUMMARY_MIN_WORDS",
    "SUMMARY_PREFERRED_MAX_WORDS",
    "assert_verified_analysis",
    "verify_analysis",
]
