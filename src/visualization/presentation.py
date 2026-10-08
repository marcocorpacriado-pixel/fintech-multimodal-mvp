"""Pure presentation helpers shared by the Dash dashboard and its tests.

These helpers format API data only. They never query SEC, call an LLM, or
recalculate financial values.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping, Sequence


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,14}$")
_WORD_RE = re.compile(r"(\w+)")
# Polarity colour (DESIGN.md positive-growth, crimson, accent-blue) and pill text.
_POLARITY_STYLE = {
    "positive": ("16, 185, 129", "#E6FFFA"),
    "negative": ("239, 68, 68", "#FFF5F5"),
    "neutral": ("56, 189, 248", "#F0F9FF"),
}
_UNSCORED_STYLE = ("148, 163, 184", "#F8FAFC")
_MUTED = "color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.04em;"
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


_CHAT_CITATION_RE = re.compile(r"\[([MPROS]\d+)\]")
_CHAT_STREAM_MARKER = "[stream interrupted]"


def chat_citation_labels(handoff: Mapping[str, Any]) -> dict[str, str]:
    """Citation tag → label, in the same order the chat API tags its context.

    Mirrors ``src.integration.chat_context.citation_index`` (the UI may not
    import integration code); a test keeps both in sync.
    """

    labels: dict[str, str] = {}
    for index, metric in enumerate(handoff.get("financial_metrics") or [], start=1):
        labels[f"M{index}"] = f"Metric: {metric.get('name')}"
    for index, item in enumerate(handoff.get("positives") or [], start=1):
        labels[f"P{index}"] = f"Positive: {item.get('finding')}"
    for index, item in enumerate(handoff.get("risks") or [], start=1):
        labels[f"R{index}"] = f"Risk: {item.get('finding')}"
    labels["O1"] = "Management outlook"
    labels["S1"] = "Executive summary"
    return labels


def cited_sources(answer: str, handoff: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Known citation tags used in an answer, in first-appearance order."""

    labels = chat_citation_labels(handoff)
    seen: dict[str, str] = {}
    for tag in _CHAT_CITATION_RE.findall(answer):
        if tag in labels and tag not in seen:
            seen[tag] = labels[tag]
    return list(seen.items())


def text_for_speech(answer: str) -> str:
    """Drop citation tags and stream markers before sending text to TTS."""

    text = answer.replace(_CHAT_STREAM_MARKER, "")
    text = re.sub(r"\s*\[[MPROS]\d+\]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def sentiment_label(outlook: Mapping[str, Any]) -> str:
    """Sentiment badge text, with the scoring model's confidence when provided."""

    label = str(outlook.get("sentiment") or "unknown").upper()
    confidence = outlook.get("confidence")
    if confidence is None:
        return label
    model = str(outlook.get("model") or "")
    model_name = "FinBERT" if "finbert" in model.lower() else model or "model"
    return f"{label} · {confidence * 100:.1f}% confidence · {model_name}"


def highlight_tokens_html(
    sentence: str,
    attributions: Sequence[Mapping[str, Any]],
    sentiment: str,
) -> str:
    """Inline HTML of ``sentence`` with attributed words as score-shaded pills.

    Attributions come from an uncased tokenizer, so words match case-insensitively.
    Every piece of filing text is HTML-escaped; only the spans are markup.
    """

    scores: dict[str, float] = {}
    for item in attributions:
        token = str(item.get("token") or "").lower()
        score = float(item.get("score") or 0.0)
        scores[token] = max(score, scores.get(token, 0.0))
    rgb, text_color = _POLARITY_STYLE.get(sentiment, _UNSCORED_STYLE)

    parts = []
    for index, piece in enumerate(_WORD_RE.split(sentence)):
        text = html.escape(piece)
        score = scores.get(piece.lower(), 0.0) if index % 2 else 0.0
        if score > 0:
            # Alpha caps at 0.50 so the pale text keeps >= 4.5:1 (WCAG AA, DESIGN.md).
            # data-tooltip feeds the instant CSS tooltip (.xai-pill in the app CSS);
            # title is the native, accessible fallback; tabindex enables keyboard focus.
            impact = f"Impact: {score * 100:.1f}%"
            parts.append(
                f'<span class="xai-pill" tabindex="0" data-tooltip="{impact}" '
                f'title="{impact} (Integrated Gradients)" '
                f"style=\"background: rgba({rgb}, {0.15 + score * 0.35:.2f}); "
                f"border: 1px solid rgba({rgb}, {0.3 + score * 0.4:.2f}); "
                f"color: {text_color}; border-radius: 4px; padding: 2px 6px; "
                f"margin: 0 2px; display: inline-block; position: relative; "
                f'line-height: 1.4; cursor: help;">{text}</span>'
            )
        else:
            parts.append(text)
    return "".join(parts)


def outlook_xai_html(outlook: Mapping[str, Any]) -> str:
    """FinBERT outlook block: polarity header, summary, cited rationale heatmap, legend.

    Built for ``st.html``; all API text is HTML-escaped.
    """

    sentiment = str(outlook.get("sentiment") or "unknown")
    rgb, _ = _POLARITY_STYLE.get(sentiment, _UNSCORED_STYLE)
    facts = []
    if outlook.get("confidence") is not None:
        facts.append(f"Confidence: {outlook['confidence'] * 100:.1f}%")
    if outlook.get("model"):
        facts.append(f"Model: {html.escape(str(outlook['model']))}")
    blocks = [
        '<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; '
        'margin-bottom: 12px;">'
        f'<span style="background: rgba({rgb}, 0.10); border: 1px solid rgba({rgb}, 0.6); '
        f"color: rgb({rgb}); border-radius: 999px; padding: 3px 12px; font-size: 12px; "
        'font-weight: 700; letter-spacing: 0.04em;">'
        f"&#9679; POLARITY: {html.escape(sentiment.upper())}</span>"
        f'<span style="{_MUTED}">{" · ".join(facts)}</span></div>',
        '<p style="color: #F8FAFC; line-height: 1.6; margin: 0 0 14px;">'
        f"{html.escape(str(outlook.get('summary') or ''))}</p>",
    ]

    sentence = outlook.get("rationale_sentence")
    if sentence:
        attributions = outlook.get("token_attributions") or []
        score = outlook.get("rationale_score")
        score_text = "" if score is None else f" · FinBERT {score * 100:.1f}%"
        blocks.append(
            f'<figure style="margin: 0; padding: 14px 18px; background: #1E293B; '
            f'border: 1px solid #334155; border-left: 3px solid rgb({rgb}); '
            'border-radius: 8px;">'
            f'<figcaption style="{_MUTED} text-transform: uppercase; margin-bottom: 8px;">'
            f"Key evidence detected{score_text}</figcaption>"
            '<blockquote style="margin: 0; color: #F8FAFC; font-size: 15px; '
            'line-height: 2.1;">'
            f'<span style="color: rgb({rgb}); font-size: 22px; line-height: 0;">&ldquo;</span>'
            f"{highlight_tokens_html(sentence, attributions, sentiment)}"
            f'<span style="color: rgb({rgb}); font-size: 22px; line-height: 0;">&rdquo;</span>'
            "</blockquote></figure>"
        )
        if attributions:
            blocks.append(
                f'<div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; '
                f'margin-top: 10px; {_MUTED}">'
                "Low impact"
                f'<span style="display: inline-block; width: 84px; height: 8px; '
                f"border-radius: 4px; background: linear-gradient(90deg, "
                f'rgba({rgb}, 0.15), rgba({rgb}, 0.5));"></span>'
                "High impact · Integrated Gradients attribution · hover a word for its weight"
                "</div>"
            )
    return "".join(blocks)


def human_source_label(filing_type: str, source_section: str | None) -> str:
    """Present a canonical SEC section before its technical identifier."""

    section = source_section or "UNSECTIONED"
    readable = _SECTION_LABELS.get(section, section.replace("_", " ").title())
    return f"{filing_type} · {readable}"


__all__ = [
    "ErrorPresentation",
    "chat_citation_labels",
    "cited_sources",
    "error_presentation",
    "filing_option_label",
    "format_display_date",
    "format_metric_period",
    "highlight_tokens_html",
    "human_source_label",
    "outlook_xai_html",
    "normalize_ticker_for_ui",
    "select_executive_metrics",
    "sentiment_label",
    "sort_filings",
    "text_for_speech",
    "verification_label",
]
