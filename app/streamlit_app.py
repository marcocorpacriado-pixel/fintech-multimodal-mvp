"""Institutional Streamlit presentation for the FastAPI analysis contract.

The UI communicates with FastAPI over HTTP only. It never imports extraction,
integration, SEC, XBRL, LLM, or audio implementation modules, and it never
recalculates financial metrics.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` only puts app/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.financial_charts import (  # noqa: E402
    NOT_AVAILABLE,
    format_change_pct,
    format_value,
    render_metrics_change_chart,
    render_metrics_comparison_chart,
)
from src.visualization.presentation import (  # noqa: E402
    error_presentation,
    filing_option_label,
    format_display_date,
    format_metric_period,
    human_source_label,
    normalize_ticker_for_ui,
    select_executive_metrics,
    sort_filings,
    verification_label,
)


API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_VOICES = ["af_heart"]
SENTIMENT_COLORS = {
    "positive": "green",
    "negative": "red",
    "mixed": "orange",
    "neutral": "gray",
    "unknown": "gray",
}


def md_escape(text: str) -> str:
    """Prevent untrusted filing text from becoming Markdown or LaTeX."""

    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<])", r"\\\1", str(text))


def parse_api_error(response: httpx.Response) -> dict[str, Any]:
    """Convert an API failure into the small safe shape needed by the UI."""

    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return {
            "code": detail["code"],
            "message": str(detail.get("message") or ""),
            "retryable": bool(detail.get("retryable")),
            "status_code": response.status_code,
        }
    return {
        "code": "INPUT_ERROR" if response.status_code == 422 else "UNKNOWN_ERROR",
        "message": "",
        "retryable": response.status_code >= 500,
        "status_code": response.status_code,
    }


def api_request(
    method: str,
    path: str,
    **kwargs: Any,
) -> tuple[httpx.Response | None, dict[str, Any] | None]:
    """Make one backend request and return data or a safe presentation error."""

    try:
        response = httpx.request(method, f"{API_URL}{path}", **kwargs)
    except httpx.HTTPError:
        return None, {
            "code": "API_UNAVAILABLE",
            "message": "",
            "retryable": True,
            "status_code": None,
        }
    if response.is_success:
        return response, None
    return None, parse_api_error(response)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_voices() -> list[str]:
    try:
        response = httpx.request("GET", f"{API_URL}/api/v1/audio/voices", timeout=30)
        return response.json()["voices"] if response.is_success else DEFAULT_VOICES
    except (httpx.HTTPError, ValueError, KeyError):
        return DEFAULT_VOICES


@st.cache_data(ttl=60, show_spinner=False)
def fetch_filings(
    ticker: str,
    filing_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fetch lightweight filing metadata; this never runs XBRL or an LLM."""

    response, error = api_request(
        "GET",
        f"/api/v1/filings/{ticker}",
        params={"filing_type": filing_type, "limit": 10},
        timeout=45,
    )
    if response is None:
        return [], error
    try:
        payload = response.json()
    except ValueError:
        return [], {
            "code": "UNKNOWN_ERROR",
            "message": "",
            "retryable": True,
            "status_code": response.status_code,
        }
    if not isinstance(payload, list):
        return [], {
            "code": "UNKNOWN_ERROR",
            "message": "",
            "retryable": True,
            "status_code": response.status_code,
        }
    return sort_filings(payload), None


def render_error(error: dict[str, Any], *, allow_retry: bool = True) -> bool:
    """Render safe error guidance and return whether retry was requested."""

    code = str(error.get("code") or "UNKNOWN_ERROR")
    copy = error_presentation(code)
    st.error(f"{copy.title} ({code})")
    st.caption(copy.guidance)
    if error.get("retryable") and allow_retry:
        st.info("The selected configuration is preserved. Retry when ready.")
        return st.button("Retry analysis", type="primary", key="retry_analysis")
    return False


def _render_filing_selector(
    ticker: str,
    filing_type: str,
) -> dict[str, Any] | None:
    with st.spinner("Loading available SEC filings"):
        filings, error = fetch_filings(ticker, filing_type)
    if error is not None:
        copy = error_presentation(str(error.get("code") or "UNKNOWN_ERROR"))
        st.error(copy.title)
        st.caption(copy.guidance)
        if error.get("retryable") and st.button(
            "Retry filing lookup", key="retry_filings"
        ):
            fetch_filings.clear()
            st.rerun()
        return None
    if not filings:
        st.info("No filings were found for this ticker and filing type.")
        return None

    labels = [filing_option_label(filing) for filing in filings]
    selected_label = st.selectbox(
        "SEC filing",
        labels,
        help="Newest filings appear first.",
    )
    selected = filings[labels.index(selected_label)]
    st.caption(f"Accession {selected['accession']}")
    return selected


def _uppercase_ticker_widget() -> None:
    value = st.session_state.get("ticker_input")
    if isinstance(value, str):
        st.session_state.ticker_input = value.strip().upper()


def render_sidebar() -> tuple[dict[str, Any] | None, bool]:
    """Render only analysis configuration; audio controls live by the summary."""

    with st.sidebar:
        st.header("Analysis configuration")
        mode = st.radio(
            "Data mode",
            ["Demo", "Real"],
            horizontal=True,
            help="Demo uses a fixture; Real contacts SEC and the configured provider.",
        )
        payload: dict[str, Any] | None
        if mode == "Real":
            st.caption("LIVE | SEC filing and configured OpenRouter model")
            raw_ticker = st.text_input(
                "Ticker",
                "AAPL",
                max_chars=15,
                key="ticker_input",
                on_change=_uppercase_ticker_widget,
            )
            ticker = normalize_ticker_for_ui(raw_ticker)
            filing_type = st.selectbox("Filing type", ["10-Q", "10-K"])
            selected: dict[str, Any] | None = None
            if ticker is None:
                st.warning("Enter a valid ticker, for example AAPL, BRK.B, or BRK-B.")
            else:
                if ticker != raw_ticker.strip():
                    st.caption(f"Using normalized ticker: {ticker}")
                selected = _render_filing_selector(ticker, filing_type)
            payload = None
            if ticker is not None and selected is not None:
                payload = {
                    "ticker": ticker,
                    "filing_date": selected["filing_date"],
                    "filing_type": selected["form"],
                    "mode": "real",
                }
        else:
            st.badge("DEMO | SYNTHETIC", color="orange")
            st.caption("Deterministic fixture | No SEC or LLM request")
            payload = {"ticker": "DEMO", "mode": "demo"}

        run = st.button("Analyze filing", type="primary", width="stretch")
    return payload, run


def render_product_header() -> None:
    st.title("Financial Intelligence Copilot")
    st.caption("SEC filings | Grounded AI | Deterministic verification")


def render_analysis_header(handoff: dict[str, Any]) -> str:
    meta = handoff["pipeline_metadata"]
    state = verification_label(handoff["verification"])
    mode_column, verification_column = st.columns([1, 4])
    with mode_column:
        if meta["analysis_mode"] == "real":
            st.badge("LIVE ANALYSIS", color="green")
        else:
            st.badge("DEMO | SYNTHETIC", color="orange")
    with verification_column:
        state_color = {
            "VERIFIED": "green",
            "VERIFIED WITH WARNINGS": "orange",
            "FAILED VERIFICATION": "red",
        }[state]
        st.badge(
            state,
            color=state_color,
        )

    company, ticker = st.columns([4, 1], vertical_alignment="bottom")
    with company:
        st.header(md_escape(handoff["company"]))
    with ticker:
        st.subheader(md_escape(handoff["ticker"]))

    facts = [
        handoff["filing_type"],
        f"Report period: {format_display_date(handoff['period'])}",
    ]
    if meta.get("filing_date"):
        facts.append(f"Filed: {format_display_date(meta['filing_date'])}")
    st.caption(" | ".join(facts))
    if meta["analysis_mode"] == "demo":
        st.caption("Deterministic synthetic fixture | No SEC or LLM request")
    return state


def _render_metric_card(metric: dict[str, Any]) -> None:
    unit = metric["unit"]
    change = metric["change_pct"]
    with st.container(border=True):
        st.metric(
            metric["name"],
            format_value(metric["current_value"], unit),
            delta=None if change is None else format_change_pct(change),
            delta_color="off",
        )
        comparison = metric["comparison_type"]
        st.caption(comparison or "NO COMPARABLE PERIOD")
        previous = format_value(metric["previous_value"], unit)
        if metric["previous_value"] is None:
            st.caption("Previous: N/A | Not available")
        else:
            st.caption(f"Previous: {previous}")


def _render_metric_grid(metrics: list[dict[str, Any]]) -> None:
    for row in (metrics[:4], metrics[4:7]):
        if not row:
            continue
        columns = st.columns(len(row))
        for column, metric in zip(columns, row, strict=True):
            with column:
                _render_metric_card(metric)


def render_metrics(metrics: list[dict[str, Any]]) -> None:
    st.subheader("Canonical financial metrics")
    st.caption("Delivered by the deterministic XBRL pipeline; no UI recalculation.")
    if not metrics:
        st.info("No canonical metrics are available.")
        return

    _render_metric_grid(metrics)
    chart_mode = st.radio(
        "Financial chart",
        ["Change %", "Values"],
        horizontal=True,
        help="Change is shown only when the pipeline provides a comparable period.",
    )
    figure = (
        render_metrics_change_chart(metrics)
        if chart_mode == "Change %"
        else render_metrics_comparison_chart(metrics)
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    unavailable = sum(metric.get("change_pct") is None for metric in metrics)
    if unavailable:
        st.caption(f"{unavailable} metric(s) have no comparable period.")

    with st.expander("Detailed metrics table"):
        st.caption(
            "QoQ compares sequential quarters. YoY_YTD compares equivalent "
            "year-to-date durations."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": metric["name"],
                    "Current": format_value(metric["current_value"], metric["unit"]),
                    "Previous": format_value(metric["previous_value"], metric["unit"]),
                    "Change": format_change_pct(metric["change_pct"]),
                    "Comparison": metric["comparison_type"] or NOT_AVAILABLE,
                    "Current period": format_metric_period(metric["current_period"]),
                    "Previous period": format_metric_period(metric["previous_period"]),
                }
                for metric in metrics
            ),
            hide_index=True,
            width="stretch",
        )


def render_executive_snapshot(handoff: dict[str, Any]) -> None:
    st.subheader("Executive snapshot")
    metrics = select_executive_metrics(handoff["financial_metrics"])
    if not metrics:
        st.info("No canonical metrics are available for the snapshot.")
    else:
        columns = st.columns(len(metrics))
        for column, metric in zip(columns, metrics, strict=True):
            with column:
                st.metric(
                    metric["name"],
                    format_value(metric["current_value"], metric["unit"]),
                    delta=(
                        None
                        if metric["change_pct"] is None
                        else format_change_pct(metric["change_pct"])
                    ),
                    delta_color="off",
                )

    positive = handoff["positives"][0]["finding"] if handoff["positives"] else None
    risk = handoff["risks"][0]["finding"] if handoff["risks"] else None
    outlook = handoff["management_outlook"]
    for column, title, value in zip(
        st.columns(3),
        ("Positive signal", "Risk to monitor", "Management outlook"),
        (
            positive or "No grounded positive development returned.",
            risk or "No grounded risk returned.",
            outlook["summary"],
        ),
        strict=True,
    ):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            if title == "Management outlook":
                sentiment = outlook["sentiment"]
                st.badge(
                    sentiment.upper(),
                    color=SENTIMENT_COLORS.get(sentiment, "gray"),
                )
            st.markdown(md_escape(value))


def _render_finding_summary(
    item: dict[str, Any],
    *,
    filing_type: str,
) -> None:
    with st.container(border=True):
        st.markdown(f"**{md_escape(item['finding'])}**")
        st.caption(human_source_label(filing_type, item.get("source_section")))
        st.caption("Exact supporting evidence is available in Sources.")


def render_findings(handoff: dict[str, Any]) -> None:
    left, right = st.columns(2)
    sections = (
        (left, "Positive developments", handoff["positives"]),
        (right, "Key risks", handoff["risks"]),
    )
    for column, title, items in sections:
        with column:
            st.subheader(title)
            if not items:
                st.caption("No sufficiently grounded findings were returned.")
            for item in items:
                _render_finding_summary(item, filing_type=handoff["filing_type"])


def render_outlook(outlook: dict[str, Any]) -> None:
    with st.container(border=True):
        heading, badge = st.columns([4, 1], vertical_alignment="center")
        with heading:
            st.subheader("Management outlook")
        with badge:
            sentiment = outlook["sentiment"]
            st.badge(
                sentiment.upper(),
                color=SENTIMENT_COLORS.get(sentiment, "gray"),
            )
        st.markdown(md_escape(outlook["summary"]))


def render_summary(handoff: dict[str, Any]) -> None:
    st.subheader("Executive summary")
    summary_column, action_column = st.columns([3, 1])
    with summary_column:
        st.markdown(md_escape(handoff["executive_summary"]))
    with action_column:
        voices = fetch_voices()
        voice = st.selectbox(
            "Voice",
            voices,
            index=voices.index("af_heart") if "af_heart" in voices else 0,
            key="summary_voice",
        )
        if st.button("Copy summary", width="stretch"):
            st.session_state.summary_copy_ready = True
        if st.button("Listen to summary", width="stretch"):
            with st.spinner("Generating summary audio"):
                response, error = api_request(
                    "POST",
                    "/api/v1/audio/summary",
                    json={"text": handoff["executive_summary"], "voice": voice},
                    timeout=300,
                )
            if response is not None:
                st.session_state.audio = response.content
                st.session_state.audio_mode = handoff["pipeline_metadata"][
                    "analysis_mode"
                ]
            elif error is not None:
                render_error(error, allow_retry=False)

    if st.session_state.get("summary_copy_ready"):
        st.caption("Use the copy control in the text block below.")
        st.code(handoff["executive_summary"], language=None, wrap_lines=True)
    if "audio" in st.session_state:
        st.audio(st.session_state.audio, format="audio/wav")
        if st.session_state.get("audio_mode") == "demo":
            st.caption("Audio generated from synthetic demo analysis.")
        else:
            st.caption("Audio generated from verified real analysis.")


def render_verification_details(verification: dict[str, Any]) -> str:
    state = verification_label(verification)
    st.caption(
        "Deterministic verification checks metrics, citations, narrative numbers, "
        "recommendation language, and summary format."
    )
    with st.expander("Verification details"):
        if not verification["issues"]:
            st.markdown("**VERIFIED** | No issues reported.")
        for issue in verification["issues"]:
            label = f"{issue['severity'].upper()} | {issue['code']} | {issue['field']}"
            show = st.warning if issue["severity"] == "warning" else st.error
            show(f"{label}: {md_escape(issue['message'])}")
    return state


def render_sources(handoff: dict[str, Any]) -> None:
    st.subheader("Evidence and provenance")
    evidence_items = [
        ("Positive", item) for item in handoff["positives"]
    ] + [("Risk", item) for item in handoff["risks"]]
    if not evidence_items:
        st.info("No finding-level evidence was returned.")
    for index, (kind, item) in enumerate(evidence_items, start=1):
        source_label = human_source_label(
            handoff["filing_type"], item.get("source_section")
        )
        with st.expander(f"{kind} {index} | {source_label}"):
            st.markdown(f"**{md_escape(item['finding'])}**")
            st.markdown(f"> {md_escape(item['evidence'])}")
            st.caption(
                f"Source type: {item['source_type']} | "
                f"Canonical section: {item.get('source_section') or NOT_AVAILABLE}"
            )
            st.caption("Technical source ID")
            st.code(item.get("source_id") or NOT_AVAILABLE, language=None, wrap_lines=True)


def render_technical_details(handoff: dict[str, Any]) -> None:
    meta = handoff["pipeline_metadata"]
    with st.expander("Technical details"):
        rows = {
            "Analysis mode": str(meta["analysis_mode"]),
            "Provider": str(meta.get("provider") or NOT_AVAILABLE),
            "Model": str(meta.get("model") or NOT_AVAILABLE),
            "Filing date": format_display_date(meta.get("filing_date")),
            "Report period": format_display_date(handoff.get("period")),
            "Retrieved evidence chunks": str(meta["retrieval_count"]),
            "Retrieved source IDs": str(len(meta["retrieved_source_ids"])),
            "Generation attempts": str(meta.get("generation_attempts", 1)),
            "Repair used": "yes" if meta.get("repair_used") else "no",
            "Verification": verification_label(handoff["verification"]),
        }
        st.dataframe(
            pd.DataFrame(rows.items(), columns=["Field", "Value"]),
            hide_index=True,
            width="stretch",
        )


def execute_analysis(payload: dict[str, Any]) -> None:
    """Run one request with a neutral, honest loading state."""

    for key in (
        "handoff",
        "audio",
        "audio_mode",
        "analysis_error",
        "summary_copy_ready",
    ):
        st.session_state.pop(key, None)
    st.session_state.last_request = payload
    started = time.perf_counter()
    st.caption(
        "The backend retrieves the filing, processes financial data, generates a "
        "grounded analysis, and applies deterministic verification."
    )
    with st.spinner("Preparing analysis"):
        response, error = api_request(
            "POST", "/api/v1/analysis", json=payload, timeout=300
        )
    elapsed = time.perf_counter() - started
    if response is not None:
        st.session_state.handoff = response.json()
        st.caption(f"Analysis completed in {elapsed:.1f}s.")
    else:
        st.session_state.analysis_error = error
        st.caption(f"Analysis stopped after {elapsed:.1f}s.")
    if error is not None:
        render_error(error, allow_retry=False)


def render_result(handoff: dict[str, Any]) -> None:
    state = render_analysis_header(handoff)
    if state == "FAILED VERIFICATION":
        st.error(
            "This result failed deterministic verification and is not displayed as "
            "a valid analysis."
        )
        render_verification_details(handoff["verification"])
        render_technical_details(handoff)
        return

    overview, financials, narrative, sources = st.tabs(
        ["Overview", "Financials", "Narrative", "Sources"]
    )
    with overview:
        render_executive_snapshot(handoff)
    with financials:
        render_metrics(handoff["financial_metrics"])
    with narrative:
        render_findings(handoff)
        render_outlook(handoff["management_outlook"])
        render_summary(handoff)
    with sources:
        render_sources(handoff)
        render_verification_details(handoff["verification"])
        render_technical_details(handoff)


def _hide_native_running_widget() -> None:
    """Hide Streamlit's header "Running..." widget.

    That built-in widget cycles through informal pictograms (cyclist, swimmer,
    wheelchair). Loading feedback is given by the neutral ``st.spinner`` blocks.
    """

    st.html("<style>[data-testid=\"stStatusWidget\"]{display:none;}</style>")


def main() -> None:
    st.set_page_config(
        page_title="Financial Intelligence Copilot",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    _hide_native_running_widget()
    render_product_header()
    payload, run = render_sidebar()

    previous_error = st.session_state.get("analysis_error")
    retry = render_error(previous_error) if previous_error and not run else False
    if run or retry:
        selected_payload = payload if run else st.session_state.get("last_request")
        if selected_payload is None:
            st.sidebar.warning("Select an available filing before running analysis.")
        else:
            execute_analysis(selected_payload)

    handoff = st.session_state.get("handoff")
    if handoff is None:
        if st.session_state.get("analysis_error") is None:
            st.info("Configure an analysis in the sidebar, then select **Analyze filing**.")
        return
    render_result(handoff)


main()
