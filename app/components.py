"""Pure Dash view builders for the analysis dashboard.

Every function turns API data (the ``AnalysisHandoff`` dict, perf events, chat
messages) into Dash components. There are no callbacks, no HTTP and no
recalculation of financial values here: formatting comes from
``src.visualization`` and the deterministic pipeline.

Untrusted text (filings, model output) is always passed as plain component
children, which React escapes; only the FinBERT heatmap is HTML, and
``outlook_xai_html`` escapes every piece of API text itself.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import plotly.graph_objects as go
from dash import dcc, html

from src.visualization.financial_charts import (
    NOT_AVAILABLE,
    format_change_pct,
    format_value,
    render_metrics_change_chart,
    render_metrics_comparison_chart,
)
from src.visualization.performance import (
    average_costs,
    event_rows,
    format_ms,
    format_usd,
    latency_chart,
    latency_rows,
    summarize,
)
from src.visualization.presentation import (
    cited_sources,
    error_presentation,
    format_display_date,
    format_metric_period,
    human_source_label,
    outlook_xai_html,
    select_executive_metrics,
    sentiment_label,
    verification_label,
)

CHAT_SUGGESTIONS = (
    "What drove the change in revenue?",
    "What are the main risks?",
    "What does management expect going forward?",
)
# Chat TTS engines: Kokoro on the container (EN/ES) or Groq Orpheus (fast, EN).
CHAT_TTS_PROVIDERS = {"local": "Local (Kokoro)", "groq": "Groq (fast)"}
SENTIMENT_COLORS = {
    "positive": "green",
    "negative": "red",
    "mixed": "orange",
    "neutral": "gray",
    "unknown": "gray",
}
VERIFICATION_COLORS = {
    "VERIFIED": "green",
    "VERIFIED WITH WARNINGS": "orange",
    "FAILED VERIFICATION": "red",
}
GRAPH_CONFIG = {"displayModeBar": False}
TEXT_PRIMARY = "#F8FAFC"
GRID_COLOR = "#334155"


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def caption(text: str, **kwargs: Any) -> html.P:
    return html.P(text, className="caption", **kwargs)


def badge(text: str, color: str = "gray") -> html.Span:
    return html.Span(text, className=f"badge badge--{color}")


def card(*children: Any, className: str = "", **kwargs: Any) -> html.Div:
    return html.Div(list(children), className=f"card {className}".strip(), **kwargs)


def notice(kind: str, *children: Any) -> html.Div:
    """Inline message box; ``kind`` is info, warning or error."""

    return html.Div(list(children), className=f"notice notice--{kind}", role="status")


def expander(summary: str, *children: Any) -> html.Details:
    return html.Details(
        [html.Summary(summary), html.Div(list(children), className="details-body")],
        className="expander",
    )


def columns(*children: Any, template: str | None = None) -> html.Div:
    template = template or f"repeat({len(children)}, minmax(0, 1fr))"
    return html.Div(list(children), className="columns", style={"gridTemplateColumns": template})


def metric_tile(label: str, value: Any, delta: str | None = None, hint: str | None = None):
    return html.Div(
        [
            html.Div(label, className="metric__label", title=hint),
            html.Div(str(value), className="metric__value"),
            html.Div(delta, className="metric__delta") if delta else None,
        ],
        className="metric",
    )


def data_table(rows: Sequence[Mapping[str, Any]]) -> html.Div:
    """Plain read-only table from a list of dicts that share their keys."""

    rows = list(rows)
    headers = list(rows[0]) if rows else []
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(header) for header in headers])),
                html.Tbody(
                    [
                        html.Tr([html.Td("" if row.get(h) is None else str(row[h])) for h in headers])
                        for row in rows
                    ]
                ),
            ],
            className="data-table",
        ),
        className="table-wrap",
    )


def themed(figure: go.Figure) -> go.Figure:
    """Dark-slate chart theme (DESIGN.md); the chart modules only set data colours."""

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY, "family": "Inter, Segoe UI, sans-serif"},
    )
    figure.update_xaxes(gridcolor=GRID_COLOR)
    figure.update_yaxes(gridcolor=GRID_COLOR)
    return figure


def graph(figure: go.Figure, **kwargs: Any) -> dcc.Graph:
    return dcc.Graph(figure=themed(figure), config=GRAPH_CONFIG, className="chart", **kwargs)


# --------------------------------------------------------------------------- #
# Errors and header
# --------------------------------------------------------------------------- #
def error_block(error: Mapping[str, Any]) -> html.Div:
    """Safe error guidance; never shows provider messages or raw payloads."""

    code = str(error.get("code") or "UNKNOWN_ERROR")
    copy = error_presentation(code)
    return notice("error", html.Strong(f"{copy.title} ({code})"), caption(copy.guidance))


def analysis_header(handoff: Mapping[str, Any], state: str) -> html.Div:
    meta = handoff["pipeline_metadata"]
    facts = [
        handoff["filing_type"],
        f"Report period: {format_display_date(handoff['period'])}",
    ]
    if meta.get("filing_date"):
        facts.append(f"Filed: {format_display_date(meta['filing_date'])}")
    live = meta["analysis_mode"] == "real"
    return html.Div(
        [
            html.Div(
                [
                    badge("LIVE ANALYSIS", "green") if live else badge("DEMO | SYNTHETIC", "orange"),
                    badge(state, VERIFICATION_COLORS[state]),
                ],
                className="badges",
            ),
            html.Div(
                [
                    html.H2(handoff["company"]),
                    html.H3(handoff["ticker"], className="ticker"),
                ],
                className="company",
            ),
            caption(" | ".join(facts)),
            None if live else caption("Deterministic synthetic fixture | No SEC or LLM request"),
        ],
        className="report-header",
    )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def _delta(metric: Mapping[str, Any]) -> str | None:
    change = metric["change_pct"]
    return None if change is None else format_change_pct(change)


def executive_snapshot(handoff: Mapping[str, Any]) -> html.Div:
    metrics = select_executive_metrics(handoff["financial_metrics"])
    if metrics:
        tiles: Any = columns(
            *[
                metric_tile(
                    m["name"], format_value(m["current_value"], m["unit"]), _delta(m)
                )
                for m in metrics
            ]
        )
    else:
        tiles = notice("info", "No canonical metrics are available for the snapshot.")

    positive = handoff["positives"][0]["finding"] if handoff["positives"] else None
    risk = handoff["risks"][0]["finding"] if handoff["risks"] else None
    outlook = handoff["management_outlook"]
    messages = (
        ("Positive signal", positive or "No grounded positive development returned."),
        ("Risk to monitor", risk or "No grounded risk returned."),
        ("Management outlook", outlook["summary"]),
    )
    cards = []
    for title, value in messages:
        cards.append(
            card(
                html.Strong(title),
                badge(
                    sentiment_label(outlook),
                    SENTIMENT_COLORS.get(outlook["sentiment"], "gray"),
                )
                if title == "Management outlook"
                else None,
                html.P(value),
            )
        )
    return html.Div(
        [html.H3("Executive snapshot"), tiles, columns(*cards)], className="section"
    )


# --------------------------------------------------------------------------- #
# Financials
# --------------------------------------------------------------------------- #
def _metric_card(metric: Mapping[str, Any]) -> html.Div:
    unit = metric["unit"]
    previous = (
        "Previous: N/A | Not available"
        if metric["previous_value"] is None
        else f"Previous: {format_value(metric['previous_value'], unit)}"
    )
    return card(
        metric_tile(metric["name"], format_value(metric["current_value"], unit), _delta(metric)),
        caption(metric["comparison_type"] or "NO COMPARABLE PERIOD"),
        caption(previous),
    )


def metrics_section(metrics: Sequence[Mapping[str, Any]]) -> html.Div:
    head = [
        html.H3("Canonical financial metrics"),
        caption("Delivered by the deterministic XBRL pipeline; no UI recalculation."),
    ]
    if not metrics:
        return html.Div([*head, notice("info", "No canonical metrics are available.")], className="section")

    metrics = list(metrics)
    grid = [columns(*[_metric_card(m) for m in row]) for row in (metrics[:4], metrics[4:7]) if row]
    charts = dcc.Tabs(
        id="chart-tabs",
        value="change",
        className="pill-tabs",
        children=[
            dcc.Tab(
                label="Change %",
                value="change",
                children=graph(render_metrics_change_chart(list(metrics))),
            ),
            dcc.Tab(
                label="Values",
                value="values",
                children=graph(render_metrics_comparison_chart(list(metrics))),
            ),
        ],
    )
    unavailable = sum(m.get("change_pct") is None for m in metrics)
    rows = [
        {
            "Metric": m["name"],
            "Current": format_value(m["current_value"], m["unit"]),
            "Previous": format_value(m["previous_value"], m["unit"]),
            "Change": format_change_pct(m["change_pct"]),
            "Comparison": m["comparison_type"] or NOT_AVAILABLE,
            "Current period": format_metric_period(m["current_period"]),
            "Previous period": format_metric_period(m["previous_period"]),
        }
        for m in metrics
    ]
    return html.Div(
        [
            *head,
            *grid,
            html.P("Financial chart", className="field-label"),
            charts,
            caption(f"{unavailable} metric(s) have no comparable period.") if unavailable else None,
            expander(
                "Detailed metrics table",
                caption(
                    "YoY compares the same period one year earlier. YoY_YTD compares "
                    "equivalent year-to-date durations."
                ),
                data_table(rows),
            ),
        ],
        className="section",
    )


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #
def _finding_card(item: Mapping[str, Any], filing_type: str) -> html.Div:
    return card(
        html.Strong(item["finding"]),
        caption(human_source_label(filing_type, item.get("source_section"))),
        caption("Exact supporting evidence is available in Sources."),
    )


def findings(handoff: Mapping[str, Any]) -> html.Div:
    blocks = []
    for title, items in (
        ("Positive developments", handoff["positives"]),
        ("Key risks", handoff["risks"]),
    ):
        blocks.append(
            html.Div(
                [
                    html.H3(title),
                    *(
                        [_finding_card(i, handoff["filing_type"]) for i in items]
                        or [caption("No sufficiently grounded findings were returned.")]
                    ),
                ]
            )
        )
    return html.Div(columns(*blocks), className="section")


def inline_html(markup: str) -> dcc.Markdown:
    """Render trusted, already-escaped presentation HTML.

    Whitespace is collapsed so a blank line inside escaped filing text cannot
    end the HTML block and let Markdown reinterpret the rest.
    """

    return dcc.Markdown(
        re.sub(r"\s+", " ", markup), dangerously_allow_html=True, className="inline-html"
    )


def outlook_block(outlook: Mapping[str, Any]) -> html.Div:
    if outlook.get("confidence") is None:  # LLM-only label: no FinBERT evidence
        body: list[Any] = [
            badge(sentiment_label(outlook), SENTIMENT_COLORS.get(outlook["sentiment"], "gray")),
            html.P(outlook["summary"]),
        ]
    else:
        body = [inline_html(outlook_xai_html(outlook))]
    return html.Div(card(html.H3("Management outlook"), *body), className="section")


def summary_section(handoff: Mapping[str, Any], voices: Sequence[str]) -> html.Div:
    default = "af_heart" if "af_heart" in voices else (voices[0] if voices else None)
    return html.Div(
        [
            html.H3("Executive summary"),
            columns(
                html.P(handoff["executive_summary"], className="summary-text"),
                html.Div(
                    [
                        html.Label("Voice", htmlFor="summary-voice", className="field-label"),
                        dcc.Dropdown(
                            id="summary-voice",
                            options=list(voices),
                            value=default,
                            clearable=False,
                            searchable=False,
                        ),
                        html.Button(
                            "Copy summary", id="summary-copy", className="btn btn--ghost btn--block"
                        ),
                        html.Button(
                            "Listen to summary", id="summary-listen", className="btn btn--ghost btn--block"
                        ),
                    ],
                    className="actions",
                ),
                template="3fr 1fr",
            ),
            html.Div(id="summary-copy-status", className="caption", role="status"),
            html.Div(id="summary-audio-box"),
        ],
        className="section",
    )


def summary_audio_view(audio: Mapping[str, Any] | None) -> list[Any]:
    if not audio:
        return []
    if audio.get("error"):
        return [error_block(audio["error"])]
    note = (
        "Audio generated from synthetic demo analysis."
        if audio.get("mode") == "demo"
        else "Audio generated from verified real analysis."
    )
    return [html.Audio(src=audio["src"], controls=True, className="audio"), caption(note)]


# --------------------------------------------------------------------------- #
# Sources and verification
# --------------------------------------------------------------------------- #
def verification_details(verification: Mapping[str, Any]) -> html.Div:
    if verification["issues"]:
        issues: list[Any] = []
        for issue in verification["issues"]:
            label = f"{issue['severity'].upper()} | {issue['code']} | {issue['field']}"
            kind = "warning" if issue["severity"] == "warning" else "error"
            issues.append(notice(kind, f"{label}: {issue['message']}"))
    else:
        issues = [html.P([html.Strong("VERIFIED"), " | No issues reported."])]
    return html.Div(
        [
            caption(
                "Deterministic verification checks metrics, citations, narrative numbers, "
                "recommendation language, and summary format."
            ),
            expander("Verification details", *issues),
        ],
        className="section",
    )


def sources_section(handoff: Mapping[str, Any]) -> html.Div:
    evidence = [("Positive", i) for i in handoff["positives"]] + [
        ("Risk", i) for i in handoff["risks"]
    ]
    blocks: list[Any] = [html.H3("Evidence and provenance")]
    if not evidence:
        blocks.append(notice("info", "No finding-level evidence was returned."))
    for index, (kind, item) in enumerate(evidence, start=1):
        label = human_source_label(handoff["filing_type"], item.get("source_section"))
        blocks.append(
            expander(
                f"{kind} {index} | {label}",
                html.Strong(item["finding"]),
                html.Blockquote(item["evidence"]),
                caption(
                    f"Source type: {item['source_type']} | "
                    f"Canonical section: {item.get('source_section') or NOT_AVAILABLE}"
                ),
                caption("Technical source ID"),
                html.Pre(item.get("source_id") or NOT_AVAILABLE),
            )
        )
    return html.Div(blocks, className="section")


def technical_details(handoff: Mapping[str, Any]) -> html.Div:
    meta = handoff["pipeline_metadata"]
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
    return html.Div(
        expander(
            "Technical details",
            data_table([{"Field": k, "Value": v} for k, v in rows.items()]),
        ),
        className="section",
    )


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
PERF_INTRO = (
    "Measured live in this session: the analysis, every chat answer, audio "
    "and voice question. LLM costs are the amounts OpenRouter reports as "
    "charged; Groq costs use its published prices; Kokoro (on-instance) is "
    "an estimate from busy CPU time."
)


def performance_section() -> html.Div:
    """Static shell: callbacks fill ``perf-summary`` and ``perf-projection``."""

    return html.Div(
        [
            html.H3("Inference latency and cost"),
            caption(PERF_INTRO),
            html.Div(id="perf-summary"),
            html.H4("Viability projection"),
            columns(
                html.Div(
                    [
                        html.Label("Sessions per month", className="field-label"),
                        dcc.Slider(
                            id="proj-sessions",
                            min=100,
                            max=10_000,
                            step=100,
                            value=1_000,
                            marks={100: "100", 5_000: "5,000", 10_000: "10,000"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
                template="1fr",
            ),
            columns(
                *[
                    html.Div(
                        [
                            html.Label(label, htmlFor=component_id, className="field-label"),
                            dcc.Input(
                                id=component_id,
                                type="number",
                                min=low,
                                max=high,
                                step=1,
                                value=value,
                                debounce=True,
                                className="input",
                            ),
                        ]
                    )
                    for component_id, label, low, high, value in (
                        ("proj-turns", "Chat turns / session", 0, 50, 5),
                        ("proj-audios", "Audio plays / session", 0, 50, 2),
                        ("proj-voices", "Voice questions / session", 0, 50, 1),
                        ("proj-minutes", "Active minutes / session", 1, 120, 10),
                    )
                ]
            ),
            html.Div(id="perf-projection"),
        ],
        className="section",
    )


def perf_summary_view(events: Sequence[Mapping[str, Any]]) -> html.Div:
    summary = summarize(events)
    blocks: list[Any] = [
        columns(
            metric_tile("Session cost", format_usd(summary["total_cost_usd"]), hint="Inference cost"),
            metric_tile("Inferences", summary["count"]),
            metric_tile("Chat 1st token", format_ms(summary["median_chat_ttft_ms"]), hint="Median"),
            metric_tile("TTS latency", format_ms(summary["median_tts_ms"]), hint="Median"),
        ),
        html.P(html.Strong("Latency vs. targets for a fluid experience")),
        data_table(latency_rows(events)),
    ]
    if events:
        blocks += [
            graph(latency_chart(list(events))),
            html.P(html.Strong("Inference log")),
            data_table(event_rows(events)),
        ]
    else:
        blocks.append(notice("info", "Run an analysis or ask the chat to start measuring."))
    return html.Div(blocks)


def projection_view(
    projection: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    cloud_run: Mapping[str, Any],
    hourly: float,
) -> html.Div:
    assumed = [kind for kind, (_, measured) in average_costs(events).items() if not measured]
    return html.Div(
        [
            columns(
                metric_tile("Cost per session", format_usd(projection["per_session"])),
                metric_tile("Monthly APIs", format_usd(projection["monthly_inference"])),
                metric_tile("Monthly infra", format_usd(projection["monthly_infra"])),
                metric_tile("Monthly total", format_usd(projection["monthly_total"])),
            ),
            caption(
                f"Cloud Run instance: {cloud_run.get('vcpu', 4):g} vCPU / "
                f"{cloud_run.get('memory_gib', 4):g} GiB ≈ {format_usd(hourly)}/hour of "
                "request time (billed per request: only the active minutes of a session "
                "count, an idle open tab costs nothing). Per-kind costs are this "
                "session's averages"
                + (f"; not measured yet, using defaults: {', '.join(assumed)}." if assumed else ".")
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Result page
# --------------------------------------------------------------------------- #
def build_result(handoff: Mapping[str, Any], voices: Sequence[str]) -> html.Div:
    state = verification_label(handoff["verification"])
    header = analysis_header(handoff, state)
    if state == "FAILED VERIFICATION":
        return html.Div(
            [
                header,
                notice(
                    "error",
                    "This result failed deterministic verification and is not displayed "
                    "as a valid analysis.",
                ),
                verification_details(handoff["verification"]),
                technical_details(handoff),
            ]
        )

    def tab(label: str, value: str, *children: Any) -> dcc.Tab:
        return dcc.Tab(
            label=label, value=value, children=html.Div(list(children), className="tab-body")
        )

    return html.Div(
        [
            header,
            dcc.Tabs(
                id="result-tabs",
                value="overview",
                className="tabs",
                children=[
                    tab("Overview", "overview", executive_snapshot(handoff)),
                    tab("Financials", "financials", metrics_section(handoff["financial_metrics"])),
                    tab(
                        "Narrative",
                        "narrative",
                        findings(handoff),
                        outlook_block(handoff["management_outlook"]),
                        summary_section(handoff, voices),
                    ),
                    tab(
                        "Sources",
                        "sources",
                        sources_section(handoff),
                        verification_details(handoff["verification"]),
                        technical_details(handoff),
                    ),
                    tab("Performance", "performance", performance_section()),
                ],
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def chat_caption(handoff: Mapping[str, Any] | None) -> str:
    if handoff and handoff["pipeline_metadata"]["analysis_mode"] == "demo":
        return "Chat uses the configured LLM over the synthetic demo analysis."
    return "Answers use only the verified results above and cite them, e.g. [M1]."


def chat_history(
    messages: Sequence[Mapping[str, Any]],
    audio_cache: Mapping[str, Mapping[str, Any]],
    provider: str,
    voice: str,
    handoff: Mapping[str, Any],
) -> list[Any]:
    """Bubbles for the whole conversation, plus a pending marker while answering."""

    bubbles: list[Any] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            bubbles.append(html.Div(message["content"], className="msg msg--user"))
            continue
        body: list[Any] = [dcc.Markdown(message["content"])]
        sources = cited_sources(message["content"], handoff)
        if sources:
            body.append(
                expander(
                    f"Sources cited ({len(sources)})",
                    *[html.P([html.Strong(tag), f" · {label}"]) for tag, label in sources],
                )
            )
        cached = audio_cache.get(f"{index}:{provider}:{voice}")
        if cached:
            body += [html.Audio(src=cached["audio"], controls=True, className="audio"), caption(cached["note"])]
        else:
            body.append(
                html.Button(
                    "Listen",
                    id={"type": "chat-listen", "index": index},
                    className="btn btn--ghost btn--small",
                )
            )
        bubbles.append(html.Div(body, className="msg msg--assistant"))
    if messages and messages[-1]["role"] == "user":
        bubbles.append(html.Div("Thinking…", className="msg msg--assistant msg--pending", role="status"))
    return bubbles


def chat_error_view(error: Mapping[str, Any] | None) -> list[Any]:
    if not error:
        return []
    code = str(error.get("code") or "UNKNOWN_ERROR")
    title = error_presentation(code).title
    if error.get("kind") == "audio":
        return [notice("error", html.Strong(f"Audio could not be generated ({code})."), caption(title))]
    return [
        notice(
            "error",
            html.Strong(f"The assistant could not answer ({code})."),
            caption(f"{title} You can ask again."),
        )
    ]
