"""Inference latency/cost presentation for the Performance tab.

Pure helpers over the per-session event log the UI keeps in session state.
Each event is a dict produced from API metrics::

    {"kind": "analysis" | "chat" | "tts" | "stt", "label": str, "provider": str,
     "latency_ms": float | None, "ttft_ms": float | None, "units": str,
     "cost_usd": float | None, "estimated": bool, "time": "HH:MM:SS"}

Latencies are end-to-end as measured by the UI (what the user waits).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Mapping, Sequence

import plotly.graph_objects as go

KIND_LABELS = {
    "analysis": "Filing analysis",
    "chat": "Chat answer",
    "tts": "Text-to-speech",
    "stt": "Speech-to-text",
}
KIND_COLORS = {
    "analysis": "#38BDF8",
    "chat": "#10B981",
    "tts": "#F59E0B",
    "stt": "#A78BFA",
}


@dataclass(frozen=True, slots=True)
class LatencyTarget:
    """UX latency budget for one interaction (documented assumption)."""

    metric: str
    kind: str
    field: str
    target_ms: float
    rationale: str


# Budgets for a fluid experience. Streaming makes "first token" the key chat
# number; the full analysis is a one-off, explicitly awaited action.
LATENCY_TARGETS: tuple[LatencyTarget, ...] = (
    LatencyTarget("Chat: first token", "chat", "ttft_ms", 2_000,
                  "Under ~2 s a streamed reply feels immediate."),
    LatencyTarget("Chat: full answer", "chat", "latency_ms", 10_000,
                  "Users keep reading while it streams."),
    LatencyTarget("Text-to-speech", "tts", "latency_ms", 5_000,
                  "Wait after pressing Listen."),
    LatencyTarget("Speech-to-text", "stt", "latency_ms", 3_000,
                  "Gap between speaking and seeing the question."),
    LatencyTarget("Filing analysis (real)", "analysis", "latency_ms", 90_000,
                  "One-off job with progress feedback."),
)

# Used when the session has no measurement yet for a kind (labelled in the UI).
DEFAULT_COST_USD = {"analysis": 0.03, "chat": 0.005, "tts": 0.002, "stt": 0.0002}


def _values(events: Sequence[Mapping[str, Any]], kind: str, field: str) -> list[float]:
    return [
        float(event[field])
        for event in events
        if event.get("kind") == kind and isinstance(event.get(field), (int, float))
    ]


def summarize(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Session totals and medians for the headline metrics."""

    costs = [float(e["cost_usd"]) for e in events if isinstance(e.get("cost_usd"), (int, float))]
    ttft = _values(events, "chat", "ttft_ms")
    tts = _values(events, "tts", "latency_ms")
    return {
        "count": len(events),
        "total_cost_usd": sum(costs),
        "median_chat_ttft_ms": median(ttft) if ttft else None,
        "median_tts_ms": median(tts) if tts else None,
    }


def latency_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per UX target: measured median vs budget."""

    rows = []
    for target in LATENCY_TARGETS:
        values = _values(events, target.kind, target.field)
        measured = median(values) if values else None
        if measured is None:
            status = "— not measured yet"
        elif measured <= target.target_ms:
            status = "✅ within target"
        else:
            status = "⚠ above target"
        rows.append(
            {
                "Interaction": target.metric,
                "Target": format_ms(target.target_ms),
                "Measured (median)": format_ms(measured),
                "Samples": len(values),
                "Status": status,
                "Why": target.rationale,
            }
        )
    return rows


def average_costs(events: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, bool]]:
    """Mean cost per kind and whether it was measured (False = default used)."""

    result = {}
    for kind, default in DEFAULT_COST_USD.items():
        values = _values(events, kind, "cost_usd")
        result[kind] = (mean(values), True) if values else (default, False)
    return result


def project_monthly_cost(
    events: Sequence[Mapping[str, Any]],
    *,
    sessions_per_month: int,
    chat_turns: int,
    audio_plays: int,
    voice_questions: int,
    session_minutes: float,
    instance_usd_per_hour: float,
) -> dict[str, float]:
    """Cost per session and per month from measured averages.

    Inference = 1 analysis + turns + audios + voice questions per session.
    Infrastructure: each session keeps the instance up for ``session_minutes``
    (sessions assumed not to overlap; Cloud Run's idle tail after the last
    session of a burst is not included).
    """

    avg = average_costs(events)
    inference = (
        avg["analysis"][0]
        + chat_turns * avg["chat"][0]
        + audio_plays * avg["tts"][0]
        + voice_questions * avg["stt"][0]
    )
    infra = session_minutes / 60 * instance_usd_per_hour
    return {
        "inference_per_session": inference,
        "infra_per_session": infra,
        "per_session": inference + infra,
        "monthly_inference": inference * sessions_per_month,
        "monthly_infra": infra * sessions_per_month,
        "monthly_total": (inference + infra) * sessions_per_month,
    }


def format_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} ms" if value < 1_000 else f"{value / 1_000:,.1f} s"


def format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "$0"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def event_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Inference log rows, newest first."""

    return [
        {
            "Time": event.get("time", ""),
            "Component": KIND_LABELS.get(str(event.get("kind")), str(event.get("kind"))),
            "Detail": event.get("label", ""),
            "Provider / model": event.get("provider", ""),
            "Latency": format_ms(event.get("latency_ms")),
            "First token": format_ms(event.get("ttft_ms")),
            "Volume": event.get("units", ""),
            "Cost": format_usd(event.get("cost_usd"))
            + (" (est.)" if event.get("estimated") else ""),
        }
        for event in reversed(events)
    ]


def latency_chart(events: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Bar per event (in order), coloured by component, with target markers."""

    figure = go.Figure()
    order = [f"#{i + 1} {KIND_LABELS.get(e.get('kind'), '')}" for i, e in enumerate(events)]
    for kind, label in KIND_LABELS.items():
        idx = [i for i, e in enumerate(events) if e.get("kind") == kind]
        if not idx:
            continue
        figure.add_trace(
            go.Bar(
                name=label,
                x=[order[i] for i in idx],
                y=[(events[i].get("latency_ms") or 0) / 1000 for i in idx],
                text=[format_ms(events[i].get("latency_ms")) for i in idx],
                textposition="outside",
                customdata=[[events[i].get("label", ""), events[i].get("provider", "")] for i in idx],
                hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                "<br>Latency: %{text}<extra></extra>",
                marker={"color": KIND_COLORS[kind], "cornerradius": 4},
            )
        )
    for target in LATENCY_TARGETS:
        if target.field != "latency_ms" or target.kind == "analysis":
            continue
        if any(e.get("kind") == target.kind for e in events):
            figure.add_hline(
                y=target.target_ms / 1000,
                line={"dash": "dot", "color": KIND_COLORS[target.kind], "width": 1},
                annotation_text=f"{target.metric} target",
                annotation_font_size=10,
            )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_title="seconds",
        xaxis={"categoryorder": "array", "categoryarray": order},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )
    return figure


__all__ = [
    "DEFAULT_COST_USD",
    "KIND_LABELS",
    "LATENCY_TARGETS",
    "average_costs",
    "event_rows",
    "format_ms",
    "format_usd",
    "latency_chart",
    "latency_rows",
    "project_monthly_cost",
    "summarize",
]
