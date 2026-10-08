"""Institutional Dash presentation for the FastAPI analysis contract.

The UI communicates with FastAPI over HTTP only. It never imports extraction,
integration, SEC, XBRL, LLM, or audio implementation modules, and it never
recalculates financial metrics.

Dash callbacks are plain, short HTTP requests (no WebSocket), so on Cloud Run
the instance is only billed while a callback is actually being served. Session
state lives in the browser (``dcc.Store``); the server keeps none.

Run locally with ``python -m app.dash_app`` (API at ``API_URL``) or in
production with ``gunicorn app.dash_app:server``.
"""

from __future__ import annotations

import base64
import binascii
import os
import sys
import time
from pathlib import Path
from typing import Any

# `python app/dash_app.py` only puts app/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402

from app.api_client import (  # noqa: E402
    API_URL,
    DEFAULT_VOICES,
    api_request,
    chat_answer,
    fetch_filings,
    fetch_pricing,
    fetch_voices,
    transcribe_audio,
)
from app.components import (  # noqa: E402
    CHAT_SUGGESTIONS,
    CHAT_TTS_PROVIDERS,
    build_result,
    caption,
    chat_caption,
    chat_error_view,
    chat_history,
    error_block,
    notice,
    perf_summary_view,
    projection_view,
    summary_audio_view,
)
from src.visualization.performance import (  # noqa: E402
    format_ms,
    project_monthly_cost,
)
from src.visualization.presentation import (  # noqa: E402
    error_presentation,
    filing_option_label,
    text_for_speech,
    verification_label,
)

ASSETS_FOLDER = str(Path(__file__).resolve().parent / "assets")
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
DEMO_PAYLOAD = {"ticker": "DEMO", "mode": "demo"}
# Tickers offered in real mode; filings for each are discovered live from SEC.
TICKERS = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com, Inc.",
    "BAC": "Bank of America Corporation",
    "GOOGL": "Alphabet Inc.",
    "GS": "The Goldman Sachs Group, Inc.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "META": "Meta Platforms, Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "UNH": "UnitedHealth Group Incorporated",
}
SPEECH_CHARS = 5000  # chat answers are clipped before text-to-speech


# --------------------------------------------------------------------------- #
# Perf events (the Performance tab log lives in the browser session)
# --------------------------------------------------------------------------- #
def perf_event(kind: str, **fields: Any) -> dict[str, Any]:
    """One inference of the session log shown in the Performance tab."""

    return {"kind": kind, "time": time.strftime("%H:%M:%S"), **fields}


def _header_float(response: Any, name: str) -> float | None:
    try:
        return float(response.headers.get(name, ""))
    except ValueError:
        return None


def tts_event(response: Any, latency_ms: float, label: str) -> dict[str, Any]:
    provider = response.headers.get("X-TTS-Provider", "local")
    audio_seconds = _header_float(response, "X-TTS-Audio-Seconds")
    chars = response.headers.get("X-TTS-Chars", "")
    units = f"{chars} chars" + (f" → {audio_seconds:.1f} s audio" if audio_seconds else "")
    if response.headers.get("X-TTS-Fallback") == "groq_unavailable":
        label += " · Groq unavailable → Kokoro"
    return perf_event(
        "tts",
        label=label,
        provider=f"{CHAT_TTS_PROVIDERS.get(provider, provider)} · "
        f"{response.headers.get('X-TTS-Voice', '')}",
        latency_ms=latency_ms,
        units=units,
        cost_usd=_header_float(response, "X-TTS-Cost-USD"),
        estimated=provider != "groq",
    )


def analysis_event(handoff: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    meta = handoff["pipeline_metadata"]
    label = f"{handoff['ticker']} {handoff['filing_type']} {handoff['period']}"
    if meta["analysis_mode"] == "demo":
        return perf_event(
            "analysis",
            label=f"{label} (demo fixture, no inference)",
            # No cost: the fixture is not an inference and must not lower the
            # measured average used by the viability projection.
            provider="fixture",
            latency_ms=latency_ms,
            units="—",
            cost_usd=None,
            estimated=False,
        )
    metrics = meta.get("metrics") or {}
    tokens = (metrics.get("prompt_tokens"), metrics.get("completion_tokens"))
    parts = [f"{metrics.get('llm_calls', 0)} LLM call(s)"]
    if all(tokens):
        parts.append(f"{tokens[0]:,} in / {tokens[1]:,} out tokens")
    if metrics.get("sec_ingestion_ms") is not None:
        parts.append(f"SEC {format_ms(metrics['sec_ingestion_ms'])}")
    if metrics.get("llm_ms") is not None:
        parts.append(f"LLM {format_ms(metrics['llm_ms'])}")
    return perf_event(
        "analysis",
        label=label,
        provider=f"OpenRouter · {meta.get('model') or 'LLM'}",
        latency_ms=latency_ms,
        units=" · ".join(parts),
        cost_usd=metrics.get("cost_usd"),
        estimated=False,
    )


# --------------------------------------------------------------------------- #
# Callback logic: plain functions (the Dash wrappers below only read ``ctx``)
# --------------------------------------------------------------------------- #
def analysis_payload(
    mode: str, ticker: str | None, accession: str | None, filings: list[dict] | None
) -> dict[str, Any] | None:
    """Request body for the selected configuration; ``None`` if incomplete."""

    if mode != "Real":
        return dict(DEMO_PAYLOAD)
    selected = next((f for f in filings or [] if f["accession"] == accession), None)
    if ticker is None or selected is None:
        return None
    return {
        "ticker": ticker,
        "filing_date": selected["filing_date"],
        "filing_type": selected["form"],
        "mode": "real",
    }


def run_analysis_logic(
    trigger: str | None,
    mode: str,
    ticker: str | None,
    accession: str | None,
    filings: list[dict] | None,
    last_request: dict | None,
) -> tuple[Any, ...]:
    """Run one analysis and reset every piece of session state tied to a report.

    Outputs: handoff, error, request, perf, chat, chat audio, summary audio,
    chat error, sidebar warning.
    """

    if trigger == "analyze":
        payload = analysis_payload(mode, ticker, accession, filings)
    elif trigger == "retry-analysis":
        payload = last_request
    else:  # first visit: run the demo once so the dashboard is never empty
        payload = dict(DEMO_PAYLOAD)
    if payload is None:
        warning = "Select an available filing before running analysis."
        return (*[no_update] * 8, warning)

    started = time.perf_counter()
    response, error = api_request("POST", "/api/v1/analysis", json=payload, timeout=300)
    elapsed_ms = (time.perf_counter() - started) * 1000
    handoff = response.json() if response is not None else None
    perf = [analysis_event(handoff, elapsed_ms)] if handoff else []
    return handoff, error, payload, perf, [], {}, None, None, ""


def filings_view_logic(
    mode: str, ticker: str | None, filing_type: str, trigger: str | None
) -> tuple[Any, ...]:
    """Outputs: options, value, filings, message, retry button class."""

    if mode != "Real" or not ticker:
        raise PreventUpdate
    if trigger == "retry-filings":
        fetch_filings.cache_clear()
    filings, error = fetch_filings(ticker, filing_type)
    hidden = "btn btn--ghost hidden"
    if error is not None:
        copy = error_presentation(str(error.get("code") or "UNKNOWN_ERROR"))
        message = notice("error", html.Strong(copy.title), caption(copy.guidance))
        return [], None, [], message, "btn btn--ghost" if error.get("retryable") else hidden
    if not filings:
        message = notice("info", "No filings were found for this ticker and filing type.")
        return [], None, [], message, hidden
    options = [{"label": filing_option_label(f), "value": f["accession"]} for f in filings]
    return options, filings[0]["accession"], filings, None, hidden


def summary_listen_logic(
    handoff: dict, voice: str | None, perf: list[dict]
) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    response, error = api_request(
        "POST",
        "/api/v1/audio/summary",
        json={"text": handoff["executive_summary"], "voice": voice or DEFAULT_VOICES[0]},
        timeout=300,
    )
    if response is None:
        return {"error": error}, perf
    event = tts_event(response, (time.perf_counter() - started) * 1000, "Executive summary")
    audio = {
        "src": _audio_uri(response.content),
        "mode": handoff["pipeline_metadata"]["analysis_mode"],
    }
    return audio, [*perf, event]


def _audio_uri(content: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(content).decode("ascii")


def chat_submit_logic(
    trigger: Any,
    text: str | None,
    voice: dict | None,
    messages: list[dict],
    perf: list[dict],
) -> tuple[Any, ...]:
    """Add the user's question to the history (transcribing a voice one first).

    Outputs: chat, chat error, perf, chat input, voice status, ask token.
    """

    if messages and messages[-1]["role"] == "user":
        raise PreventUpdate  # an answer is still in flight

    question: str | None = None
    if isinstance(trigger, dict) and trigger.get("type") == "chat-suggestion":
        question = CHAT_SUGGESTIONS[trigger["index"]]
    elif trigger == "st-voice":
        try:
            wav = base64.b64decode((voice or {}).get("wav", ""), validate=True)
        except (binascii.Error, ValueError):
            wav = b""
        body, error, latency_ms = transcribe_audio(wav) if wav else (None, {"code": "INPUT_ERROR"}, 0)
        if body is None:
            return no_update, error, no_update, no_update, "", no_update
        seconds = body.get("audio_seconds")
        perf = [
            *perf,
            perf_event(
                "stt",
                label=f"Voice question ({body.get('language', '?')})",
                provider="Groq · whisper-large-v3-turbo",
                latency_ms=latency_ms,
                units=f"{seconds:.1f} s audio" if seconds else "",
                cost_usd=body.get("cost_usd"),
                estimated=False,
            ),
        ]
        question = str(body.get("text") or "").strip() or None
    else:  # send button or Enter
        question = (text or "").strip() or None
    if question is None:
        if trigger == "st-voice":
            return no_update, {"code": "INPUT_ERROR"}, perf, no_update, "", no_update
        raise PreventUpdate

    return (
        [*messages, {"role": "user", "content": question}],
        None,
        perf,
        "",
        "",
        time.time_ns(),
    )


def chat_answer_logic(
    ask: Any, messages: list[dict], handoff: dict | None, perf: list[dict]
) -> tuple[Any, ...]:
    """Ask the chat API about the last user message. Outputs: chat, error, perf."""

    if not ask or not handoff or not messages or messages[-1]["role"] != "user":
        raise PreventUpdate
    result = chat_answer({"handoff": handoff, "messages": messages})
    if not result.text.strip():
        # Keep the history valid: it must end with an answered turn.
        error = result.error or {"code": "UNKNOWN_ERROR"}
        return messages[:-1], error, no_update
    metrics = result.metrics
    tokens = (metrics.get("prompt_tokens"), metrics.get("completion_tokens"))
    event = perf_event(
        "chat",
        label=messages[-1]["content"][:60],
        provider=f"OpenRouter · {metrics.get('model') or 'LLM'}",
        latency_ms=metrics.get("client_total_ms"),
        ttft_ms=metrics.get("client_ttft_ms"),
        units=f"{tokens[0]} in / {tokens[1]} out tokens" if all(tokens) else "",
        cost_usd=metrics.get("cost_usd"),
        estimated=False,
    )
    return [*messages, {"role": "assistant", "content": result.text}], None, [*perf, event]


def chat_listen_logic(
    index: int,
    provider: str,
    voice: str,
    messages: list[dict],
    audio_cache: dict,
    perf: list[dict],
) -> tuple[Any, ...]:
    """Synthesize one chat answer. Outputs: audio cache, error, perf."""

    if not 0 <= index < len(messages) or messages[index]["role"] != "assistant":
        raise PreventUpdate
    started = time.perf_counter()
    response, error = api_request(
        "POST",
        "/api/v1/audio/summary",
        json={
            "text": text_for_speech(messages[index]["content"])[:SPEECH_CHARS],
            "voice": voice,
            "provider": provider,
        },
        timeout=300,
    )
    if response is None:
        return no_update, {**(error or {}), "kind": "audio"}, no_update
    event = tts_event(response, (time.perf_counter() - started) * 1000, "Chat answer audio")
    used = response.headers.get("X-TTS-Provider", provider)
    note = f"Read with {CHAT_TTS_PROVIDERS.get(used, used)}"
    if response.headers.get("X-TTS-Voice"):
        note += f" · {response.headers['X-TTS-Voice']}"
    if response.headers.get("X-TTS-Fallback") == "groq_unavailable":
        note += " (Groq unavailable, e.g. free-tier daily limit)"
    elif used != provider:
        note += " (Groq has no Spanish voice)"
    cache = {**audio_cache, f"{index}:{provider}:{voice}": {"audio": _audio_uri(response.content), "note": note}}
    return cache, None, [*perf, event]


def chat_voices_logic(provider: str) -> tuple[list[str], str | None]:
    voices = fetch_voices(provider)
    if provider != "groq":
        voices = [v for v in voices if v[:1] in "ab"] or voices
    default = "af_heart" if "af_heart" in voices else (voices[0] if voices else None)
    return voices, default


def chat_panel_logic(
    trigger: str | None, handoff: dict | None, panel_class: str | None
) -> tuple[str, str, str]:
    """Open/close the floating chat. Outputs: panel, launcher and shell classes."""

    is_open = "chat-panel--open" in (panel_class or "")
    if trigger == "chat-fab":
        is_open = True
    elif trigger in ("chat-close", "st-handoff"):  # a new report starts a new chat
        is_open = False
    has_report = bool(handoff) and verification_label(handoff["verification"]) != "FAILED VERIFICATION"
    is_open = is_open and has_report
    return (
        "chat-panel chat-panel--open" if is_open else "chat-panel",
        "chat-fab" if has_report and not is_open else "chat-fab hidden",
        "app-shell app-shell--chat-open" if is_open else "app-shell",
    )


def projection_logic(
    events: list[dict], sessions: Any, turns: Any, audios: Any, voices: Any, minutes: Any
) -> Any:
    cloud_run = fetch_pricing().get("cloud_run") or {}
    hourly = float(cloud_run.get("usd_per_hour") or 0.288)
    projection = project_monthly_cost(
        events,
        sessions_per_month=int(sessions or 0),
        chat_turns=int(turns or 0),
        audio_plays=int(audios or 0),
        voice_questions=int(voices or 0),
        session_minutes=float(minutes or 1),
        instance_usd_per_hour=hourly,
    )
    return projection_view(projection, events, cloud_run, hourly)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _field(label: str, component: Any, field_id: str) -> html.Div:
    return html.Div(
        [html.Label(label, htmlFor=field_id, className="field-label"), component],
        className="field",
    )


def _sidebar() -> html.Aside:
    return html.Aside(
        [
            html.H2("Analysis configuration"),
            html.P("Data mode", className="field-label"),
            dcc.RadioItems(
                id="mode",
                options=["Demo", "Real"],
                value="Demo",
                inline=True,
                className="radio",
            ),
            html.Div(
                [
                    html.Span("DEMO | SYNTHETIC", className="badge badge--orange"),
                    caption("Deterministic fixture | No SEC or LLM request"),
                ],
                id="demo-box",
                className="mode-box",
            ),
            html.Div(
                [
                    caption("LIVE | SEC filing and configured OpenRouter model"),
                    _field(
                        "Ticker",
                        dcc.Dropdown(
                            id="ticker",
                            options=[
                                {"label": f"{t} · {name}", "value": t} for t, name in TICKERS.items()
                            ],
                            value="AAPL",
                            clearable=False,
                        ),
                        "ticker",
                    ),
                    _field(
                        "Filing type",
                        dcc.Dropdown(
                            id="filing-type",
                            options=["10-Q", "10-K"],
                            value="10-Q",
                            clearable=False,
                            searchable=False,
                        ),
                        "filing-type",
                    ),
                    dcc.Loading(
                        _field(
                            "SEC filing",
                            dcc.Dropdown(
                                id="filing-select",
                                options=[],
                                clearable=False,
                                searchable=False,
                                placeholder="Loading available SEC filings…",
                            ),
                            "filing-select",
                        ),
                        type="dot",
                        color="#38BDF8",
                    ),
                    html.Div(id="accession", className="caption"),
                    html.Div(id="filings-message"),
                    html.Button(
                        "Retry filing lookup", id="retry-filings", className="btn btn--ghost hidden"
                    ),
                ],
                id="real-box",
                className="mode-box hidden",
            ),
            html.Button("Analyze filing", id="analyze", className="btn btn--primary btn--block"),
            html.Div(id="sidebar-warning", className="notice-slot", role="status"),
        ],
        className="sidebar",
    )


def _chat_panel() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("Ask about this filing"),
                    html.Button("×", id="chat-close", className="icon-btn", title="Close chat", **{"aria-label": "Close chat"}),
                ],
                className="chat-head",
            ),
            html.Div(id="chat-caption", className="caption"),
            dcc.RadioItems(
                id="chat-tts-provider",
                options=[{"label": label, "value": key} for key, label in CHAT_TTS_PROVIDERS.items()],
                value="local",
                inline=True,
                className="radio",
            ),
            _field("English voice", dcc.Dropdown(id="chat-voice", options=DEFAULT_VOICES, value=DEFAULT_VOICES[0], clearable=False, searchable=False), "chat-voice"),
            caption(
                "Audio follows the answer's language: Spanish answers are read with a "
                "Kokoro Spanish voice (Groq has no Spanish voice)."
            ),
            html.Div(
                [
                    html.Button(
                        text,
                        id={"type": "chat-suggestion", "index": index},
                        className="btn btn--tertiary",
                    )
                    for index, text in enumerate(CHAT_SUGGESTIONS)
                ],
                id="chat-suggestions",
                className="suggestions",
            ),
            html.Div(id="chat-history", className="chat-history", role="log", **{"aria-live": "polite"}),
            html.Div(id="chat-error"),
            html.Div(
                [
                    dcc.Input(
                        id="chat-input",
                        type="text",
                        placeholder="Ask about this filing…",
                        className="input",
                        autoComplete="off",
                        debounce=False,
                        n_submit=0,
                    ),
                    html.Button("Record", id="chat-mic", className="btn btn--ghost", title="Record a voice question", **{"aria-pressed": "false"}),
                    html.Button("Send", id="chat-send", className="btn btn--primary"),
                ],
                className="chat-compose",
            ),
            html.Div(id="chat-voice-status", className="caption", role="status"),
        ],
        id="chat-panel",
        className="chat-panel",
        role="complementary",
        **{"aria-label": "Filing chat"},
    )


def serve_layout() -> html.Div:
    """Fresh layout per page load: every visit starts with a clean session."""

    return html.Div(
        [
            _sidebar(),
            html.Main(
                [
                    html.H1("Financial Intelligence Copilot"),
                    caption("SEC filings | Grounded AI | Deterministic verification"),
                    html.Div(
                        [
                            html.P("The selected configuration is preserved. Retry when ready."),
                            html.Button("Retry analysis", id="retry-analysis", className="btn btn--primary"),
                        ],
                        id="retry-wrap",
                        className="retry-wrap hidden",
                    ),
                    dcc.Loading(
                        [html.Div(id="result-area"), dcc.Store(id="st-handoff")],
                        target_components={"st-handoff": "data"},
                        custom_spinner=html.Div(
                            [
                                html.Div(className="spinner"),
                                html.P("Preparing analysis", className="spinner-title"),
                                caption(
                                    "The backend retrieves the filing, processes financial data, "
                                    "generates a grounded analysis, and applies deterministic "
                                    "verification."
                                ),
                            ],
                            className="loading-card",
                        ),
                        overlay_style={"visibility": "visible", "opacity": 0.35},
                        className="result-loading",
                    ),
                ],
                className="main",
            ),
            html.Button("Ask about this filing", id="chat-fab", className="chat-fab hidden"),
            _chat_panel(),
            dcc.Store(id="st-error"),
            dcc.Store(id="st-request"),
            dcc.Store(id="st-perf", data=[]),
            dcc.Store(id="st-filings", data=[]),
            dcc.Store(id="st-chat", data=[]),
            dcc.Store(id="st-chat-audio", data={}),
            dcc.Store(id="st-chat-error"),
            dcc.Store(id="st-chat-ask"),
            dcc.Store(id="st-summary-audio"),
            dcc.Store(id="st-voice"),
            dcc.Store(id="chat-scroll-sink"),
        ],
        id="app-shell",
        className="app-shell",
    )


app = Dash(
    __name__,
    title="Financial Intelligence Copilot",
    assets_folder=ASSETS_FOLDER,
    external_stylesheets=[FONT_URL],
    suppress_callback_exceptions=True,  # the report and its widgets are built per analysis
    update_title=None,
)
app.layout = serve_layout
server = app.server


@server.route("/_health")
def health() -> tuple[str, int]:
    """Liveness probe for CI and Cloud Run smoke checks."""

    return "ok", 200


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@app.callback(
    Output("demo-box", "className"),
    Output("real-box", "className"),
    Input("mode", "value"),
)
def toggle_mode(mode: str) -> tuple[str, str]:
    real = mode == "Real"
    return ("mode-box hidden" if real else "mode-box", "mode-box" if real else "mode-box hidden")


@app.callback(
    Output("filing-select", "options"),
    Output("filing-select", "value"),
    Output("st-filings", "data"),
    Output("filings-message", "children"),
    Output("retry-filings", "className"),
    Input("mode", "value"),
    Input("ticker", "value"),
    Input("filing-type", "value"),
    Input("retry-filings", "n_clicks"),
    prevent_initial_call=True,
)
def load_filings(mode, ticker, filing_type, _retry):
    return filings_view_logic(mode, ticker, filing_type, ctx.triggered_id)


@app.callback(
    Output("accession", "children"),
    Input("filing-select", "value"),
)
def show_accession(accession: str | None) -> str:
    return f"Accession {accession}" if accession else ""


@app.callback(
    Output("st-handoff", "data"),
    Output("st-error", "data"),
    Output("st-request", "data"),
    Output("st-perf", "data"),
    Output("st-chat", "data"),
    Output("st-chat-audio", "data"),
    Output("st-summary-audio", "data"),
    Output("st-chat-error", "data"),
    Output("sidebar-warning", "children"),
    Input("analyze", "n_clicks"),
    Input("retry-analysis", "n_clicks"),
    State("mode", "value"),
    State("ticker", "value"),
    State("filing-select", "value"),
    State("st-filings", "data"),
    State("st-request", "data"),
)
def run_analysis(_analyze, _retry, mode, ticker, accession, filings, last_request):
    outputs = run_analysis_logic(ctx.triggered_id, mode, ticker, accession, filings, last_request)
    warning = outputs[-1]
    if warning:
        return (*outputs[:-1], notice("warning", warning))
    return outputs


@app.callback(
    Output("result-area", "children"),
    Output("retry-wrap", "className"),
    Input("st-handoff", "data"),
    Input("st-error", "data"),
)
def render_result(handoff: dict | None, error: dict | None):
    if handoff:
        return build_result(handoff, fetch_voices()), "retry-wrap hidden"
    if error:
        retryable = bool(error.get("retryable"))
        return error_block(error), "retry-wrap" if retryable else "retry-wrap hidden"
    info = notice("info", "Configure an analysis in the sidebar, then select ", html.Strong("Analyze filing"), ".")
    return info, "retry-wrap hidden"


# --- Narrative: executive summary audio ------------------------------------ #
@app.callback(
    Output("st-summary-audio", "data", allow_duplicate=True),
    Output("st-perf", "data", allow_duplicate=True),
    Input("summary-listen", "n_clicks"),
    State("st-handoff", "data"),
    State("summary-voice", "value"),
    State("st-perf", "data"),
    prevent_initial_call=True,
)
def summary_listen(clicks, handoff, voice, perf):
    if not clicks or not handoff:
        raise PreventUpdate
    return summary_listen_logic(handoff, voice, perf or [])


@app.callback(
    Output("summary-audio-box", "children"),
    Input("st-summary-audio", "data"),
)
def show_summary_audio(audio):
    return summary_audio_view(audio)


app.clientside_callback(
    """
    function (clicks, handoff) {
        if (!clicks || !handoff) { return window.dash_clientside.no_update; }
        return navigator.clipboard.writeText(handoff.executive_summary).then(
            () => "Summary copied to clipboard.",
            () => "Copy failed. Select the summary text and copy it manually."
        );
    }
    """,
    Output("summary-copy-status", "children"),
    Input("summary-copy", "n_clicks"),
    State("st-handoff", "data"),
    prevent_initial_call=True,
)


# --- Performance tab ----------------------------------------------------- #
@app.callback(
    Output("perf-summary", "children"),
    Input("st-perf", "data"),
    Input("result-tabs", "value"),
)
def render_perf_summary(events, _tab):
    return perf_summary_view(events or [])


@app.callback(
    Output("perf-projection", "children"),
    Input("st-perf", "data"),
    Input("result-tabs", "value"),
    Input("proj-sessions", "value"),
    Input("proj-turns", "value"),
    Input("proj-audios", "value"),
    Input("proj-voices", "value"),
    Input("proj-minutes", "value"),
)
def render_projection(events, _tab, sessions, turns, audios, voices, minutes):
    return projection_logic(events or [], sessions, turns, audios, voices, minutes)


# --- Floating chat ----------------------------------------------------------- #
@app.callback(
    Output("chat-panel", "className"),
    Output("chat-fab", "className"),
    Output("app-shell", "className"),
    Input("chat-fab", "n_clicks"),
    Input("chat-close", "n_clicks"),
    Input("st-handoff", "data"),
    State("chat-panel", "className"),
)
def toggle_chat(_open, _close, handoff, panel_class):
    return chat_panel_logic(ctx.triggered_id, handoff, panel_class)


@app.callback(Output("chat-caption", "children"), Input("st-handoff", "data"))
def show_chat_caption(handoff):
    return chat_caption(handoff)


@app.callback(
    Output("chat-voice", "options"),
    Output("chat-voice", "value"),
    Input("chat-tts-provider", "value"),
)
def update_chat_voices(provider: str):
    return chat_voices_logic(provider)


@app.callback(
    Output("st-chat", "data", allow_duplicate=True),
    Output("st-chat-error", "data", allow_duplicate=True),
    Output("st-perf", "data", allow_duplicate=True),
    Output("chat-input", "value"),
    Output("chat-voice-status", "children"),
    Output("st-chat-ask", "data"),
    Input("chat-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    Input({"type": "chat-suggestion", "index": ALL}, "n_clicks"),
    Input("st-voice", "data"),
    State("chat-input", "value"),
    State("st-chat", "data"),
    State("st-perf", "data"),
    prevent_initial_call=True,
)
def chat_submit(_send, _enter, _suggestions, voice, text, messages, perf):
    trigger = ctx.triggered_id
    # Suggestion buttons are static, but a real click is the only non-empty value.
    if isinstance(trigger, dict) and not ctx.triggered[0]["value"]:
        raise PreventUpdate
    return chat_submit_logic(trigger, text, voice, messages or [], perf or [])


@app.callback(
    Output("st-chat", "data", allow_duplicate=True),
    Output("st-chat-error", "data", allow_duplicate=True),
    Output("st-perf", "data", allow_duplicate=True),
    Input("st-chat-ask", "data"),
    State("st-chat", "data"),
    State("st-handoff", "data"),
    State("st-perf", "data"),
    prevent_initial_call=True,
)
def answer_chat(ask, messages, handoff, perf):
    return chat_answer_logic(ask, messages or [], handoff, perf or [])


@app.callback(
    Output("st-chat-audio", "data", allow_duplicate=True),
    Output("st-chat-error", "data", allow_duplicate=True),
    Output("st-perf", "data", allow_duplicate=True),
    Input({"type": "chat-listen", "index": ALL}, "n_clicks"),
    State("chat-tts-provider", "value"),
    State("chat-voice", "value"),
    State("st-chat", "data"),
    State("st-chat-audio", "data"),
    State("st-perf", "data"),
    prevent_initial_call=True,
)
def listen_chat(_clicks, provider, voice, messages, audio_cache, perf):
    trigger = ctx.triggered_id
    # Re-rendered buttons report n_clicks=None: only a real click has a value.
    if not isinstance(trigger, dict) or not ctx.triggered[0]["value"]:
        raise PreventUpdate
    return chat_listen_logic(
        trigger["index"], provider, voice or DEFAULT_VOICES[0], messages or [], audio_cache or {}, perf or []
    )


@app.callback(
    Output("chat-history", "children"),
    Output("chat-suggestions", "className"),
    Output("chat-error", "children"),
    Input("st-chat", "data"),
    Input("st-chat-audio", "data"),
    Input("st-chat-error", "data"),
    Input("chat-tts-provider", "value"),
    Input("chat-voice", "value"),
    State("st-handoff", "data"),
)
def render_chat(messages, audio_cache, error, provider, voice, handoff):
    messages = messages or []
    history = chat_history(messages, audio_cache or {}, provider, voice, handoff) if handoff else []
    return history, "suggestions hidden" if messages else "suggestions", chat_error_view(error)


app.clientside_callback(
    """
    function (children) {
        const history = document.getElementById("chat-history");
        if (history) {
            requestAnimationFrame(() => { history.scrollTop = history.scrollHeight; });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-scroll-sink", "data"),
    Input("chat-history", "children"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
        debug=os.getenv("DASH_DEBUG", "").lower() in {"1", "true"},
    )
