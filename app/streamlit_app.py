"""Streamlit dashboard. Talks to the FastAPI backend over HTTP only.

Never import src.extraction, src.integration or src.audio here: every value
shown comes from the API's AnalysisHandoff JSON, rendered without recalculation.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` only puts app/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.financial_charts import (  # noqa: E402
    NOT_AVAILABLE,
    format_change_pct,
    format_value,
    render_metrics_comparison_chart,
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
# Base tokens live in .streamlit/config.toml; this only covers what the theme can't (DESIGN.md).
TERMINAL_CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 4rem; padding-bottom: 1.5rem; }
[data-testid="stMetricLabel"] p {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; color: #94A3B8;
}
[data-testid="stMetricValue"] { font-weight: 700; }
.st-key-report_header {
    position: sticky; top: 3.75rem; z-index: 99;
    background: #0B0F19; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;
}
.st-key-report_header h1 { padding: 0.25rem 0 0; }
</style>
"""


def md_escape(text: str) -> str:
    """Filing text is untrusted: stop `$`, `*`, `#`... from becoming markdown/LaTeX."""

    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<])", r"\\\1", text)


def show_api_error(response: httpx.Response) -> None:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict) and "code" in detail:
        st.error(f"**{detail['code']}** — {md_escape(detail['message'])}")
        if detail.get("retryable"):
            st.caption("Error transitorio: puedes volver a intentarlo.")
    elif response.status_code == 422:
        st.error("La petición no es válida. Revisa los datos introducidos.")
    else:
        st.error(f"Error inesperado de la API (HTTP {response.status_code}).")


def api_request(method: str, path: str, **kwargs) -> httpx.Response | None:
    """One HTTP call to the API; renders a clean error and returns None on failure."""

    try:
        response = httpx.request(method, f"{API_URL}{path}", **kwargs)
    except httpx.HTTPError:
        st.error(f"No se pudo conectar con la API en {API_URL}. ¿Está arrancada?")
        return None
    if response.is_success:
        return response
    show_api_error(response)
    return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_voices() -> list[str]:
    try:
        response = httpx.request("GET", f"{API_URL}/api/v1/audio/voices", timeout=30)
        return response.json()["voices"] if response.is_success else DEFAULT_VOICES
    except (httpx.HTTPError, ValueError, KeyError):
        return DEFAULT_VOICES


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_catalog() -> list[dict]:
    response = httpx.request("GET", f"{API_URL}/api/v1/filings/catalog", timeout=10)
    response.raise_for_status()
    return response.json()["companies"]


def fetch_catalog() -> list[dict]:
    """SEC-verified tickers/filings; empty on failure (exceptions are not cached)."""

    try:
        return _fetch_catalog()
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def render_sidebar() -> tuple[dict | None, bool]:
    with st.sidebar:
        st.header("Análisis")
        mode = st.radio("Modo", ["demo", "real"], horizontal=True)
        payload: dict | None
        if mode == "real":
            payload = None
            companies = fetch_catalog()
            if not companies:
                st.warning("No se pudo cargar el catálogo de filings de la API.")
            else:
                company = st.selectbox(
                    "Ticker", companies, format_func=lambda c: f"{c['ticker']} · {c['company']}"
                )
                filing_type = st.selectbox("Filing Type", ["10-Q", "10-K"])
                filing = st.selectbox(
                    "Filing",
                    company["filings"].get(filing_type, []),
                    format_func=lambda f: (
                        f"{f['filing_date']} · {f['period']} (cierre {f['period_end']})"
                    ),
                )
                if filing:
                    payload = {
                        "ticker": company["ticker"],
                        "period": filing["filing_date"],
                        "filing_type": filing_type,
                        "mode": "real",
                    }
        else:
            st.info(
                "Modo demo: la API devuelve un fixture sintético (Demo Corp). "
                "No se consulta la SEC ni ningún LLM."
            )
            payload = {"ticker": "DEMO", "period": "demo", "mode": "demo"}

        run = st.button("Ejecutar Análisis", type="primary", width="stretch")
    return payload, run


def render_header(handoff: dict) -> None:
    meta = handoff["pipeline_metadata"]
    with st.container(key="report_header"):
        if meta["analysis_mode"] == "real":
            st.badge("REAL MODE", icon=":material/verified:", color="green")
        else:
            st.badge("DEMO MODE (SYNTHETIC DATA)", icon=":material/science:", color="orange")
        st.title(f"{md_escape(handoff['company'])} ({handoff['ticker']})")
        st.caption(f"{handoff['filing_type']} · Periodo {handoff['period']}")
    if meta["analysis_mode"] != "real":
        st.warning("Datos sintéticos de demostración: no describen ninguna empresa real.")


def render_metrics(metrics: list[dict]) -> None:
    st.subheader("Métricas financieras")
    if not metrics:
        st.info("El análisis no devolvió métricas comparables.")
        return
    columns = st.columns(4)
    for index, metric in enumerate(metrics):
        unit = metric["unit"]
        change = metric["change_pct"]
        with columns[index % 4].container(border=True):
            st.metric(
                metric["name"],
                format_value(metric["current_value"], unit),
                delta=None if change is None else format_change_pct(change),
                delta_color="off",
            )
            st.badge(
                metric["comparison_type"] or "Sin comparación",
                color="blue" if metric["comparison_type"] else "gray",
            )
            st.caption(
                f"Anterior: {format_value(metric['previous_value'], unit)} · "
                f"Δ {format_change_pct(change)}"
            )
            st.caption(
                f"{metric['current_period'] or NOT_AVAILABLE} vs "
                f"{metric['previous_period'] or NOT_AVAILABLE}"
            )

    st.plotly_chart(render_metrics_comparison_chart(metrics))
    with st.expander("Ver tabla de métricas"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Métrica": m["name"],
                    "Actual": format_value(m["current_value"], m["unit"]),
                    "Anterior": format_value(m["previous_value"], m["unit"]),
                    "Cambio": format_change_pct(m["change_pct"]),
                    "Comparación": m["comparison_type"] or NOT_AVAILABLE,
                    "Periodo actual": m["current_period"] or NOT_AVAILABLE,
                    "Periodo anterior": m["previous_period"] or NOT_AVAILABLE,
                }
                for m in metrics
            ),
            hide_index=True,
        )


def render_findings(handoff: dict) -> None:
    left, right = st.columns(2)
    sections = (
        (left, ":green[✅ Drivers positivos]", handoff["positives"]),
        (right, ":orange[⚠️ Riesgos]", handoff["risks"]),
    )
    for column, title, items in sections:
        with column:
            st.subheader(title)
            if not items:
                st.caption("Sin hallazgos con evidencia suficiente.")
            for item in items:
                with st.container(border=True):
                    st.markdown(f"**{md_escape(item['finding'])}**")
                    st.markdown(f"> {md_escape(item['evidence'])}")
                    st.caption(
                        f"Sección: {item['source_section'] or NOT_AVAILABLE} · "
                        f"Fuente: {item['source_type']} · {item['source_id'] or NOT_AVAILABLE}"
                    )


def render_outlook(outlook: dict) -> None:
    st.subheader("Management Outlook")
    sentiment = outlook["sentiment"]
    st.badge(f"Sentiment: {sentiment}", color=SENTIMENT_COLORS.get(sentiment, "gray"))
    st.markdown(md_escape(outlook["summary"]))


def render_summary(handoff: dict) -> None:
    with st.container(border=True):
        st.subheader("Resumen ejecutivo")
        st.markdown(md_escape(handoff["executive_summary"]))

    with st.container(border=True):
        st.markdown("**🎧 Briefing en audio**")
        voices = fetch_voices()
        voice_col, button_col = st.columns([2, 1], vertical_alignment="bottom")
        voice = voice_col.selectbox(
            "Voz (Kokoro)", voices, index=voices.index("af_heart") if "af_heart" in voices else 0
        )
        if button_col.button("Generar audio", type="primary", width="stretch"):
            with st.spinner("Sintetizando audio (la primera vez puede descargar el modelo)..."):
                response = api_request(
                    "POST",
                    "/api/v1/audio/summary",
                    json={"text": handoff["executive_summary"], "voice": voice},
                    timeout=300,
                )
            if response is not None:
                st.session_state.audio = response.content
        if "audio" in st.session_state:
            st.audio(st.session_state.audio, format="audio/wav")
            if handoff["pipeline_metadata"]["analysis_mode"] == "demo":
                st.caption("Audio generado a partir de un análisis DEMO con datos sintéticos.")


def render_compliance(handoff: dict) -> None:
    verification = handoff["verification"]
    st.subheader("Verificación determinista")
    if verification["valid"]:
        st.success("El análisis superó la verificación determinista.")
    else:
        st.error("El análisis no superó la verificación determinista.")
    for issue in verification["issues"]:
        show = st.warning if issue["severity"] == "warning" else st.error
        show(f"**{issue['code']}** · `{issue['field']}` — {md_escape(issue['message'])}")

    meta = handoff["pipeline_metadata"]
    with st.container(border=True):
        st.markdown("**Metadatos del pipeline**")
        st.caption(
            f"Modo: {meta['analysis_mode']} · Proveedor: {meta.get('provider') or NOT_AVAILABLE}"
            f" · Modelo: {meta.get('model') or NOT_AVAILABLE}"
            f" · Chunks recuperados: {meta['retrieval_count']}"
        )
        if meta["effective_queries"] or meta["retrieved_source_ids"]:
            with st.expander("Consultas y fuentes recuperadas"):
                st.caption("Consultas: " + md_escape(", ".join(meta["effective_queries"])))
                st.caption("Fuentes: " + md_escape(", ".join(meta["retrieved_source_ids"])))


def main() -> None:
    st.set_page_config(page_title="Fintech Multimodal", page_icon="📊", layout="wide")
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)  # static constant, no user data
    payload, run = render_sidebar()

    if run:
        st.session_state.pop("handoff", None)
        st.session_state.pop("audio", None)
        if payload is None:
            st.sidebar.warning("Selecciona ticker, tipo y filing del catálogo.")
        else:
            with st.spinner("Ejecutando análisis..."):
                response = api_request("POST", "/api/v1/analysis", json=payload, timeout=300)
            if response is not None:
                st.session_state.handoff = response.json()

    handoff = st.session_state.get("handoff")
    if handoff is None:
        st.title("Fintech Multimodal")
        st.caption("Configura el análisis en la barra lateral y pulsa «Ejecutar Análisis».")
        return

    render_header(handoff)
    summary_tab, metrics_tab, findings_tab, compliance_tab = st.tabs(
        [
            "🎙️ Resumen Ejecutivo & Audio",
            "📊 Desglose Financiero & Gráficos",
            "⚖️ Drivers & Riesgos",
            "🛡️ Compliance & Verificación",
        ]
    )
    with summary_tab:
        render_summary(handoff)
    with metrics_tab:
        render_metrics(handoff["financial_metrics"])
    with findings_tab:
        render_findings(handoff)
        render_outlook(handoff["management_outlook"])
    with compliance_tab:
        render_compliance(handoff)


main()
