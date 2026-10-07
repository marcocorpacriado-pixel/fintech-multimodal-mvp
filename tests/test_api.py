from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app
from src.extraction import (
    LLMTransportError,
    ModelOutputProblem,
    ModelOutputRejectedError,
    PipelineInputError,
    SECFilingMetadata,
    SECFilingNotFoundError,
    SECServiceError,
)

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_demo_success():
    response = client.post(
        "/api/v1/analysis",
        json={"ticker": "AAPL", "mode": "demo"},
    )

    assert response.status_code == 200
    body = response.json()
    for key in (
        "company",
        "ticker",
        "financial_metrics",
        "executive_summary",
        "pipeline_metadata",
    ):
        assert key in body
    assert body["pipeline_metadata"]["analysis_mode"] == "demo"
    assert {m["comparison_type"] for m in body["financial_metrics"]} == {
        "QoQ",
        "YoY_YTD",
        None,
    }


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (PipelineInputError("bad filing path C:/secret"), 422, "INPUT_ERROR"),
        (LLMTransportError("provider body sk-or-SECRET"), 503, "LLM_PROVIDER_ERROR"),
        (RuntimeError("boom sk-or-SECRET"), 500, "UNKNOWN_ERROR"),
    ],
)
def test_analysis_error_handling(error, status, code):
    with patch("src.api.main._run_real_analysis", side_effect=error):
        response = client.post(
            "/api/v1/analysis",
            json={
                "ticker": "AAPL",
                "filing_date": "2026-07-31",
                "mode": "real",
            },
        )

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "SECRET" not in response.text
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_analysis_logs_safe_request_diagnostic_without_raw_grounding_text(
    caplog,
):
    raw = "unknown evidence_id: 'chunk:private-secret'"
    rejection = ModelOutputRejectedError(
        [ModelOutputProblem("key_risks[0].evidence_id", "INVALID_EVIDENCE_ID", raw)]
    )
    with patch("src.api.main._run_real_analysis", side_effect=rejection):
        response = client.post(
            "/api/v1/analysis",
            json={
                "ticker": "AAPL",
                "filing_date": "2026-07-31",
                "mode": "real",
            },
        )

    assert response.status_code == 422
    assert "request_id=" in caplog.text
    assert "category=GROUNDING_ERROR" in caplog.text
    assert "reason=INVALID_EVIDENCE_ID" in caplog.text
    assert "private-secret" not in caplog.text


def test_analysis_real_mode_rejects_legacy_period_field():
    response = client.post(
        "/api/v1/analysis",
        json={"ticker": "AAPL", "period": "2026-07-31", "mode": "real"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "aapl", "BRK.B", "BRK-B"])
def test_analysis_accepts_and_normalizes_valid_sec_tickers(ticker):
    handoff = api_main._run_demo_analysis()
    with patch("src.api.main._run_real_analysis", return_value=handoff) as run:
        response = client.post(
            "/api/v1/analysis",
            json={
                "ticker": ticker,
                "filing_date": "2026-07-31",
                "mode": "real",
            },
        )

    assert response.status_code == 200
    assert run.call_args.args[0].ticker == ticker.strip().upper()


@pytest.mark.parametrize(
    "ticker",
    ["", "   ", "@@@", "AAPL$", "AAPL INC", "A" * 16],
)
def test_invalid_ticker_is_rejected_before_sec_or_llm(ticker):
    with patch("src.api.main.prepare_sec_analysis_inputs") as sec, patch(
        "src.api.main.OpenRouterLLMClient.from_env"
    ) as llm:
        response = client.post(
            "/api/v1/analysis",
            json={
                "ticker": ticker,
                "filing_date": "2026-07-31",
                "mode": "real",
            },
        )

    assert response.status_code == 422
    sec.assert_not_called()
    llm.assert_not_called()


def test_real_mode_requires_filing_date():
    response = client.post(
        "/api/v1/analysis",
        json={"ticker": "AAPL", "mode": "real"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "code", "retryable"),
    [
        (SECFilingNotFoundError("missing secret"), 404, "FILING_NOT_FOUND", False),
        (SECServiceError("network secret"), 503, "SEC_INGESTION_ERROR", True),
    ],
)
def test_sec_errors_have_stable_http_semantics(error, status, code, retryable):
    with patch("src.api.main._run_real_analysis", side_effect=error):
        response = client.post(
            "/api/v1/analysis",
            json={
                "ticker": "AAPL",
                "filing_date": "2099-01-01",
                "mode": "real",
            },
        )

    assert response.status_code == status
    assert response.json()["detail"] == {
        "code": code,
        "message": response.json()["detail"]["message"],
        "retryable": retryable,
    }
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_filing_discovery_returns_lightweight_metadata_without_analysis():
    discovered = [
        SECFilingMetadata(
            ticker="AAPL",
            company="Apple Inc.",
            filing_date=date(2026, 7, 31),
            report_date=date(2026, 6, 27),
            form="10-Q",
            accession="0000320193-26-000020",
        )
    ]
    with patch("src.api.main.discover_sec_filings", return_value=discovered) as call:
        response = client.get(
            "/api/v1/filings/aapl",
            params={"filing_type": "10-Q", "limit": 5},
        )

    assert response.status_code == 200
    assert response.json() == [
        {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "filing_date": "2026-07-31",
            "report_date": "2026-06-27",
            "form": "10-Q",
            "accession": "0000320193-26-000020",
        }
    ]
    call.assert_called_once_with(ticker="aapl", form="10-Q", limit=5)


def test_real_handoff_separates_filing_date_from_report_period():
    result = api_main.AnalysisPipelineResult.model_validate_json(
        api_main.DEMO_FIXTURE_PATH.read_text(encoding="utf-8")
    )
    result.analysis.period = "2026-06-27"
    prepared = SimpleNamespace(
        filing_path="filing.txt",
        company="Apple Inc.",
        ticker="AAPL",
        filing_date=date(2026, 7, 31),
        report_period="2026-06-27",
        filing_type="10-Q",
        current_accession="0000320193-26-000020",
        current_filing=object(),
        previous_filing=object(),
    )
    llm = SimpleNamespace(model="offline-model")
    context = MagicMock()
    context.__enter__.return_value = llm
    request = api_main.AnalysisRequest(
        ticker="aapl",
        filing_date=date(2026, 7, 31),
        filing_type="10-Q",
        mode="real",
    )

    with patch(
        "src.api.main.prepare_sec_analysis_inputs", return_value=prepared
    ) as prepare, patch(
        "src.api.main.OpenRouterLLMClient.from_env", return_value=context
    ), patch("src.api.main.run_analysis_pipeline", return_value=result) as pipeline:
        handoff = api_main._run_real_analysis(request)

    prepare.assert_called_once_with(
        ticker="AAPL", filing_date=date(2026, 7, 31), form="10-Q"
    )
    assert pipeline.call_args.kwargs["period"] == "2026-06-27"
    assert handoff.period == "2026-06-27"
    assert handoff.pipeline_metadata.filing_date == date(2026, 7, 31)


def test_audio_summary_endpoint():
    fake = SimpleNamespace(audio_bytes=b"RIFFmockwavdata")
    with patch("src.api.main.synthesize", return_value=fake) as mock_synth:
        response = client.post(
            "/api/v1/audio/summary", json={"text": "Hello", "voice": "af_heart"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFmockwavdata"
    mock_synth.assert_called_once_with(text="Hello", voice="af_heart")


def test_voices_falls_back_when_kokoro_unavailable():
    with patch("src.api.main.list_voices", side_effect=RuntimeError("no model")):
        response = client.get("/api/v1/audio/voices")

    assert response.status_code == 200
    assert response.json() == {"voices": ["af_heart"]}
