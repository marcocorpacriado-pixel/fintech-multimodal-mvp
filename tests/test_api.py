from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.extraction import LLMTransportError, PipelineInputError

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_demo_success():
    response = client.post(
        "/api/v1/analysis",
        json={"ticker": "AAPL", "period": "2026-06-27", "mode": "demo"},
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
            json={"ticker": "AAPL", "period": "2026-06-27", "mode": "real"},
        )

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "SECRET" not in response.text
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_analysis_real_mode_rejects_non_iso_period():
    response = client.post(
        "/api/v1/analysis",
        json={"ticker": "AAPL", "period": "Q2 2026", "mode": "real"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INPUT_ERROR"


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
