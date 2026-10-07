"""Suite-wide guards for hermetic tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_finbert_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the LLM outlook sentiment unless a test injects a FinBERT result."""

    monkeypatch.setattr(
        "src.extraction.pipeline.classify_financial_sentiment", lambda text: None
    )
