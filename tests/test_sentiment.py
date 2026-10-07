"""FinBERT outlook sentiment, with the Hugging Face pipeline mocked (no downloads)."""

import pytest

from src.extraction import pipeline, sentiment
from src.extraction.schemas import ManagementOutlook


def _fake_classifier(label: str, score: float):
    calls: list[str] = []

    def classify(text: str, truncation: bool):
        assert truncation is True
        calls.append(text)
        return [{"label": label, "score": score}]

    return classify, calls


def test_classifier_returns_label_confidence_and_model(monkeypatch):
    classify, calls = _fake_classifier("positive", 0.942)
    monkeypatch.setattr(sentiment, "_classifier", lambda: classify)

    assert sentiment.classify_financial_sentiment("Demand should keep growing.") == {
        "sentiment": "positive",
        "confidence": 0.942,
        "model": "ProsusAI/finbert",
    }
    assert calls == ["Demand should keep growing."]


@pytest.mark.parametrize("label", ["Negative", "NEUTRAL"])
def test_labels_are_normalized(monkeypatch, label):
    classify, _ = _fake_classifier(label, 0.8)
    monkeypatch.setattr(sentiment, "_classifier", lambda: classify)

    assert sentiment.classify_financial_sentiment("text")["sentiment"] == label.lower()


def test_load_failure_falls_back_to_none(monkeypatch):
    def offline():
        raise OSError("no network to huggingface.co")

    monkeypatch.setattr(sentiment, "_classifier", offline)

    assert sentiment.classify_financial_sentiment("Outlook text.") is None


@pytest.mark.parametrize(("label", "score"), [("mixed", 0.9), ("positive", 1.7)])
def test_unexpected_output_falls_back_to_none(monkeypatch, label, score):
    classify, _ = _fake_classifier(label, score)
    monkeypatch.setattr(sentiment, "_classifier", lambda: classify)

    assert sentiment.classify_financial_sentiment("text") is None


def test_blank_text_never_loads_the_model(monkeypatch):
    monkeypatch.setattr(sentiment, "_classifier", pytest.fail)

    assert sentiment.classify_financial_sentiment("   ") is None


def _outlook(sentiment_label="positive", source_ids=("filing:chunk-1",)):
    return ManagementOutlook(
        summary="Management expects demand to remain strong.",
        sentiment=sentiment_label,
        source_ids=list(source_ids),
    )


def test_pipeline_replaces_llm_sentiment_with_finbert(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "classify_financial_sentiment",
        lambda text: {"sentiment": "neutral", "confidence": 0.71, "model": "ProsusAI/finbert"},
    )

    outlook = pipeline._classify_outlook(_outlook("mixed"))

    assert (outlook.sentiment, outlook.confidence, outlook.model) == (
        "neutral",
        0.71,
        "ProsusAI/finbert",
    )
    assert outlook.source_ids == ["filing:chunk-1"]


def test_pipeline_keeps_llm_sentiment_when_finbert_is_unavailable():
    outlook = _outlook("mixed")  # conftest makes FinBERT unavailable

    assert pipeline._classify_outlook(outlook) == outlook


@pytest.mark.parametrize(
    "outlook",
    [_outlook("unknown", ()), _outlook("positive", ())],
)
def test_ungrounded_outlook_is_never_reclassified(monkeypatch, outlook):
    monkeypatch.setattr(pipeline, "classify_financial_sentiment", pytest.fail)

    assert pipeline._classify_outlook(outlook) == outlook
