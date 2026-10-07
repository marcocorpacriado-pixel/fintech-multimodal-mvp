"""FinBERT outlook sentiment, with the Hugging Face pipeline mocked (no downloads)."""

from types import SimpleNamespace

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


def _batch_classifier(scores: dict[str, dict[str, float]]):
    def classify(sentences, top_k, truncation):
        assert top_k is None and truncation is True
        return [
            [{"label": label, "score": score} for label, score in scores[sentence].items()]
            for sentence in sentences
        ]

    return classify


def test_rationale_picks_sentence_most_aligned_with_target(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_classifier",
        lambda: _batch_classifier(
            {
                "Sales grew 8%.": {"positive": 0.96, "neutral": 0.03, "negative": 0.01},
                "The meeting is in March!": {"positive": 0.02, "neutral": 0.95, "negative": 0.03},
                "Demand may weaken?": {"positive": 0.05, "neutral": 0.10, "negative": 0.85},
            }
        ),
    )
    text = "Sales grew 8%. The meeting is in March! Demand may weaken?"

    assert sentiment.extract_sentiment_rationale(text, "positive") == {
        "rationale_sentence": "Sales grew 8%.",
        "rationale_score": 0.96,
    }
    assert sentiment.extract_sentiment_rationale(text, "negative") == {
        "rationale_sentence": "Demand may weaken?",
        "rationale_score": 0.85,
    }


def test_single_sentence_is_its_own_rationale(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_classifier",
        lambda: _batch_classifier({"Demand stays strong": {"positive": 0.9}}),
    )

    assert sentiment.extract_sentiment_rationale("Demand stays strong", "positive") == {
        "rationale_sentence": "Demand stays strong",
        "rationale_score": 0.9,
    }


@pytest.mark.parametrize(("text", "target"), [("   ", "positive"), ("Text.", "mixed")])
def test_rationale_skips_blank_text_or_unsupported_target(monkeypatch, text, target):
    monkeypatch.setattr(sentiment, "_classifier", pytest.fail)

    assert sentiment.extract_sentiment_rationale(text, target) is None


def test_rationale_falls_back_to_none_when_model_fails(monkeypatch):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(sentiment, "_classifier", offline)

    assert sentiment.extract_sentiment_rationale("A. B.", "positive") is None


def _outlook(sentiment_label="positive", source_ids=("filing:chunk-1",)):
    return ManagementOutlook(
        summary="Management expects demand to remain strong.",
        sentiment=sentiment_label,
        source_ids=list(source_ids),
    )


def test_pipeline_replaces_llm_sentiment_with_finbert_and_rationale(monkeypatch):
    rationale_calls = []
    monkeypatch.setattr(
        pipeline,
        "classify_financial_sentiment",
        lambda text: {"sentiment": "neutral", "confidence": 0.71, "model": "ProsusAI/finbert"},
    )

    def rationale(text, target_sentiment):
        rationale_calls.append((text, target_sentiment))
        return {"rationale_sentence": "Guidance is unchanged.", "rationale_score": 0.88}

    monkeypatch.setattr(pipeline, "extract_sentiment_rationale", rationale)

    outlook = pipeline._classify_outlook(
        _outlook("mixed"), evidence_text="Guidance is unchanged. Sales grew."
    )

    assert (outlook.sentiment, outlook.confidence, outlook.model) == (
        "neutral",
        0.71,
        "ProsusAI/finbert",
    )
    assert (outlook.rationale_sentence, outlook.rationale_score) == (
        "Guidance is unchanged.",
        0.88,
    )
    assert rationale_calls == [("Guidance is unchanged. Sales grew.", "neutral")]
    assert outlook.source_ids == ["filing:chunk-1"]


def test_missing_rationale_keeps_finbert_label(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "classify_financial_sentiment",
        lambda text: {"sentiment": "positive", "confidence": 0.9, "model": "ProsusAI/finbert"},
    )

    outlook = pipeline._classify_outlook(_outlook("positive"), evidence_text="x")

    assert outlook.confidence == 0.9
    assert outlook.rationale_sentence is None


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


def _tiny_finbert(tmp_path):
    """Random-weight BERT with FinBERT's label map: real gradients, no download."""

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    BertConfig = transformers.BertConfig
    BertForSequenceClassification = transformers.BertForSequenceClassification
    BertTokenizer = transformers.BertTokenizer

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sales", "increas", "##ed", "demand", "."]
        ),
        encoding="utf-8",
    )
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=3,
        label2id={"positive": 0, "negative": 1, "neutral": 2},
        id2label={0: "positive", 1: "negative", 2: "neutral"},
    )
    model = BertForSequenceClassification(config).eval()
    return SimpleNamespace(model=model, tokenizer=BertTokenizer(str(vocab)))


@pytest.mark.parametrize("target", ["positive", "negative", "neutral", None])
def test_token_attributions_merge_wordpieces_and_normalize(monkeypatch, tmp_path, target):
    tiny = _tiny_finbert(tmp_path)
    monkeypatch.setattr(sentiment, "_classifier", lambda: tiny)

    attributions = sentiment.explain_sentiment_tokens(
        "Sales increased demand.", target, steps=4
    )

    assert attributions, "some word must support the target logit"
    tokens = [item["token"] for item in attributions]
    assert set(tokens) <= {"sales", "increased", "demand"}  # no [CLS]/[SEP]/##/"."
    scores = [item["score"] for item in attributions]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0 and all(0.0 < score <= 1.0 for score in scores)
    assert all(param.grad is None for param in tiny.model.parameters())


def test_wordpieces_are_summed_and_special_tokens_dropped():
    words = sentiment._merge_wordpieces(
        ["[CLS]", "increas", "##ed", "demand", ".", "[SEP]"],
        [9.0, 0.5, 0.25, 0.3, 0.2, 9.0],
        ["[CLS]", "[SEP]", "[PAD]"],
    )

    assert words == [("increased", 0.75), ("demand", 0.3), (".", 0.2)]


def test_normalization_keeps_supporting_words_scaled_to_max():
    assert sentiment._normalize_words(
        [("increased", 0.8), ("demand", 0.2), ("risk", -0.5), (".", 0.9)]
    ) == [{"token": "increased", "score": 1.0}, {"token": "demand", "score": 0.25}]
    assert sentiment._normalize_words([("risk", -0.5)]) == []


def test_token_attribution_falls_back_to_empty_list(monkeypatch):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(sentiment, "_classifier", offline)

    assert sentiment.explain_sentiment_tokens("Demand grew.", "positive") == []
    monkeypatch.setattr(sentiment, "_classifier", pytest.fail)
    assert sentiment.explain_sentiment_tokens("  ") == []


def test_pipeline_attaches_attributions_for_rationale_sentence(monkeypatch):
    explain_calls = []
    monkeypatch.setattr(
        pipeline,
        "classify_financial_sentiment",
        lambda text: {"sentiment": "positive", "confidence": 0.9, "model": "ProsusAI/finbert"},
    )
    monkeypatch.setattr(
        pipeline,
        "extract_sentiment_rationale",
        lambda text, target_sentiment: {
            "rationale_sentence": "Demand grew.",
            "rationale_score": 0.95,
        },
    )

    def explain(text, target_sentiment=None):
        explain_calls.append((text, target_sentiment))
        return [{"token": "demand", "score": 1.0}]

    monkeypatch.setattr(pipeline, "explain_sentiment_tokens", explain)

    outlook = pipeline._classify_outlook(_outlook("positive"), evidence_text="Demand grew.")

    assert explain_calls == [("Demand grew.", "positive")]
    assert [item.model_dump() for item in outlook.token_attributions] == [
        {"token": "demand", "score": 1.0}
    ]


def test_conftest_disables_token_attribution():
    assert pipeline.explain_sentiment_tokens("Demand grew.", "positive") == []
