"""FinBERT sentiment for management outlook text (ProsusAI/finbert).

The Hugging Face pipeline is loaded lazily once per process. Any import,
download or inference failure returns ``None`` so callers keep their existing
sentiment instead of failing the analysis.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Literal, TypedDict


FINBERT_MODEL = "ProsusAI/finbert"
_LABELS = {"positive", "negative", "neutral"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Function words carry gradient mass but no financial meaning; dropped from attributions.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "its", "of", "on", "or", "our", "that", "the", "their",
        "this", "to", "was", "were", "with",
    }
)

logger = logging.getLogger(__name__)


class FinancialSentiment(TypedDict):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    model: str


class SentimentRationale(TypedDict):
    rationale_sentence: str
    rationale_score: float


class TokenAttribution(TypedDict):
    token: str
    score: float


@lru_cache(maxsize=1)
def _classifier() -> Any:
    from transformers import pipeline

    return pipeline("text-classification", model=FINBERT_MODEL)


def classify_financial_sentiment(text: str) -> FinancialSentiment | None:
    """Return FinBERT's top label and score, or ``None`` when unavailable."""

    if not text.strip():
        return None
    try:
        # ponytail: a failed load is not cached, so offline hosts retry per analysis.
        top = _classifier()(text, truncation=True)[0]
        label = str(top["label"]).lower()
        confidence = float(top["score"])
    except Exception as error:  # model, network or runtime failure
        logger.warning("FinBERT sentiment unavailable: %s", type(error).__name__)
        return None
    if label not in _LABELS or not 0.0 <= confidence <= 1.0:
        return None
    return {"sentiment": label, "confidence": confidence, "model": FINBERT_MODEL}


def extract_sentiment_rationale(
    text: str,
    target_sentiment: str,
) -> SentimentRationale | None:
    """Sentence of ``text`` that FinBERT scores highest for ``target_sentiment``.

    Sentences split on ``.``, ``!`` or ``?``; a single sentence is returned with
    its own score. ``None`` when the model is unavailable or nothing scores.
    """

    sentences = [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]
    if not sentences or target_sentiment not in _LABELS:
        return None
    try:
        scored = _classifier()(sentences, top_k=None, truncation=True)
        candidates = [
            (float(item["score"]), sentence)
            for sentence, labels in zip(sentences, scored, strict=True)
            for item in labels
            if str(item["label"]).lower() == target_sentiment
        ]
    except Exception as error:  # model, network or runtime failure
        logger.warning("FinBERT rationale unavailable: %s", type(error).__name__)
        return None
    if not candidates:
        return None
    score, sentence = max(candidates, key=lambda candidate: candidate[0])
    return {"rationale_sentence": sentence, "rationale_score": score}


def explain_sentiment_tokens(
    text: str,
    target_sentiment: str | None = None,
    *,
    steps: int = 32,
) -> list[TokenAttribution]:
    """Input x Gradient attribution of each word to one FinBERT class logit.

    The gradient of the target logit (``target_sentiment`` or the predicted
    class) is taken w.r.t. the word embeddings, averaged over ``steps`` points
    scaled from zero to the input (Integrated Gradients), then multiplied by
    the embedding and summed over the hidden size. ``steps=1`` is plain
    Input x Gradient, which saturates on BERT and tends to rank function words
    such as "management" above the sentiment-bearing ones.

    WordPiece pieces are summed back into words, special tokens, punctuation
    and ``STOPWORDS`` dropped, and only words that support the class (positive
    attribution) are returned, scaled by the maximum of what remains to
    ``(0, 1]`` and sorted by score. Any failure returns ``[]``.
    """

    if not text.strip():
        return []
    try:
        import torch

        classifier = _classifier()
        model, tokenizer = classifier.model, classifier.tokenizer
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        embeddings = model.get_input_embeddings()(encoded["input_ids"]).detach()
        alphas = torch.linspace(1.0 / steps, 1.0, steps).view(-1, 1, 1)
        with torch.enable_grad():
            # One batched forward pass over the whole zero-to-input path.
            path = (alphas * embeddings).requires_grad_()
            logits = model(
                inputs_embeds=path,
                attention_mask=encoded["attention_mask"].expand(steps, -1),
            ).logits
            target = (
                int(logits[-1].argmax())
                if target_sentiment is None
                else model.config.label2id[target_sentiment]
            )
            # autograd.grad, unlike backward(), leaves model parameter .grad untouched.
            (gradient,) = torch.autograd.grad(logits[:, target].sum(), path)
        scores = (gradient.mean(dim=0) * embeddings[0]).sum(dim=-1).tolist()
        pieces = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    except Exception as error:  # model, network, unknown label or runtime failure
        logger.warning("FinBERT token attribution unavailable: %s", type(error).__name__)
        return []
    return _normalize_words(_merge_wordpieces(pieces, scores, tokenizer.all_special_tokens))


def _merge_wordpieces(
    pieces: list[str],
    scores: list[float],
    special_tokens: list[str],
) -> list[tuple[str, float]]:
    words: list[tuple[str, float]] = []
    for piece, score in zip(pieces, scores, strict=True):
        if piece in special_tokens:
            continue
        if piece.startswith("##") and words:
            word, total = words[-1]
            words[-1] = (word + piece[2:], total + score)
        else:
            words.append((piece, score))
    return words


def _normalize_words(words: list[tuple[str, float]]) -> list[TokenAttribution]:
    supporting = [
        (word, score)
        for word, score in words
        if score > 0 and word.isalnum() and word not in STOPWORDS
    ]
    if not supporting:
        return []
    top = max(score for _, score in supporting)
    ranked = sorted(supporting, key=lambda item: item[1], reverse=True)
    return [{"token": word, "score": round(score / top, 4)} for word, score in ranked]


__all__ = [
    "FINBERT_MODEL",
    "FinancialSentiment",
    "STOPWORDS",
    "SentimentRationale",
    "TokenAttribution",
    "classify_financial_sentiment",
    "explain_sentiment_tokens",
    "extract_sentiment_rationale",
]
