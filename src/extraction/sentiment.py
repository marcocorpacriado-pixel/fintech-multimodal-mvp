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

logger = logging.getLogger(__name__)


class FinancialSentiment(TypedDict):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    model: str


class SentimentRationale(TypedDict):
    rationale_sentence: str
    rationale_score: float


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


__all__ = [
    "FINBERT_MODEL",
    "FinancialSentiment",
    "SentimentRationale",
    "classify_financial_sentiment",
    "extract_sentiment_rationale",
]
