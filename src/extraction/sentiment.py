"""FinBERT sentiment for management outlook text (ProsusAI/finbert).

The Hugging Face pipeline is loaded lazily once per process. Any import,
download or inference failure returns ``None`` so callers keep their existing
sentiment instead of failing the analysis.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, TypedDict


FINBERT_MODEL = "ProsusAI/finbert"
_LABELS = {"positive", "negative", "neutral"}

logger = logging.getLogger(__name__)


class FinancialSentiment(TypedDict):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    model: str


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


__all__ = ["FINBERT_MODEL", "FinancialSentiment", "classify_financial_sentiment"]
