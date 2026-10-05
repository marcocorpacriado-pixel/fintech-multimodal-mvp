"""Deterministic BM25 retrieval over traceable financial document chunks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .schemas import DocumentChunk, FilingType, RetrievalResult


DEFAULT_TOP_K = 5

# Small query-oriented list only. Financial nouns and modifiers are
# intentionally retained, as are negations such as "not" and "no".
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "why",
        "with",
    }
)
_TOKEN_RE = re.compile(
    r"10-[kq]\b|[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*%?",
    re.IGNORECASE,
)
_FINANCIAL_BIGRAMS = {
    ("capital", "expenditure"): "capital_expenditure",
    ("capital", "expenditures"): "capital_expenditure",
    ("cash", "flow"): "cash_flow",
    ("cash", "flows"): "cash_flow",
    ("operating", "margin"): "operating_margin",
    ("operating", "margins"): "operating_margin",
    ("risk", "factor"): "risk_factor",
    ("risk", "factors"): "risk_factor",
}


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    """A chunk paired with stable corpus position and cached tokens."""

    position: int
    chunk: DocumentChunk
    tokens: tuple[str, ...]


def tokenize_financial_text(text: str) -> list[str]:
    """Tokenize financial English without stemming or external NLP models.

    The tokenizer lowercases words, retains numeric expressions and SEC forms
    such as ``10-K``, and augments a small set of meaningful financial bigrams
    such as ``cash_flow`` while keeping their component tokens.
    """

    unigrams = [
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if token.casefold() not in _STOPWORDS
    ]
    phrase_tokens = [
        _FINANCIAL_BIGRAMS[pair]
        for pair in zip(unigrams, unigrams[1:])
        if pair in _FINANCIAL_BIGRAMS
    ]
    return [*unigrams, *phrase_tokens]


class BM25Retriever:
    """Reusable lexical retriever backed by ``rank_bm25.BM25Okapi``.

    Chunk tokenization is cached at construction. A BM25 index is built for
    the metadata-filtered candidates on each search so document frequencies
    and scores always correspond to the filtered corpus.
    """

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self._indexed_chunks = tuple(
            _IndexedChunk(
                position=position,
                chunk=chunk,
                tokens=tuple(tokenize_financial_text(chunk.text)),
            )
            for position, chunk in enumerate(chunks)
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        ticker: str | None = None,
        filing_type: FilingType | None = None,
        period: str | None = None,
        section: str | None = None,
    ) -> list[RetrievalResult]:
        """Rank chunks matching a validated query and optional metadata filters."""

        query_tokens = _validate_and_tokenize_query(query)
        _validate_top_k(top_k)

        candidates = [
            indexed
            for indexed in self._indexed_chunks
            if _matches_filters(
                indexed.chunk,
                ticker=ticker,
                filing_type=filing_type,
                period=period,
                section=section,
            )
            and indexed.tokens
        ]
        if not candidates:
            return []

        tokenized_corpus = [list(indexed.tokens) for indexed in candidates]
        index = BM25Okapi(tokenized_corpus)
        scores = index.get_scores(query_tokens)
        query_vocabulary = set(query_tokens)

        scored = [
            (float(score), indexed)
            for score, indexed in zip(scores, candidates)
            if query_vocabulary.intersection(indexed.tokens)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].position))

        return [
            RetrievalResult(chunk=indexed.chunk, score=score, rank=rank)
            for rank, (score, indexed) in enumerate(scored[:top_k], start=1)
        ]


def retrieve(
    chunks: Sequence[DocumentChunk],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    ticker: str | None = None,
    filing_type: FilingType | None = None,
    period: str | None = None,
    section: str | None = None,
) -> list[RetrievalResult]:
    """Build a temporary retriever and return ranked BM25 results."""

    return BM25Retriever(chunks).search(
        query,
        top_k=top_k,
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        section=section,
    )


def _validate_and_tokenize_query(query: str) -> list[str]:
    """Reject blank or non-searchable queries and return normalized tokens."""

    if not query.strip():
        raise ValueError("query must contain non-whitespace characters")
    tokens = tokenize_financial_text(query)
    if not tokens:
        raise ValueError("query must contain at least one searchable token")
    return tokens


def _validate_top_k(top_k: int) -> None:
    """Validate the requested result count."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")


def _matches_filters(
    chunk: DocumentChunk,
    *,
    ticker: str | None,
    filing_type: FilingType | None,
    period: str | None,
    section: str | None,
) -> bool:
    """Apply exact metadata filters before BM25 index construction."""

    return (
        (ticker is None or chunk.ticker == ticker)
        and (filing_type is None or chunk.filing_type == filing_type)
        and (period is None or chunk.period == period)
        and (section is None or chunk.section == section)
    )


__all__ = [
    "BM25Retriever",
    "DEFAULT_TOP_K",
    "retrieve",
    "tokenize_financial_text",
]
