"""Tests for deterministic BM25 financial chunk retrieval."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.extraction.retriever import (
    BM25Retriever,
    retrieve,
    tokenize_financial_text,
)
from src.extraction.schemas import DocumentChunk, FilingType, RetrievalResult


def make_chunk(
    index: int,
    text: str,
    *,
    ticker: str = "AAPL",
    filing_type: FilingType = "10-K",
    period: str = "FY 2025",
    section: str = "ITEM_7",
) -> DocumentChunk:
    """Build a traceable chunk with deterministic test metadata."""

    start = index * 1_000
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        section=section,
        text=text,
        source_id=f"source-{ticker}-{period}",
        start_char=start,
        end_char=start + len(text),
    )


@pytest.fixture
def financial_chunks() -> list[DocumentChunk]:
    """Small multi-company corpus covering common financial queries."""

    return [
        make_chunk(
            0,
            "Revenue growth accelerated because services demand increased revenue.",
        ),
        make_chunk(
            1,
            "Liquidity remained strong with cash flow and available credit facilities.",
        ),
        make_chunk(
            2,
            "Risk factors include supply chain disruption and liquidity risk.",
            section="ITEM_1A",
        ),
        make_chunk(
            3,
            "Operating margins improved while EBITDA and EPS reached 12.5% in 2026.",
            ticker="MSFT",
            filing_type="10-Q",
            period="Q1 2026",
        ),
        make_chunk(
            4,
            "Cybersecurity remains among the principal risk factors.",
            ticker="MSFT",
            filing_type="10-Q",
            period="Q1 2026",
            section="ITEM_1A",
        ),
        make_chunk(
            5,
            "Capital expenditures and capex supported factory expansion.",
            ticker="TSLA",
        ),
    ]


def test_simple_query_returns_correct_result(
    financial_chunks: list[DocumentChunk],
) -> None:
    results = retrieve(financial_chunks, "cybersecurity")

    assert results[0].chunk.chunk_id == "chunk-4"


def test_ranking_prefers_more_relevant_chunk(
    financial_chunks: list[DocumentChunk],
) -> None:
    results = retrieve(financial_chunks, "revenue growth")

    assert results[0].chunk.chunk_id == "chunk-0"


def test_top_k_one(financial_chunks: list[DocumentChunk]) -> None:
    assert len(retrieve(financial_chunks, "risk", top_k=1)) == 1


def test_top_k_larger_than_matching_corpus(
    financial_chunks: list[DocumentChunk],
) -> None:
    results = retrieve(financial_chunks, "risk", top_k=100)

    assert len(results) == 2


def test_empty_corpus_returns_empty_results() -> None:
    assert retrieve([], "liquidity") == []


@pytest.mark.parametrize("query", ["", "   \n\t"])
def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        retrieve([], query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        retrieve([], "liquidity", top_k=top_k)


def test_ticker_filter(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "risk factors", ticker="MSFT")

    assert results
    assert {result.chunk.ticker for result in results} == {"MSFT"}


def test_filing_type_filter(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "operating margins", filing_type="10-Q")

    assert results
    assert {result.chunk.filing_type for result in results} == {"10-Q"}


def test_period_filter(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "EPS", period="Q1 2026")

    assert results
    assert {result.chunk.period for result in results} == {"Q1 2026"}


def test_section_filter(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "risk", section="ITEM_1A")

    assert results
    assert {result.chunk.section for result in results} == {"ITEM_1A"}


def test_combined_filters(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(
        financial_chunks,
        "risk factors",
        ticker="MSFT",
        filing_type="10-Q",
        period="Q1 2026",
        section="ITEM_1A",
    )

    assert [result.chunk.chunk_id for result in results] == ["chunk-4"]


def test_filter_without_candidates_returns_empty(
    financial_chunks: list[DocumentChunk],
) -> None:
    assert retrieve(financial_chunks, "revenue", ticker="NVDA") == []


def test_scores_are_descending(financial_chunks: list[DocumentChunk]) -> None:
    scores = [result.score for result in retrieve(financial_chunks, "risk factors")]

    assert scores == sorted(scores, reverse=True)


def test_ranks_are_consecutive_from_one(
    financial_chunks: list[DocumentChunk],
) -> None:
    ranks = [result.rank for result in retrieve(financial_chunks, "risk", top_k=10)]

    assert ranks == list(range(1, len(ranks) + 1))


def test_retriever_is_deterministic(financial_chunks: list[DocumentChunk]) -> None:
    retriever = BM25Retriever(financial_chunks)

    first = retriever.search("risk factors")
    second = retriever.search("risk factors")

    assert [result.model_dump() for result in first] == [
        result.model_dump() for result in second
    ]


def test_tie_break_preserves_original_position() -> None:
    chunks = [
        make_chunk(9, "Identical liquidity wording."),
        make_chunk(1, "Identical liquidity wording."),
    ]

    results = retrieve(chunks, "liquidity")

    assert [result.chunk.chunk_id for result in results] == ["chunk-9", "chunk-1"]


def test_tokenizer_preserves_financial_numbers_and_forms() -> None:
    tokens = tokenize_financial_text("10-K YoY revenue rose 12.5% to 1,250 in 2026")

    assert {"10-k", "yoy", "12.5%", "1,250", "2026"} <= set(tokens)


def test_ebitda_and_eps_query(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "EBITDA EPS")

    assert results[0].chunk.chunk_id == "chunk-3"


def test_risk_factors_query(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "major risk factors")

    assert results
    assert results[0].chunk.section == "ITEM_1A"


def test_liquidity_query(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "main liquidity risks")

    assert results
    assert "liquidity" in results[0].chunk.text.casefold()


def test_revenue_query(financial_chunks: list[DocumentChunk]) -> None:
    results = retrieve(financial_chunks, "why did revenue increase")

    assert results[0].chunk.chunk_id == "chunk-0"


def test_query_without_lexical_match_returns_empty(
    financial_chunks: list[DocumentChunk],
) -> None:
    assert retrieve(financial_chunks, "derivatives hedging swaps") == []


def test_multiple_companies_are_ranked_together(
    financial_chunks: list[DocumentChunk],
) -> None:
    results = retrieve(financial_chunks, "risk")

    assert {result.chunk.ticker for result in results} == {"AAPL", "MSFT"}


def test_multiple_sections_are_ranked_together(
    financial_chunks: list[DocumentChunk],
) -> None:
    results = retrieve(financial_chunks, "liquidity risk")

    assert {result.chunk.section for result in results} == {"ITEM_7", "ITEM_1A"}


def test_financial_bigrams_are_augmented() -> None:
    tokens = tokenize_financial_text(
        "Cash flow, operating margins, risk factors, and capital expenditures."
    )

    assert {"cash_flow", "operating_margin", "risk_factor", "capital_expenditure"} <= set(
        tokens
    )


def test_retrieval_result_serializes_to_json(
    financial_chunks: list[DocumentChunk],
) -> None:
    result = retrieve(financial_chunks, "revenue", top_k=1)[0]

    assert json.loads(result.model_dump_json()) == result.model_dump()


def test_retrieval_result_rejects_invalid_rank(
    financial_chunks: list[DocumentChunk],
) -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(chunk=financial_chunks[0], score=1.0, rank=0)
