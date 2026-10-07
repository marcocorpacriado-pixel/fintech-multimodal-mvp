"""Tests for the deterministic evidence catalog."""

from __future__ import annotations

import pytest

from src.extraction.evidence_catalog import (
    MAX_EXCERPT_CHARS,
    MIN_EXCERPT_CHARS,
    EvidenceCatalogError,
    build_evidence_catalog,
)
from src.extraction.schemas import DocumentChunk, RetrievalResult


MDA_TEXT = (
    "Net sales decreased 1% during the quarter compared to the prior quarter. "
    "The decrease was driven primarily by lower Mac and iPad sales. "
    "Gross margin was $24.9 billion, or 46.5% of net sales.\n\n"
    "Liquidity remained strong because cash generation supported planned "
    "spending and capital returns. The Company's principal executive officer "
    "concluded that disclosure controls were effective."
)
RISK_TEXT = (
    "Supply chain disruption may adversely affect results. Currency "
    "volatility could reduce reported revenue in future periods."
)


def retrieval(
    chunk_id: str,
    text: str,
    *,
    rank: int,
    section: str | None = "PART_I_ITEM_2",
    ticker: str = "AAPL",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            ticker=ticker,
            filing_type="10-Q",
            period="2026-06-27",
            section=section,
            text=text,
            source_id="filing:aapl",
            start_char=0,
            end_char=len(text),
        ),
        score=1.0 / rank,
        rank=rank,
    )


def results() -> list[RetrievalResult]:
    return [
        retrieval("chunk-mda", MDA_TEXT, rank=1),
        retrieval("chunk-risk", RISK_TEXT, rank=2, section="PART_II_ITEM_1A"),
    ]


def test_ids_are_unique_ordered_and_zero_padded() -> None:
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    ids = catalog.evidence_ids
    assert len(set(ids)) == len(ids)
    assert ids[0] == "E01"
    assert list(ids) == sorted(ids)
    assert all(evidence_id.startswith("E") for evidence_id in ids)


def test_catalog_is_deterministic() -> None:
    first = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)
    second = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    assert first.items == second.items


def test_order_follows_retrieval_then_offset() -> None:
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    chunk_order = [item.source_id for item in catalog.items]
    assert chunk_order == sorted(chunk_order, key=["chunk-mda", "chunk-risk"].index)
    mda_offsets = [i.start_offset for i in catalog.items if i.source_id == "chunk-mda"]
    assert mda_offsets == sorted(mda_offsets)


def test_every_excerpt_is_an_exact_offset_slice_of_its_chunk() -> None:
    sources = {r.chunk.chunk_id: r.chunk.text for r in results()}
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    assert len(catalog) > 2
    for item in catalog.items:
        assert sources[item.source_id][item.start_offset : item.end_offset] == (
            item.excerpt
        )
        assert item.excerpt == item.excerpt.strip()


def test_mapping_reconstructs_source_section_and_metadata() -> None:
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    risk = next(item for item in catalog.items if item.source_id == "chunk-risk")
    assert catalog.get(risk.evidence_id) is risk
    assert risk.source_section == "PART_II_ITEM_1A"
    assert risk.source_type == "filing"
    assert (risk.ticker, risk.filing_type, risk.period) == (
        "AAPL",
        "10-Q",
        "2026-06-27",
    )
    assert risk.rank == 2


def test_unsectioned_chunk_gets_stable_fallback_section() -> None:
    catalog = build_evidence_catalog(
        [retrieval("c", RISK_TEXT, rank=1, section=None)], ticker="AAPL"
    )

    assert {item.source_section for item in catalog.items} == {"UNSECTIONED"}


def test_valid_and_invalid_evidence_ids() -> None:
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    assert catalog.get("E01") is not None
    assert catalog.get("E999") is None
    assert catalog.get("e01") is None
    assert catalog.get("") is None
    assert catalog.get("chunk-mda") is None  # technical ids are not selectable


def test_source_ids_for_are_unique_in_first_seen_order() -> None:
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)
    risk = next(i for i in catalog.items if i.source_id == "chunk-risk")
    mda = next(i for i in catalog.items if i.source_id == "chunk-mda")

    assert catalog.source_ids_for(
        [risk.evidence_id, mda.evidence_id, risk.evidence_id]
    ) == ["chunk-risk", "chunk-mda"]


def test_source_text_is_not_mutated_and_numbers_stay_intact() -> None:
    original = [r.model_copy(deep=True) for r in results()]
    catalog = build_evidence_catalog(results(), ticker="AAPL", max_excerpt_chars=120)

    assert [r.chunk.text for r in results()] == [r.chunk.text for r in original]
    joined = " ".join(item.excerpt for item in catalog.items)
    for figure in ("$24.9 billion", "46.5%", "1%"):
        assert figure in joined
    # A decimal point is never a sentence boundary.
    assert not any(item.excerpt.endswith("$24.") for item in catalog.items)
    assert not any(item.excerpt.startswith("9 billion") for item in catalog.items)


def test_excerpts_are_bounded_unless_a_single_unit_is_longer() -> None:
    long_unit = "word " * 400 + "end."
    catalog = build_evidence_catalog(
        [retrieval("long", MDA_TEXT + "\n\n" + long_unit, rank=1)], ticker="AAPL"
    )

    sizes = [len(item.excerpt) for item in catalog.items]
    assert max(sizes) >= len(long_unit.strip())  # never cut inside a unit
    assert sorted(sizes)[:-1] == [s for s in sorted(sizes)[:-1] if s <= MAX_EXCERPT_CHARS]


def test_short_chunk_is_kept_as_one_whole_item() -> None:
    text = "Short note."
    assert len(text) < MIN_EXCERPT_CHARS

    catalog = build_evidence_catalog([retrieval("tiny", text, rank=1)], ticker="AAPL")

    assert [item.excerpt for item in catalog.items] == [text]


def test_empty_retrieval_gives_empty_catalog() -> None:
    catalog = build_evidence_catalog([], ticker="AAPL")

    assert len(catalog) == 0
    assert not catalog
    assert catalog.evidence_ids == ()


def test_wrong_ticker_chunk_is_rejected() -> None:
    with pytest.raises(EvidenceCatalogError, match="belongs to ticker"):
        build_evidence_catalog(
            [retrieval("c", RISK_TEXT, rank=1, ticker="MSFT")], ticker="AAPL"
        )


def test_duplicate_chunk_id_is_rejected() -> None:
    duplicate = retrieval("same", RISK_TEXT, rank=1)

    with pytest.raises(EvidenceCatalogError, match="duplicate evidence chunk_id"):
        build_evidence_catalog([duplicate, duplicate], ticker="AAPL")


def test_id_width_grows_for_large_catalogs() -> None:
    text = "\n".join(f"Sentence number {i} describes an unrelated fact." for i in range(300))
    catalog = build_evidence_catalog(
        [retrieval("big", text, rank=1)], ticker="AAPL", max_excerpt_chars=60
    )

    assert len(catalog) > 99
    assert catalog.items[0].evidence_id == "E001"
    assert len({len(item.evidence_id) for item in catalog.items}) == 1
