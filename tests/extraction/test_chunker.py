"""Tests for deterministic SEC-aware document chunking."""

from __future__ import annotations

from collections import Counter

import pytest

from src.extraction.chunker import UNSECTIONED, chunk_document
from src.extraction.schemas import (
    FilingType,
    LoadedDocument,
    LoadedDocumentMetadata,
)


def make_document(
    text: str,
    *,
    filing_type: FilingType = "10-K",
    source_id: str = "filing-test",
) -> LoadedDocument:
    """Build a minimal valid source document for chunking tests."""

    return LoadedDocument(
        ticker="TEST",
        filing_type=filing_type,
        period="FY 2025",
        text=text,
        source_id=source_id,
        source_type="filing",
        metadata=LoadedDocumentMetadata(
            file_name="test.txt",
            file_extension=".txt",
            encoding="utf-8",
            byte_size=len(text.encode("utf-8")),
            content_sha256="a" * 64,
        ),
    )


def assert_exact_offsets(document: LoadedDocument) -> None:
    """Assert the central source-offset invariant for every emitted chunk."""

    for chunk in chunk_document(document, max_chars=80, overlap_chars=10):
        assert chunk.start_char is not None
        assert chunk.end_char is not None
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_document_with_one_section() -> None:
    document = make_document("ITEM 1. BUSINESS\n\nBusiness overview.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].section == "ITEM_1"


def test_document_with_multiple_sections_never_crosses_sections() -> None:
    text = (
        "ITEM 1. BUSINESS\n\nBusiness text.\n\n"
        "ITEM 1A. RISK FACTORS\n\nRisk text.\n\n"
        "ITEM 2. PROPERTIES\n\nProperty text."
    )
    chunks = chunk_document(make_document(text), max_chars=500, overlap_chars=20)

    assert [chunk.section for chunk in chunks] == ["ITEM_1", "ITEM_1A", "ITEM_2"]
    assert all(chunk.text.count("ITEM ") == 1 for chunk in chunks)


def test_distinguishes_item_1_from_item_1a() -> None:
    text = "ITEM 1. BUSINESS\n\nOne.\n\nITEM 1A. RISK FACTORS\n\nTwo."

    sections = [chunk.section for chunk in chunk_document(make_document(text))]

    assert sections == ["ITEM_1", "ITEM_1A"]


def test_distinguishes_item_7_from_item_7a() -> None:
    text = "ITEM 7. MD&A\n\nOne.\n\nITEM 7A. MARKET RISK\n\nTwo."

    sections = [chunk.section for chunk in chunk_document(make_document(text))]

    assert sections == ["ITEM_7", "ITEM_7A"]


def test_10q_items_include_part_context() -> None:
    text = (
        "PART I. FINANCIAL INFORMATION\n\n"
        "ITEM 1. FINANCIAL STATEMENTS\n\nStatements.\n\n"
        "ITEM 2. MANAGEMENT DISCUSSION\n\nDiscussion.\n\n"
        "PART II. OTHER INFORMATION\n\n"
        "ITEM 1. LEGAL PROCEEDINGS\n\nLegal.\n\n"
        "ITEM 1A. RISK FACTORS\n\nRisks."
    )

    sections = [
        chunk.section
        for chunk in chunk_document(make_document(text, filing_type="10-Q"))
    ]

    assert sections == [
        "PART_I",
        "PART_I_ITEM_1",
        "PART_I_ITEM_2",
        "PART_II",
        "PART_II_ITEM_1",
        "PART_II_ITEM_1A",
    ]


def test_table_of_contents_duplicates_are_not_section_starts() -> None:
    text = (
        "TABLE OF CONTENTS\n"
        "Item 1. Business ........ 3\n"
        "Item 1A. Risk Factors ... 9\n"
        "Item 7. MD&A ........... 30\n"
        "Item 7A. Market Risk ... 42\n"
        "Item 8. Financials ..... 45\n\n"
        "ITEM 1. BUSINESS\n\nActual business.\n\n"
        "ITEM 1A. RISK FACTORS\n\nActual risks.\n\n"
        "ITEM 7. MD&A\n\nActual discussion.\n\n"
        "ITEM 7A. MARKET RISK\n\nActual market risk.\n\n"
        "ITEM 8. FINANCIAL STATEMENTS\n\nActual statements."
    )
    chunks = chunk_document(make_document(text), max_chars=500, overlap_chars=20)

    first_item_7 = next(chunk for chunk in chunks if chunk.section == "ITEM_7")

    assert first_item_7.start_char == text.rindex("ITEM 7. MD&A")
    assert chunks[0].section == UNSECTIONED
    assert "TABLE OF CONTENTS" in chunks[0].text


def test_missing_section_is_not_invented() -> None:
    text = "ITEM 1. BUSINESS\n\nOne.\n\nITEM 3. LEGAL\n\nThree."

    sections = [chunk.section for chunk in chunk_document(make_document(text))]

    assert sections == ["ITEM_1", "ITEM_3"]
    assert "ITEM_2" not in sections


def test_text_before_first_item_is_unsectioned() -> None:
    text = "Company preamble.\n\nITEM 1. BUSINESS\n\nBusiness."

    chunks = chunk_document(make_document(text))

    assert chunks[0].section == UNSECTIONED
    assert chunks[0].text == "Company preamble."
    assert chunks[1].section == "ITEM_1"


def test_document_without_sec_headings_is_unsectioned() -> None:
    document = make_document("Narrative paragraph.\n\nAnother paragraph.")

    chunks = chunk_document(document)

    assert {chunk.section for chunk in chunks} == {UNSECTIONED}


def test_item_reference_in_prose_is_not_a_heading() -> None:
    document = make_document(
        "Item 7 of this report discusses management assumptions.\n\n"
        "The narrative continues here."
    )

    chunks = chunk_document(document)

    assert {chunk.section for chunk in chunks} == {UNSECTIONED}


def test_oversized_paragraph_is_split() -> None:
    text = "ITEM 1. BUSINESS\n" + "revenue growth " * 30
    document = make_document(text)

    chunks = chunk_document(document, max_chars=70, overlap_chars=10)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 70 for chunk in chunks)
    assert all(chunk.section == "ITEM_1" for chunk in chunks)


def test_overlap_reuses_complete_paragraph_within_section() -> None:
    text = "Alpha one.\n\nBravo two.\n\nCharlie three."
    document = make_document(text)

    chunks = chunk_document(document, max_chars=30, overlap_chars=12)

    assert len(chunks) == 2
    assert chunks[0].text == "Alpha one.\n\nBravo two."
    assert chunks[1].text == "Bravo two.\n\nCharlie three."
    assert chunks[1].start_char < chunks[0].end_char


def test_zero_overlap_produces_disjoint_chunks() -> None:
    text = "Alpha one.\n\nBravo two.\n\nCharlie three."
    chunks = chunk_document(
        make_document(text),
        max_chars=30,
        overlap_chars=0,
    )

    assert len(chunks) == 2
    assert chunks[1].text == "Charlie three."
    assert chunks[0].end_char <= chunks[1].start_char


def test_overlap_equal_to_max_chars_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than max_chars"):
        chunk_document(make_document("Short filing."), max_chars=10, overlap_chars=10)


def test_overlap_greater_than_max_chars_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than max_chars"):
        chunk_document(make_document("Short filing."), max_chars=10, overlap_chars=11)


@pytest.mark.parametrize("max_chars", [0, -1])
def test_non_positive_max_chars_is_rejected(max_chars: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_document(make_document("Short filing."), max_chars=max_chars)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        chunk_document(make_document("Short filing."), overlap_chars=-1)


def test_very_short_document_produces_one_chunk() -> None:
    chunks = chunk_document(make_document("Short filing."))

    assert len(chunks) == 1
    assert chunks[0].text == "Short filing."


def test_offsets_are_exact_for_all_chunk_types() -> None:
    document = make_document(
        "Preamble.\n\nITEM 1. BUSINESS\n\n"
        + "long narrative " * 20
        + "\n\nFinal paragraph."
    )

    assert_exact_offsets(document)


def test_chunks_are_ordered_by_source_position() -> None:
    text = "ITEM 1. BUSINESS\n\n" + "word " * 50 + "\n\nITEM 2. PROPERTIES\n\nEnd."
    chunks = chunk_document(make_document(text), max_chars=60, overlap_chars=10)
    starts = [chunk.start_char for chunk in chunks]

    assert starts == sorted(starts)


def test_chunk_ids_are_deterministic() -> None:
    document = make_document("ITEM 7. MD&A\n\nDiscussion.")

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == (
        f"{document.source_id}:ITEM_7:{first[0].start_char}:{first[0].end_char}"
    )


def test_chunks_are_not_duplicated() -> None:
    text = "One short.\n\nTwo short.\n\nThree short.\n\nFour short."
    chunks = chunk_document(make_document(text), max_chars=28, overlap_chars=12)
    identities = [
        (chunk.section, chunk.start_char, chunk.end_char, chunk.text)
        for chunk in chunks
    ]

    assert all(count == 1 for count in Counter(identities).values())


def test_every_chunk_is_reconstructable_from_offsets() -> None:
    text = (
        "Cover text.\n\n"
        "ITEM 1. BUSINESS\n\nFirst paragraph.\n\nSecond paragraph.\n\n"
        "ITEM 1A. RISK FACTORS\n\n" + "risk detail " * 30
    )
    document = make_document(text)
    chunks = chunk_document(document, max_chars=75, overlap_chars=15)

    for chunk in chunks:
        assert chunk.start_char is not None
        assert chunk.end_char is not None
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
