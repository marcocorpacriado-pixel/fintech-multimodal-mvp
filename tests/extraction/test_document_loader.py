"""Tests for narrative SEC filing ingestion."""

import codecs
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.extraction.document_loader import (
    EmptyDocumentError,
    UnsupportedDocumentFormatError,
    load_filing,
)
from src.extraction.schemas import (
    FilingType,
    LoadedDocument,
    LoadedDocumentMetadata,
)


BASE_METADATA = {
    "file_name": "sample.txt",
    "file_extension": ".txt",
    "encoding": "utf-8",
    "byte_size": 12,
    "content_sha256": "a" * 64,
}


def write_bytes(path: Path, content: bytes) -> Path:
    """Create a deterministic document fixture under pytest's temp directory."""

    path.write_bytes(content)
    return path


def load_sample(
    path: str | Path,
    *,
    ticker: str = "AAPL",
    filing_type: FilingType = "10-K",
    period: str = "FY 2025",
    source_id: str | None = None,
) -> LoadedDocument:
    """Load a test filing with stable explicit business metadata."""

    return load_filing(
        path,
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        source_id=source_id,
    )


def test_loads_utf8_txt(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", "Revenue grew €5m.".encode("utf-8"))

    document = load_sample(path)

    assert document.text == "Revenue grew €5m."
    assert document.metadata.encoding == "utf-8"


def test_loads_utf8_txt_with_bom(tmp_path: Path) -> None:
    path = write_bytes(
        tmp_path / "filing.txt",
        codecs.BOM_UTF8 + "Management outlook".encode("utf-8"),
    )

    document = load_sample(path)

    assert document.text == "Management outlook"
    assert document.metadata.encoding == "utf-8-sig"


def test_accepts_path_input(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Narrative filing")

    assert load_sample(path).metadata.file_name == "filing.txt"


def test_accepts_string_input(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Narrative filing")

    assert load_sample(str(path)).metadata.file_name == "filing.txt"


def test_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Filing not found"):
        load_sample(missing)


def test_rejects_non_file_path(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="not a file"):
        load_sample(tmp_path)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.pdf", b"Not supported in D2")

    with pytest.raises(UnsupportedDocumentFormatError, match=r"\.pdf"):
        load_sample(path)


def test_rejects_empty_document(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "empty.txt", b"")

    with pytest.raises(EmptyDocumentError, match="no narrative text"):
        load_sample(path)


def test_rejects_whitespace_only_document(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "blank.txt", b" \t\r\n  \r")

    with pytest.raises(EmptyDocumentError, match="no narrative text"):
        load_sample(path)


def test_produces_explicit_and_file_metadata(tmp_path: Path) -> None:
    raw = b"Risk factors"
    path = write_bytes(tmp_path / "Apple_2025.txt", raw)

    document = load_sample(path, ticker="MSFT", period="Q2 2026")

    assert document.ticker == "MSFT"
    assert document.filing_type == "10-K"
    assert document.period == "Q2 2026"
    assert document.source_type == "filing"
    assert document.metadata.file_name == "Apple_2025.txt"
    assert document.metadata.file_extension == ".txt"
    assert document.metadata.byte_size == len(raw)
    assert len(document.metadata.content_sha256) == 64


def test_generates_source_id(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Revenue grew")

    assert load_sample(path).source_id.startswith("sec-filing-")


def test_generated_source_id_is_deterministic(tmp_path: Path) -> None:
    first_path = write_bytes(tmp_path / "first.txt", b"Revenue grew")
    second_path = write_bytes(tmp_path / "second.txt", b"Revenue grew")

    first = load_sample(first_path)
    second = load_sample(second_path)

    assert first.source_id == second.source_id


def test_preserves_explicit_source_id(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Revenue grew")

    document = load_sample(path, source_id="0000320193-25-000079")

    assert document.source_id == "0000320193-25-000079"


def test_rejects_blank_explicit_source_id(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Revenue grew")

    with pytest.raises(ValidationError, match="non-whitespace"):
        load_sample(path, source_id="   ")


def test_normalizes_crlf_newlines(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Heading\r\n\r\nParagraph")

    assert load_sample(path).text == "Heading\n\nParagraph"


def test_normalizes_cr_newlines(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"Heading\r\rParagraph")

    assert load_sample(path).text == "Heading\n\nParagraph"


def test_loaded_document_rejects_whitespace_text() -> None:
    with pytest.raises(ValidationError, match="non-whitespace"):
        LoadedDocument(
            ticker="AAPL",
            filing_type="10-K",
            period="FY 2025",
            text=" \n\t ",
            source_id="source-1",
            source_type="filing",
            metadata=BASE_METADATA,
        )


def test_loaded_document_rejects_invalid_source_type() -> None:
    with pytest.raises(ValidationError):
        LoadedDocument(
            ticker="AAPL",
            filing_type="10-K",
            period="FY 2025",
            text="Narrative text",
            source_id="source-1",
            source_type="website",
            metadata=BASE_METADATA,
        )


def test_loaded_document_serializes_to_dict_and_json() -> None:
    document = LoadedDocument(
        ticker="AAPL",
        filing_type="10-K",
        period="FY 2025",
        text="Narrative text",
        source_id="source-1",
        source_type="filing",
        metadata=LoadedDocumentMetadata(**BASE_METADATA),
    )

    dumped = document.model_dump()

    assert json.loads(document.model_dump_json()) == dumped
    assert dumped["metadata"]["file_name"] == "sample.txt"


def test_loads_windows_1252_as_explicit_fallback(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", "Ingresos: 10 €".encode("cp1252"))

    document = load_sample(path)

    assert document.text == "Ingresos: 10 €"
    assert document.metadata.encoding == "cp1252"


def test_preserves_paragraph_structure_and_boundary_whitespace(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "filing.txt", b"  Heading\n\nParagraph  \n")

    assert load_sample(path).text == "  Heading\n\nParagraph  \n"
