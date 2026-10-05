"""Load narrative SEC filings into normalized, traceable documents.

This module is deliberately limited to ingestion. It does not infer filing
metadata, split documents, analyze content, or interact with external systems.
"""

from __future__ import annotations

import codecs
import hashlib
import json
from pathlib import Path

from .schemas import FilingType, LoadedDocument, LoadedDocumentMetadata


SUPPORTED_EXTENSIONS = frozenset({".txt"})


class UnsupportedDocumentFormatError(ValueError):
    """Raised when a document uses an unsupported file extension."""


class DocumentDecodingError(UnicodeError):
    """Raised when a document cannot be decoded with supported encodings."""


class EmptyDocumentError(ValueError):
    """Raised when a document contains no narrative text."""


def load_filing(
    path: str | Path,
    *,
    ticker: str,
    filing_type: FilingType,
    period: str,
    source_id: str | None = None,
) -> LoadedDocument:
    """Load a narrative SEC filing from disk.

    Metadata identifying the filing is always explicit; it is never inferred
    from the filename. When ``source_id`` is omitted, a stable identifier is
    generated from the normalized semantic metadata and document content.

    Args:
        path: Path to a supported narrative filing.
        ticker: Explicit company ticker.
        filing_type: SEC filing form covered by ``FilingType``.
        period: Explicit reporting period represented by the filing.
        source_id: Optional caller-provided source identifier.

    Returns:
        A validated, JSON-serializable ``LoadedDocument``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` is not a regular file.
        UnsupportedDocumentFormatError: If its extension is unsupported.
        DocumentDecodingError: If supported encodings cannot decode it.
        EmptyDocumentError: If it is empty or contains only whitespace.
    """

    filing_path = _validate_path(path)
    raw_content = filing_path.read_bytes()
    decoded_text, encoding = _decode_text(raw_content, filing_path)
    normalized_text = _normalize_newlines(decoded_text)

    if not normalized_text.strip():
        raise EmptyDocumentError(
            f"Document contains no narrative text: {filing_path}"
        )

    content_sha256 = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    metadata = LoadedDocumentMetadata(
        file_name=filing_path.name,
        file_extension=filing_path.suffix.lower(),
        encoding=encoding,
        byte_size=len(raw_content),
        content_sha256=content_sha256,
    )

    resolved_source_id = (
        source_id
        if source_id is not None
        else _generate_source_id(
            ticker=ticker,
            filing_type=filing_type,
            period=period,
            content_sha256=content_sha256,
        )
    )

    return LoadedDocument(
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        text=normalized_text,
        source_id=resolved_source_id,
        source_type="filing",
        metadata=metadata,
    )


def _validate_path(path: str | Path) -> Path:
    """Validate a filing path and its extension without reading it."""

    filing_path = Path(path)
    if not filing_path.exists():
        raise FileNotFoundError(f"Filing not found: {filing_path}")
    if not filing_path.is_file():
        raise IsADirectoryError(f"Filing path is not a file: {filing_path}")

    extension = filing_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        extension_label = extension or "<none>"
        raise UnsupportedDocumentFormatError(
            f"Unsupported filing extension {extension_label!r}; supported: {supported}"
        )
    return filing_path


def _decode_text(content: bytes, path: Path) -> tuple[str, str]:
    """Decode bytes using UTF-8 variants first, then Windows-1252."""

    if content.startswith(codecs.BOM_UTF8):
        return content.decode("utf-8-sig"), "utf-8-sig"

    for encoding in ("utf-8", "cp1252"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise DocumentDecodingError(
        f"Could not decode {path} as UTF-8, UTF-8 with BOM, or Windows-1252"
    )


def _normalize_newlines(text: str) -> str:
    """Normalize CRLF and CR line endings while preserving text structure."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def _generate_source_id(
    *,
    ticker: str,
    filing_type: FilingType,
    period: str,
    content_sha256: str,
) -> str:
    """Build a stable ID from normalized metadata and normalized content."""

    identity = {
        "content_sha256": content_sha256,
        "filing_type": filing_type,
        "period": period.strip(),
        "ticker": ticker.strip(),
    }
    canonical_identity = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()
    return f"sec-filing-{digest}"


__all__ = [
    "DocumentDecodingError",
    "EmptyDocumentError",
    "SUPPORTED_EXTENSIONS",
    "UnsupportedDocumentFormatError",
    "load_filing",
]
