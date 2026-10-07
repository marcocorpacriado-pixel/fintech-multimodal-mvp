"""Deterministic evidence catalog shared by prompting, grounding and assembly.

The catalog turns retrieved chunks into short, individually addressable excerpts
(``E01``, ``E02``, ...). Every excerpt is a literal ``chunk.text[start:end]``
slice, so it is traceable by offsets and never rewritten. The language model
selects excerpts by ``evidence_id`` only; ``source_id``, ``source_section`` and
the quoted text are rebuilt here, never copied back from model output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schemas import FilingType, RetrievalResult, SourceType


# Adjacent sentence/line units are merged into one contiguous window up to this
# size. A single unit longer than the window is kept whole (never cut).
MAX_EXCERPT_CHARS = 500
# Fragments below this size (headings, page numbers) are not useful evidence.
MIN_EXCERPT_CHARS = 30

# Unit boundaries: sentence end followed by blank space and a plausible sentence
# start, or any line break. Decimals such as "24.9" have no following space and
# are never split. Boundaries only decide where slices begin and end; they never
# alter the text inside a slice.
_BOUNDARY = re.compile(
    r"(?<=[.!?])[ \t]+(?=[A-Z0-9$(\"'‘“])|[ \t]*\n\s*"
)


class EvidenceCatalogError(ValueError):
    """Raised when retrieved evidence cannot form a trustworthy catalog."""


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One selectable excerpt and everything needed to cite it."""

    evidence_id: str
    source_id: str
    source_section: str
    excerpt: str
    source_type: SourceType
    ticker: str
    filing_type: FilingType
    period: str
    rank: int
    start_offset: int
    end_offset: int

    def prompt_payload(self) -> dict[str, Any]:
        """Minimal model-visible view; technical ids stay backend-side."""

        return {
            "evidence_id": self.evidence_id,
            "section": self.source_section,
            "text": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCatalog:
    """Ordered, immutable mapping from ``evidence_id`` to its excerpt."""

    items: tuple[EvidenceItem, ...]
    _by_id: dict[str, EvidenceItem] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        index = {item.evidence_id: item for item in self.items}
        if len(index) != len(self.items):
            raise EvidenceCatalogError("evidence ids must be unique")
        object.__setattr__(self, "_by_id", index)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.items)

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._by_id.get(evidence_id)

    def source_ids_for(self, evidence_ids: Iterable[str]) -> list[str]:
        """Unique source ids in first-seen order for already valid ids."""

        seen: dict[str, None] = {}
        for evidence_id in evidence_ids:
            seen.setdefault(self._by_id[evidence_id].source_id, None)
        return list(seen)


def build_evidence_catalog(
    retrieval_results: Sequence[RetrievalResult],
    *,
    ticker: str,
    max_excerpt_chars: int = MAX_EXCERPT_CHARS,
) -> EvidenceCatalog:
    """Segment retrieved chunks into deterministic, offset-traceable excerpts.

    Order follows retrieval order, then character offset, so the same input
    always yields the same ids. Raises ``EvidenceCatalogError`` for a chunk of
    another ticker, a repeated ``chunk_id`` or a segmentation invariant breach.
    """

    seen_chunk_ids: set[str] = set()
    pending: list[tuple[RetrievalResult, int, int]] = []
    for result in retrieval_results:
        chunk = result.chunk
        if chunk.ticker != ticker:
            raise EvidenceCatalogError(
                f"evidence chunk {chunk.chunk_id!r} belongs to ticker "
                f"{chunk.ticker!r}, not {ticker!r}"
            )
        if chunk.chunk_id in seen_chunk_ids:
            raise EvidenceCatalogError(
                f"duplicate evidence chunk_id: {chunk.chunk_id!r}"
            )
        seen_chunk_ids.add(chunk.chunk_id)
        for start, end in _excerpt_spans(chunk.text, max_excerpt_chars):
            pending.append((result, start, end))

    width = max(2, len(str(len(pending))))
    items: list[EvidenceItem] = []
    for position, (result, start, end) in enumerate(pending, start=1):
        chunk = result.chunk
        excerpt = chunk.text[start:end]
        if not (0 <= start < end <= len(chunk.text)) or excerpt != excerpt.strip():
            raise EvidenceCatalogError(
                f"excerpt offsets are inconsistent for chunk {chunk.chunk_id!r}"
            )
        items.append(
            EvidenceItem(
                evidence_id=f"E{position:0{width}d}",
                source_id=chunk.chunk_id,
                source_section=chunk.section or "UNSECTIONED",
                excerpt=excerpt,
                source_type="filing",
                ticker=chunk.ticker,
                filing_type=chunk.filing_type,
                period=chunk.period,
                rank=result.rank,
                start_offset=start,
                end_offset=end,
            )
        )
    return EvidenceCatalog(items=tuple(items))


def _excerpt_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Return trimmed ``(start, end)`` offsets of contiguous excerpt windows."""

    units: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _BOUNDARY.finditer(text):
        _append_trimmed(units, text, cursor, boundary.start())
        cursor = boundary.end()
    _append_trimmed(units, text, cursor, len(text))

    windows: list[tuple[int, int]] = []
    for start, end in units:
        if windows and end - windows[-1][0] <= max_chars:
            windows[-1] = (windows[-1][0], end)  # contiguous: keeps inner text
        else:
            windows.append((start, end))

    kept = [span for span in windows if span[1] - span[0] >= MIN_EXCERPT_CHARS]
    if kept:
        return kept
    # A short chunk is still valid evidence as one whole item.
    whole: list[tuple[int, int]] = []
    _append_trimmed(whole, text, 0, len(text))
    return whole


def _append_trimmed(
    spans: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    segment = text[start:end]
    stripped = segment.strip()
    if not stripped:
        return
    offset = start + (len(segment) - len(segment.lstrip()))
    spans.append((offset, offset + len(stripped)))


__all__ = [
    "EvidenceCatalog",
    "EvidenceCatalogError",
    "EvidenceItem",
    "MAX_EXCERPT_CHARS",
    "MIN_EXCERPT_CHARS",
    "build_evidence_catalog",
]
