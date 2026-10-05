"""SEC-aware, paragraph-preserving chunking for normalized filings.

The section detector is intentionally heuristic rather than a regulatory
parser. It recognizes conventional line-oriented SEC headings and filters
probable table-of-contents duplicates before producing traceable chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import DocumentChunk, FilingType, LoadedDocument


DEFAULT_MAX_CHARS = 4_000
DEFAULT_OVERLAP_CHARS = 400
UNSECTIONED = "UNSECTIONED"

_MAX_HEADING_CHARS = 180
_TOC_CONTEXT_CHARS = 3_000
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
_PART_RE = re.compile(r"^PART\s+(I|II)\b(.*)$", re.IGNORECASE)
_ITEM_RE = re.compile(
    r"^ITEM\s+(\d{1,2})([A-Z]?)(?=\s|[.:\-\u2013\u2014]|$)(.*)$",
    re.IGNORECASE,
)
_TOC_PAGE_RE = re.compile(r"(?:\.{2,}|\s{3,})\d+\s*$")

_TEN_K_ITEMS = {
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
}
_TEN_Q_ITEMS = {
    "I": {"1", "2", "3", "4"},
    "II": {"1", "1A", "2", "3", "4", "5", "6"},
}


@dataclass(frozen=True, slots=True)
class _Span:
    """Half-open character span over the source document."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _SectionSpan(_Span):
    """A source span assigned to one stable SEC section identifier."""

    section: str


@dataclass(frozen=True, slots=True)
class _HeadingCandidate:
    """A possible SEC heading found on a single source line."""

    start: int
    line_number: int
    section: str
    kind: str
    line: str


def chunk_document(
    document: LoadedDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Split a loaded filing into ordered, traceable SEC-aware chunks.

    Paragraphs are kept intact whenever each paragraph fits within
    ``max_chars``. Overlap reuses complete trailing paragraphs where possible;
    character-level overlap is reserved for paragraphs that already require
    direct subdivision.

    Args:
        document: Validated normalized filing from ``load_filing``.
        max_chars: Maximum number of source characters in one chunk.
        overlap_chars: Maximum same-section context repeated between chunks.

    Returns:
        Ordered ``DocumentChunk`` instances with exact source offsets.

    Raises:
        ValueError: If the size configuration is invalid.
    """

    _validate_chunk_sizes(max_chars, overlap_chars)
    section_spans = _detect_section_spans(document.text, document.filing_type)

    chunk_spans: list[_SectionSpan] = []
    for section_span in section_spans:
        for span in _chunk_section(
            document.text,
            section_span,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            chunk_spans.append(
                _SectionSpan(span.start, span.end, section_span.section)
            )

    chunks = [
        DocumentChunk(
            chunk_id=_build_chunk_id(
                document.source_id,
                span.section,
                span.start,
                span.end,
            ),
            ticker=document.ticker,
            filing_type=document.filing_type,
            period=document.period,
            section=span.section,
            text=document.text[span.start : span.end],
            source_id=document.source_id,
            start_char=span.start,
            end_char=span.end,
        )
        for span in chunk_spans
    ]

    return chunks


def _validate_chunk_sizes(max_chars: int, overlap_chars: int) -> None:
    """Reject configurations that cannot make deterministic progress."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")


def _detect_section_spans(text: str, filing_type: FilingType) -> list[_SectionSpan]:
    """Convert accepted SEC heading candidates into non-overlapping spans."""

    candidates = _find_heading_candidates(text, filing_type)
    headings = _filter_table_of_contents(text, candidates)

    if not headings:
        start, end = _trim_span(text, 0, len(text))
        return [_SectionSpan(start, end, UNSECTIONED)] if start < end else []

    sections: list[_SectionSpan] = []
    prefix_start, prefix_end = _trim_span(text, 0, headings[0].start)
    if prefix_start < prefix_end:
        sections.append(_SectionSpan(prefix_start, prefix_end, UNSECTIONED))

    for index, heading in enumerate(headings):
        raw_end = headings[index + 1].start if index + 1 < len(headings) else len(text)
        start, end = _trim_span(text, heading.start, raw_end)
        if start < end:
            sections.append(_SectionSpan(start, end, heading.section))
    return sections


def _find_heading_candidates(
    text: str,
    filing_type: FilingType,
) -> list[_HeadingCandidate]:
    """Find plausible line-oriented Part and Item headings."""

    candidates: list[_HeadingCandidate] = []
    current_part: str | None = None
    is_quarterly = filing_type in {"10-Q", "10-Q/A"}

    for line_number, match in enumerate(re.finditer(r"(?m)^.*$", text), start=1):
        raw_line = match.group(0)
        stripped = raw_line.strip()
        if not stripped or len(stripped) > _MAX_HEADING_CHARS:
            continue

        heading_start = match.start() + len(raw_line) - len(raw_line.lstrip())

        if is_quarterly:
            part = _parse_part_heading(stripped)
            if part is not None:
                current_part = part
                candidates.append(
                    _HeadingCandidate(
                        heading_start,
                        line_number,
                        f"PART_{part}",
                        "part",
                        stripped,
                    )
                )
                continue

        item = _parse_item_heading(stripped)
        if item is None or not _is_allowed_item(item, filing_type, current_part):
            continue

        if is_quarterly and current_part is not None:
            section = f"PART_{current_part}_ITEM_{item}"
        else:
            section = f"ITEM_{item}"
        candidates.append(
            _HeadingCandidate(
                heading_start,
                line_number,
                section,
                "item",
                stripped,
            )
        )
    return candidates


def _parse_part_heading(line: str) -> str | None:
    """Return a normalized 10-Q Part identifier for heading-like lines."""

    match = _PART_RE.fullmatch(line)
    if match is None:
        return None
    remainder = match.group(2)
    if remainder and not _valid_heading_remainder(remainder):
        return None
    return match.group(1).upper()


def _parse_item_heading(line: str) -> str | None:
    """Return a normalized Item identifier for heading-like lines."""

    match = _ITEM_RE.fullmatch(line)
    if match is None:
        return None
    remainder = match.group(3)
    if remainder and not _valid_heading_remainder(remainder):
        return None

    number = str(int(match.group(1)))
    suffix = match.group(2).upper()
    return f"{number}{suffix}"


def _valid_heading_remainder(remainder: str) -> bool:
    """Reject prose mentions while accepting punctuation or title-like text."""

    if not remainder.strip():
        return True

    stripped = remainder.strip()
    if stripped[0] in ".:-\u2013\u2014":
        return True
    if len(remainder) - len(remainder.lstrip()) >= 2:
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'&/-]*", stripped)
    if not words or len(words) > 16:
        return False
    significant = [word for word in words if len(word) > 2]
    return bool(significant) and all(word[0].isupper() for word in significant)


def _is_allowed_item(
    item: str,
    filing_type: FilingType,
    current_part: str | None,
) -> bool:
    """Limit detection to item numbers meaningful for the filing form."""

    if filing_type in {"10-K", "10-K/A"}:
        return item in _TEN_K_ITEMS
    if current_part is None:
        return item in _TEN_Q_ITEMS["I"] | _TEN_Q_ITEMS["II"]
    return item in _TEN_Q_ITEMS[current_part]


def _filter_table_of_contents(
    text: str,
    candidates: list[_HeadingCandidate],
) -> list[_HeadingCandidate]:
    """Discard early duplicate headings that have explicit TOC signals.

    A candidate is never discarded unless the same normalized section occurs
    later. Additional evidence must be present: a nearby "table of contents"
    marker, a page-number leader, or membership in a dense run of at least
    three Item headings. This intentionally favors false negatives over
    deleting a unique real section.
    """

    last_start_by_section = {
        candidate.section: candidate.start for candidate in candidates
    }
    dense_indexes = _dense_item_candidate_indexes(candidates)
    folded_text = text.casefold()
    accepted: list[_HeadingCandidate] = []

    for index, candidate in enumerate(candidates):
        has_later_duplicate = candidate.start < last_start_by_section[candidate.section]
        context_start = max(0, candidate.start - _TOC_CONTEXT_CHARS)
        near_toc_marker = "table of contents" in folded_text[
            context_start : candidate.start
        ]
        has_page_number = bool(_TOC_PAGE_RE.search(candidate.line))
        in_dense_item_run = index in dense_indexes

        if has_later_duplicate and (
            near_toc_marker or has_page_number or in_dense_item_run
        ):
            continue
        accepted.append(candidate)
    return accepted


def _dense_item_candidate_indexes(
    candidates: list[_HeadingCandidate],
) -> set[int]:
    """Locate runs of three Item headings separated by at most one line."""

    dense: set[int] = set()
    run: list[int] = []

    for index, candidate in enumerate(candidates):
        if candidate.kind != "item":
            if len(run) >= 3:
                dense.update(run)
            run = []
            continue

        if run and candidate.line_number - candidates[run[-1]].line_number > 2:
            if len(run) >= 3:
                dense.update(run)
            run = []
        run.append(index)

    if len(run) >= 3:
        dense.update(run)
    return dense


def _chunk_section(
    text: str,
    section: _SectionSpan,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[_Span]:
    """Chunk one section without allowing output to cross its boundary."""

    paragraphs = _paragraph_spans(text, section.start, section.end)
    chunks: list[_Span] = []
    buffer: list[_Span] = []

    for paragraph in paragraphs:
        if paragraph.end - paragraph.start > max_chars:
            if buffer:
                chunks.append(_Span(buffer[0].start, buffer[-1].end))
                buffer = []
            chunks.extend(
                _split_oversized_paragraph(
                    text,
                    paragraph,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
            continue

        if not buffer:
            buffer = [paragraph]
            continue

        if paragraph.end - buffer[0].start <= max_chars:
            buffer.append(paragraph)
            continue

        emitted = _Span(buffer[0].start, buffer[-1].end)
        chunks.append(emitted)
        overlap = _select_paragraph_overlap(
            buffer,
            emitted_end=emitted.end,
            next_paragraph=paragraph,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        buffer = [*overlap, paragraph]

    if buffer:
        chunks.append(_Span(buffer[0].start, buffer[-1].end))
    return chunks


def _paragraph_spans(text: str, start: int, end: int) -> list[_Span]:
    """Return non-empty blocks separated by blank lines."""

    paragraphs: list[_Span] = []
    cursor = start
    for separator in _BLANK_LINE_RE.finditer(text, start, end):
        paragraph_start, paragraph_end = _trim_span(text, cursor, separator.start())
        if paragraph_start < paragraph_end:
            paragraphs.append(_Span(paragraph_start, paragraph_end))
        cursor = separator.end()

    paragraph_start, paragraph_end = _trim_span(text, cursor, end)
    if paragraph_start < paragraph_end:
        paragraphs.append(_Span(paragraph_start, paragraph_end))
    return paragraphs


def _select_paragraph_overlap(
    paragraphs: list[_Span],
    *,
    emitted_end: int,
    next_paragraph: _Span,
    max_chars: int,
    overlap_chars: int,
) -> list[_Span]:
    """Select the largest suffix of complete paragraphs within both limits."""

    if overlap_chars == 0:
        return []

    selected: list[_Span] = []
    for index in range(len(paragraphs) - 1, -1, -1):
        candidate_start = paragraphs[index].start
        if emitted_end - candidate_start > overlap_chars:
            break
        if next_paragraph.end - candidate_start > max_chars:
            break
        selected = paragraphs[index:]
    return selected


def _split_oversized_paragraph(
    text: str,
    paragraph: _Span,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[_Span]:
    """Split one oversized paragraph with bounded character overlap."""

    chunks: list[_Span] = []
    start = paragraph.start

    while start < paragraph.end:
        end = min(start + max_chars, paragraph.end)
        if end < paragraph.end:
            end = _preferred_whitespace_cut(text, start, end, max_chars)
        end = _trim_span(text, start, end)[1]
        if end <= start:
            end = min(start + max_chars, paragraph.end)

        chunks.append(_Span(start, end))
        if end >= paragraph.end:
            break

        next_start = max(start + 1, end - overlap_chars)
        while next_start < paragraph.end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def _preferred_whitespace_cut(
    text: str,
    start: int,
    hard_end: int,
    max_chars: int,
) -> int:
    """Prefer a word boundary in the latter half of an oversized slice."""

    search_start = start + max_chars // 2
    for position in range(hard_end - 1, search_start - 1, -1):
        if text[position].isspace():
            return position
    return hard_end


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim boundary whitespace while retaining exact offsets into ``text``."""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _build_chunk_id(source_id: str, section: str, start: int, end: int) -> str:
    """Build a readable deterministic identifier from traceability fields."""

    return f"{source_id}:{section}:{start}:{end}"


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "UNSECTIONED",
    "chunk_document",
]
