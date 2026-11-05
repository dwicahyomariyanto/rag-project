"""Text chunking.

Strategy
--------
1. Each PDF page is whitespace-normalized into one "unit" tagged with its
   page number (this keeps citations page-accurate).
2. Units are packed greedily into chunks up to ``CHUNK_SIZE`` characters.
3. Units larger than ``CHUNK_SIZE`` are hard-split sentence-by-sentence (and
   word-wise if a single sentence is still too long) with ``CHUNK_OVERLAP``
   characters of sliding overlap between consecutive pieces.
4. Chunks smaller than ``MIN_CHUNK_CHARS`` are merged into the previous chunk.
"""

from __future__ import annotations

import re

from app.rag.loader import PageContent
from app.rag.models import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return " ".join(text.split())


def _split_long_sentence(sentence: str, size: int, overlap: int) -> list[str]:
    """Word-level sliding window for sentences longer than ``size``."""
    words = sentence.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current_len + addition > size and current:
            pieces.append(" ".join(current))
            # carry the last `overlap` characters as the next window's head
            carry: list[str] = []
            carry_len = 0
            for w in reversed(current):
                if carry_len + len(w) + 1 > overlap:
                    break
                carry.insert(0, w)
                carry_len += len(w) + 1
            current, current_len = carry, carry_len
            addition = len(word) + (1 if current else 0)
        current.append(word)
        current_len += addition

    if current:
        pieces.append(" ".join(current))
    return pieces


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split oversized text on sentence boundaries, with word-level fallback."""
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in _SENTENCE_RE.split(text):
        if not sentence:
            continue
        if len(sentence) > size:
            if current:
                pieces.append(" ".join(current))
                current, current_len = [], 0
            pieces.extend(_split_long_sentence(sentence, size, overlap))
            continue
        addition = len(sentence) + (1 if current else 0)
        if current_len + addition > size and current:
            pieces.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += addition

    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if p.strip()]


def chunk_pages(
    pages: list[PageContent],
    *,
    source_id: str,
    source_name: str,
    size: int,
    overlap: int,
    min_chars: int,
) -> list[Chunk]:
    """Turn extracted PDF pages into metadata-rich chunks."""
    overlap = max(0, min(overlap, size // 2))
    units = [
        (page.page_number, _normalize(page.text)) for page in pages
    ]
    units = [(page, text) for page, text in units if text]

    base: list[tuple[str, int, int]] = []  # (text, page_start, page_end)
    current_units: list[tuple[int, str]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_units, current_len
        if current_units:
            text = "\n\n".join(t for _, t in current_units)
            base.append((text, current_units[0][0], current_units[-1][0]))
        current_units, current_len = [], 0

    for page_number, text in units:
        if len(text) > size:
            flush()
            for piece in _hard_split(text, size, overlap):
                base.append((piece, page_number, page_number))
            continue
        addition = len(text) + (2 if current_units else 0)  # "\n\n" separator
        if current_len + addition > size and current_units:
            flush()
        current_units.append((page_number, text))
        current_len += addition
    flush()

    chunks: list[Chunk] = []
    for text, page_start, page_end in base:
        if chunks and len(text) < min_chars:
            previous = chunks[-1]
            previous.text = f"{previous.text}\n\n{text}"
            previous.page_end = max(previous.page_end, page_end)
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}-{len(chunks)}",
                text=text,
                source_id=source_id,
                source_name=source_name,
                page_start=page_start,
                page_end=page_end,
                chunk_index=len(chunks),
            )
        )
    return chunks
