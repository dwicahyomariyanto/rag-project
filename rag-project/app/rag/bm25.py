"""BM25 keyword search index (rank-bm25) over the stored chunks.

The index is rebuilt in-memory from the ChromaDB contents at startup and
after every ingest/delete. For the intended corpus sizes (up to tens of
thousands of chunks) this takes well under a second.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.rag.models import Chunk

# A compact stopword list (Indonesian + English) -- keeps BM25 focused on
# content words without pulling in heavy NLP dependencies.
_STOPWORDS = frozenset(
    """
    yang dan di ke dari untuk pada dengan adalah itu ini atau sebagai oleh karena
    jika maka agar saat telah sudah akan bisa dapat tidak bukan pula juga dalam
    para kami kita mereka anda saya nya kah lah pun per dalam antara hingga sampai
    tentang terhadap kepada dimana manakah berapa the a an and or of to in for on
    is are was were be been being this that these those it its as at by with from
    what when where how which who whom will would can could should
    """.split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer with stopword removal."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS
    ]


class BM25Index:
    """In-memory BM25 index over all stored chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._records: dict[str, Chunk] = {}

    @property
    def size(self) -> int:
        return len(self._ids)

    def rebuild(self, records: list[Chunk]) -> None:
        """(Re)build the index from the full chunk list."""
        self._records = {r.chunk_id: r for r in records}
        self._ids = [r.chunk_id for r in records]
        corpus = [tokenize(r.text) for r in records]
        self._bm25 = BM25Okapi(corpus) if records else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` (chunk_id, score) pairs, best first."""
        if self._bm25 is None or k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            (self._ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0
        ]

    def record(self, chunk_id: str) -> Chunk | None:
        return self._records.get(chunk_id)
