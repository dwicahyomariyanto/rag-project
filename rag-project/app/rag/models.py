"""Shared domain models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text stored in the knowledge base."""

    chunk_id: str
    text: str
    source_id: str
    source_name: str
    page_start: int
    page_end: int
    chunk_index: int


@dataclass
class RetrievedChunk(Chunk):
    """A chunk produced by retrieval, carrying per-stage relevance scores."""

    semantic_score: float | None = None  # cosine similarity (ChromaDB)
    bm25_score: float | None = None      # raw BM25 score
    rrf_score: float | None = None       # reciprocal rank fusion score
    rerank_score: float | None = None    # cross-encoder / lexical score

    @property
    def methods(self) -> list[str]:
        """Which retrieval channels matched this chunk."""
        out: list[str] = []
        if self.semantic_score is not None:
            out.append("semantic")
        if self.bm25_score is not None:
            out.append("bm25")
        return out

    def primary_score(self) -> float:
        """Best available relevance score (reranker wins, then semantic)."""
        for value in (self.rerank_score, self.semantic_score, self.bm25_score):
            if value is not None:
                return float(value)
        return 0.0
