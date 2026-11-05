"""Reranking backends.

* ``CrossEncoderReranker`` -- sentence-transformers cross-encoder
  (default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``). Lazy-loads the model on
  first use; falls back to ``LexicalReranker`` if the model/library cannot be
  loaded so the API never hard-fails.
* ``LexicalReranker``       -- zero-dependency token-overlap scorer (used by
  the offline test-suite).
* ``NullReranker``          -- identity (keeps RRF order, truncates only).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from app.config import Settings
from app.rag.bm25 import tokenize
from app.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Interface for rerankers."""

    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return candidates re-ordered by relevance (best first)."""


class CrossEncoderReranker(BaseReranker):
    """Semantic reranker using a cross-encoder model."""

    name = "cross_encoder"

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._failed = False

    def _ensure_model(self):
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Memuat cross-encoder '%s'...", self._model_name)
                self._model = CrossEncoder(self._model_name)
            except Exception as exc:  # library missing, model offline, etc.
                logger.warning(
                    "Cross-encoder tidak dapat dimuat (%s). "
                    "Fallback ke LexicalReranker.",
                    exc,
                )
                self._failed = True
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if len(candidates) <= 1:
            return candidates
        model = self._ensure_model()
        if model is None:
            return LexicalReranker().rerank(query, candidates)
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)


class LexicalReranker(BaseReranker):
    """Lightweight lexical scorer: weighted token overlap (no dependencies)."""

    name = "lexical"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        query_tokens = set(tokenize(query))
        for chunk in candidates:
            doc_tokens = set(tokenize(chunk.text))
            if doc_tokens and query_tokens:
                overlap = len(query_tokens & doc_tokens)
                chunk.rerank_score = overlap / math.sqrt(len(doc_tokens))
            else:
                chunk.rerank_score = 0.0
        return sorted(
            candidates, key=lambda c: c.rerank_score or 0.0, reverse=True
        )


class NullReranker(BaseReranker):
    """Passthrough reranker (no reordering)."""

    name = "none"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return candidates


def build_reranker(settings: Settings) -> BaseReranker:
    """Factory: build the reranker selected by ``RERANK_BACKEND``."""
    backend = settings.RERANK_BACKEND.strip().lower()
    if backend == "cross_encoder":
        return CrossEncoderReranker(settings.RERANK_MODEL)
    if backend == "lexical":
        return LexicalReranker()
    if backend == "none":
        return NullReranker()
    raise ValueError(
        f"RERANK_BACKEND tidak dikenal: '{settings.RERANK_BACKEND}' "
        "(pilihan valid: cross_encoder | lexical | none)"
    )
