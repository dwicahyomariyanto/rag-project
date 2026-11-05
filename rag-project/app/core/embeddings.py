"""Embedding backends.

* ``OpenAIEmbedder`` -- any OpenAI-compatible ``/v1/embeddings`` endpoint
  (OpenAI, Azure-style gateways, Ollama, LM Studio, vLLM, ...).
* ``HashEmbedder``  -- deterministic token-hashing embedder. It has NO real
  semantic quality and exists only so the whole pipeline (and pytest) can run
  offline without any API key.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BaseEmbedder(ABC):
    """Interface for embedding backends."""

    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (chunk texts)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


class OpenAIEmbedder(BaseEmbedder):
    """Client for any OpenAI-compatible embeddings endpoint."""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str, batch_size: int = 64):
        from openai import OpenAI  # local import keeps test startup fast

        if not api_key:
            raise RuntimeError(
                "EMBEDDING_API_KEY kosong. Isi .env atau set "
                "EMBEDDING_BACKEND=hash untuk mode offline testing."
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
        self.model = model
        self.batch_size = max(1, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self.batch_size]]
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbedder(BaseEmbedder):
    """Deterministic feature-hashing embedder (offline testing only).

    Cosine similarity of two HashEmbedder vectors approximates token overlap,
    which is enough to exercise retrieval logic end-to-end in tests.
    """

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = max(64, dim)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
            vec[(h >> 16) % self.dim] += sign * 0.5
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(settings: Settings) -> BaseEmbedder:
    """Factory: build the embedder selected by ``EMBEDDING_BACKEND``."""
    backend = settings.EMBEDDING_BACKEND.strip().lower()
    if backend == "api":
        return OpenAIEmbedder(
            base_url=settings.EMBEDDING_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    if backend == "hash":
        return HashEmbedder(dim=settings.EMBEDDING_DIM)
    raise ValueError(
        f"EMBEDDING_BACKEND tidak dikenal: '{settings.EMBEDDING_BACKEND}' "
        "(pilihan valid: api | hash)"
    )
