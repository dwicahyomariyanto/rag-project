"""Application configuration.

All settings are loaded from environment variables / a `.env` file placed at
the project root. See `.env.example` for a fully documented template.

Konfigurasi aplikasi dibaca dari environment variable / file `.env`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object (type-safe, validated, overridable via kwargs)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ------------------------------------------------------------------ app
    APP_NAME: str = "MiniRAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ------------------------------------------------------------------ LLM
    # LLM_BACKEND: "openai" (any OpenAI-compatible endpoint) or "fake"
    # ("fake" returns a deterministic canned answer -- offline pipeline testing).
    LLM_BACKEND: str = "openai"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    # ------------------------------------------------------------ embeddings
    # EMBEDDING_BACKEND: "api" (OpenAI-compatible /v1/embeddings) or "hash"
    # ("hash" is a deterministic token-hashing embedder for OFFLINE TESTING
    # only -- it has no semantic quality, but makes pytest runnable anywhere).
    EMBEDDING_BACKEND: str = "api"
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 512  # only used by the "hash" backend

    # -------------------------------------------------------------- chunking
    CHUNK_SIZE: int = 900        # max characters per chunk
    CHUNK_OVERLAP: int = 150     # overlap (chars) between hard-split pieces
    MIN_CHUNK_CHARS: int = 80    # chunks smaller than this get merged forward

    # -------------------------------------------------------------- retrieval
    RETRIEVAL_K_SEMANTIC: int = 20   # candidates from ChromaDB (vector search)
    RETRIEVAL_K_BM25: int = 20       # candidates from BM25 (keyword search)
    RRF_K: int = 60                  # standard Reciprocal Rank Fusion constant
    CONTEXT_TOP_K: int = 5           # final passages sent to the LLM

    # -------------------------------------------------------------- reranking
    # RERANK_BACKEND: "cross_encoder" (ms-marco MiniLM, needs
    # sentence-transformers), "lexical" (zero-dependency fallback used by the
    # offline test-suite), or "none".
    RERANK_BACKEND: str = "cross_encoder"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------------------------------------------------------------- storage
    STORAGE_DIR: Path = Path("storage")

    # ------------------------------------------------------------ derived paths
    @property
    def chroma_dir(self) -> Path:
        return Path(self.STORAGE_DIR) / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return Path(self.STORAGE_DIR) / "uploads"

    @property
    def eval_dir(self) -> Path:
        return Path(self.STORAGE_DIR) / "eval_results"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance built from the process environment."""
    return Settings()
