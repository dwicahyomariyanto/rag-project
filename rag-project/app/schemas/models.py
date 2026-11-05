"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- chat
class ChatRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=4000, description="Pertanyaan pengguna"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Jumlah passage konteks (override)"
    )


class CitationOut(BaseModel):
    id: int
    chunk_id: str
    source_name: str
    page_start: int
    page_end: int
    snippet: str


class RetrievalStatsOut(BaseModel):
    semantic_candidates: int
    bm25_candidates: int
    fused_candidates: int
    context_used: int
    retrieval_ms: float
    generation_ms: float | None = None
    total_ms: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    stats: RetrievalStatsOut


# ----------------------------------------------------------------- retrieve
class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    source_name: str
    page_start: int
    page_end: int
    text: str
    methods: list[str]
    semantic_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrieveResponse(BaseModel):
    question: str
    results: list[RetrievedChunkOut]
    stats: RetrievalStatsOut


# ---------------------------------------------------------------- documents
class IngestResponse(BaseModel):
    status: str = Field(description="'created' atau 'replaced'")
    source_id: str
    source_name: str
    num_pages: int
    num_chunks: int
    total_chunks_in_store: int
    embedding_backend: str


class SourceOut(BaseModel):
    source_id: str
    source_name: str
    num_chunks: int
    first_page: int
    last_page: int


class DeleteResponse(BaseModel):
    deleted: bool
    source_id: str


# ------------------------------------------------------------------- health
class HealthOut(BaseModel):
    status: str
    app: str
    version: str
    llm_backend: str
    llm_model: str
    embedding_backend: str
    embedding_model: str
    reranker_backend: str
    num_chunks: int
    num_sources: int
    storage_dir: str
