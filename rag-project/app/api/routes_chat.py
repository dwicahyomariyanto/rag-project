"""Chat endpoints: blocking answer, SSE streaming answer, retrieval debug."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_service
from app.rag.models import RetrievedChunk
from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunkOut,
    RetrievalStatsOut,
)
from app.service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def _chunk_out(chunk: RetrievedChunk) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        source_name=chunk.source_name,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        methods=chunk.methods,
        semantic_score=chunk.semantic_score,
        bm25_score=chunk.bm25_score,
        rrf_score=chunk.rrf_score,
        rerank_score=chunk.rerank_score,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Tanya knowledge base (jawaban lengkap + sitasi)",
)
def chat(payload: ChatRequest, service: RAGService = Depends(get_service)) -> ChatResponse:
    result = service.pipeline.answer(payload.question, top_k=payload.top_k)
    return ChatResponse(
        question=result.question,
        answer=result.answer,
        citations=[CitationOut(**c) for c in result.citations],
        stats=RetrievalStatsOut(**result.stats),
    )


@router.post(
    "/chat/stream",
    summary="Tanya knowledge base dengan jawaban streaming (Server-Sent Events)",
    description=(
        "Event SSE yang dikirim:\n"
        "- `sources`: daftar passage konteks (dikirim sekali di awal)\n"
        "- `token`: potongan jawaban `{\"delta\": \"...\"}`\n"
        "- `done`: statistik akhir\n"
        "- `error`: pesan kesalahan (jika ada)"
    ),
)
def chat_stream(
    payload: ChatRequest, service: RAGService = Depends(get_service)
) -> StreamingResponse:
    citations, tokens, stats = service.pipeline.stream_answer(
        payload.question, top_k=payload.top_k
    )

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        yield sse("sources", {"citations": citations})
        try:
            for token in tokens:
                yield sse("token", {"delta": token})
            yield sse("done", {"stats": stats})
        except Exception as exc:  # pragma: no cover
            logger.exception("Streaming gagal")
            yield sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Debug retrieval saja (tanpa LLM): lihat hasil hybrid search & skor",
)
def retrieve(
    payload: RetrieveRequest, service: RAGService = Depends(get_service)
) -> RetrieveResponse:
    context, stats = service.pipeline.retrieve(payload.question, top_k=payload.top_k)
    return RetrieveResponse(
        question=payload.question,
        results=[_chunk_out(c) for c in context],
        stats=RetrievalStatsOut(**stats),
    )
