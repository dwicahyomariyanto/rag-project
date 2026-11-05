"""The RAG orchestration pipeline.

Flow
----
retrieve:  embed query -> ChromaDB semantic search + BM25 keyword search
           -> Reciprocal Rank Fusion -> reranker -> top-K context passages
answer:    build prompt -> LLM completion -> parse [n] citations
stream:    same, but LLM tokens are streamed (consumed by the SSE endpoint)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from app.config import Settings
from app.core.embeddings import BaseEmbedder
from app.core.llm import BaseLLM
from app.rag.bm25 import BM25Index
from app.rag.hybrid import rrf_fuse
from app.rag.models import RetrievedChunk
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.rag.reranker import BaseReranker
from app.rag.vectorstore import VectorStore

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_NO_CONTEXT_ANSWER = (
    "Maaf, tidak ada potongan dokumen yang relevan ditemukan di knowledge base "
    "untuk pertanyaan ini. Pastikan dokumen terkait sudah diunggah."
)


@dataclass
class RagAnswer:
    """Result of a full (non-streaming) RAG call."""

    question: str
    answer: str
    contexts: list[RetrievedChunk] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class RAGPipeline:
    """Retrieval-augmented generation orchestrator."""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: BaseReranker,
        llm: BaseLLM,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.llm = llm

    # -------------------------------------------------------------- retrieval
    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], dict]:
        """Hybrid retrieval: semantic + BM25 -> RRF -> rerank -> top-K."""
        started = time.perf_counter()

        query_embedding = self.embedder.embed_query(question)
        semantic = self.store.query(
            query_embedding, self.settings.RETRIEVAL_K_SEMANTIC
        )
        bm25_hits = self.bm25.search(question, self.settings.RETRIEVAL_K_BM25)

        # Merge candidates from both channels.
        candidates: dict[str, RetrievedChunk] = {}
        for chunk in semantic:
            candidates[chunk.chunk_id] = chunk
        for chunk_id, score in bm25_hits:
            if chunk_id in candidates:
                candidates[chunk_id].bm25_score = score
            else:
                record = self.bm25.record(chunk_id)
                if record is not None:
                    candidates[chunk_id] = RetrievedChunk(
                        chunk_id=record.chunk_id,
                        text=record.text,
                        source_id=record.source_id,
                        source_name=record.source_name,
                        page_start=record.page_start,
                        page_end=record.page_end,
                        chunk_index=record.chunk_index,
                        bm25_score=score,
                    )

        # Reciprocal Rank Fusion over the two rankings.
        fused_ids = rrf_fuse(
            [
                [c.chunk_id for c in semantic],
                [chunk_id for chunk_id, _ in bm25_hits],
            ],
            k=self.settings.RRF_K,
        )
        for rank, chunk_id in enumerate(fused_ids, start=1):
            candidates[chunk_id].rrf_score = 1.0 / (self.settings.RRF_K + rank)

        fused = [candidates[cid] for cid in fused_ids]
        top_k = top_k or self.settings.CONTEXT_TOP_K
        context = self.reranker.rerank(question, fused)[:top_k]

        stats = {
            "semantic_candidates": len(semantic),
            "bm25_candidates": len(bm25_hits),
            "fused_candidates": len(fused),
            "context_used": len(context),
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 1),
            "generation_ms": None,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        return context, stats

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _build_messages(question: str, contexts: list[RetrievedChunk]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts)},
        ]

    @staticmethod
    def build_citations(
        answer: str, contexts: list[RetrievedChunk]
    ) -> list[dict]:
        """Extract ``[n]`` markers from the answer and map them to passages."""
        used: list[int] = []
        for match in _CITATION_RE.findall(answer):
            number = int(match)
            if 1 <= number <= len(contexts) and number not in used:
                used.append(number)
        if not used:
            # Model did not cite anything; expose the top passages instead.
            used = list(range(1, min(len(contexts), 3) + 1))
        return [
            {
                "id": number,
                "chunk_id": contexts[number - 1].chunk_id,
                "source_name": contexts[number - 1].source_name,
                "page_start": contexts[number - 1].page_start,
                "page_end": contexts[number - 1].page_end,
                "snippet": contexts[number - 1].text[:240],
            }
            for number in used
        ]

    @staticmethod
    def _context_citations(contexts: list[RetrievedChunk]) -> list[dict]:
        """Citation list for streaming (all context passages, numbered)."""
        return [
            {
                "id": index,
                "chunk_id": chunk.chunk_id,
                "source_name": chunk.source_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "snippet": chunk.text[:240],
            }
            for index, chunk in enumerate(contexts, start=1)
        ]

    # ---------------------------------------------------------------- answers
    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        """Full RAG answer with parsed citations."""
        started = time.perf_counter()
        context, stats = self.retrieve(question, top_k)
        if not context:
            stats["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return RagAnswer(
                question=question, answer=_NO_CONTEXT_ANSWER, stats=stats
            )

        messages = self._build_messages(question, context)
        generation_started = time.perf_counter()
        answer_text = self.llm.complete(messages)
        stats["generation_ms"] = round(
            (time.perf_counter() - generation_started) * 1000, 1
        )
        stats["total_ms"] = round((time.perf_counter() - started) * 1000, 1)

        return RagAnswer(
            question=question,
            answer=answer_text,
            contexts=context,
            citations=self.build_citations(answer_text, context),
            stats=stats,
        )

    def stream_answer(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[dict], Iterator[str], dict]:
        """Streaming variant.

        Returns:
            (citations, token_iterator, stats) -- ``citations`` covers all
            context passages (numbers match the ``[n]`` markers the model is
            instructed to emit); ``stats`` is final after the iterator is
            exhausted.
        """
        context, stats = self.retrieve(question, top_k)
        citations = self._context_citations(context)

        if not context:
            def no_context_tokens() -> Iterator[str]:
                for word in _NO_CONTEXT_ANSWER.split(" "):
                    yield word + " "

            return citations, no_context_tokens(), stats

        messages = self._build_messages(question, context)

        def token_stream() -> Iterator[str]:
            started = time.perf_counter()
            try:
                yield from self.llm.stream(messages)
            finally:
                stats["generation_ms"] = round(
                    (time.perf_counter() - started) * 1000, 1
                )
                stats["total_ms"] = round(
                    stats["retrieval_ms"] + stats["generation_ms"], 1
                )

        return citations, token_stream(), stats
