"""ChromaDB vector store wrapper (persistent, cosine space)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "rag_chunks"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB siap: %s (%d chunks)", persist_dir, self._collection.count()
        )

    # ------------------------------------------------------------------ write
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_id": c.source_id,
                    "source_name": c.source_name,
                    "page_start": int(c.page_start),
                    "page_end": int(c.page_end),
                    "chunk_index": int(c.chunk_index),
                }
                for c in chunks
            ],
        )

    def delete_source(
        self, source_id: str | None = None, source_name: str | None = None
    ) -> None:
        """Delete all chunks belonging to a document (by id and/or name)."""
        conditions = []
        if source_id:
            conditions.append({"source_id": source_id})
        if source_name:
            conditions.append({"source_name": source_name})
        if not conditions:
            return
        where = conditions[0] if len(conditions) == 1 else {"$or": conditions}
        try:
            self._collection.delete(where=where)
        except Exception:  # pragma: no cover - chroma raises on empty matches
            logger.debug("delete_source: tidak ada chunk yang cocok", exc_info=True)

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self._collection.count()

    def query(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        """Nearest-neighbour search; returns chunks ordered by similarity."""
        total = self._collection.count()
        if total == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    source_id=str(meta.get("source_id", "")),
                    source_name=str(meta.get("source_name", "")),
                    page_start=int(meta.get("page_start", 0)),
                    page_end=int(meta.get("page_end", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    semantic_score=1.0 - float(distance),
                )
            )
        return hits

    def all_records(self) -> list[Chunk]:
        """Return every stored chunk (used to (re)build the BM25 index)."""
        data = self._collection.get(include=["documents", "metadatas"])
        records: list[Chunk] = []
        for chunk_id, text, meta in zip(
            data["ids"], data["documents"] or [], data["metadatas"] or []
        ):
            records.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    source_id=str(meta.get("source_id", "")),
                    source_name=str(meta.get("source_name", "")),
                    page_start=int(meta.get("page_start", 0)),
                    page_end=int(meta.get("page_end", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        return records

    def list_sources(self) -> dict[str, dict]:
        """Aggregate per-document statistics (chunk count, page span)."""
        data = self._collection.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for meta in data["metadatas"] or []:
            source_id = str(meta.get("source_id", ""))
            if not source_id:
                continue
            entry = sources.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "source_name": str(meta.get("source_name", "")),
                    "num_chunks": 0,
                    "first_page": 10**9,
                    "last_page": 0,
                },
            )
            entry["num_chunks"] += 1
            entry["first_page"] = min(
                entry["first_page"], int(meta.get("page_start", 0)) or entry["first_page"]
            )
            entry["last_page"] = max(entry["last_page"], int(meta.get("page_end", 0)))
        return sources
