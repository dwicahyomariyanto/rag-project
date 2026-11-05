"""Service layer: wires every component together and owns ingestion logic."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.core.embeddings import BaseEmbedder, build_embedder
from app.core.llm import BaseLLM, build_llm
from app.rag.bm25 import BM25Index
from app.rag.chunker import chunk_pages
from app.rag.loader import extract_pdf
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import BaseReranker, build_reranker
from app.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class IngestResult:
    """Outcome of a successful PDF ingestion."""

    status: str  # "created" | "replaced"
    source_id: str
    source_name: str
    num_pages: int
    num_chunks: int
    total_chunks: int


class RAGService:
    """Owns the full RAG stack (embedder, stores, reranker, LLM, pipeline)."""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: BaseReranker,
        llm: BaseLLM,
        pipeline: RAGPipeline,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.llm = llm
        self.pipeline = pipeline

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, settings: Settings) -> "RAGService":
        """Construct every component from settings (used at app startup)."""
        Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
        embedder = build_embedder(settings)
        store = VectorStore(settings.chroma_dir)
        bm25 = BM25Index()
        bm25.rebuild(store.all_records())
        reranker = build_reranker(settings)
        llm = build_llm(settings)
        pipeline = RAGPipeline(
            settings=settings,
            embedder=embedder,
            store=store,
            bm25=bm25,
            reranker=reranker,
            llm=llm,
        )
        logger.info(
            "RAGService siap | llm=%s | embedding=%s | rerank=%s | chunks=%d",
            llm.name,
            embedder.name,
            reranker.name,
            store.count(),
        )
        return cls(settings, embedder, store, bm25, reranker, llm, pipeline)

    # -------------------------------------------------------------- ingestion
    def ingest_pdf_bytes(self, filename: str, data: bytes) -> IngestResult:
        """Extract, chunk, embed and store a PDF provided as raw bytes.

        Re-uploading the same file (same content hash) or the same filename
        replaces the previous version atomically.
        """
        if not data:
            raise ValueError("File kosong.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Ukuran file melebihi batas 50 MB.")

        source_id = hashlib.md5(data).hexdigest()[:16]
        safe_name = Path(filename).name
        upload_path = Path(self.settings.uploads_dir) / f"{source_id}-{safe_name}"
        upload_path.write_bytes(data)

        pages = extract_pdf(upload_path)
        chunks = chunk_pages(
            pages,
            source_id=source_id,
            source_name=safe_name,
            size=self.settings.CHUNK_SIZE,
            overlap=self.settings.CHUNK_OVERLAP,
            min_chars=self.settings.MIN_CHUNK_CHARS,
        )
        if not chunks:
            raise ValueError(
                f"Tidak ada potongan teks yang dihasilkan dari '{safe_name}'."
            )

        existing = self.store.list_sources()
        existed = source_id in existing or any(
            s["source_name"] == safe_name for s in existing.values()
        )

        self.store.delete_source(source_id=source_id, source_name=safe_name)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        self.rebuild_bm25()

        logger.info(
            "Ingest '%s' -> %d halaman, %d chunks (status=%s)",
            safe_name,
            len(pages),
            len(chunks),
            "replaced" if existed else "created",
        )
        return IngestResult(
            status="replaced" if existed else "created",
            source_id=source_id,
            source_name=safe_name,
            num_pages=len(pages),
            num_chunks=len(chunks),
            total_chunks=self.store.count(),
        )

    def delete_source(self, source_id: str) -> bool:
        """Remove a document (all its chunks) from the knowledge base."""
        sources = self.store.list_sources()
        if source_id not in sources:
            return False
        self.store.delete_source(source_id=source_id)
        self.rebuild_bm25()
        logger.info("Dokumen dihapus: %s", source_id)
        return True

    def rebuild_bm25(self) -> None:
        self.bm25.rebuild(self.store.all_records())

    # ----------------------------------------------------------------- health
    def health_snapshot(self) -> dict:
        sources = self.store.list_sources()
        return {
            "status": "ok",
            "app": self.settings.APP_NAME,
            "version": self.settings.APP_VERSION,
            "llm_backend": self.llm.name,
            "llm_model": self.settings.LLM_MODEL,
            "embedding_backend": self.embedder.name,
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "reranker_backend": self.reranker.name,
            "num_chunks": self.store.count(),
            "num_sources": len(sources),
            "storage_dir": str(self.settings.STORAGE_DIR),
        }
