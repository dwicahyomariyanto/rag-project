"""FastAPI application factory and ASGI entrypoint.

Run locally (from the project root):

    uvicorn app.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs   -> Swagger UI (interactive API docs)
    http://localhost:8000/       -> demo chat page
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import routes_chat, routes_documents, routes_health
from app.config import Settings, get_settings
from app.service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_DESCRIPTION = """
**MiniRAG** — REST API untuk *Retrieval-Augmented Generation* (RAG) di atas dokumen PDF.

Fitur:
* **Hybrid search**: pencarian semantik (ChromaDB) + keyword (BM25) yang digabung dengan *Reciprocal Rank Fusion*.
* **Reranking**: cross-encoder menyusun ulang kandidat terbaik sebelum dikirim ke LLM.
* **Sitasi**: setiap jawaban menyertakan rujukan `[n]` ke nama file & halaman PDF.
* **Streaming**: jawaban mengalir kata-per-kata via Server-Sent Events (`/api/chat/stream`).
* **Evaluasi RAGAS**: metrik faithfulness, relevancy, context precision & recall (`scripts/evaluate.py`).
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app (settings injectable for tests)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger.info("Membangun RAGService (embedding=%s, llm=%s)...",
                    settings.EMBEDDING_BACKEND, settings.LLM_BACKEND)
        application.state.service = RAGService.build(settings)
        logger.info("MiniRAG siap menerima request.")
        yield
        logger.info("MiniRAG dimatikan.")

    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local development convenience
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(routes_health.router)
    application.include_router(routes_documents.router)
    application.include_router(routes_chat.router)

    @application.get("/", include_in_schema=False)
    def demo_page() -> FileResponse:
        """Serve the built-in demo chat page."""
        return FileResponse(_STATIC_DIR / "index.html")

    return application


# ASGI entrypoint: `uvicorn app.main:app`
app = create_app()
