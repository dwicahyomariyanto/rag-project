"""Document management endpoints: ingest, list, delete."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_service
from app.schemas.models import DeleteResponse, IngestResponse, SourceOut
from app.service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Unggah & index satu file PDF ke knowledge base",
)
def ingest_pdf(
    file: UploadFile = File(..., description="File PDF (maks. 50 MB)"),
    service: RAGService = Depends(get_service),
) -> IngestResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file .pdf yang didukung.")
    data = file.file.read()
    try:
        result = service.ingest_pdf_bytes(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # unexpected failures
        logger.exception("Ingest gagal: %s", filename)
        raise HTTPException(status_code=500, detail=f"Ingest gagal: {exc}") from exc
    return IngestResponse(
        status=result.status,
        source_id=result.source_id,
        source_name=result.source_name,
        num_pages=result.num_pages,
        num_chunks=result.num_chunks,
        total_chunks_in_store=result.total_chunks,
        embedding_backend=service.embedder.name,
    )


@router.get(
    "",
    response_model=list[SourceOut],
    summary="Daftar semua dokumen di knowledge base",
)
def list_documents(service: RAGService = Depends(get_service)) -> list[SourceOut]:
    sources = sorted(
        service.store.list_sources().values(),
        key=lambda s: s["source_name"],
    )
    return [SourceOut(**s) for s in sources]


@router.delete(
    "/{source_id}",
    response_model=DeleteResponse,
    summary="Hapus satu dokumen beserta seluruh chunk-nya",
)
def delete_document(
    source_id: str, service: RAGService = Depends(get_service)
) -> DeleteResponse:
    if not service.delete_source(source_id):
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")
    return DeleteResponse(deleted=True, source_id=source_id)
