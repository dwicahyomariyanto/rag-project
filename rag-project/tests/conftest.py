"""Shared pytest fixtures and helpers.

The whole test-suite runs FULLY OFFLINE:
* LLM_BACKEND=fake       -> deterministic canned LLM
* EMBEDDING_BACKEND=hash -> deterministic token-hashing embedder
* RERANK_BACKEND=lexical -> zero-dependency lexical reranker
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import RAGService

# ---------------------------------------------------------------- sample PDF
# 3 short "pages" with distinct facts -- deliberately overlapping vocabulary
# so hybrid search (semantic + BM25) has something interesting to fuse.
SAMPLE_PAGES = [
    (
        "Kebijakan Cuti PT Contoh Sejahtera. "
        "Cuti tahunan karyawan tetap adalah 12 hari kerja per tahun. "
        "Setelah 5 tahun masa kerja, hak cuti bertambah menjadi 14 hari kerja. "
        "Pengajuan cuti dilakukan melalui HRIS minimal 7 hari kerja sebelumnya "
        "dan memerlukan persetujuan atasan langsung. "
        "Sisa cuti dapat ditangguhkan maksimal 6 hari kerja sampai akhir Maret."
    ),
    (
        "Kerja jarak jauh atau work from home maksimal 2 hari per minggu "
        "dengan persetujuan atasan langsung. "
        "Jam kerja resmi adalah pukul 09.00 sampai 18.00, Senin sampai Jumat. "
        "Karyawan wajib check-in melalui aplikasi HRIS paling lambat pukul 09.15. "
        "Keterlambatan lebih dari 3 kali dalam satu bulan berakibat "
        "surat peringatan tertulis."
    ),
    (
        "Reimbursement medis untuk karyawan tetap memiliki plafon "
        "Rp5.000.000 per tahun. "
        "Klaim diajukan melalui HRIS dengan melampirkan bukti kwitansi asli "
        "dan diproses dalam 10 hari kerja. "
        "Reimbursement pengembangan diri untuk buku atau kursus sebesar "
        "Rp1.500.000 per tahun dengan persetujuan atasan."
    ),
]


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Build an in-memory PDF from a list of page texts using pymupdf."""
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()  # default A4
        page.insert_textbox(
            pymupdf.Rect(72, 72, 523, 770),
            text,
            fontname="helv",
            fontsize=11,
        )
    data = doc.tobytes()
    doc.close()
    return data


def build_settings(tmp_path) -> Settings:
    """Offline settings with a throwaway storage directory."""
    return Settings(
        STORAGE_DIR=tmp_path / "storage",
        LLM_BACKEND="fake",
        EMBEDDING_BACKEND="hash",
        RERANK_BACKEND="lexical",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=60,
        MIN_CHUNK_CHARS=40,
        RETRIEVAL_K_SEMANTIC=8,
        RETRIEVAL_K_BM25=8,
        CONTEXT_TOP_K=3,
    )


def ingest_pdf(service: RAGService, filename: str, data: bytes) -> dict:
    """Ingest raw PDF bytes into a service (mirrors the API route)."""
    result = service.ingest_pdf_bytes(filename, data)
    return {
        "source_id": result.source_id,
        "source_name": result.source_name,
        "num_chunks": result.num_chunks,
        "num_pages": result.num_pages,
        "status": result.status,
    }


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    return make_pdf_bytes(SAMPLE_PAGES)


@pytest.fixture()
def service(tmp_path, sample_pdf_bytes) -> RAGService:
    """A fully built offline RAGService pre-loaded with the sample PDF."""
    svc = RAGService.build(build_settings(tmp_path))
    ingest_pdf(svc, "panduan_karyawan.pdf", sample_pdf_bytes)
    return svc


@pytest.fixture()
def client(tmp_path, sample_pdf_bytes):
    """TestClient for the FastAPI app (offline backends, temp storage)."""
    settings = build_settings(tmp_path)
    with TestClient(create_app(settings)) as test_client:
        # Pre-ingest the sample document so chat tests have context.
        test_client.post(
            "/api/documents/ingest",
            files={"file": ("panduan_karyawan.pdf", sample_pdf_bytes, "application/pdf")},
        )
        yield test_client
