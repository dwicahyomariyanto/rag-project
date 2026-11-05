"""End-to-end API tests (offline backends): ingest -> list -> chat -> stream -> delete."""

from __future__ import annotations


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_backend"] == "fake"
    assert body["embedding_backend"] == "hash"
    assert body["reranker_backend"] == "lexical"
    assert body["num_chunks"] > 0
    assert body["num_sources"] == 1


def test_ingest_pdf_and_list(client):
    # content must differ from the pre-ingested fixture doc (dedup by content hash)
    from tests.conftest import make_pdf_bytes

    different_pdf = make_pdf_bytes(
        [
            "Kebijakan parkir kantor: mobil gratis untuk karyawan tetap, "
            "motor menggunakan kartu parkir bersubsidi yang dapat diminta "
            "ke tim Admin Gedung lantai satu."
        ]
    )
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("baru.pdf", different_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] > 0
    assert body["source_name"] == "baru.pdf"
    assert body["status"] == "created"

    listing = client.get("/api/documents").json()
    names = [d["source_name"] for d in listing]
    assert "baru.pdf" in names
    assert "panduan_karyawan.pdf" in names  # pre-ingested by the fixture


def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("catatan.txt", b"ini bukan pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_ingest_rejects_corrupt_pdf(client):
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("rusak.pdf", b"%PDF-1.4 ini korup", "application/pdf")},
    )
    assert response.status_code == 400


def test_chat_full_flow(client):
    response = client.post(
        "/api/chat", json={"question": "Berapa hari cuti tahunan?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["citations"]) >= 1
    citation = body["citations"][0]
    assert citation["source_name"] == "panduan_karyawan.pdf"
    assert citation["page_start"] >= 1
    stats = body["stats"]
    assert stats["semantic_candidates"] > 0 or stats["bm25_candidates"] > 0
    assert stats["context_used"] >= 1


def test_chat_validates_empty_question(client):
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 422


def test_retrieve_debug_endpoint(client):
    response = client.post(
        "/api/retrieve", json={"question": "reimbursement medis plafon"}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) >= 1
    first = body["results"][0]
    assert first["text"]
    assert "reimbursement" in " ".join(r["text"].lower() for r in body["results"][:3])


def test_chat_stream_sse_events(client):
    response = client.post(
        "/api/chat/stream", json={"question": "Berapa plafon reimbursement medis?"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body
    # the sources event must carry parsable citation JSON
    for line in body.splitlines():
        if line.startswith("data: ") and '"citations"' in line:
            payload = line[len("data: "):]
            assert '"source_name"' in payload
            break


def test_delete_document_flow(client):
    from tests.conftest import make_pdf_bytes

    unique_pdf = make_pdf_bytes(
        [
            "Kebijakan seragam kantor: kaus polo berwarna biru pada hari Selasa "
            "dan Rabu, kemeja formal pada hari kerja lainnya. ID card wajib "
            "dipakai di area kantor."
        ]
    )
    ingest = client.post(
        "/api/documents/ingest",
        files={"file": ("hapus.pdf", unique_pdf, "application/pdf")},
    )
    source_id = ingest.json()["source_id"]
    assert client.delete(f"/api/documents/{source_id}").status_code == 200
    assert client.delete(f"/api/documents/{source_id}").status_code == 404
