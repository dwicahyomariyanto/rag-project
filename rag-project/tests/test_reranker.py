"""Unit tests for rerankers and embedders."""

from __future__ import annotations

from app.core.embeddings import HashEmbedder
from app.rag.models import RetrievedChunk
from app.rag.reranker import LexicalReranker, NullReranker


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=text, source_id="src", source_name="f.pdf",
        page_start=1, page_end=1, chunk_index=0,
    )


def test_lexical_reranker_puts_relevant_chunk_first():
    reranker = LexicalReranker()
    candidates = [
        _chunk("gaji", "Gaji dibayarkan setiap tanggal 28 setiap bulannya."),
        _chunk("cuti", "Cuti tahunan karyawan tetap adalah 12 hari kerja per tahun."),
        _chunk("wfh", "Kerja jarak jauh maksimal 2 hari per minggu."),
    ]
    reranked = reranker.rerank("berapa jumlah hari cuti tahunan", candidates)
    assert reranked[0].chunk_id == "cuti"
    assert reranked[0].rerank_score is not None


def test_null_reranker_keeps_order():
    candidates = [_chunk("a", "satu"), _chunk("b", "dua")]
    out = NullReranker().rerank("query", candidates)
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dim=256)
    v1 = embedder.embed_query("karyawan cuti tahunan")
    v2 = embedder.embed_query("karyawan cuti tahunan")
    v3 = embedder.embed_query("reimbursement medis plafon")
    assert v1 == v2
    assert v1 != v3
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert len(v1) == 256


def test_hash_embedder_similarity_tracks_token_overlap():
    embedder = HashEmbedder(dim=512)

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    q = embedder.embed_query("cuti tahunan 12 hari")
    same = embedder.embed_query("cuti tahunan karyawan 12 hari kerja")
    other = embedder.embed_query("gaji tunjangan transport bulan")
    assert cosine(q, same) > cosine(q, other)
