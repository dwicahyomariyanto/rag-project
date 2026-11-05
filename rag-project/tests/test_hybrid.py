"""Unit tests for RRF fusion and the BM25 index."""

from __future__ import annotations

from app.rag.bm25 import BM25Index, tokenize
from app.rag.hybrid import rrf_fuse
from app.rag.models import Chunk


def test_tokenize_removes_stopwords():
    tokens = tokenize("Berapa jumlah hari cuti tahunan yang diizinkan?")
    assert "yang" not in tokens
    assert "cuti" in tokens and "tahunan" in tokens


def test_rrf_prefers_items_present_in_both_lists():
    fused = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    # "b" appears high in both rankings -> must be first
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_rrf_single_list_preserves_order():
    fused = rrf_fuse([["x", "y", "z"]], k=60)
    assert fused == ["x", "y", "z"]


def test_rrf_is_deterministic():
    lists = [["a", "b", "c"], ["c", "b", "d"]]
    assert rrf_fuse(lists) == rrf_fuse(lists)


def _make_record(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, text=text, source_id="src", source_name="f.pdf",
        page_start=1, page_end=1, chunk_index=0,
    )


def test_bm25_ranks_keyword_match_first():
    records = [
        _make_record("1", "Cuti tahunan karyawan tetap adalah 12 hari kerja."),
        _make_record("2", "Gaji dibayarkan setiap tanggal 28 setiap bulannya."),
        _make_record("3", "Reimbursement medis plafon lima juta rupiah per tahun."),
    ]
    index = BM25Index()
    index.rebuild(records)
    hits = index.search("berapa hari cuti tahunan", k=3)
    assert hits[0][0] == "1"
    assert len(hits) <= 3


def test_bm25_empty_index_returns_nothing():
    index = BM25Index()
    assert index.search("apa saja", k=5) == []
