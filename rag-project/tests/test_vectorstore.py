"""Vector store tests: add/query/persist/delete round-trips (offline)."""

from __future__ import annotations

from app.rag.models import Chunk
from app.rag.vectorstore import VectorStore


def _chunk(cid: str, text: str, source_id: str = "src1") -> Chunk:
    return Chunk(
        chunk_id=cid, text=text, source_id=source_id, source_name="doc.pdf",
        page_start=1, page_end=2, chunk_index=0,
    )


def test_add_query_count_roundtrip(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    chunks = [_chunk("c1", "cuti tahunan 12 hari"), _chunk("c2", "gaji tanggal 28")]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store.add(chunks, vectors)
    assert store.count() == 2

    hits = store.query([1.0, 0.0], k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[0].semantic_score is not None
    assert hits[0].page_start == 1


def test_persistence_across_instances(tmp_path):
    store1 = VectorStore(tmp_path / "chroma")
    store1.add([_chunk("c1", "persistensi data")], [[0.5, 0.5]])
    del store1

    store2 = VectorStore(tmp_path / "chroma")
    assert store2.count() == 1
    hits = store2.query([0.5, 0.5], k=1)
    assert hits[0].chunk_id == "c1"


def test_delete_source_removes_only_that_source(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.add(
        [
            _chunk("a1", "dokumen satu", source_id="s1"),
            _chunk("a2", "dokumen satu lanjutan", source_id="s1"),
            _chunk("b1", "dokumen dua", source_id="s2"),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    sources = store.list_sources()
    assert set(sources.keys()) == {"s1", "s2"}
    assert sources["s1"]["num_chunks"] == 2

    store.delete_source(source_id="s1")
    assert store.count() == 1
    remaining = store.query([0.0, 1.0], k=5)
    assert [h.chunk_id for h in remaining] == ["b1"]


def test_query_on_empty_store_returns_empty(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    assert store.count() == 0
    assert store.query([1.0, 0.0], k=3) == []
