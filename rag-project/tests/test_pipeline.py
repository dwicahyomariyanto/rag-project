"""Pipeline tests: retrieval + FakeLLM, citations, streaming (offline)."""

from __future__ import annotations

from app.rag.pipeline import RAGPipeline


def test_retrieve_returns_context(service):
    context, stats = service.pipeline.retrieve("Berapa hari cuti tahunan?")
    assert len(context) > 0
    assert len(context) <= 3  # CONTEXT_TOP_K
    assert stats["retrieval_ms"] >= 0
    assert stats["context_used"] == len(context)
    # hybrid search: the cuti chunk must be found, and hit by BM25 at least
    texts = " ".join(c.text for c in context)
    assert "cuti" in texts.lower()
    assert any("bm25" in c.methods for c in context)


def test_answer_contains_citations(service):
    result = service.pipeline.answer("Berapa hari cuti tahunan?")
    assert result.answer
    assert len(result.citations) >= 1
    first = result.citations[0]
    assert first["id"] == 1
    assert first["source_name"] == "panduan_karyawan.pdf"
    assert first["page_start"] >= 1
    assert "total_ms" in result.stats and "generation_ms" in result.stats


def test_citation_parser_maps_numbers_to_context(service):
    contexts, _ = service.pipeline.retrieve("reimbursement medis")
    answer = "Plafonnya Rp5.000.000 [1] dan klaim lewat HRIS [1][2]."
    citations = RAGPipeline.build_citations(answer, contexts)
    assert [c["id"] for c in citations] == [1, 2]
    assert all("snippet" in c for c in citations)


def test_stream_answer_yields_tokens_and_stats(service):
    citations, tokens, stats = service.pipeline.stream_answer("jam kerja berapa?")
    collected = list(tokens)
    assert len(collected) > 3
    assert stats["generation_ms"] is not None
    assert stats["total_ms"] >= stats["retrieval_ms"]
    assert len(citations) >= 1


def test_empty_question_on_empty_store(tmp_path):
    """An empty knowledge base must produce a graceful no-context answer."""
    from tests.conftest import build_settings
    from app.service import RAGService

    svc = RAGService.build(build_settings(tmp_path))
    result = svc.pipeline.answer("apa itu cuti?")
    assert "tidak ada" in result.answer.lower()
    assert result.citations == []
