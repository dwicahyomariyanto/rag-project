"""Unit tests for the chunker (sizes, overlap, page mapping, merging)."""

from __future__ import annotations

from app.rag.chunker import _hard_split, chunk_pages
from app.rag.loader import PageContent


def test_all_chunks_respect_size():
    text = ("Karyawan mendapatkan cuti tahunan dua belas hari kerja. " * 40).strip()
    pages = [PageContent(1, text)]
    chunks = chunk_pages(
        pages, source_id="s1", source_name="a.pdf", size=300, overlap=60, min_chars=40
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300 + 5  # small separator slack


def test_hard_split_has_overlap():
    text = " ".join(f"Kalimat nomor {i} berisi informasi penting tentang cuti." for i in range(30))
    pieces = _hard_split(text, size=200, overlap=40)
    assert len(pieces) >= 2
    # consecutive pieces must share some words (the overlap window)
    tail_words = set(pieces[0].split()[-5:])
    head_words = set(pieces[1].split()[:5])
    assert tail_words & head_words, "overlap antar piece hilang"


def test_page_numbers_are_tracked():
    pages = [
        PageContent(1, "Halaman pertama membahas cuti tahunan 12 hari kerja."),
        PageContent(2, "Halaman kedua membahas reimbursement medis lima juta rupiah."),
        PageContent(3, "Halaman ketiga membahas kerja jarak jauh dua hari per minggu."),
    ]
    # size=80 forces one chunk per page (each page is ~60 chars)
    chunks = chunk_pages(
        pages, source_id="s2", source_name="b.pdf", size=80, overlap=20, min_chars=20
    )
    assert {c.page_start for c in chunks} == {1, 2, 3}
    cuti = next(c for c in chunks if "cuti" in c.text)
    assert cuti.page_start == 1
    reimbursement = next(c for c in chunks if "reimbursement" in c.text)
    assert reimbursement.page_start == 2


def test_small_pages_are_packed_into_one_chunk_with_page_span():
    pages = [
        PageContent(1, "Halaman pertama membahas cuti tahunan 12 hari kerja."),
        PageContent(2, "Halaman kedua membahas reimbursement medis lima juta rupiah."),
    ]
    chunks = chunk_pages(
        pages, source_id="s2b", source_name="b.pdf", size=500, overlap=50, min_chars=20
    )
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "cuti" in chunks[0].text and "reimbursement" in chunks[0].text


def test_tiny_chunks_are_merged():
    pages = [
        PageContent(1, "Ini paragraf utama yang cukup panjang untuk menjadi chunk sendiri dan melewati batas minimum karakter dengan mudah."),
        PageContent(2, "Pendek."),
    ]
    chunks = chunk_pages(
        pages, source_id="s3", source_name="c.pdf", size=500, overlap=50, min_chars=80
    )
    assert len(chunks) == 1
    assert "Pendek." in chunks[0].text
    assert chunks[0].page_end == 2


def test_chunk_ids_and_indices_are_stable():
    pages = [PageContent(1, "Teks halaman " + ("isi dokumen " * 200))]
    chunks = chunk_pages(
        pages, source_id="abc123", source_name="d.pdf", size=300, overlap=50, min_chars=40
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id.startswith("abc123-") for c in chunks)
    assert all(c.source_name == "d.pdf" for c in chunks)
