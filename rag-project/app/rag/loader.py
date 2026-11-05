"""PDF text extraction built on PyMuPDF (pymupdf)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Text of a single PDF page (1-based page number)."""

    page_number: int
    text: str


def extract_pdf(path: str | Path) -> list[PageContent]:
    """Extract text page-by-page from a PDF file.

    Raises:
        ValueError: if the file cannot be opened as a PDF.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pymupdf belum terpasang. Jalankan: pip install pymupdf"
        ) from exc

    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"File '{path.name}' bukan PDF yang valid: {exc}") from exc

    pages: list[PageContent] = []
    try:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(PageContent(page_number=number, text=text))
            if len(text) < 20:
                logger.warning(
                    "Halaman %d dari '%s' hampir tidak memiliki teks "
                    "(kemungkinan halaman hasil scan/gambar).",
                    number,
                    path.name,
                )
    finally:
        doc.close()

    if not any(p.text for p in pages):
        raise ValueError(
            f"Tidak ada teks yang bisa diekstrak dari '{path.name}'. "
            "Kemungkinan PDF hasil scan (image-only) dan memerlukan OCR."
        )
    return pages
