#!/usr/bin/env python3
"""CLI: ingest satu file PDF atau seluruh folder PDF ke knowledge base.

Pemakaian (dari root project):

    python scripts/ingest.py data/sample/panduan_karyawan_pt_nusantara.pdf
    python scripts/ingest.py path/to/folder_pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.service import RAGService  # noqa: E402


def collect_pdfs(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.glob("*.pdf"))
    return [target]


def main() -> None:
    if len(sys.argv) < 2:
        print("Pemakaian: python scripts/ingest.py <file.pdf | folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    pdfs = collect_pdfs(target)
    if not pdfs:
        print(f"Tidak ada PDF ditemukan di: {target}")
        sys.exit(1)

    service = RAGService.build(get_settings())
    print(f"Backend: embedding={service.embedder.name}, llm={service.llm.name}\n")

    for pdf in pdfs:
        try:
            result = service.ingest_pdf_bytes(pdf.name, pdf.read_bytes())
            print(
                f"[OK] {result.source_name}: {result.num_pages} halaman -> "
                f"{result.num_chunks} chunks ({result.status}); "
                f"total di store: {result.total_chunks}"
            )
        except ValueError as exc:
            print(f"[GAGAL] {pdf.name}: {exc}")

    print(f"\nSelesai. Total chunk di knowledge base: {service.store.count()}")


if __name__ == "__main__":
    main()
