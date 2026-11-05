#!/usr/bin/env python3
"""CLI: ajukan satu pertanyaan ke knowledge base (tanpa membuka server).

Pemakaian (dari root project):

    python scripts/query.py "Berapa plafon reimbursement medis?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.service import RAGService  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Pemakaian: python scripts/query.py "pertanyaan Anda"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    service = RAGService.build(get_settings())
    result = service.pipeline.answer(question)

    print("=" * 70)
    print("JAWABAN:")
    print(result.answer)
    print("-" * 70)
    if result.citations:
        print("SUMBER:")
        for citation in result.citations:
            page = (
                f"hal. {citation['page_start']}-{citation['page_end']}"
                if citation["page_end"] != citation["page_start"]
                else f"hal. {citation['page_start']}"
            )
            print(f"  [{citation['id']}] {citation['source_name']} ({page})")
    print("-" * 70)
    stats = result.stats
    print(
        f"STATISTIK: semantic={stats['semantic_candidates']} "
        f"bm25={stats['bm25_candidates']} fused={stats['fused_candidates']} "
        f"context={stats['context_used']} | retrieval {stats['retrieval_ms']} ms "
        f"| generasi {stats['generation_ms']} ms"
    )


if __name__ == "__main__":
    main()
