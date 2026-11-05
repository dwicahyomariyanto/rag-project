#!/usr/bin/env python3
"""Evaluasi kualitas RAG dengan kerangka RAGAS.

Untuk setiap baris di dataset evaluasi (JSONL), script ini menjalankan pipeline
RAG lengkap (retrieve -> generate), lalu menilai jawabannya dengan LLM-as-judge:

* ``faithfulness``   -- apakah jawaban setia pada konteks yang di-retrieve?
* ``response_relevancy`` -- apakah jawaban relevan dengan pertanyaan?
* ``llm_context_precision_without_reference`` -- apakah passage relevan berada di urutan atas?
* ``llm_context_recall`` -- apakah konteks menutup jawaban referensi?

Pemakaian (dari root project, .env harus berisi API key yang valid):

    python scripts/evaluate.py                          # dataset default
    python scripts/evaluate.py --limit 5                # subset
    python scripts/evaluate.py --dataset path/ke.jsonl  # dataset lain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.service import RAGService  # noqa: E402

DEFAULT_DATASET = Path("data/eval/eval_dataset.jsonl")


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if "user_input" not in item or "reference" not in item:
                    raise ValueError(
                        "Setiap baris JSONL harus punya 'user_input' dan 'reference'."
                    )
                rows.append(item)
    return rows


def build_eval_rows(service: RAGService, dataset: list[dict]) -> list[dict]:
    """Run the RAG pipeline over the dataset and collect RAGAS-formatted rows."""
    eval_rows = []
    for index, item in enumerate(dataset, start=1):
        question = item["user_input"]
        print(f"[{index}/{len(dataset)}] {question}")
        result = service.pipeline.answer(question)
        eval_rows.append(
            {
                "user_input": question,
                "retrieved_contexts": [chunk.text for chunk in result.contexts],
                "response": result.answer,
                "reference": item["reference"],
            }
        )
        time.sleep(0.2)  # be gentle with rate limits
    return eval_rows


def run_ragas(eval_rows: list[dict], settings) -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    dataset = EvaluationDataset.from_list(eval_rows)

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or "EMPTY",
            base_url=settings.LLM_BASE_URL,
            temperature=0.0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY or "EMPTY",
            base_url=settings.EMBEDDING_BASE_URL,
        )
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]
    print("\nMenjalankan penilaian RAGAS (LLM-as-judge)...\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print(result)

    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = settings.eval_dir / "ragas_report.csv"
    df = result.to_pandas()
    df.to_csv(csv_path, index=False)
    summary = {m: float(result[m]) for m in result.scores.columns} if hasattr(result, "scores") else {}
    json_path = settings.eval_dir / "ragas_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nLaporan tersimpan:\n  {csv_path}\n  {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi RAG dengan RAGAS")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah pertanyaan")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset tidak ditemukan: {args.dataset}")
        sys.exit(1)

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    settings = get_settings()
    service = RAGService.build(settings)
    eval_rows = build_eval_rows(service, dataset)

    try:
        run_ragas(eval_rows, settings)
    except ImportError as exc:
        print(
            "\nDependensi evaluasi belum terpasang. Jalankan:\n"
            "    pip install ragas langchain-openai\n"
            f"Detail: {exc}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
