"""RAGAS end-to-end evaluation (requires a real API key).

This module is EXCLUDED from the default pytest run (see pyproject.toml:
``addopts = "-m 'not ragas'"``). Run it explicitly with:

    pytest -m ragas

It needs LLM_API_KEY/EMBEDDING_API_KEY in .env (a real judge LLM is required
to grade the answers), so it is skipped automatically when they are missing.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ragas


def _has_real_llm() -> bool:
    import os

    return bool(os.environ.get("LLM_API_KEY")) and (
        os.environ.get("LLM_BACKEND", "openai") != "fake"
    )


def test_ragas_faithfulness_smoke(service):
    """2 questions -> RAGAS faithfulness & relevancy must run and be in (0, 1]."""
    if not _has_real_llm():
        pytest.skip("Butuh LLM_API_KEY asli (RAGAS memerlukan judge LLM sungguhan).")

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    questions = [
        ("Berapa hari cuti tahunan karyawan tetap?",
         "Cuti tahunan karyawan tetap adalah 12 hari kerja per tahun."),
        ("Berapa plafon reimbursement medis per tahun?",
         "Plafon reimbursement medis Rp5.000.000 per tahun."),
    ]

    rows = []
    for question, reference in questions:
        result = service.pipeline.answer(question)
        rows.append(
            {
                "user_input": question,
                "retrieved_contexts": [c.text for c in result.contexts],
                "response": result.answer,
                "reference": reference,
            }
        )

    from app.config import get_settings

    settings = get_settings()
    dataset = EvaluationDataset.from_list(rows)
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.0,
        )
    )
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge,
    )
    scores = {name: float(value) for name, value in result.items()}
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0, f"metrik {name} di luar rentang: {value}"
