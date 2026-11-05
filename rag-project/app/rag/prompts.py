"""Prompt templates for the RAG pipeline."""

from __future__ import annotations

from app.rag.models import RetrievedChunk

SYSTEM_PROMPT = """You are a precise, document-grounded question answering assistant (RAG).

STRICT RULES:
1. Answer ONLY using the numbered context passages provided in the user message.
2. Cite the passage(s) supporting each claim with bracketed numbers right after the sentence, e.g. [1] or [1][3].
3. If the context does not contain the answer, say that the information was not found in the provided documents. Never guess and never use outside knowledge.
4. Answer in the SAME language as the user's question (an Indonesian question gets an Indonesian answer).
5. Be concise and factual: at most 3 short paragraphs."""


def _page_label(chunk: RetrievedChunk) -> str:
    if chunk.page_end and chunk.page_end != chunk.page_start:
        return f"hal. {chunk.page_start}-{chunk.page_end}"
    return f"hal. {chunk.page_start}"


def build_user_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    """Compose the user message: numbered context blocks + the question."""
    blocks = []
    for index, chunk in enumerate(contexts, start=1):
        blocks.append(
            f"[Sumber {index}] ({chunk.source_name}, {_page_label(chunk)})\n{chunk.text}"
        )
    context_text = "\n\n".join(blocks)
    return f"Context passages:\n{context_text}\n\nQuestion:\n{question}"
