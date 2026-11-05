"""LLM backends.

* ``OpenAIChatLLM`` -- any OpenAI-compatible chat completions endpoint
  (OpenAI, Groq, DeepSeek, Together, Ollama, vLLM, ...). Supports both
  blocking ``complete`` and token ``stream``.
* ``FakeLLM``      -- deterministic offline LLM used by the test-suite and the
  no-API-key demo mode. It summarizes the context blocks it receives and
  always cites ``[1]``-style sources so the citation parser is exercised.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterator

from app.config import Settings

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class BaseLLM(ABC):
    """Interface for chat LLM backends."""

    name: str = "base"

    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        """Generate one full answer."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the answer token-by-token (default: single chunk)."""
        yield self.complete(messages)


class OpenAIChatLLM(BaseLLM):
    """Client for any OpenAI-compatible chat completions endpoint."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        from openai import OpenAI  # local import keeps test startup fast

        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY kosong. Isi .env atau set LLM_BACKEND=fake "
                "untuk mode offline testing."
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class FakeLLM(BaseLLM):
    """Deterministic offline LLM for pipeline/API testing (no network calls)."""

    name = "fake"

    # Matches "[Sumber 1] (file.pdf, hal. 2)\n<context text>" blocks.
    _CTX_RE = re.compile(
        r"\[Sumber (\d+)\][^\n]*\n(.*?)(?=\n\n\[Sumber |\Z)", re.DOTALL
    )

    def complete(self, messages: list[Message]) -> str:
        user = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        blocks = self._CTX_RE.findall(user)
        if not blocks:
            return (
                "Tidak ditemukan konteks relevan di dokumen untuk pertanyaan ini. "
                "Silakan unggah dokumen terlebih dahulu."
            )
        parts = []
        for number, text in blocks[:3]:
            snippet = " ".join(text.split())[:160]
            parts.append(f"Ringkasan sumber [{number}]: {snippet} [{number}]")
        return (
            "Berdasarkan dokumen yang tersedia pada knowledge base, "
            "berikut informasi yang relevan dengan pertanyaan Anda. "
            + " ".join(parts)
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for word in self.complete(messages).split(" "):
            yield word + " "


def build_llm(settings: Settings) -> BaseLLM:
    """Factory: build the LLM selected by ``LLM_BACKEND``."""
    backend = settings.LLM_BACKEND.strip().lower()
    if backend == "openai":
        return OpenAIChatLLM(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    if backend == "fake":
        return FakeLLM()
    raise ValueError(
        f"LLM_BACKEND tidak dikenal: '{settings.LLM_BACKEND}' "
        "(pilihan valid: openai | fake)"
    )
