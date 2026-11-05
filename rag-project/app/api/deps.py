"""FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import Request

from app.service import RAGService


def get_service(request: Request) -> RAGService:
    """Return the RAGService built during application startup."""
    return request.app.state.service
