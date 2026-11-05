"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists."""

from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse multiple ranked id-lists into one ranking with RRF.

    score(d) = sum over lists of 1 / (k + rank(d))   with rank starting at 1.

    Args:
        rankings: ordered lists of chunk ids (best first) per retriever.
        k: the standard RRF constant (paper default: 60).

    Returns:
        Unique chunk ids ordered by fused score (best first).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)
