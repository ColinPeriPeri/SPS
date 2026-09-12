"""Builders for the resolved recommendation.

One per tier. Both produce the same `PipelineResult`, so the writer downstream
has a single shape to render; what differs is where `referenced_sources` comes
from and what `resolution_source` says about it.
"""

from __future__ import annotations

from typing import Sequence

from .contracts import (
    SOURCE_DOCUMENTATION,
    SOURCE_HISTORICAL,
    Candidate,
    PipelineResult,
    score_to_percent,
)


def _dedupe(values: Sequence[str]) -> list[str]:
    """Preserve order, drop repeats and blanks.

    Order is retrieval order, which is descending relevance -- so the first
    citation a reviewer reads is the strongest one.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            kept.append(value)
    return kept


def success(
    recommendation: str,
    justification: str,
    top_score: float,
    candidates: Sequence[Candidate],
) -> PipelineResult:
    """Tier 1: audited recommendation plus the precedent it came from."""
    return PipelineResult(
        ai_recommendation=recommendation,
        justification=justification,
        confidence=f"{score_to_percent(top_score)}%",
        referenced_sources=_dedupe([c.sps_id for c in candidates]),
        resolution_source=SOURCE_HISTORICAL,
    )


def success_from_docs(
    recommendation: str,
    justification: str,
    top_score: float,
    chunks: Sequence,
) -> PipelineResult:
    """Tier 2: audited recommendation plus the standards sections it came from.

    The citations are taken from the retrieved chunks, not from the model's
    prose. The Judge checks that what the model wrote matches these, but the
    column itself is never the model's to author -- the same reason confidence
    and SPS IDs are supplied from measurement in Tier 1.
    """
    return PipelineResult(
        ai_recommendation=recommendation,
        justification=justification,
        confidence=f"{score_to_percent(top_score)}%",
        referenced_sources=_dedupe([c.citation for c in chunks]),
        resolution_source=SOURCE_DOCUMENTATION,
    )
