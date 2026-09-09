"""Builder for the resolved recommendation."""

from __future__ import annotations

from typing import Sequence

from .contracts import Candidate, PipelineResult, score_to_percent


def success(
    recommendation: str,
    justification: str,
    top_score: float,
    candidates: Sequence[Candidate],
) -> PipelineResult:
    """Audited recommendation plus the precedent it was synthesized from."""
    seen: set[str] = set()
    sps_ids: list[str] = []
    for candidate in candidates:
        if candidate.sps_id and candidate.sps_id not in seen:
            seen.add(candidate.sps_id)
            sps_ids.append(candidate.sps_id)
    return PipelineResult(
        ai_recommendation=recommendation,
        justification=justification,
        confidence=f"{score_to_percent(top_score)}%",
        sps_ids_referred=sps_ids,
    )
