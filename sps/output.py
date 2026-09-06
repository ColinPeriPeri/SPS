"""Builders for the Section 4 output contract.

Every exit path from the pipeline is constructed here, so the four-key schema
is guaranteed no matter which gate the ticket stopped at.
"""

from __future__ import annotations

from typing import Sequence

from .config import CONFIDENCE_THRESHOLD
from .contracts import SOLUTION_NOT_FOUND, Candidate, PipelineResult, score_to_percent

INVALID_INPUT_JUSTIFICATION = "Invalid problem statement."
NO_MATCH_JUSTIFICATION = "No historical records matched the reported problem."


def _percent(value: float | int) -> str:
    if isinstance(value, int):
        return f"{value}%"
    return f"{score_to_percent(value)}%"


SERVICE_UNAVAILABLE_JUSTIFICATION = (
    "AI service unavailable; escalate for manual review."
)


def failure(
    justification: str,
    confidence: float | int = 0,
    infrastructure_failure: bool = False,
    diagnostic: str = "",
) -> PipelineResult:
    """Any non-success exit: sentinel recommendation, empty ID list."""
    return PipelineResult(
        ai_recommendation=SOLUTION_NOT_FOUND,
        justification=justification,
        confidence=_percent(confidence),
        sps_ids_referred=[],
        infrastructure_failure=infrastructure_failure,
        diagnostic=diagnostic or justification,
    )


def invalid_input() -> PipelineResult:
    """Component B.1 rejection payload."""
    return failure(INVALID_INPUT_JUSTIFICATION, 0)


def no_matches() -> PipelineResult:
    """Vector search returned nothing at all (e.g. an empty collection)."""
    return failure(NO_MATCH_JUSTIFICATION, 0)


def below_threshold(
    top_score: float, threshold: float = CONFIDENCE_THRESHOLD
) -> PipelineResult:
    """Component B.4 gate payload -- reports the score that failed the gate."""
    return failure(
        f"Confidence below {score_to_percent(threshold)}% threshold.",
        top_score,
    )


def generation_failed(
    reason: str, top_score: float, infrastructure_failure: bool = False
) -> PipelineResult:
    """Component C circuit-breaker payload.

    The retrieval confidence was real, so it is reported honestly; the ID list
    stays empty because no vetted recommendation was produced from those records.

    On a dependency outage the reason is replaced with a generic message: the
    detail (endpoint names, missing config keys, stack context) belongs in the
    ops log, not in a field an admin -- or eventually a supplier -- may read.
    """
    return failure(
        SERVICE_UNAVAILABLE_JUSTIFICATION if infrastructure_failure else reason,
        top_score,
        infrastructure_failure=infrastructure_failure,
        # The specific cause survives here for the ops status file even though
        # the delivered Justification is deliberately generic.
        diagnostic=reason,
    )


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
        confidence=_percent(top_score),
        sps_ids_referred=sps_ids,
    )
