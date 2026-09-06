"""Component B.3 -- metadata boosting.

Pure functions: no model, no DB, no network. This is the arithmetic the
confidence gate depends on, so it is kept trivially testable.
"""

from __future__ import annotations

from typing import Sequence

from ..contracts import Candidate, IncomingTicket, SearchHit

# Boost weights fixed by the spec.
PART_NUMBER_BOOST = 0.05
ISSUE_TYPE_BOOST = 0.03
REASON_CODE_BOOST = 0.02

MAX_BOOST = PART_NUMBER_BOOST + ISSUE_TYPE_BOOST + REASON_CODE_BOOST


def _key(value: str | None) -> str:
    """Normalize a metadata value for comparison (trim + case-fold)."""
    return (value or "").strip().casefold()


def _matches(left: str | None, right: str | None) -> bool:
    """Equality that ignores blanks.

    Two records that are both missing a Part_Number are not a part match; only
    a shared, populated value earns the boost.
    """
    a, b = _key(left), _key(right)
    return bool(a) and a == b


def composite_score(hit: SearchHit, ticket: IncomingTicket) -> tuple[float, tuple[str, ...]]:
    """Return (score, applied_boost_names) for one candidate.

    Base is the cosine similarity clamped to [0, 1] -- a negative cosine is
    "unrelated", not "worse than unrelated", and must not let boosts lift an
    irrelevant record. The total is clamped to 1.0 so confidence never reports
    above 100%.
    """
    base = min(max(hit.cosine_similarity, 0.0), 1.0)
    payload = hit.payload
    applied: list[str] = []
    boost = 0.0

    if _matches(payload.get("part_number"), ticket.part_number):
        boost += PART_NUMBER_BOOST
        applied.append("part_number")
    if _matches(payload.get("issue_type"), ticket.issue_type):
        boost += ISSUE_TYPE_BOOST
        applied.append("issue_type")
    if _matches(payload.get("problem_reason_code"), ticket.problem_reason_code):
        boost += REASON_CODE_BOOST
        applied.append("problem_reason_code")

    return min(base + boost, 1.0), tuple(applied)


def to_candidate(hit: SearchHit, ticket: IncomingTicket) -> Candidate:
    score, applied = composite_score(hit, ticket)
    payload = hit.payload
    return Candidate(
        sps_id=str(payload.get("sps_id", "")),
        actual_solution=str(payload.get("actual_solution", "")),
        part_number=str(payload.get("part_number", "")),
        part_description=str(payload.get("part_description", "")),
        item_status=str(payload.get("item_status", "")),
        problem_reason_code=str(payload.get("problem_reason_code", "")),
        issue_type=str(payload.get("issue_type", "")),
        cosine_similarity=hit.cosine_similarity,
        composite_score=score,
        applied_boosts=applied,
    )


def rank_candidates(hits: Sequence[SearchHit], ticket: IncomingTicket) -> list[Candidate]:
    """Score every hit and re-sort by composite score, best first.

    Ties break on raw cosine then SPS_ID so a given query always yields the same
    ordering -- required for an auditable, reproducible recommendation.
    """
    candidates = [to_candidate(hit, ticket) for hit in hits]
    candidates.sort(
        key=lambda c: (c.composite_score, c.cosine_similarity, c.sps_id),
        reverse=True,
    )
    return candidates
