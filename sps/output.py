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


def cascade_warnings(text: str) -> str:
    """What a reviewer should read before pasting this, or '' if nothing.

    The same two scanners that BLOCK a draft in Tier 2 only ANNOTATE here, and
    the asymmetry is the point rather than an oversight. Tier 2 text is written
    by a model, so rejecting it costs a retry. Tier 1 text is the archive's own
    and is sent exactly as recorded -- rejecting it would mean refusing a
    precedent the business asked to be sent, so this names the risk and leaves
    the decision where it now sits, with the reviewer.
    """
    from .generation.transferable import misattributed_actions, untransferable_references

    findings = untransferable_references(text) + misattributed_actions(text)
    return "Contains: " + "; ".join(findings) if findings else ""


def cascade(
    solution: str,
    candidate: Candidate,
    intent_match: float,
    reason: str,
    source_list: str = "",
) -> PipelineResult:
    """Tier 1: a past solution passed through unchanged.

    `solution` is written to the sheet character for character -- no strip, no
    renumbering, no whitespace collapse. That is the requirement, and it is
    also what makes the output auditable: a reviewer can diff the cell against
    the source record and expect an exact match.
    """
    return PipelineResult(
        ai_recommendation=solution,
        justification=(
            f"Confidence is {score_to_percent(intent_match)}% against {candidate.sps_id}, "
            f"whose recorded solution is sent unchanged. {reason}".strip()
        ),
        confidence=f"{score_to_percent(intent_match)}%",
        # One record, because one record's solution was sent. Naming the other
        # four would say the answer came from all of them.
        referenced_sources=[candidate.sps_id],
        resolution_source=SOURCE_HISTORICAL,
        closest_matching_solution=source_list,
        cascade_warnings=cascade_warnings(solution),
    )


def _with_precedent(justification: str, precedent: str) -> str:
    """Put the source beneath the rationale, in the field a reviewer reads.

    The precedent also has its own column, but a reviewer checking whether a
    recommendation really follows from the record should not have to hold two
    columns in their head to do it. So it appears in both, and this is the one
    they are already looking at.

    The rationale leads here, unlike on a refusal where the precedent does. On
    a refusal the precedent IS the finding -- there is no recommendation to
    explain. On a success the rationale is the answer to "why this?" and the
    source is the evidence for it, which is the order they get read in.
    """
    if not precedent:
        return justification
    return f"{justification}\n\n{precedent}" if justification else precedent


def success(
    recommendation: str,
    justification: str,
    top_score: float,
    candidates: Sequence[Candidate],
    closest_matching_solution: str = "",
) -> PipelineResult:
    """Tier 1: audited recommendation plus the precedent it came from.

    The raw precedent rides along even on a success. Referenced_Sources names
    the SPS IDs, which tells a reviewer where to look but not what it said, and
    the whole point of showing the source is to let them check the
    recommendation against it without opening another system.
    """
    return PipelineResult(
        ai_recommendation=recommendation,
        justification=_with_precedent(justification, closest_matching_solution),
        confidence=f"{score_to_percent(top_score)}%",
        referenced_sources=_dedupe([c.sps_id for c in candidates]),
        resolution_source=SOURCE_HISTORICAL,
        closest_matching_solution=closest_matching_solution,
    )


def success_from_docs(
    recommendation: str,
    justification: str,
    top_score: float,
    chunks: Sequence,
    closest_matching_solution: str = "",
) -> PipelineResult:
    """Tier 2: audited recommendation plus the standards sections it came from.

    The citations are taken from the retrieved chunks, not from the model's
    prose. The Judge checks that what the model wrote matches these, but the
    column itself is never the model's to author -- the same reason confidence
    and SPS IDs are supplied from measurement in Tier 1.
    """
    return PipelineResult(
        ai_recommendation=recommendation,
        justification=_with_precedent(justification, closest_matching_solution),
        confidence=f"{score_to_percent(top_score)}%",
        referenced_sources=_dedupe([c.citation for c in chunks]),
        resolution_source=SOURCE_DOCUMENTATION,
        closest_matching_solution=closest_matching_solution,
    )
