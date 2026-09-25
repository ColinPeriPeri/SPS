"""Tier 1's decision: which past record is asking the same thing?

This is what replaced the Actor and Judge on the historical path. The
difference is worth stating plainly, because the whole safety posture of Tier 1
turns on it: the Actor *wrote* a recommendation and the Judge audited it, so a
bad retrieval could still be caught downstream. Nothing writes anything here.
A record scores, and if it scores highly enough its solution is sent exactly as
it was recorded. The score is the only decision in the path.

Two consequences follow from that, and both are deliberate.

The score is an LLM's, so it is not a calibrated probability. It ranks five
candidates against each other well; read as an absolute gate it means whatever
the model decides it means. `SPS_INTENT_THRESHOLD` is provisional until
measured.

And the scorer is shown problems, never solutions. It is deciding which past
problem is the same request -- showing it the answers would invite it to score
their usefulness instead, and a record whose problem matches perfectly but
whose solution is boilerplate must still score high. "The closest precedent is
empty" is a finding a reviewer needs; folding it into a low match score would
hide it behind "nothing matched".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ..contracts import Candidate, IncomingTicket
from .llm import ChatClient, LLMError
from .prompts import build_intent_messages

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    intent_match: float  # 0..1, to match every other score in the codebase
    reason: str

    @property
    def percent(self) -> int:
        from ..contracts import score_to_percent

        return score_to_percent(self.intent_match)


@dataclass(frozen=True, slots=True)
class IntentOutcome:
    """What the scorer concluded, or why it could not."""

    ticket_intent: str = ""
    scored: tuple[ScoredCandidate, ...] = ()
    failure_reason: str = ""

    @property
    def best(self) -> ScoredCandidate | None:
        return self.scored[0] if self.scored else None

    @property
    def failed(self) -> bool:
        return bool(self.failure_reason)


async def score_intent(
    client: ChatClient,
    ticket: IncomingTicket,
    candidates: Sequence[Candidate],
) -> IntentOutcome:
    """Score every candidate against the ticket's intent, best first.

    Fails closed. An unreachable or malformed scorer returns a failure rather
    than an empty score list, because the two mean opposite things: "nothing
    matched" is a legitimate business outcome the robot files, and "we could
    not tell" is an outage it should retry. Collapsing them would quietly
    convert every outage into a refusal.
    """
    if not candidates:
        return IntentOutcome()

    from ..schemas import IntentAssessment

    try:
        assessment = await client.complete_model(
            build_intent_messages(ticket, candidates), IntentAssessment
        )
    except LLMError as exc:
        logger.warning("Ticket %r: intent scoring failed: %s", ticket.sps_id, exc)
        return IntentOutcome(failure_reason=f"Intent scoring failed: {exc}")

    by_id = {c.sps_id: c for c in candidates}
    scored: list[ScoredCandidate] = []
    for entry in assessment.scores:
        candidate = by_id.get(entry.sps_id)
        if candidate is None:
            # A score for a record we never supplied. The id is the only thing
            # tying a score to a solution, so a hallucinated one cannot be
            # matched to anything and is dropped rather than guessed at.
            logger.warning(
                "Ticket %r: scorer returned unknown SPS ID %r; ignoring",
                ticket.sps_id, entry.sps_id,
            )
            continue
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                intent_match=min(max(entry.intent_match, 0), 100) / 100.0,
                reason=entry.reason.strip(),
            )
        )

    if not scored:
        # Every entry was unusable, or none came back. Not the same as a low
        # score: we have no judgement at all, so nothing may be cascaded.
        return IntentOutcome(
            ticket_intent=assessment.ticket_intent.strip(),
            failure_reason="Intent scoring returned no usable scores.",
        )

    # Ties broken by the retrieval score, then by SPS ID. The scorer works in
    # whole percents, so ties are common -- five candidates and a coarse scale
    # -- and without a tiebreak the cascaded solution would depend on dict
    # ordering. Which record gets sent to a supplier should not.
    scored.sort(
        key=lambda s: (s.intent_match, s.candidate.composite_score, s.candidate.sps_id),
        reverse=True,
    )

    logger.info(
        "Ticket %r intent %r; best %s at %d%%",
        ticket.sps_id,
        assessment.ticket_intent.strip()[:80],
        scored[0].candidate.sps_id,
        scored[0].percent,
    )
    return IntentOutcome(
        ticket_intent=assessment.ticket_intent.strip(),
        scored=tuple(scored),
    )
