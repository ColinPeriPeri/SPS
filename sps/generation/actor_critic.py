"""Component C -- the Actor / Judge generate-audit-refine loop.

Circuit breaker: MAX_ATTEMPTS total passes (1 draft + 2 refinements). A draft
that still fails the Judge on the final attempt is discarded outright; the
pipeline falls back to "Solution not found." rather than shipping unverified
text to a supplier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from ..config import LLMSettings
from ..contracts import SOLUTION_NOT_FOUND, Candidate, IncomingTicket
from .llm import ChatClient, LLMError
from .prompts import (
    build_actor_messages,
    build_judge_messages,
    build_tier2_actor_messages,
    build_tier2_judge_messages,
)
from .transferable import (
    critique_for,
    misattributed_actions,
    untransferable_references,
)

logger = logging.getLogger(__name__)

TIER_HISTORICAL = "historical"
TIER_DOCUMENTATION = "0250"


@dataclass(frozen=True, slots=True)
class Draft:
    recommendation: str
    justification: str

    @property
    def is_abstention(self) -> bool:
        """The Actor may decline when the precedent does not fit the problem."""
        return self.recommendation.strip().rstrip(".").casefold() == (
            SOLUTION_NOT_FOUND.rstrip(".").casefold()
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    critique: str = ""


@dataclass(frozen=True, slots=True)
class Grounding:
    """The evidence one loop run is allowed to draw on, and how to render it.

    Both tiers share one loop: the same circuit breaker, the same fail-closed
    Judge, the same abstention handling. Only the evidence and its prompts
    differ, so those are the parameters rather than a forked implementation --
    a second copy of the loop would be a second place for the retry limit to
    drift.
    """

    items: Sequence[Any]
    actor_messages: Callable[..., list[dict[str, str]]]
    judge_messages: Callable[..., list[dict[str, str]]]
    # What to report when the Actor declines. Tier 1 and Tier 2 decline for
    # different reasons and the status sheet should say which.
    abstention_reason: str
    tier: str = TIER_HISTORICAL


def historical_grounding(candidates: Sequence[Candidate]) -> Grounding:
    """DORMANT. Tier 1 scores intent and cascades; it no longer drafts.

    Kept so re-enabling the Actor/Judge on the historical path is a wiring
    change in resolve() rather than a rewrite. See the note in prompts.py.
    """
    return Grounding(
        items=candidates,
        actor_messages=build_actor_messages,
        judge_messages=build_judge_messages,
        abstention_reason="Historical records did not address the reported problem.",
        tier=TIER_HISTORICAL,
    )


def documentation_grounding(chunks: Sequence[Any]) -> Grounding:
    return Grounding(
        items=chunks,
        actor_messages=build_tier2_actor_messages,
        judge_messages=build_tier2_judge_messages,
        abstention_reason=(
            "The 0250 standards retrieved do not address this defect."
        ),
        tier=TIER_DOCUMENTATION,
    )


# Why the loop stopped, as a value rather than as prose.
#
# `failure_reason` is written for a person and reads differently for every
# cause, so counting causes across a batch meant pattern-matching English. The
# business question behind that count -- "are your checks the reason we get so
# few recommendations, or is it the archive?" -- deserves an answer that is
# tallied rather than argued, and these are what make it tallyable.
STOP_ACTOR_ABSTAINED = "ACTOR_ABSTAINED"
STOP_GATE_UNTRANSFERABLE = "GATE_UNTRANSFERABLE"
STOP_GATE_MISATTRIBUTED = "GATE_MISATTRIBUTED"
STOP_JUDGE_REFUSED = "JUDGE_REFUSED"
STOP_INFRASTRUCTURE = "INFRASTRUCTURE"


def _sanitised(draft: Draft, sps_id: str) -> Draft:
    """Drop a justification that carries the record's own baggage.

    Everything above audits `recommendation`; nothing audited `justification`,
    and that field is written to the workbook a reviewer reads. So a spotless
    recommendation could ship beside "Based on SPS-100, for which ESW#20033465
    was submitted; see the attachment" -- text the very same gate rejects when
    it is handed it.

    Blanked rather than rewritten, because both tiers already substitute a
    generated sentence for an empty justification. Blanking therefore needs no
    new code path, no second model call and no retry.

    And blanked rather than treated as a rejection, because the costs are not
    symmetric. The recommendation is already clean by this point; throwing it
    away over its rationale would spend a retry, and often the whole ticket, to
    fix prose that is context for the reviewer rather than the text sent on.
    """
    findings = untransferable_references(draft.justification) + misattributed_actions(
        draft.justification
    )
    if not findings:
        return draft

    logger.warning(
        "Ticket %r: justification discarded, it carried %s",
        sps_id,
        "; ".join(findings),
    )
    return replace(draft, justification="")


@dataclass(slots=True)
class LoopOutcome:
    """Result plus the audit trail an admin reviewer may need to see."""

    draft: Draft | None
    attempts: int
    critiques: list[str] = field(default_factory=list)
    failure_reason: str = ""
    # One of the STOP_* values above, or "" on success.
    stop_reason: str = ""
    # True when the loop stopped because a dependency was unavailable rather
    # than because the content failed review. The caller needs this to tell an
    # Azure outage apart from a legitimate refusal.
    infrastructure_failure: bool = False
    # Which evidence produced this outcome. The status sheet and the result
    # workbook both report it, so a reviewer can see whether a recommendation
    # came from precedent or from a standard.
    tier: str = TIER_HISTORICAL

    @property
    def succeeded(self) -> bool:
        return self.draft is not None


class ActorCriticLoop:
    def __init__(
        self,
        client: ChatClient,
        settings: LLMSettings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or LLMSettings()

    async def run(
        self,
        ticket: IncomingTicket,
        candidates: Sequence[Candidate],
    ) -> LoopOutcome:
        """Tier 1: generate from historical precedent."""
        return await self.run_grounded(ticket, historical_grounding(candidates))

    async def run_grounded(
        self,
        ticket: IncomingTicket,
        grounding: Grounding,
    ) -> LoopOutcome:
        """Generate, audit and refine against whatever evidence is supplied."""
        max_attempts = max(1, self.settings.max_attempts)
        critique: str | None = None
        previous_draft: str | None = None
        critiques: list[str] = []
        # What rejected the most recent draft. The circuit breaker reports it,
        # because "failed the audit three times" does not say whether a regex
        # or the Judge was doing the failing, and those lead to different fixes.
        last_stop = ""

        for attempt in range(1, max_attempts + 1):
            try:
                draft = await self._act(ticket, grounding, critique, previous_draft)
            except LLMError as exc:
                logger.warning("Ticket %r attempt %d: Actor failed: %s", ticket.sps_id, attempt, exc)
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=f"Generation failed: {exc}",
                    stop_reason=STOP_INFRASTRUCTURE,
                    infrastructure_failure=True,
                    tier=grounding.tier,
                )

            if draft.is_abstention:
                # The Actor judged the precedent inapplicable. Retrying would
                # only pressure it into inventing something.
                logger.info("Ticket %r attempt %d: Actor abstained", ticket.sps_id, attempt)
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=grounding.abstention_reason,
                    stop_reason=STOP_ACTOR_ABSTAINED,
                    tier=grounding.tier,
                )

            # Deterministic gate, deliberately BEFORE the Judge. It is local and
            # free where the Judge is a paid network call, and unlike the Judge
            # it cannot talk itself round: CHECK 1 has just taught the model
            # that anything drawn from the source is acceptable, which is
            # precisely why "ESW#20033465 is submitted for these issues" passed
            # an audit it should have failed. A rewrite is fed back through the
            # same critique path the Judge uses, so the circuit breaker still
            # bounds the retries.
            leaks = untransferable_references(draft.recommendation)
            misattributed = misattributed_actions(draft.recommendation)
            if leaks or misattributed:
                logger.info(
                    "Ticket %r attempt %d rejected locally: %s",
                    ticket.sps_id, attempt, "; ".join(leaks + misattributed),
                )
                # One critique covering both, so a draft with both problems is
                # rewritten once rather than burning two of its three attempts.
                feedback = critique_for(leaks, misattributed)
                # Leaks win when both fired. Arbitrary but fixed, so the tally
                # is stable; `failure_reason` still names every fragment found.
                last_stop = (
                    STOP_GATE_UNTRANSFERABLE if leaks else STOP_GATE_MISATTRIBUTED
                )
                critiques.append(feedback)
                critique = feedback
                previous_draft = draft.recommendation
                continue

            try:
                verdict = await self._judge(ticket, grounding, draft.recommendation)
            except LLMError as exc:
                logger.warning("Ticket %r attempt %d: Judge failed: %s", ticket.sps_id, attempt, exc)
                # An unverified draft is never shipped: a Judge outage fails closed.
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=f"Compliance audit unavailable: {exc}",
                    stop_reason=STOP_INFRASTRUCTURE,
                    infrastructure_failure=True,
                    tier=grounding.tier,
                )

            if verdict.passed:
                logger.info("Ticket %r: draft passed audit on attempt %d", ticket.sps_id, attempt)
                return LoopOutcome(
                    draft=_sanitised(draft, ticket.sps_id),
                    attempts=attempt,
                    critiques=critiques,
                    tier=grounding.tier,
                )

            logger.info(
                "Ticket %r attempt %d rejected: %s", ticket.sps_id, attempt, verdict.critique
            )
            last_stop = STOP_JUDGE_REFUSED
            critiques.append(verdict.critique)
            critique = verdict.critique
            previous_draft = draft.recommendation

        logger.warning(
            "Ticket %r: circuit breaker tripped after %d attempts", ticket.sps_id, max_attempts
        )
        return LoopOutcome(
            draft=None,
            attempts=max_attempts,
            critiques=critiques,
            failure_reason=(
                f"Draft failed the compliance audit on all {max_attempts} attempts."
                + (f" Last critique: {critiques[-1]}" if critiques else "")
            ),
            stop_reason=last_stop or STOP_JUDGE_REFUSED,
            tier=grounding.tier,
        )

    async def _act(
        self,
        ticket: IncomingTicket,
        grounding: Grounding,
        critique: str | None,
        previous_draft: str | None,
    ) -> Draft:
        from ..schemas import ActorDraft

        messages = grounding.actor_messages(
            ticket, grounding.items, critique, previous_draft
        )
        drafted = await self.client.complete_model(messages, ActorDraft)
        recommendation = drafted.recommendation.strip()
        if not recommendation:
            raise LLMError("Actor returned an empty recommendation")
        return Draft(
            recommendation=recommendation,
            justification=drafted.justification.strip(),
        )

    async def _judge(
        self,
        ticket: IncomingTicket,
        grounding: Grounding,
        draft: str,
    ) -> Verdict:
        from ..schemas import JudgeVerdict

        messages = grounding.judge_messages(ticket, grounding.items, draft)
        # The Literal["PASS", "FAIL"] on JudgeVerdict means an unrecognised
        # status fails validation and raises rather than being read as a pass.
        verdict = await self.client.complete_model(messages, JudgeVerdict)
        if verdict.status == "PASS":
            return Verdict(passed=True)
        return Verdict(
            passed=False,
            critique=verdict.critique.strip()
            or "The draft was rejected without a stated reason.",
        )
