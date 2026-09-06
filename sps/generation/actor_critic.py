"""Component C -- the Actor / Judge generate-audit-refine loop.

Circuit breaker: MAX_ATTEMPTS total passes (1 draft + 2 refinements). A draft
that still fails the Judge on the final attempt is discarded outright; the
pipeline falls back to "Solution not found." rather than shipping unverified
text to a supplier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..config import LLMSettings
from ..contracts import SOLUTION_NOT_FOUND, Candidate, IncomingTicket
from .llm import ChatClient, LLMError
from .prompts import build_actor_messages, build_judge_messages

logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class LoopOutcome:
    """Result plus the audit trail an admin reviewer may need to see."""

    draft: Draft | None
    attempts: int
    critiques: list[str] = field(default_factory=list)
    failure_reason: str = ""
    # True when the loop stopped because a dependency was unavailable rather
    # than because the content failed review. The caller needs this to tell an
    # Azure outage apart from a legitimate refusal.
    infrastructure_failure: bool = False

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
        max_attempts = max(1, self.settings.max_attempts)
        critique: str | None = None
        previous_draft: str | None = None
        critiques: list[str] = []

        for attempt in range(1, max_attempts + 1):
            try:
                draft = await self._act(ticket, candidates, critique, previous_draft)
            except LLMError as exc:
                logger.warning("Ticket %r attempt %d: Actor failed: %s", ticket.sps_id, attempt, exc)
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=f"Generation failed: {exc}",
                    infrastructure_failure=True,
                )

            if draft.is_abstention:
                # The Actor judged the precedent inapplicable. Retrying would
                # only pressure it into inventing something.
                logger.info("Ticket %r attempt %d: Actor abstained", ticket.sps_id, attempt)
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason="Historical records did not address the reported problem.",
                )

            try:
                verdict = await self._judge(ticket, candidates, draft.recommendation)
            except LLMError as exc:
                logger.warning("Ticket %r attempt %d: Judge failed: %s", ticket.sps_id, attempt, exc)
                # An unverified draft is never shipped: a Judge outage fails closed.
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=f"Compliance audit unavailable: {exc}",
                    infrastructure_failure=True,
                )

            if verdict.passed:
                logger.info("Ticket %r: draft passed audit on attempt %d", ticket.sps_id, attempt)
                return LoopOutcome(draft=draft, attempts=attempt, critiques=critiques)

            logger.info(
                "Ticket %r attempt %d rejected: %s", ticket.sps_id, attempt, verdict.critique
            )
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
            ),
        )

    async def _act(
        self,
        ticket: IncomingTicket,
        candidates: Sequence[Candidate],
        critique: str | None,
        previous_draft: str | None,
    ) -> Draft:
        from ..schemas import ActorDraft

        messages = build_actor_messages(ticket, candidates, critique, previous_draft)
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
        candidates: Sequence[Candidate],
        draft: str,
    ) -> Verdict:
        from ..schemas import JudgeVerdict

        messages = build_judge_messages(ticket, candidates, draft)
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
