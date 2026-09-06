"""End-to-end orchestration: Component B gate -> Component C loop -> contract.

The pipeline is deliberately the only place the three components meet, and it
has exactly one public method so every ticket follows the same path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .contracts import IncomingTicket, PipelineResult
from .generation.actor_critic import ActorCriticLoop
from .generation.llm import AzureOpenAIChatClient, ChatClient
from .output import below_threshold, generation_failed, invalid_input, no_matches, success
from .retrieval.retriever import Retriever, RetrievalStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SPSPipeline:
    retriever: Retriever
    loop: ActorCriticLoop
    settings: Settings

    @classmethod
    def build(
        cls,
        store,
        embedder=None,
        client: ChatClient | None = None,
        settings: Settings | None = None,
    ) -> "SPSPipeline":
        """Wire the default production stack (BGE + Azure OpenAI)."""
        settings = settings or Settings.from_env()
        if embedder is None:
            from .embedding import BGEEmbedder

            embedder = BGEEmbedder(settings.embedding)
        return cls(
            retriever=Retriever(embedder, store, settings.retrieval),
            loop=ActorCriticLoop(client or AzureOpenAIChatClient(settings.llm), settings.llm),
            settings=settings,
        )

    async def process(self, ticket: IncomingTicket) -> PipelineResult:
        """Run one ticket to a contract-shaped result. Never raises."""
        try:
            return await self._process(ticket)
        except Exception:
            # A supplier-facing service must always emit the contract; an
            # unexpected fault becomes a graceful failure, logged for ops.
            logger.exception("Ticket %r: unhandled pipeline error", ticket.sps_id)
            return generation_failed("", 0.0, infrastructure_failure=True)

    async def _process(self, ticket: IncomingTicket) -> PipelineResult:
        outcome = self.retriever.retrieve(ticket)

        if outcome.status is RetrievalStatus.INVALID_INPUT:
            return invalid_input()
        if outcome.status is RetrievalStatus.NO_MATCHES:
            return no_matches()
        if outcome.status is RetrievalStatus.BELOW_THRESHOLD:
            return below_threshold(
                outcome.top_score, self.settings.retrieval.confidence_threshold
            )

        result = await self.loop.run(ticket, outcome.qualifying)
        if not result.succeeded or result.draft is None:
            if result.infrastructure_failure:
                # Detail stays in the log; the contract carries a generic reason
                # and the caller gets a non-zero exit to raise on.
                logger.error("Ticket %r: %s", ticket.sps_id, result.failure_reason)
            return generation_failed(
                result.failure_reason,
                outcome.top_score,
                infrastructure_failure=result.infrastructure_failure,
            )

        justification = result.draft.justification or (
            f"Synthesized from {len(outcome.qualifying)} historical SPS record(s) "
            f"matching at {outcome.confidence_percent}% confidence."
        )
        return success(
            recommendation=result.draft.recommendation,
            justification=justification,
            top_score=outcome.top_score,
            candidates=outcome.qualifying,
        )

    async def process_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Convenience entry point: raw ticket dict in, contract dict out."""
        result = await self.process(IncomingTicket.from_dict(payload))
        return result.to_contract()
