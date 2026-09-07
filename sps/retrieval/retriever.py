"""Component B -- input validation, vector retrieval and the confidence gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from ..config import RetrievalSettings
from ..contracts import Candidate, IncomingTicket, score_to_percent
from ..embedding import Embedder
from ..vectorstore.base import VectorStore
from .scoring import rank_candidates

logger = logging.getLogger(__name__)


class RetrievalStatus(str, Enum):
    OK = "ok"
    INVALID_INPUT = "invalid_input"
    NO_MATCHES = "no_matches"
    BELOW_THRESHOLD = "below_threshold"


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    status: RetrievalStatus
    candidates: list[Candidate]
    top_score: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status is RetrievalStatus.OK

    @property
    def confidence_percent(self) -> int:
        return score_to_percent(self.top_score)

    @property
    def qualifying(self) -> list[Candidate]:
        """Candidates at or above the gate -- the Actor's entire context."""
        return self.candidates


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings or RetrievalSettings()

    def validate(self, ticket: IncomingTicket) -> bool:
        """Component B.1 -- non-empty Problem_Description of >= 10 characters."""
        text = (ticket.problem_description or "").strip()
        return len(text) >= self.settings.min_query_length

    def retrieve(self, ticket: IncomingTicket) -> RetrievalOutcome:
        if not self.validate(ticket):
            logger.info("Rejecting ticket %r: problem statement too short", ticket.sps_id)
            return RetrievalOutcome(status=RetrievalStatus.INVALID_INPUT, candidates=[])

        # Hard filter: semantic similarity is only ever computed against history
        # for the same part. A ticket with no part number searches the whole
        # index rather than being pinned to records with a blank one.
        part_number = (ticket.part_number or "").strip() or None

        vector = self.embedder.embed_query(ticket.problem_description.strip())
        hits = self.store.search(vector, limit=self.settings.top_k, part_number=part_number)
        if not hits:
            # With the filter on, this now also means "no history for this part",
            # which is a far more common outcome than an empty index.
            logger.info(
                "Ticket %r: vector search returned no candidates (part_number=%r)",
                ticket.sps_id,
                part_number,
            )
            return RetrievalOutcome(status=RetrievalStatus.NO_MATCHES, candidates=[])

        ranked = rank_candidates(hits, ticket)
        top_score = ranked[0].composite_score

        # Component B.4: abort before any LLM call when the best match is weak.
        if top_score < self.settings.confidence_threshold:
            logger.info(
                "Ticket %r: top composite %.4f below %.2f threshold; skipping generation",
                ticket.sps_id,
                top_score,
                self.settings.confidence_threshold,
            )
            return RetrievalOutcome(
                status=RetrievalStatus.BELOW_THRESHOLD,
                candidates=[],
                top_score=top_score,
            )

        qualifying = [
            candidate
            for candidate in ranked
            if candidate.composite_score >= self.settings.confidence_threshold
        ][: self.settings.max_context_records]

        logger.info(
            "Ticket %r: %d/%d candidates at or above threshold, top=%.4f",
            ticket.sps_id,
            len(qualifying),
            len(ranked),
            top_score,
        )
        return RetrievalOutcome(
            status=RetrievalStatus.OK,
            candidates=qualifying,
            top_score=top_score,
        )
