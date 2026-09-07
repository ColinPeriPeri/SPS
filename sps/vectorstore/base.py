"""Vector store abstraction.

The pipeline depends on this Protocol, not on Qdrant, so the backend can be
swapped for Milvus / Pinecone / pgvector without touching Components B or C.
"""

from __future__ import annotations

import uuid
from typing import Protocol, Sequence, runtime_checkable

from ..contracts import SearchHit, VectorPoint

# Stable namespace so a given SPS_ID always maps to the same point ID. Re-running
# the indexer over a modified record therefore *updates in place* instead of
# accumulating duplicate vectors.
SPS_NAMESPACE = uuid.UUID("6f3b6d2e-9a1f-4b8a-9d3f-1c5a7e2b4d80")


def point_id_for(sps_id: str) -> str:
    """Deterministic UUIDv5 point ID derived from the SPS_ID."""
    return str(uuid.uuid5(SPS_NAMESPACE, sps_id))


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection with cosine distance if absent (idempotent)."""

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        """Insert or replace points, keyed by point_id_for(sps_id)."""

    def delete(self, sps_ids: Sequence[str]) -> None:
        """Evict points whose records are no longer indexable."""

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        part_number: str | None = None,
    ) -> list[SearchHit]:
        """Cosine-similarity nearest neighbours, best first.

        `part_number` restricts the search to records carrying exactly that
        value, so semantic similarity is only ever computed against history for
        the same part. A blank or None value applies no filter -- a ticket that
        arrives without a part number must still be searchable, and filtering on
        "" would match only records whose part number is also empty.
        """

    def find_by_content_hash(self, hashes: Sequence[str]) -> dict[str, str]:
        """Map content_hash -> sps_id for pairs already present in the index.

        This is what makes deduplication survive across indexing runs: without
        it, an identical problem-solution pair arriving next week under a new
        SPS_ID would be indexed as a second vector, polluting retrieval with
        semantically identical neighbours.
        """

    def count(self) -> int:
        """Number of indexed points."""
