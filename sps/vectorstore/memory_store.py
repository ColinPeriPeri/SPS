"""In-memory VectorStore for tests and local dry runs.

Same semantics as the Qdrant adapter (cosine similarity, upsert-by-SPS_ID) with
no server required.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..contracts import SearchHit, VectorPoint
from .base import point_id_for


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._points: dict[str, tuple[list[float], dict[str, str]]] = {}
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        for point in points:
            self._points[point_id_for(point.sps_id)] = (list(point.vector), dict(point.payload))

    def delete(self, sps_ids: Sequence[str]) -> None:
        for sps_id in sps_ids:
            self._points.pop(point_id_for(sps_id), None)

    def search(self, vector: Sequence[float], limit: int) -> list[SearchHit]:
        scored = [
            SearchHit(payload=payload, cosine_similarity=cosine_similarity(vector, stored))
            for stored, payload in self._points.values()
        ]
        scored.sort(key=lambda hit: hit.cosine_similarity, reverse=True)
        return scored[:limit]

    def find_by_content_hash(self, hashes: Sequence[str]) -> dict[str, str]:
        wanted = set(hashes)
        found: dict[str, str] = {}
        for _, payload in self._points.values():
            digest = payload.get("content_hash", "")
            if digest in wanted:
                found[digest] = payload.get("sps_id", "")
        return found

    def count(self) -> int:
        return len(self._points)
