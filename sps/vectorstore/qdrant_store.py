"""Qdrant-backed VectorStore (cosine distance + metadata payloads).

LEGACY PATH -- not on the resolver flow.

The primary entry point is now `scripts/run_resolver.py`, which filters a
history file by part number and embeds the survivors per ticket, so there is
no persistent index to build or maintain. This module is retained for
`service/run_inference.py` and for a future return to a persistent index;
nothing on the resolver path imports it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..config import VectorStoreSettings
from ..contracts import SearchHit, VectorPoint
from .base import point_id_for


class VectorStoreBusyError(RuntimeError):
    """Embedded storage is locked by another process."""


class QdrantVectorStore:
    def __init__(self, settings: VectorStoreSettings | None = None, client=None) -> None:
        self.settings = settings or VectorStoreSettings()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.settings.embedded:
                # Embedded mode: in-process, persistent local directory, no
                # server. Qdrant holds an exclusive lock on it, so a second
                # concurrent process fails fast here rather than corrupting
                # anything -- surfaced as a clear error for the caller to log.
                path = Path(self.settings.path)
                path.mkdir(parents=True, exist_ok=True)
                try:
                    self._client = QdrantClient(path=str(path))
                except RuntimeError as exc:
                    raise VectorStoreBusyError(
                        f"Qdrant storage at {path} is already open by another "
                        f"process. Embedded mode allows a single writer; ensure "
                        f"the indexer and the inference run are not concurrent. "
                        f"({exc})"
                    ) from exc
            else:
                self._client = QdrantClient(
                    url=self.settings.url,
                    api_key=self.settings.api_key or None,
                    timeout=60,
                )
        return self._client

    def close(self) -> None:
        """Release the embedded-mode directory lock.

        A no-op in server mode. The CLI calls this so a Performer invocation
        never leaves the storage locked for the next scheduled process.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # already closed / server mode
                pass
            self._client = None

    # content_hash is indexed because the indexer filters on it once per batch;
    # the rest support the metadata filters used by admin tooling.
    INDEXED_FIELDS = (
        "content_hash",
        "part_number",
        "issue_type",
        "problem_reason_code",
        "item_status",
    )

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not self.client.collection_exists(self.settings.collection):
            self.client.create_collection(
                collection_name=self.settings.collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

        # Run on every call, not just at creation: a collection built before
        # content_hash existed still needs that index before the cross-run
        # dedup filter can use it. Creating an index that already exists is a
        # no-op on the server but raises on some versions, so it is tolerated.
        for field in self.INDEXED_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:  # already indexed
                pass

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.settings.collection,
            points=[
                PointStruct(
                    id=point_id_for(point.sps_id),
                    vector=list(point.vector),
                    payload=point.payload,
                )
                for point in points
            ],
            wait=True,
        )

    def delete(self, sps_ids: Sequence[str]) -> None:
        """Evict points, tolerating IDs that were never indexed.

        The indexer evicts records that failed sanitization or were superseded
        by a duplicate, and on a first load those points were never written. A
        Qdrant *server* treats deleting an unknown ID as a no-op; **embedded
        Qdrant raises KeyError**. So the bulk delete is attempted first, and on
        failure the set is narrowed to what actually exists -- one extra call,
        not one per ID, and only on the path that would otherwise fail.
        """
        if not sps_ids:
            return
        from qdrant_client.models import PointIdsList

        ids = [point_id_for(i) for i in sps_ids]
        try:
            self.client.delete(
                collection_name=self.settings.collection,
                points_selector=PointIdsList(points=ids),
                wait=True,
            )
        except (KeyError, ValueError):
            existing = [
                point.id
                for point in self.client.retrieve(
                    collection_name=self.settings.collection,
                    ids=ids,
                    with_payload=False,
                    with_vectors=False,
                )
            ]
            if not existing:
                return
            self.client.delete(
                collection_name=self.settings.collection,
                points_selector=PointIdsList(points=existing),
                wait=True,
            )

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        part_number: str | None = None,
    ) -> list[SearchHit]:
        query_filter = self._part_number_filter(part_number)

        # query_points() was added in qdrant-client 1.10 and search() is
        # deprecated from that release on, so pick whichever the installed
        # client actually has. The pinned 1.9.1 only has search().
        if hasattr(self.client, "query_points"):
            points = self.client.query_points(
                collection_name=self.settings.collection,
                query=list(vector),
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            ).points
        else:
            points = self.client.search(
                collection_name=self.settings.collection,
                query_vector=list(vector),
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            )
        return [
            SearchHit(payload=dict(p.payload or {}), cosine_similarity=float(p.score))
            for p in points
        ]

    @staticmethod
    def _part_number_filter(part_number: str | None):
        """Exact-match filter on the part_number payload key, or None.

        Blank is deliberately treated as "no filter" rather than "match blank":
        a ticket submitted without a part number must still reach the whole
        index, and MatchValue("") would instead pin it to records whose part
        number is also empty.
        """
        from ..contracts import normalize_part_number

        wanted = normalize_part_number(part_number)
        if not wanted:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Normalised on both sides: payloads are written canonically at ingest,
        # so matching the canonical query value keeps "pn-1000" from reporting
        # NO_MATCHES for a part that is genuinely indexed.
        return Filter(
            must=[FieldCondition(key="part_number", match=MatchValue(value=wanted))]
        )

    def find_by_content_hash(self, hashes: Sequence[str]) -> dict[str, str]:
        """One filtered scroll per batch, not one lookup per record."""
        if not hashes:
            return {}
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = Filter(
            must=[FieldCondition(key="content_hash", match=MatchAny(any=list(hashes)))]
        )
        found: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.settings.collection,
                scroll_filter=query_filter,
                limit=min(len(hashes), 256),
                offset=offset,
                with_payload=["content_hash", "sps_id"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                digest = payload.get("content_hash")
                if digest:
                    found.setdefault(digest, payload.get("sps_id", ""))
            if offset is None:
                return found

    def count(self) -> int:
        return int(self.client.count(self.settings.collection, exact=True).count)
