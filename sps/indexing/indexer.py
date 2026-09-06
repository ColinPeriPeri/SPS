"""Component A -- offline incremental indexing engine.

Run periodically (nightly/weekly). Never rebuilds: it reads only the delta past
the stored high-water mark, and streams it through RAM-safe micro-batches.
"""

from __future__ import annotations

import gc
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, Sequence

from ..config import MAX_INDEX_BATCH, MIN_INDEX_BATCH, IndexingSettings
from ..contracts import SourceRecord, VectorPoint
from ..embedding import Embedder
from ..sanitize import SanitizeReport, cleanse, content_hash, deduplicate
from ..vectorstore.base import VectorStore
from .source import RecordSource
from .watermark import Watermark, WatermarkStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexRunReport:
    batches: int = 0
    indexed: int = 0
    evicted: int = 0
    cross_run_duplicates: int = 0
    sanitize: SanitizeReport = field(default_factory=SanitizeReport)
    watermark: Watermark | None = None

    def as_dict(self) -> dict:
        return {
            "batches": self.batches,
            "indexed": self.indexed,
            "evicted": self.evicted,
            "cross_run_duplicates": self.cross_run_duplicates,
            "sanitize": self.sanitize.as_dict(),
            "watermark": self.watermark.as_dict() if self.watermark else None,
        }


class DedupeLedger:
    """Run-scoped ledger enforcing one vector per problem-solution pair.

    Batch-local dedup is not enough: duplicates routinely straddle a batch
    boundary. The ledger carries hashes across the whole run and, because the
    source stream is ordered ascending by (Last_Modified_Date, SPS_ID), the
    later SPS_ID always wins -- matching the spec's "keep the latest".
    """

    __slots__ = ("_by_hash", "_by_id")

    def __init__(self) -> None:
        self._by_hash: dict[str, tuple[tuple[datetime, str], str]] = {}
        self._by_id: dict[str, str] = {}

    def register(self, record: SourceRecord) -> tuple[bool, str | None]:
        """Return (should_index, sps_id_to_evict)."""
        digest = content_hash(record.problem_description, record.actual_solution)

        # This SPS_ID was seen earlier in the run under different content (it was
        # edited); retire its stale hash so it cannot be evicted by a later
        # record that happens to match the old text.
        stale = self._by_id.get(record.sps_id)
        if stale is not None and stale != digest:
            self._by_hash.pop(stale, None)

        evict: str | None = None
        incumbent = self._by_hash.get(digest)
        if incumbent is not None and incumbent[1] != record.sps_id:
            if record.sort_key() > incumbent[0]:
                evict = incumbent[1]  # this record is newer; drop the older one
            else:
                return False, None  # incumbent is newer; skip this duplicate

        self._by_hash[digest] = (record.sort_key(), record.sps_id)
        self._by_id[record.sps_id] = digest
        return True, evict


def _batched(iterable: Iterable[SourceRecord], size: int) -> Iterator[list[SourceRecord]]:
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


class IncrementalIndexer:
    def __init__(
        self,
        source: RecordSource,
        embedder: Embedder,
        store: VectorStore,
        settings: IndexingSettings | None = None,
        watermark_store: WatermarkStore | None = None,
    ) -> None:
        self.settings = settings or IndexingSettings()
        if not (MIN_INDEX_BATCH <= self.settings.batch_size <= MAX_INDEX_BATCH):
            raise ValueError(
                "batch_size must be between "
                f"{MIN_INDEX_BATCH} and {MAX_INDEX_BATCH}, got {self.settings.batch_size}"
            )
        self.source = source
        self.embedder = embedder
        self.store = store
        self.watermarks = watermark_store or WatermarkStore(self.settings.watermark_path)

    def run(self) -> IndexRunReport:
        start_mark = self.watermarks.read()
        logger.info(
            "Indexing delta since %s (SPS_ID > %r), batch_size=%d",
            start_mark.last_modified_date.isoformat(),
            start_mark.sps_id,
            self.settings.batch_size,
        )
        self.store.ensure_collection(self.embedder.dimension)

        # A run starting from the initial mark reads the entire table, so the
        # in-run ledger already sees every record and the per-batch store lookup
        # is pure overhead. That lookup is a filtered scan, which is linear in
        # collection size on embedded Qdrant (payload indexes are a no-op there),
        # so skipping it removes roughly half an hour from a 300k backfill.
        full_rebuild = start_mark == Watermark.initial()
        if full_rebuild:
            logger.info("Full rebuild detected: cross-run hash lookup disabled for this run")

        ledger = DedupeLedger()
        totals = SanitizeReport()
        batches = 0
        indexed = 0
        evicted = 0
        cross_run_total = 0
        mark = start_mark

        stream = self.source.fetch_since(start_mark, self.settings.batch_size)
        for batch in _batched(stream, self.settings.batch_size):
            batches += 1
            # Advance past every row read, including rows we drop, so rejected
            # records are not re-fetched on every subsequent run.
            batch_mark = self._max_mark(batch, mark)

            count, evictions, cross_run, report = self._process_batch(
                batch, ledger, cross_run_lookup=not full_rebuild
            )
            indexed += count
            evicted += evictions
            cross_run_total += cross_run
            totals = totals.merge(report)

            # Commit the mark per batch: a crash mid-run resumes here instead of
            # restarting the whole delta.
            mark = batch_mark
            self.watermarks.write(mark)

            logger.info(
                "batch %d: read=%d indexed=%d evicted=%d watermark=%s",
                batches,
                len(batch),
                count,
                evictions,
                mark.last_modified_date.isoformat(),
            )

            # Release the batch and its vectors before pulling the next chunk --
            # the guard against OOM on a 10-12 GB CPU host.
            del batch
            gc.collect()

        final = IndexRunReport(
            batches=batches,
            indexed=indexed,
            evicted=evicted,
            cross_run_duplicates=cross_run_total,
            sanitize=totals,
            watermark=mark,
        )
        logger.info("Index run complete: %s", final.as_dict())
        return final

    def _process_batch(
        self,
        batch: Sequence[SourceRecord],
        ledger: DedupeLedger,
        cross_run_lookup: bool = True,
    ) -> tuple[int, int, int, SanitizeReport]:
        kept, rejected, report = cleanse(batch, self.settings.min_text_length)
        survivors, superseded = deduplicate(kept)

        to_index: list[SourceRecord] = []
        evictions: list[str] = list(rejected) + list(superseded)
        for record in survivors:
            should_index, evict_id = ledger.register(record)
            if evict_id:
                evictions.append(evict_id)
            if should_index:
                to_index.append(record)
            else:
                evictions.append(record.sps_id)

        report = SanitizeReport(
            received=report.received,
            dropped_missing_id=report.dropped_missing_id,
            dropped_short_problem=report.dropped_short_problem,
            dropped_short_solution=report.dropped_short_solution,
            dropped_duplicate=len(kept) - len(to_index),
            kept=len(to_index),
        )

        cross_run = 0
        if to_index and cross_run_lookup:
            # Cross-run dedup. The ledger above only sees this run; an identical
            # pair indexed weeks ago under a different SPS_ID is caught here, by
            # one filtered lookup per batch against the persisted content_hash.
            #
            # The incoming record always wins: the delta query only returns rows
            # past the watermark, and everything already indexed was read at or
            # below it, so an incoming record is strictly newer by the
            # (Last_Modified_Date, SPS_ID) ordering.
            by_digest = {record.content_digest(): record for record in to_index}
            for digest, incumbent_id in self.store.find_by_content_hash(
                list(by_digest)
            ).items():
                record = by_digest.get(digest)
                if record is not None and incumbent_id and incumbent_id != record.sps_id:
                    evictions.append(incumbent_id)
                    cross_run += 1

        if to_index:
            # Component A.4: the embedded string is the cleansed
            # Problem_Description and nothing else.
            vectors = self.embedder.embed_passages(
                [record.problem_description for record in to_index]
            )
            self.store.upsert(
                [
                    VectorPoint(sps_id=record.sps_id, vector=vector, payload=record.payload())
                    for record, vector in zip(to_index, vectors)
                ]
            )
            del vectors

        # Evict points whose records became unusable or were superseded, so the
        # incremental index stays equivalent to a full rebuild.
        unique_evictions = sorted({i for i in evictions if i})
        if unique_evictions:
            self.store.delete(unique_evictions)

        return len(to_index), len(unique_evictions), cross_run, report

    @staticmethod
    def _max_mark(batch: Sequence[SourceRecord], current: Watermark) -> Watermark:
        best = (current.last_modified_date, current.sps_id)
        for record in batch:
            if record.last_modified_date is None:
                continue
            key = record.sort_key()
            if key > best:
                best = key
        return Watermark(last_modified_date=best[0], sps_id=best[1])
