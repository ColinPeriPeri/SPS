"""Component A -- sanitization, dedup, micro-batching, delta tracking."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sps.config import IndexingSettings
from sps.indexing import IncrementalIndexer, InMemoryRecordSource, Watermark, WatermarkStore
from sps.sanitize import cleanse, content_hash, deduplicate, normalize_text
from sps.vectorstore import InMemoryVectorStore, point_id_for
from tests.conftest import BASE_TIME, TokenOverlapEmbedder, make_record

# --------------------------------------------------------------------------
# A.2 -- sanitization
# --------------------------------------------------------------------------


def test_drops_rows_with_short_or_missing_text():
    records = [
        make_record("S1"),                                  # valid
        make_record("S2", problem="too short"),             # 9 chars
        make_record("S3", solution="fixed"),                # 5 chars
        make_record("S4", problem=""),                      # empty
        make_record("S5", solution=""),                     # empty
    ]
    kept, rejected, report = cleanse(records)

    assert [r.sps_id for r in kept] == ["S1"]
    assert set(rejected) == {"S2", "S3", "S4", "S5"}
    assert report.received == 5
    assert report.kept == 1


def test_boundary_of_fifteen_characters_is_inclusive():
    exactly_15 = "123456789012345"
    assert len(exactly_15) == 15
    kept, _, _ = cleanse([make_record("S1", problem=exactly_15, solution=exactly_15)])
    assert len(kept) == 1

    kept, _, _ = cleanse([make_record("S2", problem="12345678901234", solution=exactly_15)])
    assert kept == []


def test_whitespace_only_text_cannot_pass_the_length_gate():
    kept, rejected, _ = cleanse([make_record("S1", problem="   \n\t   " * 5)])
    assert kept == []
    assert rejected == ["S1"]


def test_normalization_collapses_whitespace():
    assert normalize_text("  weld   seam \n cracked ") == "weld seam cracked"


# --------------------------------------------------------------------------
# A.2 -- deduplication
# --------------------------------------------------------------------------


def test_deduplicate_keeps_the_latest_sps_id():
    old = make_record("S1", minutes=0)
    new = make_record("S9", minutes=60)  # same problem + solution text
    survivors, superseded = deduplicate([old, new])

    assert [r.sps_id for r in survivors] == ["S9"]
    assert superseded == ["S1"]


def test_deduplicate_is_order_independent():
    old = make_record("S1", minutes=0)
    new = make_record("S9", minutes=60)
    for ordering in ([old, new], [new, old]):
        survivors, superseded = deduplicate(ordering)
        assert [r.sps_id for r in survivors] == ["S9"]
        assert superseded == ["S1"]


def test_content_hash_ignores_case_and_whitespace_but_not_substance():
    assert content_hash("Weld  cracked", "Rework it") == content_hash("weld cracked", "rework it")
    assert content_hash("Weld cracked", "Rework it") != content_hash("Weld cracked", "Scrap it")


def test_distinct_pairs_are_not_collapsed():
    a = make_record("S1")
    b = make_record("S2", solution="Scrap the part and issue a replacement lot")
    survivors, superseded = deduplicate([a, b])
    assert len(survivors) == 2
    assert superseded == []


# --------------------------------------------------------------------------
# A.1 -- high-water mark
# --------------------------------------------------------------------------


def test_watermark_roundtrip(tmp_path):
    store = WatermarkStore(tmp_path / "state" / "watermark.json")
    assert store.read().last_modified_date.year == 1970  # cold start

    mark = Watermark(last_modified_date=datetime(2026, 3, 1, tzinfo=timezone.utc), sps_id="S42")
    store.write(mark)
    assert store.read() == mark


def test_corrupt_watermark_falls_back_to_full_rebuild(tmp_path):
    path = tmp_path / "watermark.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert WatermarkStore(path).read().last_modified_date.year == 1970


def test_delta_source_returns_only_records_after_the_mark():
    records = [make_record(f"S{i}", minutes=i) for i in range(5)]
    source = InMemoryRecordSource(records)
    mark = Watermark(last_modified_date=BASE_TIME.replace(minute=2), sps_id="S2")

    assert [r.sps_id for r in source.fetch_since(mark, 10)] == ["S3", "S4"]


# --------------------------------------------------------------------------
# A.3 -- micro-batching and the indexer as a whole
# --------------------------------------------------------------------------


def _indexer(records, tmp_path, batch_size=250, store=None, embedder=None):
    settings = IndexingSettings(
        batch_size=batch_size,
        watermark_path=str(tmp_path / "watermark.json"),
    )
    return IncrementalIndexer(
        source=InMemoryRecordSource(records),
        embedder=embedder or TokenOverlapEmbedder(),
        store=store or InMemoryVectorStore(),
        settings=settings,
    )


def test_batch_size_outside_the_ram_safe_band_is_rejected():
    for bad in (1, 100, 249, 501, 5000):
        with pytest.raises(ValueError, match="RAM-safe|between"):
            IndexingSettings(batch_size=bad)


def test_batch_size_inside_the_band_is_accepted():
    for good in (250, 400, 500):
        assert IndexingSettings(batch_size=good).batch_size == good


def test_indexer_processes_in_micro_batches(tmp_path):
    embedder = TokenOverlapEmbedder()
    records = [
        make_record(f"S{i:04d}", problem=f"Defect number {i} found on the bracket", minutes=i)
        for i in range(600)
    ]
    indexer = _indexer(records, tmp_path, batch_size=250, embedder=embedder)
    report = indexer.run()

    assert report.indexed == 600
    assert report.batches == 3  # 250 + 250 + 100
    # No single embed call ever exceeds the micro-batch ceiling.
    assert max(len(call) for call in embedder.passage_calls) <= 250


def test_only_the_problem_description_is_embedded(tmp_path):
    embedder = TokenOverlapEmbedder()
    record = make_record("S1")
    _indexer([record], tmp_path, embedder=embedder).run()

    embedded = embedder.passage_calls[0]
    assert embedded == [record.problem_description]
    # A.4: no metadata and no solution text leak into the embedding string.
    assert record.actual_solution not in embedded[0]
    assert record.part_number not in embedded[0]


def test_payload_carries_exactly_the_specified_schema(tmp_path):
    store = InMemoryVectorStore()
    _indexer([make_record("S1")], tmp_path, store=store).run()

    _, payload = store._points[point_id_for("S1")]
    assert set(payload) == {
        "sps_id",
        "content_hash",
        "actual_solution",
        "part_number",
        "part_description",
        "item_status",
        "problem_reason_code",
        "issue_type",
    }
    assert payload["sps_id"] == "S1"
    # SHA-256 of the sanitized problem + solution pair.
    assert len(payload["content_hash"]) == 64
    assert payload["content_hash"] == content_hash(
        "Bracket weld seam cracking observed during incoming inspection",
        "Rework the weld seam and re-inspect before shipment",
    )


def test_second_run_indexes_only_the_delta(tmp_path):
    store = InMemoryVectorStore()
    first = [make_record(f"S{i}", problem=f"Issue {i} on the bracket weld", minutes=i)
             for i in range(3)]

    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    embedder = TokenOverlapEmbedder()

    run_one = IncrementalIndexer(InMemoryRecordSource(first), embedder, store, settings)
    assert run_one.run().indexed == 3

    # Same source, nothing new: the watermark suppresses a rebuild.
    run_two = IncrementalIndexer(InMemoryRecordSource(first), embedder, store, settings)
    report = run_two.run()
    assert report.indexed == 0
    assert report.batches == 0

    # A new record arrives; only it is processed.
    extended = first + [make_record("S9", problem="Fresh corrosion on the flange", minutes=99)]
    run_three = IncrementalIndexer(InMemoryRecordSource(extended), embedder, store, settings)
    assert run_three.run().indexed == 1
    assert store.count() == 4


def test_duplicates_spanning_batches_keep_only_the_latest(tmp_path):
    store = InMemoryVectorStore()
    # 250 fillers, then the same problem/solution pair twice at different times,
    # placed so the pair straddles the batch boundary.
    records = [
        make_record(f"F{i:04d}", problem=f"Filler defect {i} on the housing", minutes=i)
        for i in range(249)
    ]
    records.append(make_record("OLD", minutes=500))
    records.append(make_record("NEW", minutes=600))

    report = _indexer(records, tmp_path, batch_size=250, store=store).run()

    assert report.batches == 2
    assert point_id_for("NEW") in store._points
    assert point_id_for("OLD") not in store._points  # superseded and evicted


def test_records_that_become_invalid_are_evicted(tmp_path):
    store = InMemoryVectorStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    embedder = TokenOverlapEmbedder()

    IncrementalIndexer(
        InMemoryRecordSource([make_record("S1", minutes=0)]), embedder, store, settings
    ).run()
    assert store.count() == 1

    # The solution field is later blanked in the source system.
    IncrementalIndexer(
        InMemoryRecordSource([make_record("S1", solution="", minutes=10)]),
        embedder, store, settings,
    ).run()
    assert store.count() == 0


def test_watermark_advances_past_rows_that_were_dropped(tmp_path):
    store = InMemoryVectorStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    # Every row is unusable, but the run must still not re-read them forever.
    records = [make_record(f"B{i}", solution="no", minutes=i) for i in range(3)]

    report = IncrementalIndexer(
        InMemoryRecordSource(records), TokenOverlapEmbedder(), store, settings
    ).run()

    assert report.indexed == 0
    assert report.watermark.sps_id == "B2"
    assert WatermarkStore(settings.watermark_path).read().sps_id == "B2"


def test_modified_record_updates_in_place_rather_than_duplicating(tmp_path):
    store = InMemoryVectorStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    embedder = TokenOverlapEmbedder()

    IncrementalIndexer(
        InMemoryRecordSource([make_record("S1", minutes=0)]), embedder, store, settings
    ).run()
    IncrementalIndexer(
        InMemoryRecordSource(
            [make_record("S1", solution="Replace the bracket assembly entirely", minutes=10)]
        ),
        embedder, store, settings,
    ).run()

    assert store.count() == 1
    _, payload = store._points[point_id_for("S1")]
    assert payload["actual_solution"] == "Replace the bracket assembly entirely"


# --------------------------------------------------------------------------
# A.2 -- cross-run deduplication via the persisted content_hash
# --------------------------------------------------------------------------


def test_duplicate_arriving_in_a_later_run_is_deduplicated(tmp_path):
    """The gap the in-run ledger cannot see: an identical pair submitted weeks
    later under a new SPS_ID must replace the incumbent, not join it."""
    store = InMemoryVectorStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    embedder = TokenOverlapEmbedder()

    IncrementalIndexer(
        InMemoryRecordSource([make_record("SPS-1", minutes=0)]), embedder, store, settings
    ).run()
    assert store.count() == 1

    # Next week's delta: same problem and solution text, new SPS_ID.
    report = IncrementalIndexer(
        InMemoryRecordSource([make_record("SPS-9", minutes=10_000)]),
        embedder, store, settings,
    ).run()

    assert report.indexed == 1
    assert report.cross_run_duplicates == 1
    assert store.count() == 1                            # no vector pollution
    assert point_id_for("SPS-9") in store._points
    assert point_id_for("SPS-1") not in store._points    # incumbent evicted


def test_cross_run_dedup_does_not_evict_a_records_own_point(tmp_path):
    """Re-indexing an unchanged record must be an update, not delete-and-add."""
    store = InMemoryVectorStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    embedder = TokenOverlapEmbedder()

    IncrementalIndexer(
        InMemoryRecordSource([make_record("SPS-1", minutes=0)]), embedder, store, settings
    ).run()

    # Same SPS_ID, same content, later timestamp (e.g. an unrelated column changed).
    report = IncrementalIndexer(
        InMemoryRecordSource([make_record("SPS-1", minutes=500)]),
        embedder, store, settings,
    ).run()

    assert report.cross_run_duplicates == 0
    assert store.count() == 1
    assert point_id_for("SPS-1") in store._points


def test_distinct_records_across_runs_are_both_kept(tmp_path):
    store = InMemoryVectorStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    embedder = TokenOverlapEmbedder()

    IncrementalIndexer(
        InMemoryRecordSource([make_record("SPS-1", minutes=0)]), embedder, store, settings
    ).run()
    report = IncrementalIndexer(
        InMemoryRecordSource(
            [make_record("SPS-2", problem="Flange face corrosion found at goods-in", minutes=100)]
        ),
        embedder, store, settings,
    ).run()

    assert report.cross_run_duplicates == 0
    assert store.count() == 2


class _CountingStore(InMemoryVectorStore):
    """Records how many times, and how widely, the hash lookup is called."""

    def __init__(self) -> None:
        super().__init__()
        self.lookup_sizes: list[int] = []

    def find_by_content_hash(self, hashes):
        self.lookup_sizes.append(len(hashes))
        return super().find_by_content_hash(hashes)


def _six_hundred_records():
    return [
        make_record(f"S{i:04d}", problem=f"Distinct defect {i} on the housing", minutes=i)
        for i in range(600)
    ]


def test_content_hash_lookup_is_one_call_per_batch(tmp_path):
    """At ~300k records the lookup must not degrade into per-record queries."""
    store = _CountingStore()
    settings = IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json"))
    # Establish a mark so this counts as an incremental delta, not a rebuild.
    WatermarkStore(settings.watermark_path).write(
        Watermark(last_modified_date=BASE_TIME, sps_id="")
    )

    IncrementalIndexer(
        InMemoryRecordSource(_six_hundred_records()), TokenOverlapEmbedder(), store, settings
    ).run()

    assert store.lookup_sizes == [250, 250, 100]  # one per batch, not per record


def test_full_rebuild_skips_the_cross_run_lookup(tmp_path):
    """A rebuild reads the whole table, so the in-run ledger already sees every
    record and the per-batch store scan is pure cost -- on embedded Qdrant that
    scan is linear in collection size, so skipping it matters at 300k."""
    store = _CountingStore()
    report = _indexer(_six_hundred_records(), tmp_path, batch_size=250, store=store).run()

    assert store.lookup_sizes == []   # no lookups at all
    assert report.indexed == 600      # indexing itself is unaffected
    assert store.count() == 600


def test_rebuild_still_deduplicates_within_the_run(tmp_path):
    """Skipping the store lookup must not weaken dedup: the ledger covers it."""
    store = _CountingStore()
    records = _six_hundred_records()
    records.append(make_record("DUP", problem=records[0].problem_description,
                               solution=records[0].actual_solution, minutes=9999))

    report = _indexer(records, tmp_path, batch_size=250, store=store).run()

    assert store.lookup_sizes == []
    assert point_id_for("DUP") in store._points
    assert point_id_for("S0000") not in store._points  # superseded by the later duplicate
    # The duplicate straddles a batch boundary, so S0000 was already written when
    # DUP arrived: the ledger evicts the older point rather than dropping the
    # newer record, which is why this counts under `evicted`, not
    # `dropped_duplicate`.
    assert report.evicted == 1
    assert store.count() == 600


def test_hash_changes_when_the_solution_is_edited():
    a = make_record("S1")
    b = make_record("S1", solution="Scrap the lot and ship replacements from stock")
    assert a.content_digest() != b.content_digest()


def test_hash_is_insensitive_to_cosmetic_reformatting():
    a = make_record("S1", problem="Weld  seam\ncracked", solution="Rework the seam")
    b = make_record("S2", problem="weld seam cracked", solution="REWORK THE SEAM")
    assert a.content_digest() == b.content_digest()
