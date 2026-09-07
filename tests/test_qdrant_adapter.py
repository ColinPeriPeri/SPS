"""QdrantVectorStore against a real Qdrant engine.

qdrant-client ships a local mode (":memory:") that runs the actual filtering and
similarity code in-process, so the adapter is validated for real -- no server,
no mocks. Skipped when qdrant-client is not installed.
"""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from sps.config import IndexingSettings, RetrievalSettings, VectorStoreSettings  # noqa: E402
from sps.contracts import IncomingTicket  # noqa: E402
from sps.indexing import IncrementalIndexer, InMemoryRecordSource  # noqa: E402
from sps.retrieval import RetrievalStatus, Retriever  # noqa: E402
from sps.vectorstore import point_id_for  # noqa: E402
from sps.vectorstore.qdrant_store import QdrantVectorStore  # noqa: E402
from tests.conftest import TokenOverlapEmbedder, make_record  # noqa: E402


@pytest.fixture
def store():
    from qdrant_client import QdrantClient

    return QdrantVectorStore(
        settings=VectorStoreSettings(collection="test_sps"),
        client=QdrantClient(":memory:"),
    )


@pytest.fixture
def embedder():
    return TokenOverlapEmbedder()


def _index(records, store, embedder, tmp_path):
    return IncrementalIndexer(
        source=InMemoryRecordSource(records),
        embedder=embedder,
        store=store,
        settings=IndexingSettings(batch_size=250, watermark_path=str(tmp_path / "wm.json")),
    ).run()


def test_collection_is_created_with_cosine_distance(store, embedder):
    from qdrant_client.models import Distance

    store.ensure_collection(embedder.dimension)
    info = store.client.get_collection("test_sps")
    params = info.config.params.vectors
    assert params.size == embedder.dimension
    assert params.distance == Distance.COSINE


def test_ensure_collection_is_idempotent(store, embedder):
    store.ensure_collection(embedder.dimension)
    store.ensure_collection(embedder.dimension)  # must not raise on re-index
    assert store.count() == 0


def test_upsert_search_and_payload_roundtrip(store, embedder, tmp_path):
    records = [
        make_record("SPS-1", problem="Bracket weld seam cracking at incoming inspection"),
        make_record("SPS-2", problem="Outer carton label misprint on the packaging",
                    solution="Reprint the carton labels before dispatch",
                    part_number="PN-2000", issue_type="Packaging"),
    ]
    report = _index(records, store, embedder, tmp_path)
    assert report.indexed == 2
    assert store.count() == 2

    hits = store.search(embedder.embed_query("Bracket weld seam cracking at incoming inspection"), limit=5)
    assert hits
    top = hits[0]
    assert top.payload["sps_id"] == "SPS-1"
    assert top.cosine_similarity == pytest.approx(1.0, abs=1e-5)
    # The full approved payload survives the round trip.
    assert set(top.payload) == {
        "sps_id", "content_hash", "actual_solution", "part_number",
        "part_description", "item_status", "problem_reason_code", "issue_type",
    }


def test_search_returns_results_best_first(store, embedder, tmp_path):
    records = [
        make_record("SPS-1", problem="Bracket weld seam cracking at incoming inspection"),
        make_record("SPS-2", problem="Bracket weld seam cracking found later",
                    solution="Rework the seam and re-inspect the bracket"),
        make_record("SPS-3", problem="Completely unrelated carton label misprint",
                    solution="Reprint the carton labels before dispatch"),
    ]
    _index(records, store, embedder, tmp_path)

    hits = store.search(embedder.embed_query("Bracket weld seam cracking at incoming inspection"), limit=3)
    scores = [h.cosine_similarity for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].payload["sps_id"] == "SPS-1"


def test_delete_removes_points(store, embedder, tmp_path):
    _index([make_record("SPS-1"), make_record("SPS-2", problem="Flange corrosion at goods-in",
                                              solution="Clean and re-protect the flange face")],
           store, embedder, tmp_path)
    assert store.count() == 2

    store.delete(["SPS-1"])
    assert store.count() == 1
    remaining = store.search(embedder.embed_query("Flange corrosion at goods-in"), limit=5)
    assert remaining[0].payload["sps_id"] == "SPS-2"


def test_find_by_content_hash_filter_works(store, embedder, tmp_path):
    record = make_record("SPS-1")
    _index([record], store, embedder, tmp_path)

    digest = record.content_digest()
    assert store.find_by_content_hash([digest]) == {digest: "SPS-1"}
    assert store.find_by_content_hash(["0" * 64]) == {}
    assert store.find_by_content_hash([]) == {}


def test_cross_run_dedup_against_real_qdrant(store, embedder, tmp_path):
    """The approved schema change, exercised end to end on the real engine."""
    _index([make_record("SPS-1", minutes=0)], store, embedder, tmp_path)
    assert store.count() == 1

    report = _index([make_record("SPS-9", minutes=10_000)], store, embedder, tmp_path)

    assert report.cross_run_duplicates == 1
    assert store.count() == 1
    hits = store.search(embedder.embed_query(make_record("SPS-9").problem_description), limit=5)
    assert hits[0].payload["sps_id"] == "SPS-9"


def test_modified_record_updates_in_place(store, embedder, tmp_path):
    _index([make_record("SPS-1", minutes=0)], store, embedder, tmp_path)
    _index([make_record("SPS-1", solution="Replace the bracket assembly entirely", minutes=10)],
           store, embedder, tmp_path)

    assert store.count() == 1
    hits = store.search(embedder.embed_query(make_record("SPS-1").problem_description), limit=1)
    assert hits[0].payload["actual_solution"] == "Replace the bracket assembly entirely"


def test_retriever_end_to_end_on_real_qdrant(store, embedder, tmp_path):
    problem = "Bracket weld seam cracking observed during incoming inspection"
    _index(
        [
            make_record("SPS-1", problem=problem),
            make_record("SPS-2", problem="Outer carton label misprint on the packaging",
                        solution="Reprint the carton labels before dispatch",
                        part_number="PN-2000", issue_type="Packaging"),
        ],
        store, embedder, tmp_path,
    )

    retriever = Retriever(embedder, store, RetrievalSettings())
    outcome = retriever.retrieve(
        IncomingTicket(problem_description=problem, part_number="PN-1000", issue_type="Quality")
    )

    assert outcome.status is RetrievalStatus.OK
    assert [c.sps_id for c in outcome.candidates] == ["SPS-1"]
    assert outcome.confidence_percent == 100


def test_weak_match_is_gated_on_real_qdrant(store, embedder, tmp_path):
    _index([make_record("SPS-1", problem="Outer carton label misprint on the packaging",
                        solution="Reprint the carton labels before dispatch")],
           store, embedder, tmp_path)

    outcome = Retriever(embedder, store, RetrievalSettings()).retrieve(
        IncomingTicket(problem_description="Hydraulic pump pressure fluctuating on the test run")
    )
    assert outcome.status is RetrievalStatus.BELOW_THRESHOLD


def test_point_ids_are_deterministic_uuid5(store, embedder, tmp_path):
    _index([make_record("SPS-1")], store, embedder, tmp_path)
    retrieved = store.client.retrieve("test_sps", ids=[point_id_for("SPS-1")])
    assert len(retrieved) == 1
    assert retrieved[0].payload["sps_id"] == "SPS-1"


# --------------------------------------------------------------------------
# Embedded mode -- the UiPath deployment shape: no server, local directory
# --------------------------------------------------------------------------


def _embedded(tmp_path, name="embedded_sps"):
    return QdrantVectorStore(
        settings=VectorStoreSettings(collection=name, path=str(tmp_path / "qdrant_data"))
    )


def test_settings_select_embedded_mode_from_a_path():
    server = VectorStoreSettings()
    embedded = VectorStoreSettings(path="./qdrant_data")

    assert not server.embedded and "server:" in server.describe()
    assert embedded.embedded and "embedded:" in embedded.describe()


def test_embedded_mode_persists_across_processes(tmp_path, embedder):
    """The whole point of a local path: the index survives the process exiting."""
    store = _embedded(tmp_path)
    _index([make_record("SPS-1"), make_record("SPS-2", problem="Flange corrosion at goods-in",
                                              solution="Clean and re-protect the flange face")],
           store, embedder, tmp_path)
    assert store.count() == 2
    store.close()

    reopened = _embedded(tmp_path)
    assert reopened.count() == 2
    hits = reopened.search(embedder.embed_query(make_record("SPS-1").problem_description), limit=1)
    assert hits[0].payload["sps_id"] == "SPS-1"
    reopened.close()


def test_embedded_directory_is_created_on_demand(tmp_path, embedder):
    target = tmp_path / "nested" / "deeper" / "qdrant_data"
    store = QdrantVectorStore(settings=VectorStoreSettings(collection="c", path=str(target)))
    store.ensure_collection(embedder.dimension)
    assert target.exists()
    store.close()


def test_close_releases_the_lock_for_the_next_process(tmp_path, embedder):
    """UiPath alternates the indexer and the Performer against one directory, so
    a run that does not release the lock blocks the next scheduled process."""
    first = _embedded(tmp_path)
    first.ensure_collection(embedder.dimension)
    first.close()

    second = _embedded(tmp_path)          # must not raise
    second.ensure_collection(embedder.dimension)
    assert second.count() == 0
    second.close()


def test_concurrent_open_fails_loudly(tmp_path, embedder):
    """Embedded Qdrant is single-writer. A second opener must get a clear,
    actionable error rather than silent corruption."""
    from sps.vectorstore import VectorStoreBusyError

    first = _embedded(tmp_path)
    first.ensure_collection(embedder.dimension)
    try:
        second = _embedded(tmp_path)
        with pytest.raises(VectorStoreBusyError, match="already open by another process"):
            second.ensure_collection(embedder.dimension)
    finally:
        first.close()


def test_close_is_safe_to_call_twice(tmp_path, embedder):
    store = _embedded(tmp_path)
    store.ensure_collection(embedder.dimension)
    store.close()
    store.close()  # idempotent


# --------------------------------------------------------------------------
# Eviction of never-indexed IDs
# --------------------------------------------------------------------------


def test_deleting_a_never_indexed_id_is_a_no_op(store, embedder, tmp_path):
    """The indexer evicts records that failed sanitization or were superseded,
    and on a first load those points were never written. A Qdrant server treats
    that as a no-op; embedded Qdrant raises KeyError."""
    _index([make_record("SPS-1")], store, embedder, tmp_path)
    assert store.count() == 1

    store.delete(["NEVER-INDEXED"])          # must not raise
    assert store.count() == 1


def test_deleting_a_mix_of_known_and_unknown_ids(store, embedder, tmp_path):
    _index(
        [make_record("SPS-1"),
         make_record("SPS-2", problem="Flange corrosion at goods-in",
                     solution="Clean and re-protect the flange face")],
        store, embedder, tmp_path,
    )

    store.delete(["SPS-1", "NEVER-INDEXED", "ALSO-MISSING"])

    assert store.count() == 1
    remaining = store.search(embedder.embed_query("Flange corrosion at goods-in"), limit=5)
    assert remaining[0].payload["sps_id"] == "SPS-2"


def test_indexing_a_file_whose_rows_are_all_rejected(store, embedder, tmp_path):
    """Every row fails sanitization, so every eviction targets an ID that was
    never indexed -- the exact shape that used to raise on embedded Qdrant."""
    records = [make_record(f"BAD-{i}", solution="no", minutes=i) for i in range(3)]
    report = _index(records, store, embedder, tmp_path)

    assert report.indexed == 0
    assert store.count() == 0


# --------------------------------------------------------------------------
# Hard part_number filter, against the real Qdrant filtering engine
# --------------------------------------------------------------------------


def _mixed_parts(store, embedder, tmp_path):
    problem = "Bracket weld seam cracking observed during incoming inspection"
    _index(
        [
            make_record("SAME", problem=problem, part_number="PN-1000"),
            make_record("OTHER", problem=problem, part_number="PN-2000",
                        solution="Scrap the lot and ship replacements from stock"),
            make_record("BLANK", problem=problem, part_number="",
                        solution="Return the unit to the supplier for analysis"),
        ],
        store, embedder, tmp_path,
    )
    return embedder.embed_query(problem)


def test_qdrant_filter_returns_only_the_requested_part(store, embedder, tmp_path):
    vector = _mixed_parts(store, embedder, tmp_path)
    hits = store.search(vector, limit=15, part_number="PN-1000")
    assert [h.payload["sps_id"] for h in hits] == ["SAME"]


def test_qdrant_unfiltered_search_still_sees_everything(store, embedder, tmp_path):
    vector = _mixed_parts(store, embedder, tmp_path)
    assert len(store.search(vector, limit=15)) == 3
    assert len(store.search(vector, limit=15, part_number=None)) == 3
    assert len(store.search(vector, limit=15, part_number="   ")) == 3


def test_qdrant_filter_is_exact_not_prefix(store, embedder, tmp_path):
    vector = _mixed_parts(store, embedder, tmp_path)
    assert store.search(vector, limit=15, part_number="PN-100") == []
    assert store.search(vector, limit=15, part_number="PN") == []


def test_qdrant_filter_is_case_insensitive_via_normalisation(store, embedder, tmp_path):
    """Payloads are canonicalised at ingest, so the query is canonicalised too:
    a lower-case part number must not report NO_MATCHES for an indexed part."""
    vector = _mixed_parts(store, embedder, tmp_path)
    for spelling in ("PN-1000", "pn-1000", "  Pn-1000  "):
        hits = store.search(vector, limit=15, part_number=spelling)
        assert [h.payload["sps_id"] for h in hits] == ["SAME"], spelling


def test_qdrant_filter_on_an_unknown_part_returns_nothing(store, embedder, tmp_path):
    vector = _mixed_parts(store, embedder, tmp_path)
    assert store.search(vector, limit=15, part_number="PN-DOES-NOT-EXIST") == []


def test_qdrant_filter_object_shape():
    """The filter must be an exact MatchValue on the part_number payload key."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    built = QdrantVectorStore._part_number_filter("PN-1000")
    assert isinstance(built, Filter)
    condition = built.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "part_number"
    assert condition.match == MatchValue(value="PN-1000")

    for blank in (None, "", "  "):
        assert QdrantVectorStore._part_number_filter(blank) is None


def test_qdrant_end_to_end_retrieval_is_part_scoped(store, embedder, tmp_path):
    vector = _mixed_parts(store, embedder, tmp_path)
    del vector

    retriever = Retriever(embedder, store, RetrievalSettings())
    outcome = retriever.retrieve(
        IncomingTicket(
            problem_description="Bracket weld seam cracking observed during incoming inspection",
            part_number="PN-2000",
        )
    )
    assert [c.sps_id for c in outcome.candidates] == ["OTHER"]
