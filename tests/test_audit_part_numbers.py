"""The part_number drift audit and repair utility.

Exercised against a real Qdrant engine in local mode, because the whole point of
the tool is that `set_payload` rewrites one field without disturbing the vector
or the rest of the payload -- a property a mock could not demonstrate.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("qdrant_client")

import scripts.audit_part_numbers as audit  # noqa: E402
from sps.config import VectorStoreSettings  # noqa: E402
from sps.contracts import VectorPoint  # noqa: E402
from sps.vectorstore.qdrant_store import QdrantVectorStore  # noqa: E402
from tests.conftest import TokenOverlapEmbedder  # noqa: E402

COLLECTION = "sps"


@pytest.fixture
def embedder():
    return TokenOverlapEmbedder()


def seed(path, embedder, spec):
    """spec: [(part_number, how_many), ...]"""
    store = QdrantVectorStore(
        settings=VectorStoreSettings(collection=COLLECTION, path=str(path))
    )
    store.ensure_collection(embedder.dimension)
    points, n = [], 0
    for value, count in spec:
        for _ in range(count):
            n += 1
            points.append(
                VectorPoint(
                    sps_id=f"S{n:05d}",
                    vector=embedder.embed_passages([f"defect {n} on the housing"])[0],
                    payload={
                        "sps_id": f"S{n:05d}",
                        "content_hash": f"{n:064d}",
                        "actual_solution": "Rework it and re-inspect.",
                        "part_number": value,
                        "part_description": "Mounting bracket",
                        "item_status": "Active",
                        "problem_reason_code": "RC-WELD",
                        "issue_type": "Quality",
                    },
                )
            )
    store.upsert(points)
    return store


DIRTY = [("pn-1000", 4), ("PN-1000 ", 2), ("Pn-2000", 1), ("PN-1000", 3), ("", 2)]


# ------------------------------------------------------------------- scanning


def test_scan_finds_every_drift_pattern(tmp_path, embedder):
    store = seed(tmp_path / "q", embedder, DIRTY)
    found = audit.scan(store.client, COLLECTION)

    assert found.total == 12
    assert found.blank == 2
    assert {k: len(v) for k, v in found.drift.items()} == {
        "pn-1000": 4,
        "PN-1000 ": 2,
        "Pn-2000": 1,
    }
    store.close()


def test_already_canonical_values_are_not_flagged(tmp_path, embedder):
    store = seed(tmp_path / "q", embedder, [("PN-1000", 5), ("PN-2000", 3)])
    found = audit.scan(store.client, COLLECTION)

    assert (found.total, found.blank, found.drift, found.affected) == (8, 0, {}, 0)
    store.close()


def test_blank_part_numbers_are_counted_but_never_rewritten(tmp_path, embedder):
    """Blank is legitimately 'no part number', not a misspelling of one --
    rewriting it would invent data."""
    store = seed(tmp_path / "q", embedder, [("", 4), ("   ", 2), ("PN-1000", 1)])
    found = audit.scan(store.client, COLLECTION)

    assert found.total == 7
    assert found.blank == 6
    assert found.drift == {}
    store.close()


def test_scan_pages_through_more_than_one_scroll_page(tmp_path, embedder, monkeypatch):
    monkeypatch.setattr(audit, "SCROLL_PAGE", 5)
    store = seed(tmp_path / "q", embedder, [("pn-1", 13)])
    found = audit.scan(store.client, COLLECTION)

    assert found.total == 13
    assert len(found.drift["pn-1"]) == 13
    store.close()


# ------------------------------------------------------------------- repairing


def test_repair_makes_every_value_canonical(tmp_path, embedder):
    store = seed(tmp_path / "q", embedder, DIRTY)
    found = audit.scan(store.client, COLLECTION)

    fixed = audit.repair(store.client, COLLECTION, found.drift)
    assert fixed == 7

    after = audit.scan(store.client, COLLECTION)
    assert after.drift == {}
    assert (after.total, after.blank) == (12, 2)
    store.close()


def test_repair_leaves_the_rest_of_the_payload_untouched(tmp_path, embedder):
    store = seed(tmp_path / "q", embedder, [("pn-1000", 3)])
    audit.repair(store.client, COLLECTION, audit.scan(store.client, COLLECTION).drift)

    hit = store.search(embedder.embed_query("defect 1 on the housing"), limit=1)[0]
    assert hit.payload["part_number"] == "PN-1000"
    assert set(hit.payload) == {
        "sps_id", "content_hash", "actual_solution", "part_number",
        "part_description", "item_status", "problem_reason_code", "issue_type",
    }
    assert hit.payload["actual_solution"] == "Rework it and re-inspect."
    assert hit.payload["problem_reason_code"] == "RC-WELD"
    store.close()


def test_repair_does_not_touch_vectors_or_point_count(tmp_path, embedder):
    """The saving that justifies the tool: no re-embedding."""
    store = seed(tmp_path / "q", embedder, [("pn-1000", 4)])
    vector = embedder.embed_query("defect 1 on the housing")
    before = [h.cosine_similarity for h in store.search(vector, limit=10)]

    audit.repair(store.client, COLLECTION, audit.scan(store.client, COLLECTION).drift)

    after = [h.cosine_similarity for h in store.search(vector, limit=10)]
    assert store.count() == 4
    assert before == after
    store.close()


def test_repair_restores_reachability(tmp_path, embedder):
    """The symptom the tool exists to cure: history invisible to its own part."""
    store = seed(tmp_path / "q", embedder, [("pn-1000", 6), ("PN-1000", 2)])
    vector = embedder.embed_query("defect 1 on the housing")

    assert len(store.search(vector, limit=50, part_number="PN-1000")) == 2

    audit.repair(store.client, COLLECTION, audit.scan(store.client, COLLECTION).drift)

    assert len(store.search(vector, limit=50, part_number="PN-1000")) == 8
    store.close()


def test_repair_batches_by_target_value(tmp_path, embedder, monkeypatch):
    """One call per distinct value (chunked), not one per point."""
    calls = []
    store = seed(tmp_path / "q", embedder, [("pn-1000", 7), ("pn-2000", 2)])
    real = store.client.set_payload

    def counting(**kwargs):
        calls.append(len(kwargs["points"]))
        return real(**kwargs)

    monkeypatch.setattr(store.client, "set_payload", counting)
    audit.repair(store.client, COLLECTION, audit.scan(store.client, COLLECTION).drift)

    assert sorted(calls) == [2, 7]  # two calls, nine points
    store.close()


def test_large_groups_are_chunked(tmp_path, embedder, monkeypatch):
    monkeypatch.setattr(audit, "WRITE_CHUNK", 3)
    calls = []
    store = seed(tmp_path / "q", embedder, [("pn-1000", 7)])
    real = store.client.set_payload

    def counting(**kwargs):
        calls.append(len(kwargs["points"]))
        return real(**kwargs)

    monkeypatch.setattr(store.client, "set_payload", counting)
    audit.repair(store.client, COLLECTION, audit.scan(store.client, COLLECTION).drift)

    assert calls == [3, 3, 1]
    store.close()


# ------------------------------------------------------------------------ CLI


@pytest.fixture
def env(monkeypatch, tmp_path):
    def _env(path):
        monkeypatch.setenv("SPS_QDRANT_PATH", str(path))
        monkeypatch.setenv("SPS_COLLECTION", COLLECTION)
        monkeypatch.delenv("SPS_QDRANT_URL", raising=False)

    return _env


def report_from(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip())


def test_audit_is_read_only_without_fix(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, DIRTY).close()
    env(path)

    assert audit.main([]) == audit.EXIT_OK
    report = report_from(capsys)
    assert report["non_canonical_points"] == 7
    assert report["fixed"] == 0
    assert report["remaining"] == 7

    # Nothing was written: a second audit sees exactly the same drift.
    assert audit.main([]) == audit.EXIT_OK
    assert report_from(capsys)["non_canonical_points"] == 7


def test_fix_repairs_and_reverifies(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, DIRTY).close()
    env(path)

    assert audit.main(["--fix"]) == audit.EXIT_OK
    report = report_from(capsys)
    assert report["fixed"] == 7
    assert report["remaining"] == 0

    assert audit.main([]) == audit.EXIT_OK
    assert report_from(capsys)["non_canonical_points"] == 0


def test_report_includes_the_blank_rate(tmp_path, embedder, env, capsys):
    """Blank history is unreachable to any ticket that supplies a part number,
    so the rate is worth knowing before an evaluation run."""
    path = tmp_path / "q"
    seed(path, embedder, [("PN-1000", 6), ("", 2)]).close()
    env(path)

    audit.main([])
    report = report_from(capsys)
    assert report["blank_part_number"] == 2
    assert report["blank_pct"] == 25.0


def test_report_lists_drift_patterns_most_common_first(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, [("pn-a", 1), ("pn-b", 5), ("pn-c", 3)]).close()
    env(path)

    audit.main([])
    examples = report_from(capsys)["examples"]
    assert [e["stored"] for e in examples] == ["pn-b", "pn-c", "pn-a"]
    assert examples[0]["canonical"] == "PN-B"


def test_samples_limits_the_listed_patterns(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, [(f"pn-{i}", 1) for i in range(8)]).close()
    env(path)

    audit.main(["--samples", "3"])
    report = report_from(capsys)
    assert len(report["examples"]) == 3
    assert report["distinct_drift_values"] == 8  # the count is not truncated


def test_clean_index_reports_nothing_to_do(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, [("PN-1000", 4)]).close()
    env(path)

    assert audit.main([]) == audit.EXIT_OK
    report = report_from(capsys)
    assert report["non_canonical_points"] == 0
    assert report["distinct_drift_values"] == 0


def test_missing_collection_is_a_config_error(tmp_path, env, capsys):
    env(tmp_path / "empty")
    assert audit.main([]) == audit.EXIT_CONFIG


def test_stdout_is_pure_json(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, DIRTY).close()
    env(path)

    audit.main(["--verbose"])
    json.loads(capsys.readouterr().out.strip())  # logs go to stderr, so this parses


def test_purely_numeric_part_numbers_are_counted(tmp_path, embedder):
    """A purely numeric part number is where Excel destroys leading zeros in the
    sheet itself. Nothing downstream can recover them, so the audit surfaces the
    count rather than pretending the assumption always holds."""
    store = seed(tmp_path / "q", embedder, [("0012-43951", 3), ("001243951", 2), ("PN-1", 1)])
    found = audit.scan(store.client, COLLECTION)

    assert found.purely_numeric == 2      # only the digits-only value
    assert found.affected == 0            # neither is drifted
    store.close()


def test_the_house_format_is_already_canonical(tmp_path, embedder, env, capsys):
    """Alphanumeric part numbers with no stray spaces: both canonicalisation
    steps are no-ops, so the audit confirms rather than repairs."""
    path = tmp_path / "q"
    seed(path, embedder, [("0012-43951", 5), ("0034-11020", 3)]).close()
    env(path)

    assert audit.main([]) == audit.EXIT_OK
    report = report_from(capsys)
    assert report["non_canonical_points"] == 0
    assert report["purely_numeric_part_numbers"] == 0
    assert report["blank_part_number"] == 0


def test_report_carries_the_numeric_count(tmp_path, embedder, env, capsys):
    path = tmp_path / "q"
    seed(path, embedder, [("001243951", 4), ("0012-43951", 1)]).close()
    env(path)

    audit.main([])
    assert report_from(capsys)["purely_numeric_part_numbers"] == 4
