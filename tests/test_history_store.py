"""Commodity families and the history store.

The properties that carry the risk: a family is searched in full, so the best
match is never dropped for being old; an unchanged export is never re-read; a
record is embedded once, and never compared across two embedding spaces; and
every reason a ticket falls back to its own part's history is said out loud.
"""

from __future__ import annotations

import csv
import logging

import pytest

import sps
from sps import (
    HistoryRetriever,
    IncomingTicket,
    MatchScope,
    MemoryHistoryStore,
    SqliteHistoryStore,
    commodity_code,
    injected_encoder,
    load_commodity_groups,
    match_scope,
)
from tests.conftest import TokenOverlapEmbedder
from tests.test_resolver import PART, PROBLEM, row, write_csv_history


def write_groups(path, rows, header=("Commodity_Code", "Group")):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


class CountingEmbedder(TokenOverlapEmbedder):
    """Counts the passages it is asked to encode, across calls."""

    @property
    def passages_encoded(self) -> int:
        return sum(len(call) for call in self.passage_calls)


def ticket(part=PART, problem=PROBLEM):
    return IncomingTicket(problem_description=problem, part_number=part)


@pytest.fixture(params=["sqlite", "memory"])
def store_kind(request, monkeypatch):
    """Every retriever test runs against both stores: ranking must not depend
    on which one answered."""
    monkeypatch.setenv("SPS_HISTORY_STORE", request.param)
    return request.param


# --------------------------------------------------------------- the code


@pytest.mark.parametrize("part, code", [
    ("0022-43951", "0022"),
    (" 0022-43951 ", "0022"),
    ("abc1-x-2", "ABC1"),
    ("0022", ""),
    ("ABC123", ""),
    ("-43951", ""),
    ("", ""),
])
def test_the_commodity_code_is_the_part_number_up_to_its_first_hyphen(part, code):
    assert commodity_code(part) == code


# ------------------------------------------------------------ groups file


def test_groups_are_symmetric_and_ignore_case_in_the_name(tmp_path):
    groups = load_commodity_groups(write_groups(tmp_path / "g.csv", [
        ("0011", "FASTENERS"), ("0022", "Fasteners"), ("0034", "PACKAGING"),
    ]))
    assert groups.family("0011") == ("FASTENERS", ("0011", "0022"))
    assert groups.family("0022") == ("FASTENERS", ("0011", "0022"))
    assert groups.family("0034") == ("PACKAGING", ("0034",))
    assert groups.family("0099") is None


def test_a_code_in_two_groups_is_reported_not_resolved(tmp_path, caplog):
    path = write_groups(tmp_path / "g.csv", [
        ("0011", "FASTENERS"), ("0022", "FASTENERS"), ("0022", "BRACKETS"),
    ])
    with caplog.at_level(logging.ERROR):
        groups = load_commodity_groups(path)

    assert groups.conflicts == {"0022": ("BRACKETS", "FASTENERS")}
    assert groups.family("0022") is None
    # The code's partner keeps its family, minus the conflicted code.
    assert groups.family("0011") == ("FASTENERS", ("0011",))
    assert "0022" in caplog.text


def test_an_xlsx_groups_file_warns_when_a_code_was_stored_as_a_number(tmp_path, caplog):
    from openpyxl import Workbook

    book = Workbook()
    book.active.append(["Commodity_Code", "Group"])
    book.active.append([11, "FASTENERS"])      # typed 0011 into a General cell
    book.active.append(["0022", "FASTENERS"])
    book.save(tmp_path / "g.xlsx")

    with caplog.at_level(logging.WARNING):
        groups = load_commodity_groups(tmp_path / "g.xlsx")
    assert "leading zeros" in caplog.text
    assert groups.family("0022") == ("FASTENERS", ("0022", "11"))


def test_a_groups_file_without_its_columns_is_an_error(tmp_path):
    with pytest.raises(sps.CommodityGroupsError, match="Group"):
        load_commodity_groups(write_groups(tmp_path / "g.csv", [("0011",)], header=("Commodity_Code",)))


# --------------------------------------------------------------- the scope


@pytest.mark.parametrize("setup, part, expected", [
    ("none", PART, "no commodity groups file is configured"),
    ("file", "ABC123", "ABC123 has no hyphen, so no commodity code"),
    ("file", "0099-11111", "commodity code 0099 is not in g.csv"),
    ("conflict", PART, "commodity code 0012 is listed under BRACKETS and FASTENERS"),
    ("missing", PART, "the commodity groups file could not be used"),
])
def test_every_fallback_says_why(tmp_path, setup, part, expected):
    rows = [("0012", "FASTENERS"), ("0022", "FASTENERS")]
    if setup == "conflict":
        rows.append(("0012", "BRACKETS"))
    path = None if setup == "none" else write_groups(tmp_path / "g.csv", rows)
    if setup == "missing":
        path = tmp_path / "absent.csv"

    scope = match_scope(part, path)

    assert not scope.is_family
    assert scope.note.startswith("Matched on the part number only:")
    assert expected in scope.note


def test_a_mapped_code_scopes_the_whole_family(tmp_path):
    path = write_groups(tmp_path / "g.csv", [("0012", "FASTENERS"), ("0022", "FASTENERS")])
    scope = match_scope(PART, path)
    assert scope.codes == ("0012", "0022")
    assert scope.note == "Matched across commodity family FASTENERS (0012, 0022)."


# --------------------------------------------------------------- the store


def _retriever(history, store=None, groups=None, threshold=0.0, embedder=None, **kw):
    return HistoryRetriever(
        history_path=history,
        embedder=embedder or TokenOverlapEmbedder(),
        confidence_threshold=threshold,
        commodity_file=groups,
        store=store,
        **kw,
    )


def test_a_family_is_searched_across_parts(tmp_path, store_kind):
    history = write_csv_history(tmp_path / "h.csv", [
        row("OWN", problem="Paint flaking on the enclosure door after shipment"),
        row("SIBLING", part="0022-77777"),
        row("OUTSIDER", part="0034-11020", minutes=1),
    ])
    groups = write_groups(tmp_path / "g.csv", [("0012", "FASTENERS"), ("0022", "FASTENERS")])

    retriever = _retriever(history, groups=groups)
    found = retriever.retrieve(ticket())

    assert [c.sps_id for c in found] == ["SIBLING", "OWN"]
    assert found[0].exact_part is False and found[1].exact_part is True
    assert retriever.stats.family == "FASTENERS"
    assert retriever.stats.part_matches == 2


def test_without_groups_only_the_part_is_searched(tmp_path, store_kind):
    history = write_csv_history(tmp_path / "h.csv", [row("OWN"), row("SIBLING", part="0022-77777")])
    found = _retriever(history).retrieve(ticket())
    assert [c.sps_id for c in found] == ["OWN"]


def test_at_an_equal_score_the_own_part_ranks_first(tmp_path, store_kind):
    history = write_csv_history(tmp_path / "h.csv", [
        row("SPS-9", part="0022-77777"), row("SPS-1"),  # identical text, so identical cosine
    ])
    groups = write_groups(tmp_path / "g.csv", [("0012", "F"), ("0022", "F")])
    found = _retriever(history, groups=groups).retrieve(ticket())
    assert [c.sps_id for c in found] == ["SPS-1", "SPS-9"]


def test_one_entry_per_sps_id_across_a_family(tmp_path, store_kind):
    """The scorer's answers are matched back by SPS ID, so the same record
    under two part numbers must reach it once."""
    history = write_csv_history(tmp_path / "h.csv", [
        row("SPS-1", part="0022-77777"), row("SPS-1"), row("SPS-2", minutes=1),
    ])
    groups = write_groups(tmp_path / "g.csv", [("0012", "F"), ("0022", "F")])
    found = _retriever(history, groups=groups).retrieve(ticket())
    ids = [c.sps_id for c in found]
    assert ids.count("SPS-1") == 1
    assert found[ids.index("SPS-1")].exact_part is True


def test_the_oldest_record_in_a_30k_family_can_win(tmp_path):
    """The invisible regression the cap would have caused: at family scale,
    keeping the newest 300 drops the oldest record even when it is the match."""
    filler = "Outer carton label misprinted on shipment packaging lot"
    rows = [row("SPS-OLDEST", minutes=0)]
    rows += [row(f"SPS-{i}", part=f"0022-{i:05d}", problem=f"{filler} {i}", minutes=i)
             for i in range(1, 30_000)]
    history = write_csv_history(tmp_path / "h.csv", rows)
    groups = write_groups(tmp_path / "g.csv", [("0012", "F"), ("0022", "F")])

    retriever = _retriever(history, groups=groups, threshold=0.5)
    found = retriever.retrieve(ticket())

    assert found[0].sps_id == "SPS-OLDEST"
    assert retriever.stats.considered == 30_000


def test_an_unchanged_export_is_not_read_again(tmp_path, monkeypatch):
    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1"), row("SPS-2", minutes=1)])
    embedder = CountingEmbedder()
    _retriever(history, embedder=embedder).retrieve(ticket())
    encoded = embedder.passages_encoded

    reads = []
    real = sps.read_history
    monkeypatch.setattr(sps, "read_history", lambda p: reads.append(p) or real(p))
    again = _retriever(history, embedder=embedder)
    again.retrieve(ticket())

    assert reads == []
    assert embedder.passages_encoded == encoded
    assert again.stats.embedded_now == 0


def test_appending_rows_embeds_only_the_new_ones(tmp_path):
    base = [row(f"SPS-{i}", problem=f"{PROBLEM} variant {i}", minutes=i) for i in range(50)]
    history = write_csv_history(tmp_path / "h.csv", base)
    embedder = CountingEmbedder()
    store = SqliteHistoryStore(tmp_path / "s.sqlite")
    _retriever(history, store=store, embedder=embedder).retrieve(ticket())
    assert embedder.passages_encoded == 50

    added = [row(f"NEW-{i}", problem=f"{PROBLEM} fresh {i}", minutes=100 + i) for i in range(10)]
    history = write_csv_history(tmp_path / "h.csv", base + added)
    incremental = _retriever(history, store=store, embedder=embedder)
    incremental.retrieve(ticket())

    assert embedder.passages_encoded == 60       # only the 10 new texts
    fresh = _retriever(history, store=SqliteHistoryStore(tmp_path / "fresh.sqlite"))
    fresh.retrieve(ticket())
    assert [c.sps_id for c in incremental.stats.top_candidates] == \
        [c.sps_id for c in fresh.stats.top_candidates]
    assert incremental.stats.considered == fresh.stats.considered == 60


def test_a_row_deleted_from_the_export_leaves_the_store(tmp_path):
    store = SqliteHistoryStore(tmp_path / "s.sqlite")
    history = write_csv_history(tmp_path / "h.csv", [row("KEEP"), row("GONE", minutes=1)])
    store.sync(history)
    write_csv_history(tmp_path / "h.csv", [row("KEEP")])

    store.sync(history)

    assert [r.sps_id for r in store.all_rows()] == ["KEEP"]


def test_vectors_from_another_deployment_are_never_used(tmp_path):
    store = SqliteHistoryStore(tmp_path / "s.sqlite")
    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])
    store.sync(history)
    first = CountingEmbedder()
    store.fetch(MatchScope(PART), injected_encoder(first), PROBLEM)

    class OtherDeployment(CountingEmbedder):
        pass

    second = OtherDeployment()
    store.fetch(MatchScope(PART), injected_encoder(second), PROBLEM)
    assert second.passages_encoded == 1   # re-embedded, not borrowed


def test_a_store_written_under_another_schema_is_rebuilt(tmp_path):
    location = tmp_path / "s.sqlite"
    store = SqliteHistoryStore(location)
    store.sync(write_csv_history(tmp_path / "h.csv", [row("SPS-1")]))
    store._db.execute("UPDATE meta SET value = '0' WHERE key = 'schema'")
    store.close()

    reopened = SqliteHistoryStore(location)
    assert reopened.rows_count() == 0               # rebuilt, so the next sync re-reads
    assert reopened.sync(tmp_path / "h.csv").reread


def test_the_store_lives_beside_the_history_unless_told_otherwise(tmp_path, monkeypatch):
    monkeypatch.delenv("SPS_HISTORY_STORE_DIR")
    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])
    assert sps.history_store_path(history).parent == tmp_path.resolve()


def test_a_store_on_a_network_share_moves_to_local_disk(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sps, "_is_network_path", lambda path: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    with caplog.at_level(logging.WARNING):
        location = sps.history_store_path(tmp_path / "h.csv")
    assert location.parent == tmp_path / "local" / "SPS" / "history-store"
    assert "network share" in caplog.text


def test_an_unusable_store_falls_back_to_memory_and_still_answers(tmp_path, monkeypatch, caplog):
    def refuse(location):
        raise PermissionError("read-only folder")

    monkeypatch.setattr(sps, "SqliteHistoryStore", refuse)
    with caplog.at_level(logging.WARNING):
        store = sps.open_history_store(tmp_path / "h.csv")
    assert isinstance(store, MemoryHistoryStore)
    assert "read-only folder" in caplog.text

    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])
    assert [c.sps_id for c in _retriever(history, store=store).retrieve(ticket())] == ["SPS-1"]


def test_an_unknown_store_setting_is_a_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SPS_HISTORY_STORE", "qdrant")
    with pytest.raises(sps.HistoryStoreConfigError, match="sqlite"):
        sps.open_history_store(tmp_path / "h.csv")


def test_an_unknown_part_needs_no_embedding_at_all(tmp_path, monkeypatch):
    """Rows are stored on sync; vectors only when a ticket needs them. So a
    part with no history still ends NO_MATCHES with Azure unconfigured, as it
    did before the store existed."""
    for name in ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY", "AZURE_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])
    retriever = HistoryRetriever(history_path=history)
    assert retriever.retrieve(ticket(part="0099-99999")) == []
    assert retriever.stats.part_matches == 0


# ---------------------------------------------------------------- embedding


def test_a_long_batch_is_sent_in_chunks_and_comes_back_in_order():
    from tests.test_azure_embeddings import CONFIGURED

    calls = []

    class Client:
        class embeddings:
            @staticmethod
            def create(*, model, input):
                calls.append(len(input))

                class Item:
                    def __init__(self, index, value):
                        self.index, self.embedding = index, [value, 1.0]

                class Response:
                    data = [Item(i, float(text)) for i, text in enumerate(input)]
                return Response()

    texts = [str(i) for i in range(5000)]
    vectors = sps.AzureEmbedder(CONFIGURED, client=Client()).embed_batch(texts)

    assert calls == [2048, 2048, 904]
    ratios = [v[0] / v[1] for v in vectors]
    assert ratios == pytest.approx([float(i) for i in range(5000)])


# ------------------------------------------------------------------ ingest


def test_ingest_embeds_everything_once_and_then_nothing(tmp_path, azure_embeddings, capsys):
    history = write_csv_history(tmp_path / "h.csv", [
        row(f"SPS-{i}", problem=f"{PROBLEM} variant {i}", minutes=i) for i in range(5)
    ])

    assert sps.main(["ingest", "--history", str(history)]) == sps.EXIT_OK
    assert "embedded: 5 new problem text(s); 0 already stored" in capsys.readouterr().out

    assert sps.main(["ingest", "--history", str(history)]) == sps.EXIT_OK
    out = capsys.readouterr().out
    assert "(file unchanged)" in out
    assert "embedded: 0 new problem text(s); 5 already stored" in out


@pytest.mark.parametrize("problem", ["missing file", "not configured", "memory store"])
def test_ingest_refuses_what_a_person_must_fix(tmp_path, azure_embeddings, monkeypatch, problem):
    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])
    if problem == "missing file":
        history = tmp_path / "absent.csv"
    elif problem == "not configured":
        monkeypatch.delenv("AZURE_EMBEDDING_API_KEY")
    else:
        monkeypatch.setenv("SPS_HISTORY_STORE", "memory")
    assert sps.main(["ingest", "--history", str(history)]) == sps.EXIT_BAD_INPUT


def test_both_stores_implement_the_seam(tmp_path):
    """A server-backed store would implement the same two methods; the
    retriever ranks identically whichever answers."""
    assert isinstance(MemoryHistoryStore(), sps.HistoryStore)
    assert isinstance(SqliteHistoryStore(tmp_path / "s.sqlite"), sps.HistoryStore)


def test_the_sample_groups_file_makes_a_family_of_0012_and_0013():
    """samples/commodity_groups.csv is what the docs walk through: a ticket on
    0013-11111, a part with no history of its own, answered from 0012's."""
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "samples" / "commodity_groups.csv"
    scope = match_scope("0013-11111", sample)
    assert (scope.group, scope.codes) == ("BRACKETS", ("0012", "0013"))


def test_an_unwritable_store_folder_falls_back_to_the_local_one(tmp_path, monkeypatch, caplog):
    """Before memory: a memory store embeds a whole family on every ticket."""
    real = sps.SqliteHistoryStore
    blocked = tmp_path / "readonly"

    def store(location):
        if blocked in location.parents:
            raise PermissionError("read-only folder")
        return real(location)

    monkeypatch.setattr(sps, "SqliteHistoryStore", store)
    monkeypatch.setenv("SPS_HISTORY_STORE_DIR", str(blocked))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    with caplog.at_level(logging.WARNING):
        opened = sps.open_history_store(tmp_path / "h.csv")
    assert isinstance(opened, real)
    assert (tmp_path / "local" / "SPS" / "history-store") in __import__("pathlib").Path(opened.location).parents
    assert "using" in caplog.text and "read-only folder" in caplog.text


def test_a_ticket_never_waits_for_another_process_writing_the_store(tmp_path):
    """WAL lets readers through a writer, but only if opening the store is
    itself read-only. It was not: every ticket waited out the writer."""
    import subprocess
    import sys
    import time

    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])
    location = tmp_path / "s.sqlite"
    first = SqliteHistoryStore(location)
    HistoryRetriever(history_path=history, embedder=TokenOverlapEmbedder(), store=first,
                     confidence_threshold=0.0).retrieve(ticket())
    first.close()

    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sqlite3, sys, time\n"
         "db = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
         "db.execute('BEGIN IMMEDIATE'); print('locked', flush=True); time.sleep(30)",
         str(location)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        started = time.perf_counter()
        store = SqliteHistoryStore(location)
        found = HistoryRetriever(history_path=history, embedder=TokenOverlapEmbedder(),
                                 store=store, confidence_threshold=0.0).retrieve(ticket())
        elapsed = time.perf_counter() - started
        store.close()
    finally:
        holder.kill()
        holder.wait()
    assert [c.sps_id for c in found] == ["SPS-1"]
    assert elapsed < 2.0
