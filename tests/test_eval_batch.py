"""The batch evaluator: discovery, per-case isolation, and the score columns.

The utility exists for threshold calibration, so the load-bearing tests here are
the ones asserting a score is reported for a case the gate *rejected*. A batch
that scored only its successes would hide the exact rows you need to decide
whether the threshold sits in the right place.

resolve() is replaced throughout: these tests are about the batch, not the
pipeline, which tests/test_resolver.py already covers.
"""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import scripts.run_eval_batch as batch  # noqa: E402
import scripts.run_resolver as resolver  # noqa: E402
from sps.file_reader import UnsupportedFileType  # noqa: E402
from tests.test_resolver import HEADERS, PART, PROBLEM, read_sheet, row  # noqa: E402


def write_ticket_csv(path, part=PART, problem=PROBLEM):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["SPS_ID", "Part_Number", "Issue_Type", "Problem_Description"])
        writer.writerow(["T-1", part, "Quality", problem])
    return path


def write_history_csv(path, rows=None):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADERS)
        writer.writerows(rows if rows is not None else [row("SPS-1")])
    return path


def write_run_list(path, rows, headers=("Test_ID", "Ticket_File", "History_File")):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def make_dir(tmp_path, ids, with_history=()):
    directory = tmp_path / "cases"
    directory.mkdir(exist_ok=True)
    for test_id in ids:
        write_ticket_csv(directory / f"{test_id}_ticket.csv")
    for test_id in with_history:
        write_history_csv(directory / f"{test_id}_history.csv")
    return directory


def outcome(code=resolver.CODE_SUCCESS_HISTORICAL, **overrides):
    fields = dict(
        reason="ok",
        exit_code=resolver.EXIT_OK,
        embedding_model="local:BAAI/bge-small-en-v1.5",
        top_score=0.95,
        threshold_used=0.89,
        candidates_considered=3,
    )
    fields.update(overrides)
    return resolver.ResolveOutcome(code, **fields)


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted resolve(); returns the list it records calls into.

    Outcomes are consumed in order and the last one repeats, so a batch of five
    identical cases needs one argument.
    """

    def install(*outcomes):
        queue = list(outcomes)
        calls = []

        def fake_resolve(args, output_dir):
            calls.append(args)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(resolver, "resolve", fake_resolve)
        return calls

    return install


# --------------------------------------------------------------- discovery


def test_a_ticket_pairs_with_its_own_history(tmp_path):
    directory = make_dir(tmp_path, ["case01"], ["case01"])
    (case,) = batch.discover_from_dir(directory, None)

    assert case.test_id == "case01"
    assert case.ticket.name == "case01_ticket.csv"
    assert case.history.name == "case01_history.csv"


def test_a_case_without_its_own_history_uses_the_shared_one(tmp_path):
    """Fifty tickets against one extract is the usual shape."""
    directory = make_dir(tmp_path, ["case01", "case02"], ["case02"])
    shared = write_history_csv(tmp_path / "master.csv")

    by_id = {c.test_id: c for c in batch.discover_from_dir(directory, shared)}

    assert by_id["case01"].history == shared
    # A case that ships its own history means to use it.
    assert by_id["case02"].history.name == "case02_history.csv"


def test_a_case_with_no_history_anywhere_is_kept_not_skipped(tmp_path):
    """A case that never ran is not the same as a case that failed, so it gets
    a row saying so rather than quietly disappearing from the count."""
    directory = make_dir(tmp_path, ["case01"])
    (case,) = batch.discover_from_dir(directory, None)
    assert case.history is None

    result = batch.run_case(case, tmp_path / "work", None)

    assert result.status_code == "INVALID_INPUT"
    assert "case01" in result.reason
    assert "--history-file" in result.reason


def test_unsupported_files_in_the_directory_are_ignored(tmp_path):
    directory = make_dir(tmp_path, ["case01"])
    for noise in ("notes.pdf", "README.md", "case02_ticket.txt", "thumbs.db"):
        (directory / noise).write_text("x", encoding="utf-8")

    assert [c.test_id for c in batch.discover_from_dir(directory, None)] == ["case01"]


def test_the_ticket_marker_is_matched_case_insensitively(tmp_path):
    directory = tmp_path / "cases"
    directory.mkdir()
    write_ticket_csv(directory / "Case01_Ticket.csv")

    assert [c.test_id for c in batch.discover_from_dir(directory, None)] == ["Case01"]


def test_cases_come_back_in_a_stable_order(tmp_path):
    """Sorted, so two runs of the same set produce diffable workbooks."""
    directory = make_dir(tmp_path, ["case03", "case01", "case02"])
    shared = write_history_csv(tmp_path / "master.csv")

    ids = [c.test_id for c in batch.discover_from_dir(directory, shared)]
    assert ids == ["case01", "case02", "case03"]


def test_a_history_with_no_ticket_is_ignored_but_said_out_loud(tmp_path, caplog):
    """Usually a typo in a filename, which is worth a warning: the case the
    author thought they wrote is not in the batch."""
    directory = make_dir(tmp_path, ["case01"], ["case01", "case07"])

    with caplog.at_level("WARNING"):
        cases = batch.discover_from_dir(directory, None)

    assert [c.test_id for c in cases] == ["case01"]
    assert "case07" in caplog.text


# -------------------------------------------------------------- the run list


def test_run_list_paths_resolve_against_the_list(tmp_path):
    """So a test set assembled on one laptop runs on another."""
    nested = tmp_path / "set"
    nested.mkdir()
    write_ticket_csv(nested / "a_ticket.csv")
    write_history_csv(nested / "a_history.csv")
    listing = write_run_list(nested / "runs.csv", [["A1", "a_ticket.csv", "a_history.csv"]])

    (case,) = batch.discover_from_list(listing, None)

    assert case.test_id == "A1"
    assert case.ticket == nested / "a_ticket.csv"
    assert case.history == nested / "a_history.csv"


def test_a_blank_history_cell_falls_back_to_the_shared_file(tmp_path):
    listing = write_run_list(tmp_path / "runs.csv", [["A1", "a_ticket.csv", ""]])
    shared = write_history_csv(tmp_path / "master.csv")

    (case,) = batch.discover_from_list(listing, shared)
    assert case.history == shared


def test_a_blank_history_cell_with_no_shared_file_is_no_history(tmp_path):
    listing = write_run_list(tmp_path / "runs.csv", [["A1", "a_ticket.csv", ""]])

    (case,) = batch.discover_from_list(listing, None)
    assert case.history is None


def test_the_run_list_must_name_its_columns(tmp_path):
    listing = write_run_list(tmp_path / "runs.csv", [["A1", "a.csv"]], headers=("id", "file"))

    with pytest.raises(UnsupportedFileType, match="test_id"):
        batch.discover_from_list(listing, None)


def test_run_list_headers_tolerate_spacing_and_case(tmp_path):
    listing = write_run_list(
        tmp_path / "runs.csv",
        [["A1", "a_ticket.csv", ""]],
        headers=("test id", "TICKET FILE", "History File"),
    )

    (case,) = batch.discover_from_list(listing, None)
    assert case.test_id == "A1"
    assert case.ticket.name == "a_ticket.csv"


def test_blank_rows_in_the_run_list_are_skipped(tmp_path):
    listing = write_run_list(
        tmp_path / "runs.csv", [["A1", "a_ticket.csv", ""], ["", "", ""]]
    )
    shared = write_history_csv(tmp_path / "master.csv")

    assert len(batch.discover_from_list(listing, shared)) == 1


# ------------------------------------------------------------------ one case


def test_a_rejected_case_still_reports_its_score(tmp_path, scripted):
    """The reason this utility exists. Without the score of a case the gate
    turned away, there is nothing to calibrate the gate against."""
    scripted(outcome(resolver.CODE_BELOW_THRESHOLD, reason="too low", top_score=0.6123))
    case = batch.Case("case04", tmp_path / "t.csv", tmp_path / "h.csv")

    result = batch.run_case(case, tmp_path / "work", None)

    assert result.status == "FAIL"
    assert result.status_code == resolver.CODE_BELOW_THRESHOLD
    assert result.confidence_score == 0.6123
    assert result.threshold_applied == 0.89
    assert result.cleared == "NO"
    assert result.candidates == 3


def test_a_success_is_marked_pass(tmp_path, scripted):
    scripted(outcome())
    case = batch.Case("c", tmp_path / "t.csv", tmp_path / "h.csv")

    result = batch.run_case(case, tmp_path / "work", None)

    assert result.status == "PASS"
    assert result.cleared == "YES"


def test_an_unencoded_run_leaves_the_score_blank_rather_than_zero(tmp_path, scripted):
    """Blank and 0.0 say different things: nothing was scored, versus scored at
    zero. A zero here would join the distribution and drag every percentile
    down."""
    scripted(resolver.ResolveOutcome(resolver.CODE_INVALID_INPUT, "bad part", resolver.EXIT_OK))
    case = batch.Case("c", tmp_path / "t.csv", tmp_path / "h.csv")

    result = batch.run_case(case, tmp_path / "work", None)

    assert result.confidence_score == ""
    assert result.threshold_applied == ""
    assert result.cleared == ""
    assert result.candidates == ""


def test_a_raising_case_becomes_a_row(tmp_path, scripted):
    scripted(RuntimeError("model exploded"))
    case = batch.Case("c", tmp_path / "t.csv", tmp_path / "h.csv")

    result = batch.run_case(case, tmp_path / "work", None)

    assert result.status_code == "INFRASTRUCTURE_ERROR"
    assert "model exploded" in result.reason


def test_the_threshold_override_reaches_the_resolver(tmp_path, scripted):
    calls = scripted(outcome())
    case = batch.Case("c", tmp_path / "t.csv", tmp_path / "h.csv")

    batch.run_case(case, tmp_path / "work", 0.42)

    assert calls[0].threshold == 0.42


def test_each_case_gets_its_own_output_directory(tmp_path, scripted):
    """Otherwise case 2 would overwrite case 1's status.xlsx, and a failure
    would be unattributable after the fact."""
    calls = scripted(outcome())
    work = tmp_path / "work"

    batch.run_case(batch.Case("case01", tmp_path / "t.csv", tmp_path / "h.csv"), work, None)
    batch.run_case(batch.Case("case02", tmp_path / "t.csv", tmp_path / "h.csv"), work, None)

    assert [c.output_dir for c in calls] == [str(work / "case01"), str(work / "case02")]


# ----------------------------------------------------------------- the batch


def _run(tmp_path, directory, shared, extra=()):
    out = tmp_path / "out"
    argv = ["--test-dir", str(directory), "--history-file", str(shared),
            "--output-dir", str(out), *extra]
    return batch.main(argv), out


def test_the_workbook_carries_every_column_in_order(tmp_path, scripted):
    directory = make_dir(tmp_path, ["case01", "case02"])
    shared = write_history_csv(tmp_path / "master.csv")
    scripted(outcome(), outcome(resolver.CODE_BELOW_THRESHOLD, top_score=0.5))

    code, out = _run(tmp_path, directory, shared)

    assert code == batch.EXIT_OK
    sheet = read_sheet(out / batch.RESULTS_FILE)
    assert list(sheet.columns) == list(batch.RESULT_COLUMNS)
    assert list(sheet["Test_ID"]) == ["case01", "case02"]
    assert list(sheet["Status"]) == ["PASS", "FAIL"]
    assert list(sheet["Cleared_Threshold"]) == ["YES", "NO"]
    assert [float(v) for v in sheet["Confidence_Score"]] == [0.95, 0.5]
    assert list(sheet["Ticket_File"]) == ["case01_ticket.csv", "case02_ticket.csv"]
    assert set(sheet["History_File"]) == {"master.csv"}


def test_a_failing_case_does_not_stop_the_ones_after_it(tmp_path, scripted):
    """Fifty cases and an hour of runtime: losing the batch to case three is
    not an acceptable way to find out case three is broken."""
    directory = make_dir(tmp_path, ["case01", "case02", "case03"])
    shared = write_history_csv(tmp_path / "master.csv")
    scripted(RuntimeError("boom"), outcome(), outcome())

    code, out = _run(tmp_path, directory, shared)

    sheet = read_sheet(out / batch.RESULTS_FILE)
    assert list(sheet["Status"]) == ["FAIL", "PASS", "PASS"]
    assert code == batch.EXIT_INFRASTRUCTURE


def test_a_gated_case_is_a_result_not_an_error(tmp_path, scripted):
    """Exit 0: the batch ran. Cases legitimately failing their gate is the
    finding, not a fault."""
    directory = make_dir(tmp_path, ["case01"])
    shared = write_history_csv(tmp_path / "master.csv")
    scripted(outcome(resolver.CODE_BELOW_THRESHOLD, top_score=0.4))

    code, _ = _run(tmp_path, directory, shared)
    assert code == batch.EXIT_OK


def test_an_empty_directory_is_a_bad_input(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()

    argv = ["--test-dir", str(directory), "--output-dir", str(tmp_path / "out")]
    assert batch.main(argv) == batch.EXIT_BAD_INPUT


def test_a_missing_directory_is_a_bad_input(tmp_path):
    argv = ["--test-dir", str(tmp_path / "absent"), "--output-dir", str(tmp_path / "out")]
    assert batch.main(argv) == batch.EXIT_BAD_INPUT


def test_a_shared_history_of_the_wrong_type_is_caught_before_any_case_runs(tmp_path, scripted):
    """Fifty cases would otherwise each fail the same way, fifty times over."""
    directory = make_dir(tmp_path, ["case01"])
    bad = tmp_path / "master.pdf"
    bad.write_text("x", encoding="utf-8")
    calls = scripted(outcome())

    code, _ = _run(tmp_path, directory, bad)

    assert code == batch.EXIT_BAD_INPUT
    assert calls == []


def test_a_run_list_drives_the_batch(tmp_path, scripted):
    write_ticket_csv(tmp_path / "a_ticket.csv")
    write_history_csv(tmp_path / "a_history.csv")
    listing = write_run_list(tmp_path / "runs.csv", [["A1", "a_ticket.csv", "a_history.csv"]])
    scripted(outcome())
    out = tmp_path / "out"

    code = batch.main(["--run-list", str(listing), "--output-dir", str(out)])

    assert code == batch.EXIT_OK
    assert list(read_sheet(out / batch.RESULTS_FILE)["Test_ID"]) == ["A1"]


def test_a_source_is_required(tmp_path):
    with pytest.raises(SystemExit):
        batch.main(["--output-dir", str(tmp_path)])


# ---------------------------------------------------------------- the digest


def test_the_digest_reports_the_spread(tmp_path):
    rows = [
        batch.Row(test_id=str(i), status_code="X", confidence_score=score)
        for i, score in enumerate([0.10, 0.50, 0.90, 0.95])
    ]

    text = batch.summarise(rows)

    assert "min 0.1000" in text
    assert "max 0.9500" in text
    assert "4 scored case(s)" in text


def test_the_digest_survives_a_batch_that_scored_nothing():
    text = batch.summarise([batch.Row(test_id="a", status_code="INVALID_INPUT")])

    assert "1 case(s)" in text
    assert "score distribution" not in text


def test_a_mixed_encoder_batch_is_called_out(tmp_path):
    """Azure for some cases and the local fallback for others pools two
    embedding spaces into one set of percentiles. Nothing else reports it:
    falling back is normal behaviour, not an error."""
    rows = [
        batch.Row(test_id="a", status_code="X", confidence_score=0.95,
                  embedding_model="local:BAAI/bge-small-en-v1.5"),
        batch.Row(test_id="b", status_code="X", confidence_score=0.42,
                  embedding_model="azure:text-embedding-3-large"),
    ]

    text = batch.summarise(rows)

    assert "WARNING" in text
    assert "azure:text-embedding-3-large" in text


def test_one_encoder_throughout_draws_no_warning():
    rows = [
        batch.Row(test_id=str(i), status_code="X", confidence_score=0.9,
                  embedding_model="azure:text-embedding-3-large")
        for i in range(3)
    ]

    assert "WARNING" not in batch.summarise(rows)


# ------------------------------------------------------------- configuration


def test_the_batch_loads_dotenv_like_the_resolver_does(tmp_path, scripted, monkeypatch):
    """The batch IS the Azure calibration run. Without .env it would fall back
    to the local encoder for every case and measure the wrong distribution --
    silently, since a fallback is not an error."""
    loaded = []
    monkeypatch.setattr(resolver, "_load_dotenv", lambda: loaded.append(True))
    directory = make_dir(tmp_path, ["case01"])
    shared = write_history_csv(tmp_path / "master.csv")
    scripted(outcome())

    _run(tmp_path, directory, shared)

    assert loaded == [True]
