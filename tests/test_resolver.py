"""The in-memory resolver: validation, part filtering, capping, dual workbooks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")
pytest.importorskip("numpy")

import scripts.run_resolver as resolver  # noqa: E402
from sps.contracts import IncomingTicket  # noqa: E402
from sps.retrieval.in_memory import (  # noqa: E402
    HistoryError,
    InMemoryRetriever,
    build_header_map,
    cap_to_newest,
    load_matching_history,
)
from sps.validators import normalize_part_number, validate_ticket  # noqa: E402
from tests.conftest import TokenOverlapEmbedder  # noqa: E402

PROBLEM = "Bracket weld seam cracking observed during incoming inspection"
PART = "0012-43951"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

HEADERS = ["SPS_ID", "Part_Number", "Issue_Type", "Problem_Description",
           "Solution_Text", "Last_Modified_Date"]


def row(sps_id, part=PART, problem=PROBLEM, solution=None, minutes=0):
    return [
        sps_id, part, "Quality", problem,
        solution or f"Rework the weld seam and re-inspect bracket {sps_id}.",
        (BASE + timedelta(minutes=minutes)).isoformat(),
    ]


def write_history(path, rows, headers=HEADERS):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(headers)
    for r in rows:
        sheet.append(r)
    book.save(path)
    return path


def write_csv_history(path, rows, headers=HEADERS):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
    return path


def write_ticket(path, part=PART, problem=PROBLEM):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(["SPS_ID", "Part_Number", "Issue_Type", "Problem_Description"])
    sheet.append(["T-1", part, "Quality", problem])
    book.save(path)
    return path


def read_sheet(path):
    import pandas as pd

    return pd.read_excel(path, dtype=str).fillna("")


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "part,expected",
    [("", "INVALID_INPUT"), ("   ", "INVALID_INPUT"), ("---", "INVALID_INPUT"),
     ("​", "INVALID_INPUT"), (None, "INVALID_INPUT"), (PART, None)],
)
def test_part_number_gatekeeping(part, expected):
    error = validate_ticket(part, PROBLEM)
    assert (error.code if error else None) == expected


def test_invisible_characters_are_stripped_not_rejected():
    """A zero-width space pasted from a web form must not turn a valid part
    number into a total miss."""
    for junk in ("​", "﻿", "­", "⁠"):
        assert normalize_part_number(f"{junk}0012-43951{junk}") == PART


def test_non_breaking_and_interior_spaces_are_removed():
    assert normalize_part_number("0012 -43951") == PART
    assert normalize_part_number("0012 - 43951") == PART


def test_structural_delimiters_stay_distinct():
    """0012-43951 and 0012/43951 are different parts, not spellings of one."""
    variants = {normalize_part_number(v) for v in
                ("0012-43951", "0012/43951", "0012_43951", "0012.43951")}
    assert len(variants) == 4


def test_validation_runs_before_anything_expensive(tmp_path, monkeypatch):
    """A malformed ticket must not load a model or scan the history."""
    called = []
    monkeypatch.setattr(
        resolver, "read_ticket", lambda p: {"Part_Number": "", "Problem_Description": PROBLEM}
    )
    import sps.retrieval.in_memory as engine

    monkeypatch.setattr(engine, "load_matching_history",
                        lambda *a, **k: called.append("scanned") or [])

    write_history(tmp_path / "h.xlsx", [row("S1")])
    args = resolver.parse_args([
        "--ticket-file", str(write_ticket(tmp_path / "t.xlsx")),
        "--history-file", str(tmp_path / "h.xlsx"),
        "--output-dir", str(tmp_path / "out"),
    ])
    code, _, exit_code = resolver.resolve(args, tmp_path / "out")

    assert code == resolver.CODE_INVALID_INPUT
    assert exit_code == resolver.EXIT_OK
    assert called == []


# ------------------------------------------------------------ history loading


def test_only_rows_for_the_requested_part_are_kept(tmp_path):
    path = write_history(tmp_path / "h.xlsx", [
        row("SAME-1"), row("SAME-2", minutes=1),
        row("OTHER", part="0034-11020"),
    ])
    kept = load_matching_history(path, PART)
    assert [r.sps_id for r in kept] == ["SAME-1", "SAME-2"]


def test_the_dual_check_canonicalises_the_stored_value(tmp_path):
    """Drift in the file must not hide a genuine match."""
    path = write_history(tmp_path / "h.xlsx", [
        row("A", part=" 0012-43951 "), row("B", part="0012​-43951", minutes=1),
    ])
    assert len(load_matching_history(path, "0012-43951")) == 2


def test_short_text_rows_are_dropped(tmp_path):
    path = write_history(tmp_path / "h.xlsx", [
        row("GOOD"), row("SHORT", solution="TBD", minutes=1), row("NOPROB", problem="x", minutes=2),
    ])
    assert [r.sps_id for r in load_matching_history(path, PART)] == ["GOOD"]


def test_csv_and_xlsx_history_agree(tmp_path):
    rows = [row("A"), row("B", minutes=1), row("C", part="0034-11020")]
    from_xlsx = load_matching_history(write_history(tmp_path / "h.xlsx", rows), PART)
    from_csv = load_matching_history(write_csv_history(tmp_path / "h.csv", rows), PART)
    assert [r.sps_id for r in from_xlsx] == [r.sps_id for r in from_csv] == ["A", "B"]


def test_missing_required_column_is_reported(tmp_path):
    path = write_history(
        tmp_path / "h.xlsx",
        [["S1", PART, "Quality"]],
        headers=["SPS_ID", "Part_Number", "Issue_Type"],
    )
    with pytest.raises(HistoryError, match="Problem_Description"):
        load_matching_history(path, PART)


def test_missing_history_file_is_reported(tmp_path):
    with pytest.raises(HistoryError, match="not found"):
        load_matching_history(tmp_path / "nope.csv", PART)


def test_header_map_accepts_either_solution_spelling():
    for header in ("Solution_Text", "Actual_Solution"):
        mapping = build_header_map(["SPS_ID", "Part_Number", "Problem_Description", header])
        assert "actual_solution" in mapping


# --------------------------------------------------------------------- cap


def test_cap_keeps_the_most_recent(tmp_path):
    rows = load_matching_history(
        write_history(tmp_path / "h.xlsx", [row(f"S{i}", minutes=i) for i in range(10)]),
        PART,
    )
    capped = cap_to_newest(rows, limit=3)
    assert [r.sps_id for r in capped] == ["S9", "S8", "S7"]


def test_cap_is_a_no_op_below_the_limit(tmp_path):
    rows = load_matching_history(
        write_history(tmp_path / "h.xlsx", [row(f"S{i}", minutes=i) for i in range(4)]), PART
    )
    assert cap_to_newest(rows, limit=300) is rows


# ----------------------------------------------------------------- ranking


def _retriever(tmp_path, rows, threshold=0.5):
    return InMemoryRetriever(
        embedder=TokenOverlapEmbedder(),
        history_path=write_history(tmp_path / "h.xlsx", rows),
        confidence_threshold=threshold,
    )


def test_ranking_is_cosine_only(tmp_path):
    """No metadata boosting: the part is already an exact filter."""
    r = _retriever(tmp_path, [row("EXACT"), row("OTHER", problem="Totally different text here", minutes=1)])
    candidates = r.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))

    assert candidates[0].sps_id == "EXACT"
    assert candidates[0].composite_score == candidates[0].cosine_similarity
    assert candidates[0].applied_boosts == ()


def test_a_perfect_match_on_another_part_is_never_returned(tmp_path):
    r = _retriever(tmp_path, [row("WRONG-PART", part="0034-11020"), row("RIGHT", minutes=1)])
    candidates = r.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))
    assert [c.sps_id for c in candidates] == ["RIGHT"]


def test_threshold_filters_the_result(tmp_path):
    r = _retriever(tmp_path, [row("WEAK", problem="Completely unrelated packaging text")], threshold=0.9)
    assert r.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART)) == []
    assert r.stats.usable == 1        # history existed
    assert r.stats.top_score < 0.9    # it just was not similar enough


def test_stats_distinguish_no_history_from_no_similarity(tmp_path):
    r = _retriever(tmp_path, [row("A", part="0034-11020")])
    assert r.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART)) == []
    assert r.stats.part_matches == 0
    assert r.stats.usable == 0


# ------------------------------------------------------------- CLI workbooks


class _Draft:
    recommendation = "1. Rework the weld seam.\n2. Re-inspect before shipment."
    justification = "Drawn from SPS-1001 for this part."


class _Outcome:
    succeeded = True
    draft = _Draft()
    infrastructure_failure = False
    failure_reason = ""


@pytest.fixture
def passing_llm(monkeypatch):
    """Stub the Actor-Critic loop so the PASS path can be exercised offline."""
    class Loop:
        def __init__(self, *a, **k):
            pass

        async def run(self, ticket, candidates):
            return _Outcome()

    monkeypatch.setattr("sps.generation.ActorCriticLoop", Loop)
    monkeypatch.setattr("sps.generation.AzureOpenAIChatClient", lambda *a, **k: object())


def run_cli(tmp_path, rows=None, part=PART, problem=PROBLEM, threshold=0.5):
    out = tmp_path / "out"
    write_history(tmp_path / "h.xlsx", rows if rows is not None else [row("SPS-1001")])
    write_ticket(tmp_path / "t.xlsx", part=part, problem=problem)
    code = resolver.main([
        "--ticket-file", str(tmp_path / "t.xlsx"),
        "--history-file", str(tmp_path / "h.xlsx"),
        "--output-dir", str(out),
        "--threshold", str(threshold),
    ])
    return code, out


def test_status_workbook_is_always_written(tmp_path, passing_llm):
    for part, problem in ((PART, PROBLEM), ("", PROBLEM), (PART, "short"), ("0099-1", PROBLEM)):
        _, out = run_cli(tmp_path, part=part, problem=problem)
        from service.excel_output import STATUS_COLUMNS

        sheet = read_sheet(out / "status.xlsx")
        assert list(sheet.columns) == list(STATUS_COLUMNS)
        assert len(sheet) == 1


def test_success_writes_both_workbooks(tmp_path, passing_llm):
    code, out = run_cli(tmp_path)

    assert code == resolver.EXIT_OK
    status = read_sheet(out / "status.xlsx").iloc[0]
    assert status["Status"] == "PASS"
    assert status["Status_Code"] == "SUCCESS"

    result = read_sheet(out / "output.xlsx")
    assert list(result.columns) == [
        "Part_Number", "AI_Recommendation", "Justification",
        "Confidence_Score", "Referenced_SPS_IDs",
    ]
    assert result.iloc[0]["Part_Number"] == PART
    assert result.iloc[0]["Referenced_SPS_IDs"] == "SPS-1001"
    assert result.iloc[0]["Confidence_Score"].endswith("%")


def test_failure_writes_status_only(tmp_path, passing_llm):
    code, out = run_cli(tmp_path, part="")
    assert code == resolver.EXIT_OK
    assert (out / "status.xlsx").exists()
    assert not (out / "output.xlsx").exists()


def test_unhandled_error_still_writes_status(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "resolve", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    code, out = run_cli(tmp_path)

    assert code == resolver.EXIT_INFRASTRUCTURE
    status = read_sheet(out / "status.xlsx").iloc[0]
    assert status["Status"] == "FAIL"
    assert status["Status_Code"] == "INFRASTRUCTURE_ERROR"


def test_stale_workbooks_are_cleared_before_work(tmp_path, passing_llm):
    out = tmp_path / "out"
    out.mkdir()
    from service.excel_output import RESULT_COLUMNS, write_rows

    write_rows(out / "output.xlsx", RESULT_COLUMNS, [{"Part_Number": "STALE"}])
    run_cli(tmp_path, part="")            # a run that must not produce output.xlsx

    assert not (out / "output.xlsx").exists()


def test_status_codes_cover_each_outcome(tmp_path, passing_llm):
    cases = [
        (dict(part=""), "INVALID_INPUT"),
        (dict(problem="short"), "INVALID_INPUT"),
        (dict(part="0099-99999"), "NO_MATCHES"),
        (dict(threshold=0.99), "BELOW_CONFIDENCE_THRESHOLD"),
        (dict(), "SUCCESS"),
    ]
    for kwargs, expected in cases:
        _, out = run_cli(tmp_path, **kwargs)
        assert read_sheet(out / "status.xlsx").iloc[0]["Status_Code"] == expected, kwargs


def test_reason_is_a_single_line(tmp_path, passing_llm):
    """Newlines would break a one-row-per-run sheet for anyone reading it back."""
    _, out = run_cli(tmp_path, part="0099-99999")
    assert "\n" not in read_sheet(out / "status.xlsx").iloc[0]["Reason"]


# ------------------------------------------------------------------ threshold


def test_default_threshold_is_calibrated_for_bge_small():
    """0.89, not the spec's 0.75 or bge-large's 0.82: bge-small scores higher on
    the same texts, so carrying a lower number over would loosen the gate.

    Now that the bge-large path is gone there is only one threshold in the
    codebase, and it lives beside the engine that applies it."""
    from sps.retrieval.in_memory import DEFAULT_CONFIDENCE_THRESHOLD, InMemoryRetriever

    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.89
    assert resolver.DEFAULT_THRESHOLD == 0.89
    assert InMemoryRetriever.__dataclass_fields__["confidence_threshold"].default == 0.89


def test_threshold_precedence(tmp_path, monkeypatch, passing_llm):
    """--threshold beats the environment, which beats the built-in default."""
    # The history problem only partly overlaps the ticket's, so the score lands
    # between the two thresholds under test rather than at a perfect 1.0.
    write_history(tmp_path / "h.xlsx",
                  [row("SPS-1001", problem="Weld seam cracking found on the bracket")])
    write_ticket(tmp_path / "t.xlsx")
    common = ["--ticket-file", str(tmp_path / "t.xlsx"),
              "--history-file", str(tmp_path / "h.xlsx")]

    monkeypatch.delenv("SPS_CONFIDENCE_THRESHOLD", raising=False)
    args = resolver.parse_args(common)
    assert args.threshold is None          # falls through to the default

    monkeypatch.setenv("SPS_CONFIDENCE_THRESHOLD", "0.95")
    out = tmp_path / "env"
    resolver.main(common + ["--output-dir", str(out)])
    assert read_sheet(out / "status.xlsx").iloc[0]["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"

    out2 = tmp_path / "flag"
    resolver.main(common + ["--output-dir", str(out2), "--threshold", "0.1"])
    assert read_sheet(out2 / "status.xlsx").iloc[0]["Status_Code"] == "SUCCESS"
