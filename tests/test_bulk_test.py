"""The bulk test runner: a sheet of tickets in, the same sheet plus answers out.

The properties that carry the risk are about the sheet, not the pipeline, which
tests/test_resolver.py already covers: every original column survives verbatim,
the output has exactly one row per input row in the same order, and the input
file is never touched. A bulk run that silently dropped or reordered rows would
be worse than one that failed.
"""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import scripts.run_bulk_test as bulk  # noqa: E402
import scripts.run_resolver as resolver  # noqa: E402
from tests.test_resolver import (  # noqa: E402
    NEAR_PROBLEM, PART, PROBLEM, read_sheet, row, write_history,
)

TICKET_COLUMNS = ["SPS_ID", "Part_Number", "Issue_Type", "Problem_Description", "Supplier"]


@pytest.fixture(autouse=True)
def _stubbed_azure(azure_embeddings):
    """Azure is the only encoder, so every row needs the stub to embed at all."""


class _Draft:
    recommendation = "1. Rework the weld seam and re-inspect."
    justification = "Drawn from SPS-1001."


class _Outcome:
    draft = _Draft()
    attempts = 1
    critiques: list = []
    failure_reason = ""
    infrastructure_failure = False
    tier = "historical"
    succeeded = True


@pytest.fixture
def passing_llm(monkeypatch):
    class Loop:
        def __init__(self, *a, **k):
            pass

        async def run(self, ticket, candidates):
            return _Outcome()

        async def run_grounded(self, ticket, grounding):
            return _Outcome()

    monkeypatch.setattr("sps.generation.ActorCriticLoop", Loop)
    monkeypatch.setattr("sps.generation.AzureOpenAIChatClient", lambda *a, **k: object())


def write_tickets(path, rows, columns=TICKET_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def ticket_row(sps_id="T-1", part=PART, issue="Quality", problem=PROBLEM, supplier="Acme"):
    return [sps_id, part, issue, problem, supplier]


def run(tmp_path, rows, history_rows=None, extra=(), columns=TICKET_COLUMNS):
    tickets = write_tickets(tmp_path / "tickets.csv", rows, columns)
    history = write_history(
        tmp_path / "h.xlsx", history_rows if history_rows is not None else [row("SPS-1001")]
    )
    out = tmp_path / "results.xlsx"
    code = bulk.main([
        "--tickets", str(tickets),
        "--history", str(history),
        "--output", str(out),
        "--threshold", "0.5",
        *extra,
    ])
    return code, out


# ----------------------------------------------------------------- the sheet


def test_every_original_column_survives_verbatim(tmp_path, passing_llm):
    """Including ones the pipeline never looks at: the sheet is the reviewer's
    reference, so whatever they put in it has to come back."""
    code, out = run(tmp_path, [ticket_row(supplier="Northgate Packaging")])

    assert code == bulk.EXIT_OK
    result = read_sheet(out).iloc[0]
    assert result["SPS_ID"] == "T-1"
    assert result["Part_Number"] == PART
    assert result["Supplier"] == "Northgate Packaging"
    assert result["Problem_Description"] == PROBLEM


def test_the_original_columns_come_first_and_keep_their_order(tmp_path, passing_llm):
    _, out = run(tmp_path, [ticket_row()])

    columns = list(read_sheet(out).columns)
    assert columns[: len(TICKET_COLUMNS)] == TICKET_COLUMNS
    assert columns[len(TICKET_COLUMNS):] == list(bulk.RESULT_COLUMNS)


def test_one_output_row_per_input_row_in_order(tmp_path, passing_llm):
    rows = [ticket_row(sps_id=f"T-{i}") for i in range(1, 6)]
    _, out = run(tmp_path, rows)

    sheet = read_sheet(out)
    assert len(sheet) == 5
    assert list(sheet["SPS_ID"]) == ["T-1", "T-2", "T-3", "T-4", "T-5"]


def test_the_input_workbook_is_never_modified(tmp_path, passing_llm):
    """It stays a clean, re-runnable fixture."""
    tickets = write_tickets(tmp_path / "tickets.csv", [ticket_row()])
    before = tickets.read_bytes()

    run(tmp_path, [ticket_row()])

    assert tickets.read_bytes() == before


def test_blank_rows_are_skipped(tmp_path, passing_llm):
    _, out = run(tmp_path, [ticket_row(), ["", "", "", "", ""], ticket_row(sps_id="T-2")])

    assert list(read_sheet(out)["SPS_ID"]) == ["T-1", "T-2"]


def test_a_whole_float_part_number_is_not_corrupted(tmp_path, passing_llm):
    """Excel holds every number as a double, so a numeric part number arrives as
    1243951.0 and would otherwise be written back as an id matching nothing."""
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(TICKET_COLUMNS)
    sheet.append(["T-1", 1243951.0, "Quality", PROBLEM, "Acme"])
    tickets = tmp_path / "tickets.xlsx"
    book.save(tickets)

    columns, rows = bulk.read_tickets(tickets)
    assert rows[0]["Part_Number"] == "1243951"


def test_a_clashing_column_name_is_suffixed(tmp_path, passing_llm, capsys):
    """A sheet that already has Status must not end up with two of them."""
    columns = TICKET_COLUMNS + ["Status"]
    _, out = run(
        tmp_path,
        [ticket_row() + ["already here"]],
        extra=(),
        columns=columns,
    )

    sheet = read_sheet(out)
    assert sheet.iloc[0]["Status"] == "already here"
    assert "Status_AI" in sheet.columns
    assert sheet.iloc[0]["Status_AI"] in {"PASS", "FAIL"}


def test_the_default_output_sits_beside_the_input():
    from pathlib import Path

    assert bulk.default_output(Path("d:/x/tickets.xlsx")).name == "tickets_results.xlsx"


# ---------------------------------------------------------------- the answers


def test_a_resolved_row_carries_the_recommendation(tmp_path, passing_llm):
    _, out = run(tmp_path, [ticket_row()])

    result = read_sheet(out).iloc[0]
    assert result["Status"] == "PASS"
    assert result["Status_Code"] == "SUCCESS_HISTORICAL"
    assert "Rework the weld seam" in result["AI_Recommendation"]
    assert result["Resolution_Source"] == "HISTORICAL_DATA"
    assert result["Referenced_Sources"] == "SPS-1001"


def test_an_unresolved_row_says_so_rather_than_going_blank(tmp_path, passing_llm):
    """The whole point of the sheet: a ticket that found nothing still has a
    row, and the row says what happened."""
    _, out = run(tmp_path, [ticket_row(part="0099-99999")])

    result = read_sheet(out).iloc[0]
    assert result["Status"] == "FAIL"
    assert result["Status_Code"] == "NO_MATCHES"
    assert result["AI_Recommendation"] == "Solution not found."
    assert result["Resolution_Source"] == "NONE"


def test_nothing_scored_leaves_the_score_blank(tmp_path, passing_llm):
    """An unknown part never reaches the encoder. 0.0 would read as a match
    that scored badly, which is a different finding."""
    _, out = run(tmp_path, [ticket_row(part="0099-99999")])

    assert read_sheet(out).iloc[0]["Tier1_Score"] == ""


def test_a_gated_row_reports_how_close_it_came(tmp_path, passing_llm):
    """Which is what tells you whether the threshold is the problem."""
    tickets = write_tickets(tmp_path / "tickets.csv", [ticket_row()])
    history = write_history(tmp_path / "h.xlsx", [row("SPS-1001", problem=NEAR_PROBLEM)])
    out = tmp_path / "results.xlsx"

    bulk.main([
        "--tickets", str(tickets), "--history", str(history),
        "--output", str(out), "--threshold", "0.99",
    ])

    result = read_sheet(out).iloc[0]
    assert result["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"
    assert float(result["Tier1_Score"]) > 0.9
    assert result["AI_Recommendation"] == "Solution not found."


def test_each_row_is_resolved_independently(tmp_path, passing_llm):
    """A mix of outcomes in one sheet, each landing on its own row."""
    _, out = run(tmp_path, [
        ticket_row(sps_id="T-1"),
        ticket_row(sps_id="T-2", part="0099-99999"),
        ticket_row(sps_id="T-3", part=""),
    ])

    codes = dict(zip(read_sheet(out)["SPS_ID"], read_sheet(out)["Status_Code"]))
    assert codes["T-1"] == "SUCCESS_HISTORICAL"
    assert codes["T-2"] == "NO_MATCHES"
    assert codes["T-3"] == "INVALID_INPUT"


# ------------------------------------------------------------- not attempted


def test_limit_leaves_the_rest_aligned_and_marked(tmp_path, passing_llm):
    """The sheet must still line up with the input after a trial run."""
    rows = [ticket_row(sps_id=f"T-{i}") for i in range(1, 5)]
    _, out = run(tmp_path, rows, extra=["--limit", "2"])

    sheet = read_sheet(out)
    assert len(sheet) == 4
    assert list(sheet["Status"])[:2] == ["PASS", "PASS"]
    assert list(sheet["Status"])[2:] == [bulk.NOT_RUN, bulk.NOT_RUN]
    assert "--limit" in sheet.iloc[2]["Reason"]


def test_repeated_infrastructure_errors_stop_the_run(tmp_path, monkeypatch, passing_llm):
    """Wrong credentials would otherwise burn one failing call per row for a
    whole sheet, and every row would carry the same useless message."""
    monkeypatch.setattr(
        resolver, "resolve",
        lambda *a: resolver.ResolveOutcome(
            resolver.CODE_INFRASTRUCTURE, "Azure unreachable", resolver.EXIT_INFRASTRUCTURE
        ),
    )
    rows = [ticket_row(sps_id=f"T-{i}") for i in range(1, 7)]

    code, out = run(tmp_path, rows, extra=["--stop-after-errors", "2"])

    sheet = read_sheet(out)
    assert code == bulk.EXIT_INFRASTRUCTURE
    assert len(sheet) == 6, "every input row still has a row"
    assert list(sheet["Status_Code"])[:2] == ["INFRASTRUCTURE_ERROR"] * 2
    assert list(sheet["Status"])[2:] == [bulk.NOT_RUN] * 4
    assert "consecutive infrastructure errors" in sheet.iloc[2]["Reason"]


def test_an_isolated_error_does_not_stop_the_run(tmp_path, passing_llm):
    """Only CONSECUTIVE failures trip it: one bad row in fifty is a result."""
    rows = [
        ticket_row(sps_id="T-1", part="0099-99999"),
        ticket_row(sps_id="T-2"),
        ticket_row(sps_id="T-3"),
    ]
    code, out = run(tmp_path, rows, extra=["--stop-after-errors", "1"])

    assert code == bulk.EXIT_OK
    assert bulk.NOT_RUN not in list(read_sheet(out)["Status"])


def test_a_row_that_raises_becomes_a_row(tmp_path, monkeypatch, passing_llm):
    calls = []

    def explode(*a):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("row exploded")
        return resolver.ResolveOutcome(resolver.CODE_NO_MATCHES, "nothing", resolver.EXIT_OK)

    monkeypatch.setattr(resolver, "resolve", explode)
    code, out = run(tmp_path, [ticket_row(sps_id="T-1"), ticket_row(sps_id="T-2")])

    sheet = read_sheet(out)
    assert "row exploded" in sheet.iloc[0]["Reason"]
    assert sheet.iloc[1]["Status_Code"] == "NO_MATCHES"
    assert code == bulk.EXIT_INFRASTRUCTURE


# ------------------------------------------------------------------- inputs


def test_an_unsupported_ticket_type_is_refused(tmp_path):
    bad = tmp_path / "tickets.pdf"
    bad.write_text("x", encoding="utf-8")
    history = write_history(tmp_path / "h.xlsx", [row("SPS-1001")])

    assert bulk.main([
        "--tickets", str(bad), "--history", str(history),
        "--output", str(tmp_path / "out.xlsx"),
    ]) == bulk.EXIT_BAD_INPUT


def test_a_missing_ticket_file_is_refused(tmp_path):
    history = write_history(tmp_path / "h.xlsx", [row("SPS-1001")])

    assert bulk.main([
        "--tickets", str(tmp_path / "absent.csv"), "--history", str(history),
        "--output", str(tmp_path / "out.xlsx"),
    ]) == bulk.EXIT_BAD_INPUT


def test_a_sheet_with_no_ticket_rows_is_refused(tmp_path):
    tickets = write_tickets(tmp_path / "tickets.csv", [])
    history = write_history(tmp_path / "h.xlsx", [row("SPS-1001")])

    assert bulk.main([
        "--tickets", str(tickets), "--history", str(history),
        "--output", str(tmp_path / "out.xlsx"),
    ]) == bulk.EXIT_BAD_INPUT
