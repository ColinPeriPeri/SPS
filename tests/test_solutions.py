"""Up to three solutions per ticket.

Every record that clears the intent gate, best first, up to MAX_SOLUTIONS --
each sent exactly as recorded, on its own row, with its own confidence and
source. Fewer clear it, fewer are sent; none, and the refusal is unchanged.
"""

from __future__ import annotations

import asyncio
import csv

import pytest

import sps
from sps import IncomingTicket, MAX_SOLUTIONS, NO_RECOMMENDATION, score_intent
from tests.conftest import ScriptedChatClient
from tests.test_history_store import write_groups
from tests.test_intent_matching import assessment, candidate
from tests.test_resolver import PART, PROBLEM, read_sheet, row, write_history, write_ticket


@pytest.fixture(autouse=True)
def _stubbed_azure(azure_embeddings):
    pass


@pytest.fixture
def intent_scores(monkeypatch):
    """Set the intent score each SPS ID receives; record what the scorer saw."""
    scores: dict[str, float] = {}
    seen: list[list[str]] = []

    async def fake_score_intent(client, ticket, candidates):
        from sps import IntentOutcome, ScoredCandidate

        seen.append([c.sps_id for c in candidates])
        scored = [
            ScoredCandidate(candidate=c, intent_match=scores.get(c.sps_id, 0.05), reason="stub")
            for c in candidates
        ]
        scored.sort(key=lambda s: (s.intent_match, s.candidate.exact_part), reverse=True)
        return IntentOutcome(ticket_intent="stubbed intent", scored=tuple(scored))

    monkeypatch.setattr("sps.AzureOpenAIChatClient", lambda *a, **k: object())
    monkeypatch.setattr("sps.score_intent", fake_score_intent)
    return scores, seen


def records(n):
    return [
        row(f"SPS-{i}", solution=f"Solution {i}: rework the weld seam and re-inspect.", minutes=i)
        for i in range(1, n + 1)
    ]


def resolve(tmp_path, rows, *extra):
    out = tmp_path / "out"
    write_history(tmp_path / "h.xlsx", rows)
    write_ticket(tmp_path / "t.xlsx")
    code = sps.resolve_main([
        "--ticket-file", str(tmp_path / "t.xlsx"),
        "--history-file", str(tmp_path / "h.xlsx"),
        "--output-dir", str(out),
        "--threshold", "0.0",
        *extra,
    ])
    return code, out


# --------------------------------------------------------------- ordering


def test_an_equal_intent_score_goes_to_the_own_part():
    own = candidate("SPS-OWN", score=0.60)
    sibling = candidate("SPS-SIB", score=0.90)
    object.__setattr__(sibling, "exact_part", False)
    client = ScriptedChatClient([assessment(**{"SPS-OWN": 90, "SPS-SIB": 90})])
    outcome = asyncio.run(score_intent(client, IncomingTicket(PROBLEM, part_number=PART),
                                       [sibling, own]))
    assert [s.candidate.sps_id for s in outcome.scored] == ["SPS-OWN", "SPS-SIB"]


def test_a_higher_scoring_sibling_still_goes_first():
    own = candidate("SPS-OWN", score=0.90)
    sibling = candidate("SPS-SIB", score=0.60)
    object.__setattr__(sibling, "exact_part", False)
    client = ScriptedChatClient([assessment(**{"SPS-OWN": 90, "SPS-SIB": 95})])
    outcome = asyncio.run(score_intent(client, IncomingTicket(PROBLEM, part_number=PART),
                                       [own, sibling]))
    assert [s.candidate.sps_id for s in outcome.scored] == ["SPS-SIB", "SPS-OWN"]


# ---------------------------------------------------------------- the rows


def test_every_record_over_the_gate_is_sent_best_first(tmp_path, intent_scores):
    scores, _ = intent_scores
    scores.update({"SPS-1": 0.80, "SPS-2": 0.95, "SPS-3": 0.90, "SPS-4": 0.70})

    code, out = resolve(tmp_path, records(4))

    assert code == sps.EXIT_OK
    rows = read_sheet(out / "output.xlsx")
    assert [r["Solution_Rank"] for r in rows] == ["1", "2", "3"]
    assert [r["Referenced_Sources"] for r in rows] == ["SPS-2", "SPS-3", "SPS-1"]
    assert [r["Confidence_Score"] for r in rows] == ["95%", "90%", "80%"]
    # Each verbatim, each its own.
    assert [r["AI_Recommendation"] for r in rows] == [
        "Solution 2: rework the weld seam and re-inspect.",
        "Solution 3: rework the weld seam and re-inspect.",
        "Solution 1: rework the weld seam and re-inspect.",
    ]
    status = read_sheet(out / "status.xlsx")
    assert len(status) == 1
    assert "Intent matched 3 record(s) at or above 75%" in status[0]["Reason"]
    assert "SPS-2 (95%), SPS-3 (90%), SPS-1 (80%)" in status[0]["Reason"]


def test_fewer_over_the_gate_means_fewer_rows(tmp_path, intent_scores):
    scores, _ = intent_scores
    scores.update({"SPS-1": 0.95})
    _, out = resolve(tmp_path, records(4))
    rows = read_sheet(out / "output.xlsx")
    assert [(r["Solution_Rank"], r["Referenced_Sources"]) for r in rows] == [("1", "SPS-1")]
    assert "its solution was sent unchanged" in read_sheet(out / "status.xlsx")[0]["Reason"]


def test_none_over_the_gate_is_the_refusal_it_always_was(tmp_path, intent_scores):
    scores, _ = intent_scores
    scores.update({"SPS-1": 0.60, "SPS-2": 0.50})
    _, out = resolve(tmp_path, records(2))
    rows = read_sheet(out / "output.xlsx")
    assert len(rows) == 1
    assert rows[0]["AI_Recommendation"] == NO_RECOMMENDATION
    assert rows[0]["Solution_Rank"] == ""


def test_never_more_than_three(tmp_path, intent_scores):
    scores, _ = intent_scores
    scores.update({f"SPS-{i}": 0.99 - i / 100 for i in range(1, 6)})
    _, out = resolve(tmp_path, records(5))
    assert len(read_sheet(out / "output.xlsx")) == MAX_SOLUTIONS == 3


def test_the_shortlist_size_is_sps_intent_top_k(tmp_path, intent_scores, monkeypatch):
    _, seen = intent_scores
    (tmp_path / "default").mkdir()
    (tmp_path / "two").mkdir()
    resolve(tmp_path / "default", records(7))
    monkeypatch.setenv("SPS_INTENT_TOP_K", "2")
    resolve(tmp_path / "two", records(7))
    assert [len(s) for s in seen] == [5, 2]


def test_a_solution_from_a_sibling_part_says_so(tmp_path, intent_scores):
    scores, _ = intent_scores
    scores.update({"SPS-SIB": 0.95})
    groups = write_groups(tmp_path / "g.csv", [("0012", "FASTENERS"), ("0022", "FASTENERS")])
    _, out = resolve(tmp_path, [row("SPS-SIB", part="0022-77777")],
                     "--commodity-file", str(groups))

    result = read_sheet(out / "output.xlsx")[0]
    assert result["Referenced_Sources"] == "SPS-SIB"
    assert "recorded for 0022-77777, another part in the same commodity family" in \
        result["Justification"]
    assert "commodity family FASTENERS" in read_sheet(out / "status.xlsx")[0]["Reason"]


# ------------------------------------------------------------ bulk and eval


def test_the_bulk_sheet_gives_each_solution_a_row(tmp_path, intent_scores, capsys):
    scores, _ = intent_scores
    scores.update({"SPS-1": 0.95, "SPS-2": 0.90})
    history = write_history(tmp_path / "h.xlsx", records(3))
    tickets = tmp_path / "tickets.csv"
    with open(tickets, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["SPS_ID", "Part_Number", "Problem_Description"])
        writer.writerow(["T-1", PART, PROBLEM])
        writer.writerow(["T-2", "0099-99999", PROBLEM])

    out = tmp_path / "results.xlsx"
    sps.bulk_main(["--tickets", str(tickets), "--history", str(history), "--output", str(out),
                   "--threshold", "0.0"])

    sheet = read_sheet(out)
    assert [(r["SPS_ID"], r["Solution_Rank"]) for r in sheet] == [("T-1", "1"), ("T-1", "2"), ("T-2", "")]
    # The summary counts tickets, not rows.
    err = capsys.readouterr().err
    assert "2 row(s)" in err and "1 of 2 produced a recommendation" in err


def test_the_evaluator_reports_the_intent_score(tmp_path, intent_scores):
    scores, _ = intent_scores
    scores.update({"SPS-1": 0.95, "SPS-2": 0.90})
    history = write_history(tmp_path / "h.xlsx", records(2))
    ticket_file = write_ticket(tmp_path / "case01_ticket.xlsx")
    case = sps.Case("case01", ticket_file, history)

    row_ = sps.run_case(case, tmp_path / "work", None)

    assert row_.intent_score == 0.95
    assert row_.intent_threshold == 0.75
    assert row_.solutions == 2
