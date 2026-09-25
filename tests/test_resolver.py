"""The in-memory resolver: validation, part filtering, capping, dual workbooks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")
pytest.importorskip("numpy")

import scripts.run_resolver as resolver
import sps.contracts as resolver_contracts  # noqa: E402
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


@pytest.fixture(autouse=True)
def _stubbed_azure(azure_embeddings):
    """Azure is the only encoder now, so every test in this module embeds
    through the stub. A test that wants the unconfigured path deletes the
    variables itself."""


PROBLEM = "Bracket weld seam cracking observed during incoming inspection"
# A near-duplicate: 0.9354 against PROBLEM through the stubbed encoder, so a
# 0.99 gate rejects it and a 0.5 gate does not. Identical text scores exactly
# 1.0 -- Azure applies no query-instruction prefix, unlike bge, so query and
# passage vectors coincide -- and would clear any threshold below 1.
NEAR_PROBLEM = "Bracket weld seam cracking observed during incoming"
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


def write_ticket(path, part=PART, problem=PROBLEM, reason_code=""):
    """The reason code is blank by default, which is what most tickets carry.

    Tier 2 only runs for configured reason codes, so a blank one keeps these
    tests on the Tier-1 path they are actually about."""
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(
        ["SPS_ID", "Part_Number", "Issue_Type", "Problem_Description", "Problem_Reason_Code"]
    )
    sheet.append(["T-1", part, "Quality", problem, reason_code])
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


def test_a_blank_part_number_falls_through_to_tier_two(tmp_path, passing_llm, monkeypatch):
    """It used to be terminal. Tier 1 filters history by exact part so it still
    cannot run, but Tier 2 searches standards by defect text and never looks at
    the part number -- so the ticket is no longer dead, it is Tier 2's."""
    monkeypatch.setenv("SPS_TIER2_REASON_CODES", "RC-1")
    _, out = run_cli(tmp_path, part="")

    status = read_sheet(out / "status.xlsx").iloc[0]
    assert status["Status_Code"] != "INVALID_INPUT"
    assert "part number" in status["Reason"].casefold()


def test_a_blank_problem_description_is_still_terminal(tmp_path, passing_llm):
    """Both tiers match on that text, so there is nothing to search with on
    either side. Falling through would be searching for nothing."""
    _, out = run_cli(tmp_path, problem="")

    assert read_sheet(out / "status.xlsx").iloc[0]["Status_Code"] == "INVALID_INPUT"


def test_validation_runs_before_anything_expensive(tmp_path, monkeypatch):
    """A malformed ticket must not load a model or scan the history."""
    called = []
    # A blank DESCRIPTION, not a blank part number: a blank part now falls
    # through to Tier 2 rather than stopping, so it no longer demonstrates the
    # early gate. A blank description is still terminal and still cheap.
    monkeypatch.setattr(
        resolver, "read_ticket", lambda p: {"Part_Number": PART, "Problem_Description": ""}
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
    outcome = resolver.resolve(args, tmp_path / "out")

    assert outcome.code == resolver.CODE_INVALID_INPUT
    assert outcome.exit_code == resolver.EXIT_OK
    assert called == []
    # Nothing was encoded, so no model is named and no score exists -- both are
    # themselves information.
    assert outcome.embedding_model == ""
    assert outcome.top_score == 0.0


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


# What the stubbed intent scorer awards the top candidate. Comfortably above
# any threshold these tests set, so a change to the default gate cannot quietly
# flip a test that is about something else.
STUB_INTENT = 0.95


@pytest.fixture
def passing_llm(monkeypatch):
    """Stub the LLM so the PASS path can be exercised offline.

    Tier 1 no longer drafts anything: it scores intent and, above the gate,
    sends the matched record's solution unchanged. So what makes a run succeed
    here is a high intent score, not a draft. The loop is still stubbed because
    Tier 2 uses it.
    """
    class Loop:
        def __init__(self, *a, **k):
            pass

        async def run(self, ticket, candidates):
            return _Outcome()

        async def run_grounded(self, ticket, grounding):
            return _Outcome()

    async def fake_score_intent(client, ticket, candidates):
        from sps.generation.intent import IntentOutcome, ScoredCandidate

        return IntentOutcome(
            ticket_intent="stubbed ticket intent",
            scored=tuple(
                ScoredCandidate(
                    candidate=c,
                    intent_match=STUB_INTENT if i == 0 else 0.05,
                    reason="stubbed",
                )
                for i, c in enumerate(candidates)
            ),
        )

    monkeypatch.setattr("sps.generation.ActorCriticLoop", Loop)
    monkeypatch.setattr("sps.generation.AzureOpenAIChatClient", lambda *a, **k: object())
    monkeypatch.setattr("sps.generation.score_intent", fake_score_intent)


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
    assert status["Status_Code"] == "SUCCESS_HISTORICAL"

    result = read_sheet(out / "output.xlsx")
    assert list(result.columns) == [
        "Part_Number", "AI_Recommendation", "Justification",
        "Confidence_Score", "Referenced_Sources", "Resolution_Source",
        "Closest_Matching_Solution", "Cascade_Warnings",
    ]
    assert result.iloc[0]["Part_Number"] == PART
    assert result.iloc[0]["Referenced_Sources"] == "SPS-1001"
    assert result.iloc[0]["Resolution_Source"] == "HISTORICAL_DATA"
    assert result.iloc[0]["Confidence_Score"].endswith("%")


def test_a_concluded_failure_still_writes_a_result_row(tmp_path, passing_llm):
    """A caller merges output.xlsx into its own records, so every run that
    reached a conclusion needs a row. Without one, "nothing matched" and "this
    ticket was never processed" look identical downstream."""
    code, out = run_cli(tmp_path, part="")

    assert code == resolver.EXIT_OK
    assert (out / "status.xlsx").exists()
    assert (out / "output.xlsx").exists()

    row = read_sheet(out / "output.xlsx").iloc[0]
    assert row["AI_Recommendation"] == resolver_contracts.NO_RECOMMENDATION
    assert row["Resolution_Source"] == "NONE"
    assert row["Referenced_Sources"] == ""


def test_the_unresolved_row_carries_the_reason(tmp_path, passing_llm):
    """Justification is for the reviewer; the diagnostics stay in status.xlsx.

    These were the same string, so a DEA reviewer opening the portal read
    "Best historical match 0.8234 is below the 0.50 threshold across 12
    candidate(s)". The numbers did not go away -- the robot and support still
    read them in Reason -- they stopped being the business-facing text.
    """
    _, out = run_cli(tmp_path, part="0099-99999")

    row = read_sheet(out / "output.xlsx").iloc[0]
    status = read_sheet(out / "status.xlsx").iloc[0]

    assert row["Justification"] == (
        "Not much historical data to infer the solution or recommendation."
    )
    # The diagnostic is still recorded, just not here.
    assert "No usable history for part 0099-99999" in status["Reason"]
    assert "No usable history" not in row["Justification"]
    assert "threshold" not in row["Justification"]


def test_nothing_scored_leaves_the_confidence_blank(tmp_path, passing_llm):
    """An unknown part never reaches the encoder. Reporting 0% would read as a
    near miss that scored badly, which is a different finding."""
    _, out = run_cli(tmp_path, part="0099-99999")

    assert read_sheet(out / "output.xlsx").iloc[0]["Confidence_Score"] == ""


def test_a_gated_ticket_reports_the_score_it_reached(tmp_path, passing_llm):
    """The opposite case: something was measured, and how close it came is the
    whole reason a reviewer would look at the row."""
    _, out = run_cli(
        tmp_path,
        rows=[row("SPS-1001", problem=NEAR_PROBLEM)],
        threshold=0.99,
    )

    result = read_sheet(out / "output.xlsx").iloc[0]
    assert result["AI_Recommendation"] == resolver_contracts.NO_RECOMMENDATION
    assert result["Confidence_Score"].endswith("%")
    assert result["Confidence_Score"] != "0%"


def test_a_success_still_writes_the_real_recommendation(tmp_path, passing_llm):
    """The row is only synthesised when there is nothing to report."""
    _, out = run_cli(tmp_path)

    row_ = read_sheet(out / "output.xlsx").iloc[0]
    assert row_["AI_Recommendation"] != resolver_contracts.NO_RECOMMENDATION
    assert row_["Resolution_Source"] == "HISTORICAL_DATA"
    assert row_["Referenced_Sources"] == "SPS-1001"


def test_unhandled_error_still_writes_status(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "resolve", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    code, out = run_cli(tmp_path)

    assert code == resolver.EXIT_INFRASTRUCTURE
    status = read_sheet(out / "status.xlsx").iloc[0]
    assert status["Status"] == "FAIL"
    assert status["Status_Code"] == "INFRASTRUCTURE_ERROR"


def test_a_stale_result_row_is_replaced_not_left(tmp_path, passing_llm):
    """The previous run's answer must never be read as this one's."""
    out = tmp_path / "out"
    out.mkdir()
    from service.excel_output import RESULT_COLUMNS, write_rows

    write_rows(out / "output.xlsx", RESULT_COLUMNS, [{"Part_Number": "STALE"}])
    run_cli(tmp_path, part="")

    row_ = read_sheet(out / "output.xlsx").iloc[0]
    assert row_["Part_Number"] != "STALE"
    assert row_["AI_Recommendation"] == resolver_contracts.NO_RECOMMENDATION


def test_an_infrastructure_fault_leaves_no_result_row(tmp_path, monkeypatch, passing_llm):
    """It reached no conclusion. Writing "Solution not found." for an Azure
    outage would record a verdict the pipeline never formed, and the robot is
    meant to retry it -- so a stale row from a previous run is cleared too."""
    out = tmp_path / "out"
    out.mkdir()
    from service.excel_output import RESULT_COLUMNS, write_rows

    write_rows(out / "output.xlsx", RESULT_COLUMNS, [{"Part_Number": "STALE"}])
    monkeypatch.setattr(
        resolver, "resolve",
        lambda *a: resolver.ResolveOutcome(
            resolver.CODE_INFRASTRUCTURE, "Azure unreachable", resolver.EXIT_INFRASTRUCTURE
        ),
    )
    code, _ = run_cli(tmp_path)

    assert code == resolver.EXIT_INFRASTRUCTURE
    assert not (out / "output.xlsx").exists()
    assert (out / "status.xlsx").exists()


def test_status_codes_cover_each_outcome(tmp_path, passing_llm):
    cases = [
        # A blank part number is no longer INVALID_INPUT: Tier 1 cannot run
        # without it, but Tier 2 does not need it, so the ticket falls through
        # and reports what Tier 2 made of it.
        (dict(problem=""), "INVALID_INPUT"),
        (dict(problem="short"), "INVALID_INPUT"),
        # With no 0250 corpus loaded, Tier 2 retrieves nothing and the code
        # is Tier 1's own -- identical to the behaviour before Tier 2 existed.
        (dict(part="0099-99999"), "NO_MATCHES"),
        (dict(rows=[row("SPS-1001", problem=NEAR_PROBLEM)], threshold=0.99),
         "BELOW_CONFIDENCE_THRESHOLD"),
        (dict(), "SUCCESS_HISTORICAL"),
    ]
    for kwargs, expected in cases:
        _, out = run_cli(tmp_path, **kwargs)
        assert read_sheet(out / "status.xlsx").iloc[0]["Status_Code"] == expected, kwargs


def test_reason_is_a_single_line(tmp_path, passing_llm):
    """Newlines would break a one-row-per-run sheet for anyone reading it back."""
    _, out = run_cli(tmp_path, part="0099-99999")
    assert "\n" not in read_sheet(out / "status.xlsx").iloc[0]["Reason"]


# ------------------------------------------------------------------ threshold


def test_each_embedding_space_has_its_own_threshold():
    """A threshold belongs to one embedding space. Two encoders means two
    numbers, and the gate must apply whichever produced the vectors."""
    from sps.retrieval.in_memory import (
        AZURE_EMBEDDING_THRESHOLD,
        LOCAL_EMBEDDING_THRESHOLD,
        InMemoryRetriever,
    )

    assert LOCAL_EMBEDDING_THRESHOLD == 0.89
    fields = InMemoryRetriever.__dataclass_fields__
    assert fields["local_threshold"].default == LOCAL_EMBEDDING_THRESHOLD
    assert fields["azure_threshold"].default == AZURE_EMBEDDING_THRESHOLD
    # The Azure value is deliberately not pinned to a literal. It is a recall
    # filter feeding the intent scorer rather than a decision, so it is
    # expected to move; what must not move is that the two spaces keep
    # separate numbers.
    assert AZURE_EMBEDDING_THRESHOLD != LOCAL_EMBEDDING_THRESHOLD
    # No override by default: the backend decides.
    assert fields["confidence_threshold"].default is None
    assert resolver.DEFAULT_THRESHOLD == 0.89


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

    monkeypatch.setenv("SPS_CONFIDENCE_THRESHOLD", "0.99")
    out = tmp_path / "env"
    resolver.main(common + ["--output-dir", str(out)])
    status = read_sheet(out / "status.xlsx").iloc[0]
    assert status["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"
    assert "0.99 threshold" in status["Reason"]

    out2 = tmp_path / "flag"
    resolver.main(common + ["--output-dir", str(out2), "--threshold", "0.1"])
    assert read_sheet(out2 / "status.xlsx").iloc[0]["Status_Code"] == "SUCCESS_HISTORICAL"
