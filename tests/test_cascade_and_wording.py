"""What a reviewer sees when there is no recommendation.

DEA copies AI_Recommendation into the SPS portal by hand and it goes to the
supplier from there, essentially unedited. Two consequences run through every
test here: that field must never carry archive text that has not been checked,
and the reviewer still needs to see the archive text somewhere, or a refusal
tells them nothing they can act on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")
pytest.importorskip("numpy")

import scripts.run_resolver as resolver  # noqa: E402
from sps.contracts import NO_RECOMMENDATION, SOLUTION_NOT_FOUND  # noqa: E402
from tests.test_resolver import (  # noqa: E402
    NEAR_PROBLEM,
    PART,
    PROBLEM,
    read_sheet,
    row,
    run_cli,
)

# Imported for its side effect of registering here: a fixture defined in
# another test module is not visible to this one unless it is in the namespace.
from tests.test_resolver import passing_llm  # noqa: E402,F401

# The two shapes that made these guards necessary, verbatim from real output.
LEAKY_SOLUTION = (
    "1. For the feedback please see the attachment. 2. Per discussed, please "
    "rework as attachment shown. 4. ESW#20033465 is submitted for these issues."
)
MISATTRIBUTED_SOLUTION = "1. Issue an ESW. 2. Do not ship the parts until the ESW is fully approved."


@pytest.fixture(autouse=True)
def _stubbed_azure(azure_embeddings):
    """Azure is the only encoder, so everything here embeds through the stub."""


# --------------------------------------------------------------- the banner


def test_a_gated_match_is_cascaded_and_marked_weak(tmp_path, passing_llm):
    """The row the gate threw away is the one a reviewer most wants to read.

    Before this, `retrieve()` returned only what qualified, so on a gated run
    the near-miss text existed nowhere at all and the reviewer was told "no
    match" about a record that scored 0.93.
    """
    _, out = run_cli(
        tmp_path,
        rows=[row("SPS-1001", problem=NEAR_PROBLEM)],
        threshold=0.99,
    )
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert "SPS-1001" in cell
    assert resolver.WEAK_BANNER in cell
    assert "below the confidence gate" in cell


def test_a_resolved_row_lists_its_sources_rather_than_quoting_them(tmp_path, passing_llm):
    """Up to fifteen records reach the Actor and a recommendation may combine
    several, so quoting one would read as THE source and invite a reviewer to
    check a step against a record it did not come from."""
    _, out = run_cli(tmp_path, threshold=0.5)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert cell.startswith("Synthesized from")
    assert "SPS-1001" in cell
    assert "%" in cell          # the per-record score Referenced_Sources cannot carry
    assert "[RAW HISTORY" not in cell


def test_the_banner_says_the_text_is_unchecked(tmp_path, passing_llm):
    """The banner is a control, not a caption. DEA pastes what looks pasteable,
    so a field holding raw archive text has to disqualify itself up front.

    Only reachable on a refusal now -- a resolved row carries a source list,
    which has no archive text in it to mislabel."""
    _, out = run_cli(tmp_path, rows=[row("SPS-1001", problem=NEAR_PROBLEM)], threshold=0.99)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert cell.startswith("[RAW HISTORY")
    assert "WEAK MATCH" in cell


def test_an_unknown_part_cascades_nothing(tmp_path, passing_llm):
    """No precedent means an empty cell, not a banner over nothing."""
    _, out = run_cli(tmp_path, part="0099-99999")
    assert read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"] == ""


# ------------------------------------------------- the copy-paste safety line


def test_raw_archive_text_never_reaches_the_recommendation(tmp_path, passing_llm):
    """The property the whole design turns on.

    A history row whose entire solution is boilerplate must not put that
    boilerplate into the field DEA copies. It may -- and must -- appear in the
    cascade column, which is the point: the reviewer sees it, the supplier does
    not, and nobody has to take our word for which is which.
    """
    _, out = run_cli(
        tmp_path,
        # NEAR_PROBLEM, not the default: identical text scores exactly 1.0 on
        # the stubbed encoder, so a 0.99 gate would not actually gate and this
        # would silently test the resolved path instead.
        rows=[row("SPS-1001", problem=NEAR_PROBLEM, solution=LEAKY_SOLUTION)],
        threshold=0.99,
    )
    result = read_sheet(out / "output.xlsx").iloc[0]

    recommendation = result["AI_Recommendation"]
    for forbidden in ("ESW#", "attachment", "Per discussed"):
        assert forbidden not in recommendation

    # Present where the reviewer can weigh it.
    assert "ESW#20033465" in result["Closest_Matching_Solution"]
    assert "ESW#20033465" in result["Justification"]


# ------------------------------------------------------------- the wording


def test_the_refusal_reads_as_business_copy(tmp_path, passing_llm):
    _, out = run_cli(tmp_path, part="0099-99999")
    row_ = read_sheet(out / "output.xlsx").iloc[0]

    assert row_["AI_Recommendation"] == "No recommendation available."
    assert row_["Justification"] == (
        "Not much historical data to infer the solution or recommendation."
    )


def test_the_justification_leads_with_the_precedent(tmp_path, passing_llm):
    """Precedent first, explanation second -- the order the reviewer asked for,
    because they judge the match before they read why we would not use it."""
    _, out = run_cli(
        tmp_path,
        rows=[row("SPS-1001", problem=NEAR_PROBLEM)],
        threshold=0.99,
    )
    justification = read_sheet(out / "output.xlsx").iloc[0]["Justification"]

    assert justification.startswith("[RAW HISTORY")
    assert "not similar enough to rely on" in justification
    # The precedent comes before the explanation, not after it.
    assert justification.index("SPS-1001") < justification.index("not similar enough")


def test_no_diagnostics_leak_into_the_business_field(tmp_path, passing_llm):
    """Thresholds and cosines belong in status.xlsx, which is where the robot
    and support read them. They were being shown to DEA as well."""
    _, out = run_cli(
        tmp_path,
        rows=[row("SPS-1001", problem=NEAR_PROBLEM)],
        threshold=0.99,
    )
    justification = read_sheet(out / "output.xlsx").iloc[0]["Justification"]
    reason = read_sheet(out / "status.xlsx").iloc[0]["Reason"]

    for jargon in ("threshold", "cosine", "candidate(s)", "[Azure]"):
        assert jargon not in justification
    assert "threshold" in reason  # still recorded, just not there


# ------------------------------------------------ the sentinel stays internal


def test_the_protocol_sentinel_never_reaches_a_reader(tmp_path, passing_llm):
    """Two strings doing two jobs. The Actor still emits "Solution not found.";
    nobody outside the pipeline should ever see that phrasing."""
    for part in ("0099-99999", PART):
        _, out = run_cli(tmp_path, part=part)
        sheet = read_sheet(out / "output.xlsx").iloc[0]
        assert SOLUTION_NOT_FOUND not in str(sheet["AI_Recommendation"])
        assert SOLUTION_NOT_FOUND not in str(sheet["Justification"])


def test_the_two_strings_are_actually_different():
    """A guard against someone 'tidying' them back into one constant."""
    assert SOLUTION_NOT_FOUND != NO_RECOMMENDATION


def test_no_module_spells_the_sentinel_by_hand():
    """Three modules hardcoded it, so changing the constant left the prompts,
    the critique and the tool schema quietly stale. Comments are exempt: they
    describe the value rather than depend on it."""
    import ast
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path("sps").rglob("*.py")):
        if path.name == "contracts.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Docstrings are Constant nodes too, so only flag short strings --
            # a prose paragraph mentioning it is documentation, not a duplicate.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if SOLUTION_NOT_FOUND in node.value and len(node.value) < 200:
                    offenders.append(f"{path}: {node.value[:60]!r}")
    assert not offenders, "hardcoded sentinel: " + "; ".join(offenders)


# ------------------------------------------- the precedent on the success path
#
# The refusal path led with the precedent from the start. The success path did
# not carry it into Justification at all -- it reached the column and stopped
# there, so a reviewer checking whether a recommendation really followed from
# the record had to read two columns to find out.






def test_no_precedent_means_no_empty_banner(tmp_path, passing_llm):
    """An unknown part has nothing to cascade. The justification should read as
    a sentence, not as a heading over blank space."""
    _, out = run_cli(tmp_path, part="0099-99999")
    justification = read_sheet(out / "output.xlsx").iloc[0]["Justification"]

    assert "[RAW HISTORY" not in justification
    assert justification == (
        "Not much historical data to infer the solution or recommendation."
    )


# ------------------------------------------------------- more than one record
#
# Up to fifteen qualified records reach the Actor, each tagged with its own
# match score, and a recommendation may draw on any of them. Everything above
# was written against a single record and would not have noticed.


def five_records():
    """Five rows for the same part, descending similarity to the ticket.

    None is a verbatim copy of the ticket. Identical text scores exactly 1.0
    on the stubbed encoder, and one such row would clear any gate below 1 --
    silently turning a test of the refusal path into a test of the other one.
    """
    problems = [
        "Bracket weld seam cracking observed during incoming",
        "Bracket weld seam cracking noted at incoming inspection",
        "Bracket weld seam cracking noted at goods-in",
        "Bracket weld cracking seen on arrival",
        "Bracket weld issue found",
    ]
    return [
        row(f"SPS-{1001 + i}", problem=text, solution=f"Step {i}: rework the seam.")
        for i, text in enumerate(problems)
    ]


def scores_in(text):
    """Every NN% in the order it appears."""
    import re

    return [int(n) for n in re.findall(r"\((\d+)%\)", text)]


def test_a_resolved_row_names_every_record_it_drew_on(tmp_path, passing_llm):
    """Not just the best. The Actor saw all of them and the answer may combine
    several, so naming one would misreport where it came from."""
    _, out = run_cli(tmp_path, rows=five_records(), threshold=0.1)
    result = read_sheet(out / "output.xlsx").iloc[0]

    listed = result["Closest_Matching_Solution"]
    assert listed.startswith("Synthesized from 5 record(s)")
    for i in range(5):
        assert f"SPS-{1001 + i}" in listed
    # Referenced_Sources carries the same ids; this adds the scores it cannot.
    assert len(scores_in(listed)) == 5


def test_the_source_list_is_ordered_best_first(tmp_path, passing_llm):
    """So a reviewer who opens only one opens the one most worth opening."""
    _, out = run_cli(tmp_path, rows=five_records(), threshold=0.1)
    listed = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    scores = scores_in(listed)
    assert len(scores) == 5
    assert scores == sorted(scores, reverse=True), listed


def test_a_refusal_shows_three_records_not_one_and_not_all(tmp_path, passing_llm):
    """Three is the chosen middle: a near-miss is not hidden behind a
    marginally better one, and the cell stays readable."""
    _, out = run_cli(tmp_path, rows=five_records(), threshold=0.99)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert "3 closest record(s)" in cell
    assert cell.count("SPS-") == 3

    # The three highest-scoring, and whole records rather than a truncation.
    # Deliberately not asserted by id: the encoder decides the ranking, and
    # pinning SPS-1001..1003 would be asserting that the ids happen to sort the
    # same way the scores do.
    shown = scores_in(cell)
    assert shown == sorted(shown, reverse=True)
    # Whole records, not a truncation of a longer list.
    assert cell.count("rework the seam.") == 3


def test_every_shown_record_carries_its_own_score(tmp_path, passing_llm):
    """A reviewer choosing between three records needs to know which matched
    best; three unlabelled blocks of prose do not say."""
    _, out = run_cli(tmp_path, rows=five_records(), threshold=0.99)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert len(scores_in(cell)) == 3


def test_the_banner_appears_once_not_per_record(tmp_path, passing_llm):
    """It heads the block. Repeating it between records would read as three
    separate warnings rather than one label on one field."""
    _, out = run_cli(tmp_path, rows=five_records(), threshold=0.99)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert cell.count("RAW HISTORY") == 1


# ------------------------------------------------------- the verbatim cascade
#
# The bargain changed shape. It used to be: archive text may never reach the
# field DEA copies. It is now: above the gate that text IS the field DEA
# copies, sent exactly as recorded, and a warning column says what is in it.
#
# So what these pin is no longer an absence. It is that the pass-through is
# genuinely a pass-through, and that nothing risky travels silently.


def test_the_matched_solution_is_sent_character_for_character(tmp_path, passing_llm):
    """Not "similar to", not "cleaned up". A reviewer must be able to diff this
    cell against the source record and find no difference at all."""
    _, out = run_cli(tmp_path, rows=[row("SPS-1001", solution=LEAKY_SOLUTION)], threshold=0.5)
    recommendation = read_sheet(out / "output.xlsx").iloc[0]["AI_Recommendation"]

    assert recommendation == LEAKY_SOLUTION


def test_the_shape_of_the_solution_survives_the_cascade(tmp_path, passing_llm):
    """Blank lines, double spaces and the numbering style all come through.

    One thing IS normalised, and it is worth stating rather than discovering:
    the history reader strips each cell at load, as it does for every field, so
    whitespace at the very start and end of the cell does not survive. Nothing
    between the first and last character is touched -- which is what makes the
    cascade diffable against the source.
    """
    awkward = "1)  Rework the seam.\n\n2)   Re-inspect under 10x."
    _, out = run_cli(tmp_path, rows=[row("SPS-1001", solution=awkward)], threshold=0.5)

    assert read_sheet(out / "output.xlsx").iloc[0]["AI_Recommendation"] == awkward


def test_only_the_cell_edges_are_trimmed(tmp_path, passing_llm):
    """The one normalisation, pinned so it stays the only one."""
    padded = "   Rework the seam and re-inspect.   "
    _, out = run_cli(tmp_path, rows=[row("SPS-1001", solution=padded)], threshold=0.5)

    assert read_sheet(out / "output.xlsx").iloc[0]["AI_Recommendation"] == padded.strip()


def test_risky_content_is_flagged_but_not_blocked(tmp_path, passing_llm):
    """The whole design in one test. The text goes out unchanged AND the
    reviewer is told what is in it -- the flag replaces the gate that used to
    refuse it, and replaces nothing else."""
    _, out = run_cli(tmp_path, rows=[row("SPS-1001", solution=LEAKY_SOLUTION)], threshold=0.5)
    result = read_sheet(out / "output.xlsx").iloc[0]

    assert "ESW#20033465" in result["AI_Recommendation"]      # not blocked
    assert "ESW#20033465" in result["Cascade_Warnings"]       # not silent
    assert "attachment" in result["Cascade_Warnings"]


def test_a_clean_solution_is_flagged_with_nothing(tmp_path, passing_llm):
    """A warning on every row is a warning on no row."""
    clean = "Grind out the cracked seam to sound metal, then re-weld and re-inspect."
    _, out = run_cli(tmp_path, rows=[row("SPS-1001", solution=clean)], threshold=0.5)
    result = read_sheet(out / "output.xlsx").iloc[0]

    assert result["AI_Recommendation"] == clean
    assert result["Cascade_Warnings"] == ""


def test_a_cascade_names_only_the_record_it_came_from(tmp_path, passing_llm):
    """One record's solution was sent, so one record is cited. The others were
    considered and rejected, and listing them would say the answer came from
    all five."""
    rows_ = [row("SPS-1001"), row("SPS-1002", problem=NEAR_PROBLEM)]
    _, out = run_cli(tmp_path, rows=rows_, threshold=0.3)
    result = read_sheet(out / "output.xlsx").iloc[0]

    assert result["Referenced_Sources"] == "SPS-1001"
    # The rest are still visible, with their scores, in the source list.
    assert "SPS-1002" in result["Closest_Matching_Solution"]


def test_the_success_justification_names_the_confidence_and_the_source(tmp_path, passing_llm):
    _, out = run_cli(tmp_path, threshold=0.5)
    justification = read_sheet(out / "output.xlsx").iloc[0]["Justification"]

    assert "95%" in justification
    assert "SPS-1001" in justification
    assert "sent unchanged" in justification
