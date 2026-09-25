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


def test_a_cleared_match_is_cascaded_without_the_weak_marker(tmp_path, passing_llm):
    _, out = run_cli(tmp_path, threshold=0.5)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert "SPS-1001" in cell
    assert resolver.RAW_BANNER in cell
    assert "WEAK MATCH" not in cell


def test_the_banner_says_the_text_is_unchecked(tmp_path, passing_llm):
    """The banner is a control, not a caption. DEA pastes what looks pasteable,
    so a field holding raw archive text has to disqualify itself up front."""
    _, out = run_cli(tmp_path)
    cell = read_sheet(out / "output.xlsx").iloc[0]["Closest_Matching_Solution"]

    assert cell.startswith("[RAW HISTORY")
    assert "not checked for supplier use" in cell


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
        rows=[row("SPS-1001", solution=LEAKY_SOLUTION)],
        threshold=0.99,  # gated, so the cascade is populated and no LLM runs
    )
    result = read_sheet(out / "output.xlsx").iloc[0]

    recommendation = result["AI_Recommendation"]
    for forbidden in ("ESW#", "attachment", "Per discussed"):
        assert forbidden not in recommendation

    # Present where the reviewer can weigh it.
    assert "ESW#20033465" in result["Closest_Matching_Solution"]


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
