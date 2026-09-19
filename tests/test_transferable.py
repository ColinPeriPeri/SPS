"""The deterministic transferability gate.

Two halves carry the risk, and the second is the larger one. Catching
`ESW#20033465` is easy; not catching a part number, a tolerance or a duration is
what decides whether this gate is usable at all. A scanner that rejects good
drafts costs three Actor round trips and then refuses a ticket that should have
been answered.
"""

from __future__ import annotations

import pytest

from sps.generation.transferable import (
    KNOWN_TRACKING_PREFIXES,
    critique_for,
    untransferable_references,
)

# The real thing, from two live tickets.
LIVE_A = (
    "1. See the feedback in the attachment.\n"
    "2. Per discussed, rework as attachment shown.\n"
    "3. After rework, provide photos and related data; it will be waived if the "
    "result is ok."
)
LIVE_B = (
    "1. For the feedback please see the attachment.\n"
    "2. Per discussed, please rework as attachment shown.\n"
    "3. After rework, please provide photos and related data, we will waive it "
    "if the result is ok.\n"
    "4. ESW#20033465 is submitted for these issues."
)


# ------------------------------------------------------------ the real cases


def test_the_live_recommendations_are_both_rejected():
    """Neither of these should ever have reached a supplier."""
    assert untransferable_references(LIVE_A)
    assert untransferable_references(LIVE_B)


def test_the_leaked_work_request_is_caught():
    findings = untransferable_references(LIVE_B)
    assert any("ESW#20033465" in f for f in findings)
    assert any("tracking identifier" in f for f in findings)


def test_the_attachment_references_are_caught():
    findings = untransferable_references(LIVE_A)
    assert any("attachment" in f.lower() for f in findings)


def test_the_prior_conversation_is_caught():
    findings = untransferable_references(LIVE_A)
    assert any("prior conversation" in f for f in findings)


def test_a_repeated_reference_is_reported_once():
    """LIVE_A says 'attachment' twice. The critique should not say it twice."""
    findings = [f for f in untransferable_references(LIVE_A) if "attachment" in f.lower()]
    assert len(findings) == 1


# ------------------------------------------------------- tracking identifiers


@pytest.mark.parametrize(
    "text",
    [
        "ESW#20033465 is submitted.",
        "ESW #20033465 is submitted.",
        "ESW-20033465 raised.",
        "esw#20033465 raised.",
        "CAR 5566 was opened.",
        "NCR#1234 applies.",
        "SCAR-99887 issued.",
        "Raised under MRB 4321.",
    ],
)
def test_internal_tracking_numbers_are_caught(text):
    assert untransferable_references(text), text


def test_an_unknown_prefix_with_a_hash_is_still_caught():
    """The prefix list cannot be exhaustive, so LETTERS#DIGITS is caught too."""
    assert untransferable_references("QNX#889900 is open.")


def test_the_prefix_list_is_extendable_without_touching_a_regex():
    assert "ESW" in KNOWN_TRACKING_PREFIXES
    assert isinstance(KNOWN_TRACKING_PREFIXES, tuple)


# ------------------------------------------------- what must NOT be flagged


@pytest.mark.parametrize(
    "text",
    [
        # The single most important one: a part number is digit-led.
        "Rework part 0012-43951 and re-inspect.",
        "Applies to 0034-11020 only.",
        # Measurements and tolerances.
        "Grind out to sound metal plus 25 mm beyond each visible end.",
        "Porosity shall not exceed 2 percent by area over any 25 mm.",
        "Torque to 12.5 Nm.",
        "Surface roughness Ra 1.6 micrometres or better.",
        # Durations, not dates.
        "Re-inspect within 30 days of rework.",
        "Hold for 24 hours before shipment.",
        # "attach" as a verb is an action the supplier can take.
        "Attach photos of the reworked seam to your response.",
        # Ordinary technical prose.
        "Segregate the affected lot and quarantine it pending disposition.",
        "Re-weld to the original joint profile and re-inspect by dye penetrant.",
        "Provide photos and related data after rework.",
        "Reprint the carton labels and re-apply them before dispatch.",
    ],
)
def test_legitimate_technical_text_is_left_alone(text):
    assert untransferable_references(text) == [], text


def test_a_small_lot_count_is_not_a_lot_number():
    """'lot 5' is a quantity; 'lot 449120' is an identifier."""
    assert untransferable_references("Segregate lot 5.") == []
    assert untransferable_references("Segregate lot 449120.")


def test_a_bare_number_is_not_an_identifier():
    assert untransferable_references("Inspect all 400 units.") == []


# ------------------------------------------------------------------- dates


def test_a_calendar_date_is_caught():
    assert untransferable_references("Agreed on 2026-03-15.")
    assert untransferable_references("Agreed on 15/03/2026.")


def test_a_part_number_is_not_mistaken_for_a_date():
    """0012-43951 has the digit-dash-digit shape a loose date pattern eats."""
    assert untransferable_references("Part 0012-43951 refers.") == []


# ------------------------------------------------------------------ plumbing


def test_empty_text_is_clean():
    assert untransferable_references("") == []
    assert untransferable_references(None) == []


def test_the_critique_names_what_to_remove_and_why():
    """It is handed to the Actor, so it has to be actionable."""
    critique = critique_for(untransferable_references(LIVE_B))

    assert "ESW#20033465" in critique
    # The instruction that matters: the Actor has just been told by CHECK 1 that
    # anything from the source is fine, so this has to say otherwise.
    assert "does not make these transferable" in critique
    assert "Solution not found." in critique


def test_the_critique_offers_abstention_as_the_way_out():
    critique = critique_for(untransferable_references(LIVE_A))
    assert "no action the supplier could actually perform" in critique


def test_one_fragment_is_reported_once_under_its_best_category():
    """ESW#20033465 matches both the known-prefix rule and the generic
    LETTERS#DIGITS rule. Reporting it twice makes one problem read as two."""
    findings = [f for f in untransferable_references(LIVE_B) if "20033465" in f]

    assert len(findings) == 1
    assert findings[0].startswith("internal tracking identifier")


# ------------------------------------------------------- misattributed actions
#
# From two NTK tickets about an ESC whose engraved revision did not match its
# spec. Both requested an ESW; both came back telling the supplier to issue one.
# Issuance is the customer's action, and the historical Solution_Text was an
# internal engineer's own note, so its imperatives address the wrong party.

NTK_RECOMMENDATION = (
    "1. Issue an ESW.\n"
    "2. Do not ship the parts until the ESW is fully approved."
)


def test_the_live_esw_recommendation_is_rejected():
    from sps.generation.transferable import misattributed_actions

    assert misattributed_actions(NTK_RECOMMENDATION)


def test_the_correctly_addressed_step_survives():
    """Step 2 was right: the supplier controls shipment. A gate that eats it is
    worse than no gate at all."""
    from sps.generation.transferable import misattributed_actions

    assert misattributed_actions(
        "Do not ship the parts until the ESW is fully approved."
    ) == []


@pytest.mark.parametrize(
    "text",
    [
        "1. Issue an ESW.",
        "1. Issue ESW.",
        "Please issue an ESW for this unit.",
        "Rework the part and issue an ESW.",
        "Approve the deviation before shipment.",
        "Waive the requirement for this lot.",
        "Grant an MRB disposition.",
    ],
)
def test_customer_actions_stated_as_instructions_are_caught(text):
    from sps.generation.transferable import misattributed_actions

    assert misattributed_actions(text), text


@pytest.mark.parametrize(
    "text",
    [
        # The outcome, not the instruction. Base-form verbs are what make an
        # imperative, so these must all survive.
        "Do not ship the parts until the ESW is fully approved.",
        "An ESW has been requested; await approval before shipping.",
        "An ESW will be issued once the review completes.",
        "Hold the shipment until the deviation is granted.",
        # Requesting is the supplier's own action -- the tickets show them doing
        # exactly this.
        "Request an ESW before shipment.",
        "Submit a waiver request with the inspection data.",
        # Ordinary supplier work, no internal instrument in sight.
        "Re-engrave the unit with the correct marking prior to shipment.",
        "Re-inspect each unit and provide the measurement data.",
    ],
)
def test_outcomes_and_supplier_actions_are_left_alone(text):
    from sps.generation.transferable import misattributed_actions

    assert misattributed_actions(text) == [], text


def test_the_instrument_list_is_extendable():
    from sps.generation.transferable import INTERNAL_ACTION_VERBS, INTERNAL_INSTRUMENTS

    assert "ESW" in INTERNAL_INSTRUMENTS
    assert "issue" in INTERNAL_ACTION_VERBS
    # Requesting is the supplier's to do.
    assert "request" not in INTERNAL_ACTION_VERBS


def test_the_critique_gives_the_rephrasing_not_only_the_rejection():
    from sps.generation.transferable import misattributed_actions

    critique = critique_for([], misattributed_actions(NTK_RECOMMENDATION))

    assert "cannot grant them" in critique
    assert "An ESW has been requested" in critique, "show the required wording"
    assert "internal engineer's own note" in critique, "explain why it read that way"


def test_both_kinds_arrive_in_one_critique():
    """A draft with both problems is rewritten once, not twice -- it only gets
    three attempts."""
    from sps.generation.transferable import misattributed_actions

    combined = "1. Issue an ESW.\n2. Per discussed, see the attachment."
    critique = critique_for(
        untransferable_references(combined), misattributed_actions(combined)
    )

    assert "attachment" in critique.lower()
    assert "Issue an ESW" in critique
    assert critique.count("Solution not found.") == 1, "one way out, stated once"


@pytest.mark.parametrize(
    "text",
    [
        # Straight from live history. The customer stating an outcome the
        # supplier can expect is useful text, and an earlier version of the
        # bare-verb rule ate it.
        "we will waive it if the result is ok",
        "it will be waived if the result is ok",
        "3. After rework, provide photos; we will waive it if the result is ok.",
    ],
)
def test_a_stated_outcome_is_not_an_instruction(text):
    from sps.generation.transferable import misattributed_actions

    assert misattributed_actions(text) == [], text


@pytest.mark.parametrize(
    "text",
    ["Waive the requirement for this lot.", "1. Waive the requirement.", "Please waive it."],
)
def test_a_bare_customer_verb_is_caught_in_imperative_position(text):
    """Position is the only thing separating "Waive the requirement" from
    "we will waive it": both are the base form."""
    from sps.generation.transferable import misattributed_actions

    assert misattributed_actions(text), text
