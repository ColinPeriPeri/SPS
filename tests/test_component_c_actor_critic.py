"""Component C -- Actor, Judge, refinement and the circuit breaker."""

from __future__ import annotations

import json

import pytest

# Component C is schema-constrained: the Actor and Judge responses are
# validated through pydantic models, so pydantic is a hard requirement here.
# Components A and B stay dependency-free.
pytest.importorskip("pydantic")

from sps.config import LLMSettings
from sps.contracts import Candidate, IncomingTicket
from sps.generation import ActorCriticLoop, LLMError, parse_json_response
from sps.generation.prompts import build_actor_messages, build_judge_messages
from tests.conftest import ScriptedChatClient

TICKET = IncomingTicket(
    problem_description="Bracket weld seam cracking observed during incoming inspection",
    sps_id="T-1",
    part_number="PN-1000",
)

CANDIDATES = [
    Candidate(
        sps_id="SPS-100",
        actual_solution="Rework the weld seam. Re-inspect under 10x magnification.",
        part_number="PN-1000",
        part_description="Mounting bracket",
        item_status="Active",
        problem_reason_code="RC-WELD",
        issue_type="Quality",
        cosine_similarity=0.90,
        composite_score=0.95,
    ),
    Candidate(
        sps_id="SPS-101",
        actual_solution="Scrap the affected lot and ship a replacement.",
        part_number="PN-1000",
        part_description="Mounting bracket",
        item_status="Active",
        problem_reason_code="RC-WELD",
        issue_type="Quality",
        cosine_similarity=0.84,
        composite_score=0.89,
    ),
]


def actor(recommendation: str, justification: str = "Based on SPS-100.") -> str:
    return json.dumps({"recommendation": recommendation, "justification": justification})


PASS = json.dumps({"status": "PASS"})


def fail(critique: str) -> str:
    return json.dumps({"status": "FAIL", "critique": critique})


def loop(*responses, max_attempts: int = 3):
    client = ScriptedChatClient(responses)
    return ActorCriticLoop(client, LLMSettings(max_attempts=max_attempts)), client


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


async def test_draft_passing_the_judge_is_returned_immediately():
    engine, client = loop(actor("1. Rework the weld seam."), PASS)
    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert result.attempts == 1
    assert result.draft.recommendation == "1. Rework the weld seam."
    assert client.call_count == 2  # one Actor call, one Judge call


# --------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------


async def test_failed_draft_is_refined_and_can_pass_on_retry():
    engine, client = loop(
        actor("1. Rework the weld seam. 2. Torque to 40 Nm."),
        fail("'Torque to 40 Nm' does not appear in the historical solutions."),
        actor("1. Rework the weld seam."),
        PASS,
    )
    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert result.attempts == 2
    assert result.critiques == ["'Torque to 40 Nm' does not appear in the historical solutions."]


async def test_critique_and_rejected_draft_are_fed_back_to_the_actor():
    engine, client = loop(
        actor("1. Torque to 40 Nm."),
        fail("Remove the invented torque value."),
        actor("1. Rework the weld seam."),
        PASS,
    )
    await engine.run(TICKET, CANDIDATES)

    second_actor_prompt = client.calls[2][1]["content"]
    assert "Remove the invented torque value." in second_actor_prompt
    assert "1. Torque to 40 Nm." in second_actor_prompt
    assert "REJECTED DRAFT" in second_actor_prompt


async def test_third_attempt_can_still_succeed():
    engine, _ = loop(
        actor("draft one"), fail("c1"),
        actor("draft two"), fail("c2"),
        actor("draft three"), PASS,
    )
    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert result.attempts == 3


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


async def test_circuit_breaker_stops_after_three_attempts():
    engine, client = loop(
        actor("draft one"), fail("c1"),
        actor("draft two"), fail("c2"),
        actor("draft three"), fail("c3"),
    )
    result = await engine.run(TICKET, CANDIDATES)

    assert not result.succeeded
    assert result.draft is None
    assert result.attempts == 3
    assert result.critiques == ["c1", "c2", "c3"]
    # Exactly 3 Actor + 3 Judge calls: no fourth attempt is made.
    assert client.call_count == 6
    assert client.responses == []


async def test_abstention_short_circuits_without_retrying():
    engine, client = loop(actor("Solution not found."))
    result = await engine.run(TICKET, CANDIDATES)

    assert not result.succeeded
    assert result.attempts == 1
    assert client.call_count == 1  # the Judge is never consulted


# --------------------------------------------------------------------------
# Failure modes fail closed
# --------------------------------------------------------------------------


async def test_judge_outage_does_not_ship_an_unaudited_draft():
    class BrokenJudge(ScriptedChatClient):
        async def complete(self, messages):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return actor("1. Rework the weld seam.")
            raise LLMError("Azure timeout")

    engine = ActorCriticLoop(BrokenJudge([]), LLMSettings())
    result = await engine.run(TICKET, CANDIDATES)

    assert not result.succeeded
    assert "audit unavailable" in result.failure_reason.lower()


async def test_unrecognised_judge_status_is_not_treated_as_a_pass():
    engine, _ = loop(actor("1. Rework."), json.dumps({"status": "MAYBE"}))
    result = await engine.run(TICKET, CANDIDATES)
    assert not result.succeeded


async def test_actor_returning_no_recommendation_fails_gracefully():
    engine, _ = loop(json.dumps({"justification": "nothing here"}))
    result = await engine.run(TICKET, CANDIDATES)

    assert not result.succeeded
    assert "Generation failed" in result.failure_reason


async def test_judge_failing_without_a_critique_still_yields_actionable_text():
    engine, _ = loop(
        actor("d1"), json.dumps({"status": "FAIL"}),
        actor("d2"), PASS,
    )
    result = await engine.run(TICKET, CANDIDATES)
    assert result.succeeded
    assert result.critiques[0]


# --------------------------------------------------------------------------
# JSON tolerance
# --------------------------------------------------------------------------


def test_parses_json_wrapped_in_a_code_fence():
    parsed = parse_json_response('```json\n{"status": "PASS"}\n```')
    assert parsed == {"status": "PASS"}


@pytest.mark.parametrize("raw", ["", "   ", "not json at all", "[1, 2, 3]"])
def test_unusable_model_output_raises(raw):
    with pytest.raises(LLMError):
        parse_json_response(raw)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_actor_context_contains_solutions_but_no_hidden_metadata():
    messages = build_actor_messages(TICKET, CANDIDATES)
    user = messages[1]["content"]

    assert TICKET.problem_description in user
    assert "Rework the weld seam." in user
    assert "SPS-100" in user
    # The Actor is never shown internal-ish payload fields it could echo.
    assert "Mounting bracket" not in user
    assert "RC-WELD" not in user


def test_actor_system_prompt_states_the_grounding_and_leakage_rules():
    system = build_actor_messages(TICKET, CANDIDATES)[0]["content"]
    assert "ZERO external domain knowledge" in system
    assert "EXTERNAL SUPPLIER" in system


def test_judge_prompt_carries_both_mandated_checks():
    system = build_judge_messages(TICKET, CANDIDATES, "some draft")[0]["content"]
    assert "DOMAIN HALLUCINATION" in system
    assert "INTERNAL TOOL LEAKAGE" in system

    user = build_judge_messages(TICKET, CANDIDATES, "some draft")[1]["content"]
    assert "some draft" in user
    assert "Rework the weld seam." in user  # the source of truth it audits against


def test_history_metadata_cannot_reach_the_actor_or_judge():
    """The Actor is shown the incoming problem and the historical solution text,
    nothing else. Candidate carries metadata, but the prompt builders read named
    attributes only, so a field added to history can never reach the model."""
    from sps.retrieval.in_memory import HistoryRow, _to_candidate

    secret = "INTERNAL-MES-REF-9931"
    candidate = _to_candidate(
        HistoryRow(
            sps_id="SPS-100",
            part_number="0012-43951",
            problem_description="Weld seam cracking on the bracket",
            actual_solution="Rework the weld seam and re-inspect.",
            issue_type=secret,
            problem_reason_code=secret,
            part_description=secret,
            item_status=secret,
        ),
        0.93,
    )

    for messages in (
        build_actor_messages(TICKET, [candidate]),
        build_judge_messages(TICKET, [candidate], "draft"),
    ):
        blob = "".join(m["content"] for m in messages)
        assert secret not in blob
    # What the model *is* shown: the SPS ID and the solution text.
    user = build_actor_messages(TICKET, [candidate])[1]["content"]
    assert "SPS-100" in user
    assert "Rework the weld seam" in user


# --------------------------------------------------------------------------
# The transferability gate
#
# From two live tickets. Both were faithful restatements of historical
# Solution_Text, both passed the Judge, and both reached a supplier: one citing
# an internal work request raised for a different issue, both pointing at an
# attachment that does not exist. The Judge checks provenance, and by that
# measure they were correct.
# --------------------------------------------------------------------------

LEAKING_DRAFT = (
    "1. For the feedback please see the attachment.\n"
    "2. Per discussed, please rework as attachment shown.\n"
    "3. After rework, please provide photos and related data.\n"
    "4. ESW#20033465 is submitted for these issues."
)
CLEAN_DRAFT = "1. Rework the weld seam. 2. Provide photos and related data after rework."


async def test_a_leaking_draft_never_reaches_the_judge():
    """The gate runs first because it is local and free, where the Judge is a
    paid call that had already passed this exact text."""
    engine, client = loop(actor(LEAKING_DRAFT), actor(LEAKING_DRAFT), actor(LEAKING_DRAFT))

    result = await engine.run(TICKET, CANDIDATES)

    assert not result.succeeded
    # Three Actor calls and no Judge call: a PASS was never scripted, and the
    # ScriptedChatClient would have raised had one been requested.
    assert client.call_count == 3


async def test_the_leaked_work_request_trips_the_circuit_breaker():
    engine, _ = loop(actor(LEAKING_DRAFT), actor(LEAKING_DRAFT), actor(LEAKING_DRAFT))

    result = await engine.run(TICKET, CANDIDATES)

    assert result.draft is None, "this must never reach a supplier"
    assert result.attempts == 3
    assert "ESW#20033465" in " ".join(result.critiques)


async def test_the_failure_reason_says_which_gate_kept_failing():
    """A reviewer reading Reason needs to tell a hallucination apart from a
    reference carried forward: they want different follow-ups."""
    engine, _ = loop(actor(LEAKING_DRAFT), actor(LEAKING_DRAFT), actor(LEAKING_DRAFT))

    result = await engine.run(TICKET, CANDIDATES)

    assert "ESW#20033465" in result.failure_reason


async def test_the_actor_is_told_exactly_what_to_remove():
    engine, _ = loop(actor(LEAKING_DRAFT), actor(LEAKING_DRAFT), actor(LEAKING_DRAFT))

    result = await engine.run(TICKET, CANDIDATES)
    critique = result.critiques[0]

    assert "attachment" in critique.lower()
    assert "Per discussed" in critique
    # The instruction that undoes what CHECK 1 taught it.
    assert "does not make these transferable" in critique


async def test_a_rewrite_that_drops_the_references_passes():
    """The gate is a rewrite prompt, not only a refusal."""
    engine, client = loop(actor(LEAKING_DRAFT), actor(CLEAN_DRAFT), PASS)

    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert result.attempts == 2
    assert "ESW" not in result.draft.recommendation
    assert "attachment" not in result.draft.recommendation.lower()


async def test_the_rejected_draft_is_shown_to_the_actor_on_the_retry():
    engine, client = loop(actor(LEAKING_DRAFT), actor(CLEAN_DRAFT), PASS)

    await engine.run(TICKET, CANDIDATES)

    retry = client.calls[1][-1]["content"]
    assert "ESW#20033465" in retry, "the actor must see what it wrote"
    assert "AUDITOR CRITIQUE" in retry


async def test_a_clean_draft_is_untouched_by_the_gate():
    """The gate must not cost a round trip on text that was always fine."""
    engine, client = loop(actor(CLEAN_DRAFT), PASS)

    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert result.attempts == 1
    assert client.call_count == 2, "one Actor call, one Judge call"


# --------------------------------------------------------------------------
# Misattributed actions
#
# Two NTK tickets requested an ESW; both answers told the supplier to issue one.
# Issuance is the customer's action. The historical Solution_Text is an internal
# engineer's own to-do note, so its imperatives address the wrong party.
# --------------------------------------------------------------------------

MISATTRIBUTED_DRAFT = (
    "1. Issue an ESW.\n2. Do not ship the parts until the ESW is fully approved."
)
REPHRASED_DRAFT = (
    "1. An ESW has been requested.\n"
    "2. Do not ship the parts until the ESW is fully approved."
)


async def test_telling_the_supplier_to_issue_an_esw_is_rejected():
    engine, client = loop(
        actor(MISATTRIBUTED_DRAFT), actor(MISATTRIBUTED_DRAFT), actor(MISATTRIBUTED_DRAFT)
    )

    result = await engine.run(TICKET, CANDIDATES)

    assert result.draft is None
    # No Judge call was scripted, and none was needed: the gate is local.
    assert client.call_count == 3


async def test_the_critique_shows_the_required_rephrasing():
    engine, _ = loop(actor(MISATTRIBUTED_DRAFT), actor(REPHRASED_DRAFT), PASS)

    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert "An ESW has been requested" in result.critiques[0]
    assert "cannot grant them" in result.critiques[0]


async def test_rephrasing_as_an_awaited_outcome_passes():
    """The fix is a rewrite, not only a refusal -- and step 2 was always fine."""
    engine, _ = loop(actor(MISATTRIBUTED_DRAFT), actor(REPHRASED_DRAFT), PASS)

    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert "Do not ship the parts" in result.draft.recommendation
    assert "Issue an ESW" not in result.draft.recommendation


async def test_both_kinds_of_problem_cost_one_attempt_not_two():
    """A draft that leaks a reference AND misattributes an action gets one
    combined critique. Three attempts is not many to spend."""
    both = "1. Issue an ESW.\n2. Per discussed, see the attachment."
    engine, _ = loop(actor(both), actor(REPHRASED_DRAFT), PASS)

    result = await engine.run(TICKET, CANDIDATES)

    assert result.succeeded
    assert result.attempts == 2
    assert "attachment" in result.critiques[0].lower()
    assert "Issue an ESW" in result.critiques[0]
