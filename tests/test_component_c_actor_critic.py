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
