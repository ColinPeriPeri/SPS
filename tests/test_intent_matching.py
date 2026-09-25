"""Intent scoring, and the reason-code gate on Tier 2.

Tier 1 no longer writes anything. It scores, and above the gate it sends the
matched record's solution exactly as recorded. That makes the score the only
decision in the path, so these tests are mostly about the ways a score can be
wrong or missing, and what the pipeline does about each.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("pydantic")

from sps.config import IntentSettings  # noqa: E402
from sps.contracts import Candidate, IncomingTicket  # noqa: E402
from sps.generation.intent import IntentOutcome, score_intent  # noqa: E402
from sps.generation.llm import LLMError  # noqa: E402
from sps.generation.prompts import build_intent_messages  # noqa: E402
from tests.conftest import ScriptedChatClient  # noqa: E402

TICKET = IncomingTicket(
    problem_description=(
        "Lot received against PO 45001. During incoming inspection the bracket "
        "weld seam was found cracked. We have no experience with this rework "
        "and request approval to ship these units as-is."
    ),
    sps_id="T-1",
    part_number="PN-1000",
    problem_reason_code="RC-EQUIV",
)


def candidate(sps_id, problem="Weld seam cracking at goods-in", solution="Rework it.", score=0.8):
    return Candidate(
        sps_id=sps_id,
        actual_solution=solution,
        problem_description=problem,
        part_number="PN-1000",
        part_description="Bracket",
        item_status="Active",
        problem_reason_code="RC-EQUIV",
        issue_type="Quality",
        cosine_similarity=score,
        composite_score=score,
    )


CANDIDATES = [candidate("SPS-100"), candidate("SPS-101", score=0.7)]


def assessment(intent="Requests use-as-is approval", **scores):
    return json.dumps({
        "ticket_intent": intent,
        "scores": [
            {"sps_id": k, "intent_match": v, "reason": "because"} for k, v in scores.items()
        ],
    })


_DEFAULT = object()


def run(*responses, candidates=_DEFAULT):
    # A sentinel, not `candidates or CANDIDATES`. An empty list is falsy and
    # would silently become the default -- which is exactly the case the
    # no-candidates test is trying to exercise.
    client = ScriptedChatClient(list(responses))
    if candidates is _DEFAULT:
        candidates = CANDIDATES
    return asyncio.run(score_intent(client, TICKET, candidates)), client


# ------------------------------------------------------------ what it returns


def test_scores_come_back_scaled_to_the_same_range_as_everything_else():
    """0-100 from the model, 0..1 internally. Every other score in the codebase
    is a fraction, and one field in percent would eventually be compared
    against one that is not."""
    outcome, _ = run(assessment(**{"SPS-100": 82, "SPS-101": 40}))

    assert outcome.best.intent_match == pytest.approx(0.82)
    assert outcome.best.percent == 82


def test_the_best_match_is_first_whatever_order_the_model_answered_in():
    outcome, _ = run(assessment(**{"SPS-101": 91, "SPS-100": 12}))

    assert outcome.best.candidate.sps_id == "SPS-101"
    assert [s.candidate.sps_id for s in outcome.scored] == ["SPS-101", "SPS-100"]


def test_the_ticket_intent_is_kept():
    """Recorded so a wrong match can be traced to a wrong reading, rather than
    being an unexplained number."""
    outcome, _ = run(assessment(intent="Requests use-as-is", **{"SPS-100": 80, "SPS-101": 10}))

    assert outcome.ticket_intent == "Requests use-as-is"


def test_a_tie_is_broken_by_retrieval_score_not_by_luck():
    """The scorer works in whole percents across five candidates, so ties are
    common. Which record gets sent to a supplier must not depend on dict
    ordering."""
    outcome, _ = run(assessment(**{"SPS-101": 80, "SPS-100": 80}))

    # SPS-100 retrieved higher (0.8 against 0.7), so it wins the tie.
    assert outcome.best.candidate.sps_id == "SPS-100"


# --------------------------------------------------------------- what it costs


def test_no_candidates_means_no_call():
    """A ticket with no history costs nothing, as it did before."""
    outcome, client = run(candidates=[])

    assert client.call_count == 0
    assert outcome.best is None
    assert not outcome.failed


def test_every_candidate_is_scored_in_one_call():
    _, client = run(assessment(**{"SPS-100": 80, "SPS-101": 10}))

    assert client.call_count == 1


# ------------------------------------------------------------- failing closed


def test_an_unreachable_scorer_is_a_failure_not_an_empty_result():
    """The distinction the whole error path turns on. "Nothing matched" is a
    business outcome the robot files; "we could not tell" is an outage it
    should retry. Collapsing them turns every outage into a silent refusal."""
    outcome, _ = run("not json at all")

    assert outcome.failed
    assert outcome.best is None


def test_no_usable_scores_is_also_a_failure():
    """A response that parsed but named nothing we supplied leaves us with no
    judgement at all -- which is not the same as a low one."""
    outcome, _ = run(assessment(**{"SPS-999": 90}))

    assert outcome.failed


def test_a_score_for_a_record_we_never_sent_is_dropped():
    """The SPS ID is the only thing tying a score to a solution, so one we did
    not supply cannot be matched to anything and must not be guessed at."""
    outcome, _ = run(assessment(**{"SPS-100": 80, "SPS-999": 99}))

    assert [s.candidate.sps_id for s in outcome.scored] == ["SPS-100"]
    assert not outcome.failed


@pytest.mark.parametrize("value, expected", [(150, 1.0), (-20, 0.0)])
def test_an_out_of_range_score_is_clamped(value, expected):
    """Pydantic bounds it, but the clamp stays: a score out of range must never
    become a confidence above 100% on a sheet someone reads."""
    from sps.generation.intent import ScoredCandidate

    scored = ScoredCandidate(
        candidate=CANDIDATES[0],
        intent_match=min(max(value, 0), 100) / 100.0,
        reason="",
    )
    assert scored.intent_match == expected


# ------------------------------------------------------------------ the prompt


def test_the_scorer_is_shown_problems_and_never_solutions():
    """Rule 1, enforced structurally rather than asked for. It is deciding
    which past PROBLEM asks the same question; showing it the answers would
    invite it to score their usefulness instead -- and a record whose problem
    matches perfectly but whose solution is boilerplate must still score high,
    because "the closest precedent is empty" is a finding."""
    messages = build_intent_messages(TICKET, [candidate("SPS-100", solution="SECRET FIX")])
    blob = json.dumps(messages)

    assert "SECRET FIX" not in blob
    assert "Weld seam cracking at goods-in" in blob
    assert "SPS-100" in blob


def test_the_prompt_teaches_the_distinction_it_has_to_draw():
    """Two tickets, same defect, opposite requests. That pair is the whole
    reason this step exists, so it is in the prompt as a worked example."""
    from sps.generation.prompts import INTENT_SYSTEM_PROMPT

    assert "as-is" in INTENT_SYSTEM_PROMPT
    assert "re-engrave" in INTENT_SYSTEM_PROMPT
    assert "Score the PROBLEM, never the solution" in INTENT_SYSTEM_PROMPT


# -------------------------------------------------------- the reason-code gate


@pytest.mark.parametrize(
    "configured, code, allowed",
    [
        ("RC-1,RC-2", "RC-1", True),
        ("RC-1,RC-2", "RC-2", True),
        ("RC-1,RC-2", "RC-3", False),
        ("RC-1,RC-2", "", False),
        # Case and padding are the two ways a hand-edited .env goes wrong.
        ("rc-1", "RC-1", True),
        ("RC-1", "  rc-1  ", True),
        (" RC-1 , RC-2 ", "RC-2", True),
        # Nothing configured means nothing allowed -- not everything allowed.
        ("", "RC-1", False),
        ("", "", False),
    ],
)
def test_the_reason_code_gate(monkeypatch, configured, code, allowed):
    monkeypatch.setenv("SPS_TIER2_REASON_CODES", configured)
    assert IntentSettings.from_env().tier2_allowed(code) is allowed


def test_an_empty_list_is_read_as_none_configured(monkeypatch):
    """The direction of the default matters. Reading an absent list as "all
    codes" would mean a mistyped variable name silently removes the
    restriction; reading it as "none" means a lost line silently disables the
    tier. The second is the safer failure, and the resolver says which it hit."""
    monkeypatch.delenv("SPS_TIER2_REASON_CODES", raising=False)
    settings = IntentSettings.from_env()

    assert settings.tier2_reason_codes == ()
    assert not settings.tier2_allowed("anything")


def test_the_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("SPS_INTENT_THRESHOLD", "0.9")
    assert IntentSettings.from_env().threshold == 0.9


def test_the_threshold_has_a_default(monkeypatch):
    monkeypatch.delenv("SPS_INTENT_THRESHOLD", raising=False)
    assert IntentSettings.from_env().threshold == 0.75
