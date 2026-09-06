"""End-to-end: every exit path must emit the Section 4 JSON contract."""

from __future__ import annotations

import json

import pytest

# Component C is schema-constrained: the Actor and Judge responses are
# validated through pydantic models, so pydantic is a hard requirement here.
# Components A and B stay dependency-free.
pytest.importorskip("pydantic")

from sps.config import LLMSettings, RetrievalSettings, Settings
from sps.contracts import SOLUTION_NOT_FOUND, IncomingTicket, VectorPoint, score_to_percent
from sps.generation import ActorCriticLoop
from sps.pipeline import SPSPipeline
from sps.retrieval import Retriever
from sps.vectorstore import InMemoryVectorStore
from tests.conftest import ScriptedChatClient, TokenOverlapEmbedder

PROBLEM = "Bracket weld seam cracking observed during incoming inspection"

CONTRACT_KEYS = {"AI_Recommendation", "Justification", "Confidence", "SPS_IDs_Referred"}


def assert_valid_contract(payload: dict) -> None:
    """The schema guarantee every caller depends on."""
    assert set(payload) == CONTRACT_KEYS
    assert isinstance(payload["AI_Recommendation"], str) and payload["AI_Recommendation"]
    assert isinstance(payload["Justification"], str) and payload["Justification"]
    assert isinstance(payload["Confidence"], str)
    assert payload["Confidence"].endswith("%")
    assert payload["Confidence"][:-1].isdigit()
    assert isinstance(payload["SPS_IDs_Referred"], list)
    assert all(isinstance(i, str) for i in payload["SPS_IDs_Referred"])
    json.dumps(payload)  # must be serialisable as-is


def build(responses, problems=None, settings=None):
    embedder = TokenOverlapEmbedder()
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dimension)

    problems = problems if problems is not None else {"SPS-100": PROBLEM}
    if problems:
        store.upsert(
            [
                VectorPoint(
                    sps_id=sps_id,
                    vector=embedder.embed_passages([text])[0],
                    payload={
                        "sps_id": sps_id,
                        "actual_solution": "Rework the weld seam and re-inspect.",
                        "part_number": "PN-1000",
                        "part_description": "Mounting bracket",
                        "item_status": "Active",
                        "problem_reason_code": "RC-WELD",
                        "issue_type": "Quality",
                    },
                )
                for sps_id, text in problems.items()
            ]
        )

    settings = settings or Settings(retrieval=RetrievalSettings(), llm=LLMSettings())
    client = ScriptedChatClient(responses)
    return SPSPipeline(
        retriever=Retriever(embedder, store, settings.retrieval),
        loop=ActorCriticLoop(client, settings.llm),
        settings=settings,
    ), client


def actor(recommendation, justification="Drawn from SPS-100, the same weld defect."):
    return json.dumps({"recommendation": recommendation, "justification": justification})


PASS = json.dumps({"status": "PASS"})
FAIL = json.dumps({"status": "FAIL", "critique": "Invented a torque specification."})


# --------------------------------------------------------------------------


async def test_successful_recommendation():
    pipeline, client = build([actor("1. Rework the weld seam. 2. Re-inspect."), PASS])
    payload = await pipeline.process_dict({"problem_description": PROBLEM, "part_number": "PN-1000"})

    assert_valid_contract(payload)
    assert payload["AI_Recommendation"] == "1. Rework the weld seam. 2. Re-inspect."
    assert payload["SPS_IDs_Referred"] == ["SPS-100"]
    assert payload["Confidence"] == "100%"


async def test_invalid_input_payload_matches_the_spec_exactly():
    pipeline, client = build([])
    payload = await pipeline.process_dict({"problem_description": "short"})

    assert payload == {
        "AI_Recommendation": "Solution not found.",
        "Justification": "Invalid problem statement.",
        "Confidence": "0%",
        "SPS_IDs_Referred": [],
    }
    assert client.call_count == 0  # no LLM spend on a malformed ticket


async def test_below_threshold_payload_matches_the_spec_exactly():
    pipeline, client = build(
        [], problems={"SPS-200": "Carton label misprint on the outer packaging"}
    )
    payload = await pipeline.process_dict({"problem_description": PROBLEM})

    assert_valid_contract(payload)
    assert payload["AI_Recommendation"] == SOLUTION_NOT_FOUND
    assert payload["Justification"] == "Confidence below 75% threshold."
    assert payload["SPS_IDs_Referred"] == []
    assert int(payload["Confidence"][:-1]) < 75
    assert client.call_count == 0  # the gate fires before any generation


async def test_circuit_breaker_falls_back_to_solution_not_found():
    pipeline, client = build(
        [actor("d1"), FAIL, actor("d2"), FAIL, actor("d3"), FAIL]
    )
    payload = await pipeline.process_dict({"problem_description": PROBLEM})

    assert_valid_contract(payload)
    assert payload["AI_Recommendation"] == SOLUTION_NOT_FOUND
    assert payload["SPS_IDs_Referred"] == []
    assert "3 attempts" in payload["Justification"]
    assert client.call_count == 6


async def test_empty_index_returns_the_failure_contract():
    pipeline, _ = build([], problems={})
    payload = await pipeline.process_dict({"problem_description": PROBLEM})

    assert_valid_contract(payload)
    assert payload["AI_Recommendation"] == SOLUTION_NOT_FOUND
    assert payload["Confidence"] == "0%"


async def test_unexpected_error_still_yields_the_contract():
    class Exploding:
        dimension = 128

        def embed_query(self, text):
            raise RuntimeError("model crashed")

    pipeline, _ = build([])
    pipeline.retriever.embedder = Exploding()
    payload = await pipeline.process_dict({"problem_description": PROBLEM})

    assert_valid_contract(payload)
    assert payload["AI_Recommendation"] == SOLUTION_NOT_FOUND


async def test_multiple_qualifying_records_are_all_cited():
    pipeline, _ = build(
        [actor("1. Rework the weld seam."), PASS],
        problems={"SPS-100": PROBLEM, "SPS-101": PROBLEM},
    )
    payload = await pipeline.process_dict({"problem_description": PROBLEM})

    assert sorted(payload["SPS_IDs_Referred"]) == ["SPS-100", "SPS-101"]


async def test_confidence_never_reports_the_threshold_it_failed():
    """A 0.7499 score must render as 74%, never as a misleading 75%."""
    assert score_to_percent(0.7499) == 74
    assert score_to_percent(0.75) == 75
    assert score_to_percent(1.0) == 100
    assert score_to_percent(-0.2) == 0


@pytest.mark.parametrize(
    "ticket",
    [
        {"problem_description": ""},
        {"problem_description": None},
        {},
        {"problem_description": "   \n  "},
    ],
)
async def test_malformed_tickets_never_raise(ticket):
    pipeline, _ = build([])
    payload = await pipeline.process_dict(ticket)
    assert_valid_contract(payload)
    assert payload["Justification"] == "Invalid problem statement."


async def test_result_object_and_contract_agree():
    pipeline, _ = build([actor("1. Rework."), PASS])
    result = await pipeline.process(IncomingTicket(problem_description=PROBLEM))

    assert result.succeeded
    assert result.to_contract()["AI_Recommendation"] == result.ai_recommendation
