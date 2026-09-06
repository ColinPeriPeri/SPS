"""Component B -- input validation, metadata boosting, confidence gate."""

from __future__ import annotations

import pytest

from sps.config import RetrievalSettings
from sps.contracts import IncomingTicket, SearchHit, VectorPoint
from sps.retrieval import RetrievalStatus, Retriever, composite_score, rank_candidates
from sps.retrieval.scoring import ISSUE_TYPE_BOOST, PART_NUMBER_BOOST, REASON_CODE_BOOST
from sps.vectorstore import InMemoryVectorStore
from tests.conftest import TokenOverlapEmbedder

TICKET = IncomingTicket(
    problem_description="Bracket weld seam cracking observed during incoming inspection",
    sps_id="T-1",
    part_number="PN-1000",
    issue_type="Quality",
    problem_reason_code="RC-WELD",
)


def hit(similarity: float, **payload) -> SearchHit:
    base = {
        "sps_id": payload.pop("sps_id", "H1"),
        "actual_solution": "Rework the weld seam and re-inspect",
        "part_number": "",
        "part_description": "",
        "item_status": "",
        "problem_reason_code": "",
        "issue_type": "",
    }
    base.update(payload)
    return SearchHit(payload=base, cosine_similarity=similarity)


# --------------------------------------------------------------------------
# B.1 -- input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "short", "123456789"])
def test_rejects_problem_statements_under_ten_characters(text):
    retriever = Retriever(TokenOverlapEmbedder(), InMemoryVectorStore())
    outcome = retriever.retrieve(IncomingTicket(problem_description=text))

    assert outcome.status is RetrievalStatus.INVALID_INPUT
    assert outcome.candidates == []


def test_accepts_exactly_ten_characters():
    retriever = Retriever(TokenOverlapEmbedder(), InMemoryVectorStore())
    assert retriever.validate(IncomingTicket(problem_description="1234567890"))


def test_invalid_input_never_touches_the_vector_store():
    embedder = TokenOverlapEmbedder()
    Retriever(embedder, InMemoryVectorStore()).retrieve(IncomingTicket(problem_description="no"))
    assert embedder.query_calls == []  # aborted before embedding


# --------------------------------------------------------------------------
# B.3 -- metadata boosting arithmetic
# --------------------------------------------------------------------------


def test_each_boost_is_applied_at_its_specified_weight():
    score, applied = composite_score(hit(0.50, part_number="PN-1000"), TICKET)
    assert score == pytest.approx(0.50 + PART_NUMBER_BOOST)
    assert applied == ("part_number",)

    score, applied = composite_score(hit(0.50, issue_type="Quality"), TICKET)
    assert score == pytest.approx(0.50 + ISSUE_TYPE_BOOST)

    score, applied = composite_score(hit(0.50, problem_reason_code="RC-WELD"), TICKET)
    assert score == pytest.approx(0.50 + REASON_CODE_BOOST)


def test_boosts_stack_to_the_full_ten_points():
    score, applied = composite_score(
        hit(0.60, part_number="PN-1000", issue_type="Quality", problem_reason_code="RC-WELD"),
        TICKET,
    )
    assert score == pytest.approx(0.70)
    assert set(applied) == {"part_number", "issue_type", "problem_reason_code"}


def test_no_boost_without_a_match():
    score, applied = composite_score(hit(0.60, part_number="PN-9999"), TICKET)
    assert score == pytest.approx(0.60)
    assert applied == ()


def test_two_blank_fields_do_not_count_as_a_match():
    """Both sides missing a Part_Number is absence of evidence, not a match."""
    blank_ticket = IncomingTicket(problem_description="a" * 20)  # all metadata empty
    score, applied = composite_score(hit(0.60), blank_ticket)
    assert score == pytest.approx(0.60)
    assert applied == ()


def test_matching_ignores_case_and_surrounding_whitespace():
    score, _ = composite_score(hit(0.50, part_number="  pn-1000 "), TICKET)
    assert score == pytest.approx(0.50 + PART_NUMBER_BOOST)


def test_score_is_clamped_to_one():
    score, _ = composite_score(
        hit(0.99, part_number="PN-1000", issue_type="Quality", problem_reason_code="RC-WELD"),
        TICKET,
    )
    assert score == 1.0  # never reports above 100%


def test_negative_cosine_is_floored_at_zero():
    score, _ = composite_score(hit(-0.4, part_number="PN-1000"), TICKET)
    assert score == pytest.approx(PART_NUMBER_BOOST)


def test_boosting_reorders_the_candidate_list():
    hits = [
        hit(0.80, sps_id="A"),                        # 0.80, no metadata match
        hit(0.74, sps_id="B", part_number="PN-1000",  # 0.74 + 0.10 = 0.84
            issue_type="Quality", problem_reason_code="RC-WELD"),
    ]
    ranked = rank_candidates(hits, TICKET)

    assert [c.sps_id for c in ranked] == ["B", "A"]
    assert ranked[0].composite_score == pytest.approx(0.84)


def test_ranking_is_deterministic_on_ties():
    hits = [hit(0.80, sps_id="Z"), hit(0.80, sps_id="A"), hit(0.80, sps_id="M")]
    assert [c.sps_id for c in rank_candidates(hits, TICKET)] == ["Z", "M", "A"]


# --------------------------------------------------------------------------
# B.2 / B.4 -- retrieval and the 75% gate
# --------------------------------------------------------------------------


def _store_with(problems: dict[str, str], embedder, **payload_overrides):
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dimension)
    points = []
    for sps_id, problem in problems.items():
        payload = {
            "sps_id": sps_id,
            "actual_solution": f"Historical fix for {sps_id}",
            "part_number": "",
            "part_description": "",
            "item_status": "Active",
            "problem_reason_code": "",
            "issue_type": "",
        }
        payload.update(payload_overrides)
        points.append(
            VectorPoint(sps_id=sps_id, vector=embedder.embed_passages([problem])[0], payload=payload)
        )
    store.upsert(points)
    return store


def test_retrieves_at_most_top_k_candidates():
    embedder = TokenOverlapEmbedder()
    store = _store_with(
        {f"S{i}": f"Bracket weld seam cracking variant {i}" for i in range(40)}, embedder
    )
    retriever = Retriever(embedder, store, RetrievalSettings(confidence_threshold=0.0))
    outcome = retriever.retrieve(TICKET)

    assert len(outcome.candidates) <= 15


def test_gate_blocks_generation_below_threshold():
    embedder = TokenOverlapEmbedder()
    store = _store_with({"S1": "Completely unrelated packaging label misprint"}, embedder)
    outcome = Retriever(embedder, store).retrieve(TICKET)

    assert outcome.status is RetrievalStatus.BELOW_THRESHOLD
    assert outcome.candidates == []          # nothing is handed to the LLM
    assert outcome.top_score < 0.75


def test_gate_passes_a_strong_match():
    embedder = TokenOverlapEmbedder()
    store = _store_with({"S1": TICKET.problem_description}, embedder)
    outcome = Retriever(embedder, store).retrieve(TICKET)

    assert outcome.status is RetrievalStatus.OK
    assert outcome.top_score == pytest.approx(1.0)
    assert [c.sps_id for c in outcome.candidates] == ["S1"]


def test_only_candidates_at_or_above_threshold_reach_the_actor():
    embedder = TokenOverlapEmbedder()
    store = _store_with(
        {
            "STRONG": TICKET.problem_description,
            "WEAK": "Unrelated packaging label misprint on the outer carton",
        },
        embedder,
    )
    outcome = Retriever(embedder, store).retrieve(TICKET)

    assert [c.sps_id for c in outcome.candidates] == ["STRONG"]


def test_empty_index_reports_no_matches():
    embedder = TokenOverlapEmbedder()
    outcome = Retriever(embedder, InMemoryVectorStore()).retrieve(TICKET)
    assert outcome.status is RetrievalStatus.NO_MATCHES


def test_boost_can_lift_a_candidate_through_the_gate():
    """A 0.73 cosine match on the same part number clears 0.75."""
    hits = [hit(0.73, sps_id="A", part_number="PN-1000", issue_type="Quality")]
    ranked = rank_candidates(hits, TICKET)
    assert ranked[0].composite_score == pytest.approx(0.81)
    assert ranked[0].composite_score >= 0.75
