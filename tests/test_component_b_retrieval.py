"""Component B -- input validation, metadata boosting, confidence gate."""

from __future__ import annotations

import pytest

from sps.config import RetrievalSettings
from sps.contracts import IncomingTicket, SearchHit, VectorPoint
from sps.retrieval import RetrievalStatus, Retriever, composite_score, rank_candidates
from sps.retrieval.scoring import ISSUE_TYPE_BOOST, MAX_BOOST, REASON_CODE_BOOST
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


def test_part_number_is_no_longer_scored():
    """It is a hard filter on the search, so every candidate reaching scoring
    already matches it. Scoring it too would add a constant to every result and
    silently soften the confidence gate by 0.05."""
    score, applied = composite_score(hit(0.50, part_number="PN-1000"), TICKET)
    assert score == pytest.approx(0.50)
    assert applied == ()


def test_each_boost_is_applied_at_its_specified_weight():
    score, applied = composite_score(hit(0.50, issue_type="Quality"), TICKET)
    assert score == pytest.approx(0.50 + ISSUE_TYPE_BOOST)

    score, applied = composite_score(hit(0.50, problem_reason_code="RC-WELD"), TICKET)
    assert score == pytest.approx(0.50 + REASON_CODE_BOOST)


def test_boosts_stack_to_the_full_five_points():
    score, applied = composite_score(
        hit(0.60, part_number="PN-1000", issue_type="Quality", problem_reason_code="RC-WELD"),
        TICKET,
    )
    assert MAX_BOOST == pytest.approx(0.05)
    assert score == pytest.approx(0.65)
    assert set(applied) == {"issue_type", "problem_reason_code"}


def test_no_boost_without_a_match():
    score, applied = composite_score(hit(0.60, issue_type="Packaging"), TICKET)
    assert score == pytest.approx(0.60)
    assert applied == ()


def test_two_blank_fields_do_not_count_as_a_match():
    """Both sides missing a Part_Number is absence of evidence, not a match."""
    blank_ticket = IncomingTicket(problem_description="a" * 20)  # all metadata empty
    score, applied = composite_score(hit(0.60), blank_ticket)
    assert score == pytest.approx(0.60)
    assert applied == ()


def test_matching_ignores_case_and_surrounding_whitespace():
    score, _ = composite_score(hit(0.50, issue_type="  quality "), TICKET)
    assert score == pytest.approx(0.50 + ISSUE_TYPE_BOOST)


def test_score_is_clamped_to_one():
    score, _ = composite_score(
        hit(0.99, issue_type="Quality", problem_reason_code="RC-WELD"), TICKET
    )
    assert score == 1.0  # never reports above 100%


def test_negative_cosine_is_floored_at_zero():
    score, _ = composite_score(hit(-0.4, issue_type="Quality"), TICKET)
    assert score == pytest.approx(ISSUE_TYPE_BOOST)


def test_boosting_reorders_the_candidate_list():
    hits = [
        hit(0.80, sps_id="A"),                       # 0.80, no metadata match
        hit(0.77, sps_id="B",                        # 0.77 + 0.05 = 0.82
            issue_type="Quality", problem_reason_code="RC-WELD"),
    ]
    ranked = rank_candidates(hits, TICKET)

    assert [c.sps_id for c in ranked] == ["B", "A"]
    assert ranked[0].composite_score == pytest.approx(0.82)


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
            # Matches TICKET.part_number: the store now hard-filters on it, so a
            # blank here would exclude every record before scoring and these
            # tests would be asserting the filter rather than the gate.
            "part_number": "PN-1000",
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
    """A 0.73 cosine match on the same issue type and reason code clears 0.75."""
    hits = [hit(0.73, sps_id="A", issue_type="Quality", problem_reason_code="RC-WELD")]
    ranked = rank_candidates(hits, TICKET)
    assert ranked[0].composite_score == pytest.approx(0.78)
    assert ranked[0].composite_score >= 0.75


# --------------------------------------------------------------------------
# B.2 -- hard part_number filter on the vector search
# --------------------------------------------------------------------------


def _mixed_parts_store(embedder):
    """Same problem text under three different part numbers."""
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dimension)
    store.upsert(
        [
            VectorPoint(
                sps_id=sps_id,
                vector=embedder.embed_passages([TICKET.problem_description])[0],
                payload={
                    "sps_id": sps_id,
                    "content_hash": "0" * 64,
                    "actual_solution": f"Historical fix for {sps_id}",
                    "part_number": part,
                    "part_description": "",
                    "item_status": "Active",
                    "problem_reason_code": "",
                    "issue_type": "Quality",
                },
            )
            for sps_id, part in (("SAME", "PN-1000"), ("OTHER", "PN-2000"), ("BLANK", ""))
        ]
    )
    return store


def test_search_returns_only_the_requested_part(embedder):
    store = _mixed_parts_store(embedder)
    vector = embedder.embed_query(TICKET.problem_description)

    hits = store.search(vector, limit=15, part_number="PN-1000")
    assert [h.payload["sps_id"] for h in hits] == ["SAME"]


def test_blank_part_number_applies_no_filter(embedder):
    """A ticket with no part number must still reach the whole index, not be
    pinned to records whose part number is also blank."""
    store = _mixed_parts_store(embedder)
    vector = embedder.embed_query(TICKET.problem_description)

    for blank in (None, "", "   "):
        hits = store.search(vector, limit=15, part_number=blank)
        assert len(hits) == 3, f"{blank!r} should not filter"


def test_filter_is_exact_not_prefix_or_substring(embedder):
    store = _mixed_parts_store(embedder)
    vector = embedder.embed_query(TICKET.problem_description)

    assert store.search(vector, limit=15, part_number="PN-100") == []
    assert store.search(vector, limit=15, part_number="PN") == []


def test_filter_is_case_insensitive_via_normalisation(embedder):
    """Part numbers are canonicalised at ingest and on the query, so a
    lower-case ticket must not report NO_MATCHES for an indexed part."""
    store = _mixed_parts_store(embedder)
    vector = embedder.embed_query(TICKET.problem_description)

    for spelling in ("PN-1000", "pn-1000", "  Pn-1000  "):
        hits = store.search(vector, limit=15, part_number=spelling)
        assert [h.payload["sps_id"] for h in hits] == ["SAME"], spelling


def test_surrounding_whitespace_on_the_part_number_is_tolerated(embedder):
    store = _mixed_parts_store(embedder)
    vector = embedder.embed_query(TICKET.problem_description)
    hits = store.search(vector, limit=15, part_number="  PN-1000  ")
    assert [h.payload["sps_id"] for h in hits] == ["SAME"]


def test_retriever_passes_the_tickets_part_number_down(embedder):
    seen = {}

    class RecordingStore(InMemoryVectorStore):
        def search(self, vector, limit, part_number=None):
            seen["part_number"] = part_number
            return super().search(vector, limit, part_number)

    store = RecordingStore()
    store.ensure_collection(embedder.dimension)
    Retriever(embedder, store).retrieve(TICKET)
    assert seen["part_number"] == "PN-1000"


def test_retriever_sends_none_when_the_ticket_has_no_part_number(embedder):
    seen = {}

    class RecordingStore(InMemoryVectorStore):
        def search(self, vector, limit, part_number=None):
            seen["part_number"] = part_number
            return super().search(vector, limit, part_number)

    store = RecordingStore()
    store.ensure_collection(embedder.dimension)
    Retriever(embedder, store).retrieve(
        IncomingTicket(problem_description=TICKET.problem_description)
    )
    assert seen["part_number"] is None


def test_a_perfect_text_match_on_the_wrong_part_is_excluded(embedder):
    """The point of the change: identical problem text under a different part
    number must not be offered as precedent."""
    store = _mixed_parts_store(embedder)
    outcome = Retriever(embedder, store).retrieve(
        IncomingTicket(problem_description=TICKET.problem_description, part_number="PN-2000")
    )
    assert [c.sps_id for c in outcome.candidates] == ["OTHER"]


def test_no_history_for_the_part_reports_no_matches(embedder):
    """A new part with no precedent is now a common outcome, not just an
    empty index."""
    store = _mixed_parts_store(embedder)
    outcome = Retriever(embedder, store).retrieve(
        IncomingTicket(problem_description=TICKET.problem_description, part_number="PN-NEW")
    )
    assert outcome.status is RetrievalStatus.NO_MATCHES
    assert outcome.candidates == []


def test_records_with_a_blank_part_number_are_unreachable_when_filtering(embedder):
    """Documented consequence: history that never recorded a part number can no
    longer be retrieved by any ticket that supplies one."""
    store = _mixed_parts_store(embedder)
    vector = embedder.embed_query(TICKET.problem_description)

    assert [h.payload["sps_id"] for h in store.search(vector, limit=15, part_number="PN-1000")] == [
        "SAME"
    ]
    assert "BLANK" not in [
        h.payload["sps_id"] for h in store.search(vector, limit=15, part_number="PN-1000")
    ]


def test_confidence_is_pure_cosine_when_only_the_part_matches(embedder):
    """The gate is no longer softened by a boost that fired on everything: a
    part-filtered candidate with no other metadata match scores its raw cosine."""
    store = _mixed_parts_store(embedder)
    outcome = Retriever(embedder, store).retrieve(
        IncomingTicket(problem_description=TICKET.problem_description, part_number="PN-1000")
    )

    assert outcome.candidates
    assert all(c.applied_boosts == () for c in outcome.candidates)
    assert outcome.top_score == pytest.approx(outcome.candidates[0].cosine_similarity)


# --------------------------------------------------------------------------
# Ticket payload keys and part-number canonicalisation
# --------------------------------------------------------------------------


SQL_STYLE = {
    "SPS_ID": "SPS-999",
    "Problem_Description": "Weld seam cracking on the mounting bracket",
    "Part_Number": "PN-123",
    "Issue_Type": "Quality",
    "Problem_Reason_Code": "RC-WELD",
}


def test_sql_style_payload_keys_are_accepted():
    """Every other interface names fields the SQL way, so a hand-written ticket
    naturally uses Problem_Description. Reading only the lower-case spelling
    produced an empty ticket that exited 0 looking like a real refusal."""
    ticket = IncomingTicket.from_dict(SQL_STYLE)

    assert ticket.problem_description == "Weld seam cracking on the mounting bracket"
    assert ticket.sps_id == "SPS-999"
    assert ticket.part_number == "PN-123"
    assert ticket.issue_type == "Quality"
    assert ticket.problem_reason_code == "RC-WELD"


def test_lowercase_payload_keys_still_work():
    ticket = IncomingTicket.from_dict(
        {"problem_description": "Weld seam cracking", "part_number": "PN-123"}
    )
    assert ticket.problem_description == "Weld seam cracking"
    assert ticket.part_number == "PN-123"


def test_payload_keys_ignore_spacing_and_mixed_case():
    ticket = IncomingTicket.from_dict(
        {"problem description": "Weld seam cracking", "PART_NUMBER": "PN-123"}
    )
    assert ticket.problem_description == "Weld seam cracking"
    assert ticket.part_number == "PN-123"


def test_a_sql_style_ticket_is_not_silently_judged_invalid():
    """The exact smoke-test payload: 13 characters, over the 10-char minimum,
    which used to be reported as an invalid problem statement."""
    ticket = IncomingTicket.from_dict(
        {"SPS_ID": "SPS-999", "Problem_Description": "Weld cracking", "Part_Number": "PN-123"}
    )
    retriever = Retriever(TokenOverlapEmbedder(), InMemoryVectorStore())
    assert retriever.validate(ticket)


def test_incoming_part_numbers_are_canonicalised():
    for spelling in ("PN-123", "pn-123", "  Pn-123  "):
        assert IncomingTicket(problem_description="a" * 20, part_number=spelling).part_number == (
            "PN-123"
        )


def test_ingested_part_numbers_are_canonicalised():
    from sps.contracts import SourceRecord

    record = SourceRecord(
        sps_id="S1",
        problem_description="a" * 30,
        actual_solution="b" * 30,
        part_number="  pn-123 ",
    )
    assert record.part_number == "PN-123"
    assert record.payload()["part_number"] == "PN-123"


def test_canonicalisation_makes_both_sides_agree(embedder):
    """The point of normalising at ingest and on the query: a lower-case ticket
    finds an upper-case indexed record."""
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dimension)
    store.upsert(
        [
            VectorPoint(
                sps_id="S1",
                vector=embedder.embed_passages([TICKET.problem_description])[0],
                payload={
                    "sps_id": "S1",
                    "content_hash": "0" * 64,
                    "actual_solution": "Rework the weld seam and re-inspect.",
                    "part_number": "PN-1000",
                    "part_description": "",
                    "item_status": "Active",
                    "problem_reason_code": "",
                    "issue_type": "",
                },
            )
        ]
    )

    outcome = Retriever(embedder, store).retrieve(
        IncomingTicket.from_dict(
            {"Problem_Description": TICKET.problem_description, "Part_Number": "pn-1000"}
        )
    )
    assert [c.sps_id for c in outcome.candidates] == ["S1"]
