"""The real BAAI/bge-large-en-v1.5 model.

Opt-in: these load ~1.3 GB of weights, so they are skipped unless
SPS_MODEL_TESTS=1 is set. Everything else in the suite uses a deterministic
stand-in.

    SPS_MODEL_TESTS=1 python -m pytest tests/test_real_embedder.py -q

The properties pinned here are the ones the pipeline's arithmetic assumes:
1024 dimensions, unit-length vectors (so the vector DB's cosine distance is a
plain dot product and the 0.75 gate is calibrated on cosine), and the BGE query
instruction applied to queries but never to indexed passages.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest

if not os.environ.get("SPS_MODEL_TESTS"):
    pytest.skip("set SPS_MODEL_TESTS=1 to run the real-model tests", allow_module_level=True)

pytest.importorskip("sentence_transformers")

from sps.config import BGE_QUERY_INSTRUCTION, EmbeddingSettings  # noqa: E402
from sps.contracts import IncomingTicket  # noqa: E402
from sps.embedding import BGEEmbedder  # noqa: E402
from sps.retrieval import RetrievalStatus, Retriever  # noqa: E402
from sps.vectorstore import InMemoryVectorStore  # noqa: E402
from sps.contracts import VectorPoint  # noqa: E402

TOL = 1e-5

PROBLEM = "Bracket weld seam cracking observed during incoming inspection"
PARAPHRASE = "Cracks found in the weld seam of the mounting bracket at goods-in"
UNRELATED = "Outer carton label misprinted on the shipment packaging"


def l2(vector) -> float:
    return math.sqrt(sum(v * v for v in vector))


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b) -> float:
    na, nb = l2(a), l2(b)
    return dot(a, b) / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def embedder() -> BGEEmbedder:
    model = BGEEmbedder(EmbeddingSettings())
    _ = model.model  # load once for the whole module
    return model


@pytest.fixture(scope="module")
def plain_embedder(embedder) -> BGEEmbedder:
    """Same weights, query-instruction prefix disabled."""
    other = BGEEmbedder(replace(embedder.settings, use_query_instruction=False))
    other._model = embedder.model
    return other


# ------------------------------------------------------------------ dimensions


def test_model_produces_1024_dimensions(embedder):
    assert embedder.model.get_sentence_embedding_dimension() == 1024
    assert embedder.dimension == 1024


def test_passage_and_query_vectors_are_1024_dim(embedder):
    passages = embedder.embed_passages([PROBLEM, UNRELATED])
    assert [len(v) for v in passages] == [1024, 1024]
    assert len(embedder.embed_query(PROBLEM)) == 1024


def test_dimension_mismatch_is_caught_at_load():
    wrong = BGEEmbedder(replace(EmbeddingSettings(), dimension=768))
    with pytest.raises(ValueError, match="expected 768"):
        _ = wrong.model


# --------------------------------------------------------------- normalization


def test_vectors_are_unit_length(embedder):
    for vector in embedder.embed_passages([PROBLEM, PARAPHRASE, UNRELATED]):
        assert abs(l2(vector) - 1.0) < TOL
    assert abs(l2(embedder.embed_query(PROBLEM)) - 1.0) < TOL


def test_dot_product_equals_cosine(embedder):
    """The assumption behind Qdrant COSINE distance and the 0.75 gate."""
    a, b = embedder.embed_passages([PROBLEM, PARAPHRASE])
    assert abs(dot(a, b) - cosine(a, b)) < TOL


def test_self_similarity_is_one(embedder):
    vector = embedder.embed_passages([PROBLEM])[0]
    assert cosine(vector, vector) == pytest.approx(1.0, abs=TOL)


# ----------------------------------------------------------- query instruction


def test_prefix_is_actually_applied_to_queries(embedder, plain_embedder):
    assert cosine(embedder.embed_query(PROBLEM), plain_embedder.embed_query(PROBLEM)) < 0.9999


def test_passages_never_receive_the_prefix(embedder, plain_embedder):
    """Component A.4: the embedded string is the Problem_Description alone."""
    passage = embedder.embed_passages([PROBLEM])[0]
    unprefixed_query = plain_embedder.embed_query(PROBLEM)
    assert cosine(passage, unprefixed_query) > 1 - TOL


def test_prefix_constant_matches_the_bge_convention():
    assert BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "


def test_disabling_the_prefix_makes_query_and_passage_identical(plain_embedder):
    query = plain_embedder.embed_query(PROBLEM)
    passage = plain_embedder.embed_passages([PROBLEM])[0]
    assert cosine(query, passage) > 1 - TOL


# ------------------------------------------------------- retrieval calibration


def test_related_text_outranks_unrelated_text(embedder):
    query = embedder.embed_query(PROBLEM)
    exact, paraphrase, unrelated = embedder.embed_passages([PROBLEM, PARAPHRASE, UNRELATED])

    assert cosine(query, exact) > cosine(query, paraphrase) > cosine(query, unrelated)


def test_gate_admits_an_exact_match_and_blocks_an_unrelated_one(embedder):
    query = embedder.embed_query(PROBLEM)
    exact, unrelated = embedder.embed_passages([PROBLEM, UNRELATED])

    assert cosine(query, exact) >= 0.75
    assert cosine(query, unrelated) < 0.75


def test_end_to_end_retrieval_with_the_real_model(embedder):
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dimension)
    corpus = {"SPS-1": PROBLEM, "SPS-2": UNRELATED}
    store.upsert(
        [
            VectorPoint(
                sps_id=sps_id,
                vector=embedder.embed_passages([text])[0],
                payload={
                    "sps_id": sps_id,
                    "content_hash": "0" * 64,
                    "actual_solution": "Rework the weld seam and re-inspect.",
                    "part_number": "PN-1000",
                    "part_description": "Mounting bracket",
                    "item_status": "Active",
                    "problem_reason_code": "RC-WELD",
                    "issue_type": "Quality",
                },
            )
            for sps_id, text in corpus.items()
        ]
    )

    outcome = Retriever(embedder, store).retrieve(
        IncomingTicket(problem_description=PARAPHRASE, part_number="PN-1000")
    )
    assert outcome.status is RetrievalStatus.OK
    assert outcome.candidates[0].sps_id == "SPS-1"


# ---------------------------------------------------------------- determinism


def test_encoding_is_deterministic(embedder):
    first = embedder.embed_query(PROBLEM)
    second = embedder.embed_query(PROBLEM)
    assert all(abs(a - b) < 1e-9 for a, b in zip(first, second))


def test_batching_does_not_change_vectors(embedder):
    """A record must embed identically whether it lands alone or mid-batch --
    otherwise a micro-batch boundary would perturb the index."""
    alone = embedder.embed_passages([PROBLEM])[0]
    in_batch = embedder.embed_passages([UNRELATED, PROBLEM, PARAPHRASE])[1]
    assert cosine(alone, in_batch) > 1 - TOL
