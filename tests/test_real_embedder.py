"""The real BAAI/bge-small-en-v1.5 model.

Opt-in: these load ~130 MB of weights, so they are skipped unless
SPS_MODEL_TESTS=1 is set. Everything else in the suite uses a deterministic
stand-in.

    SPS_MODEL_TESTS=1 python -m pytest tests/test_real_embedder.py -q

The properties pinned here are the ones the pipeline's arithmetic assumes:
384 dimensions, unit-length vectors (so the NumPy matmul ranking is the cosine
the 0.89 gate is calibrated on), and the BGE query instruction applied to
queries but never to the history passages.
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
from sps.embedding import BGEEmbedder  # noqa: E402

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


def test_model_produces_384_dimensions(embedder):
    assert embedder.model.get_sentence_embedding_dimension() == 384
    assert embedder.dimension == 384


def test_passage_and_query_vectors_are_384_dim(embedder):
    passages = embedder.embed_passages([PROBLEM, UNRELATED])
    assert [len(v) for v in passages] == [384, 384]
    assert len(embedder.embed_query(PROBLEM)) == 384


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
    """The assumption behind the matmul ranking and the 0.89 gate."""
    a, b = embedder.embed_passages([PROBLEM, PARAPHRASE])
    assert abs(dot(a, b) - cosine(a, b)) < TOL


def test_self_similarity_is_one(embedder):
    vector = embedder.embed_passages([PROBLEM])[0]
    assert cosine(vector, vector) == pytest.approx(1.0, abs=TOL)


# ----------------------------------------------------------- query instruction


def test_prefix_is_actually_applied_to_queries(embedder, plain_embedder):
    assert cosine(embedder.embed_query(PROBLEM), plain_embedder.embed_query(PROBLEM)) < 0.9999


def test_passages_never_receive_the_prefix(embedder, plain_embedder):
    """The embedded string is the Problem_Description alone."""
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

    assert cosine(query, exact) >= 0.89
    assert cosine(query, unrelated) < 0.89


# ---------------------------------------------------------------- determinism


def test_encoding_is_deterministic(embedder):
    first = embedder.embed_query(PROBLEM)
    second = embedder.embed_query(PROBLEM)
    assert all(abs(a - b) < 1e-9 for a, b in zip(first, second))


def test_batching_does_not_change_vectors(embedder):
    """A record must embed identically alone or mid-batch -- otherwise the
    candidate batch boundary would perturb its score."""
    alone = embedder.embed_passages([PROBLEM])[0]
    in_batch = embedder.embed_passages([UNRELATED, PROBLEM, PARAPHRASE])[1]
    assert cosine(alone, in_batch) > 1 - TOL
