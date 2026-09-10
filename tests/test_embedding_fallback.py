"""Azure-primary / local-fallback embedding, and the dual threshold.

Three properties carry the risk here:

* vectors are never mixed between the two models;
* the local model is not loaded when Azure succeeds;
* the gate applies the threshold belonging to whichever model answered.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("numpy")
pytest.importorskip("openpyxl")

from sps.config import AzureEmbeddingSettings  # noqa: E402
from sps.contracts import IncomingTicket  # noqa: E402
from sps.embedding import AzureEmbedder, AzureEmbeddingError  # noqa: E402
from sps.retrieval.in_memory import (  # noqa: E402
    AZURE_BACKEND,
    AZURE_EMBEDDING_THRESHOLD,
    LOCAL_BACKEND,
    LOCAL_EMBEDDING_THRESHOLD,
    InMemoryRetriever,
)
from tests.conftest import TokenOverlapEmbedder  # noqa: E402
from tests.test_resolver import PART, PROBLEM, row, write_history  # noqa: E402

CONFIGURED = AzureEmbeddingSettings(
    endpoint="https://x.openai.azure.com/", api_key="k", deployment="text-embedding-3-small"
)


class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _Response:
    def __init__(self, data):
        self.data = data


class FakeAzureClient:
    """Mimics the SDK surface AzureEmbedder touches."""

    def __init__(self, vectors=None, error=None, shuffle=False, short=False):
        self.calls = []
        outer = self

        class _Embeddings:
            def create(self, *, model, input):
                outer.calls.append(list(input))
                if error is not None:
                    raise error
                data = [
                    _Item(i, vectors[i] if vectors else [1.0, 0.0, 0.0])
                    for i in range(len(input))
                ]
                if short:
                    data = data[:-1]
                if shuffle:
                    data = list(reversed(data))
                return _Response(data)

        self.embeddings = _Embeddings()


# ------------------------------------------------------------- AzureEmbedder


def test_vectors_are_normalised():
    client = FakeAzureClient(vectors=[[3.0, 4.0, 0.0]])
    out = AzureEmbedder(CONFIGURED, client=client).embed_batch(["x"])
    assert out[0] == pytest.approx([0.6, 0.8, 0.0])


def test_no_bge_prefix_is_added():
    """The instruction prefix is a bge convention. Sending it to Azure would
    inject a constant meaningless string into every query."""
    client = FakeAzureClient()
    AzureEmbedder(CONFIGURED, client=client).embed_batch(["weld seam cracking"])
    assert client.calls[0] == ["weld seam cracking"]
    assert "Represent this sentence" not in client.calls[0][0]


def test_response_is_reordered_by_index():
    """A silently reordered batch would pair every candidate with another
    candidate's score."""
    client = FakeAzureClient(vectors=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], shuffle=True)
    out = AzureEmbedder(CONFIGURED, client=client).embed_batch(["a", "b", "c"])
    assert out == [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]


def test_a_short_response_is_an_error_not_a_silent_mismatch():
    client = FakeAzureClient(short=True)
    with pytest.raises(AzureEmbeddingError, match="Expected 3 embeddings"):
        AzureEmbedder(CONFIGURED, client=client).embed_batch(["a", "b", "c"])


@pytest.mark.parametrize(
    "error",
    [ConnectionError("connection reset"), TimeoutError("timed out"),
     RuntimeError("429 rate limit exceeded"), PermissionError("401 invalid key")],
)
def test_every_transport_failure_becomes_one_error_type(error):
    client = FakeAzureClient(error=error)
    with pytest.raises(AzureEmbeddingError):
        AzureEmbedder(CONFIGURED, client=client).embed_batch(["a"])


def test_unconfigured_deployment_names_what_is_missing():
    with pytest.raises(AzureEmbeddingError) as info:
        AzureEmbedder(AzureEmbeddingSettings()).embed_batch(["a"])
    assert "AZURE_EMBEDDING_ENDPOINT" in str(info.value)


def test_credentials_never_appear_in_the_error():
    settings = AzureEmbeddingSettings(
        endpoint="https://x/", api_key="sk-super-secret", deployment="d"
    )
    client = FakeAzureClient(error=RuntimeError("boom"))
    with pytest.raises(AzureEmbeddingError) as info:
        AzureEmbedder(settings, client=client).embed_batch(["a"])
    assert "sk-super-secret" not in str(info.value)


# ------------------------------------------------------------------ fallback


def _retriever(tmp_path, monkeypatch, *, azure_client=None, azure_settings=CONFIGURED, rows=None):
    """Retriever whose Azure path is served by a fake, tracking local builds."""
    built = []

    def factory():
        built.append(1)
        return TokenOverlapEmbedder()

    if azure_client is not None:
        import sps.embedding as embedding_module

        real = embedding_module.AzureEmbedder
        monkeypatch.setattr(
            embedding_module,
            "AzureEmbedder",
            lambda settings, _real=real: _real(settings, client=azure_client),
        )

    retriever = InMemoryRetriever(
        history_path=write_history(tmp_path / "h.xlsx", rows or [row("SPS-1"), row("SPS-2", minutes=1)]),
        local_embedder_factory=factory,
        azure_settings=azure_settings,
    )
    return retriever, built


def test_azure_success_never_builds_the_local_model(tmp_path, monkeypatch):
    """The whole point of the primary path: no ~15 s torch cold start."""
    client = FakeAzureClient(vectors=[[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    retriever, built = _retriever(tmp_path, monkeypatch, azure_client=client)

    retriever.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))

    assert built == [], "local embedder must not be constructed when Azure answers"
    assert retriever.stats.backend == AZURE_BACKEND
    assert retriever.stats.backend_detail == "text-embedding-3-small"


def test_azure_sends_query_and_candidates_in_one_request(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0, 0]] * 3)
    retriever, _ = _retriever(tmp_path, monkeypatch, azure_client=client)

    retriever.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))

    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent[0] == PROBLEM          # query first, so its position is known
    assert len(sent) == 3              # query + two candidates


def test_azure_failure_falls_back_and_logs_the_marker(tmp_path, monkeypatch, caplog):
    client = FakeAzureClient(error=ConnectionError("connection reset by peer"))
    retriever, built = _retriever(tmp_path, monkeypatch, azure_client=client)

    with caplog.at_level(logging.WARNING):
        candidates = retriever.retrieve(
            IncomingTicket(problem_description=PROBLEM, part_number=PART)
        )

    assert "AZURE_EMBEDDING_FAILED_FALLING_BACK" in caplog.text
    assert built == [1], "fallback must build the local model exactly once"
    assert retriever.stats.backend == LOCAL_BACKEND
    assert "connection reset" in retriever.stats.fallback_reason
    assert candidates, "the fallback must still produce a usable result"


def test_unconfigured_azure_falls_back_without_a_request(tmp_path, monkeypatch, caplog):
    retriever, built = _retriever(
        tmp_path, monkeypatch, azure_settings=AzureEmbeddingSettings()
    )
    with caplog.at_level(logging.WARNING):
        retriever.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))

    assert "AZURE_EMBEDDING_FAILED_FALLING_BACK" in caplog.text
    assert "AZURE_EMBEDDING_ENDPOINT" in retriever.stats.fallback_reason
    assert built == [1]
    assert retriever.stats.backend == LOCAL_BACKEND


def test_partial_azure_output_is_never_mixed_with_local(tmp_path, monkeypatch):
    """A short response is discarded entirely; the batch is re-encoded locally
    so query and candidates share one embedding space."""
    client = FakeAzureClient(short=True)
    retriever, built = _retriever(tmp_path, monkeypatch, azure_client=client)

    candidates = retriever.retrieve(
        IncomingTicket(problem_description=PROBLEM, part_number=PART)
    )

    assert retriever.stats.backend == LOCAL_BACKEND
    assert built == [1]
    # A local exact match scores 1.0. A mixed-space cosine could not.
    assert candidates[0].cosine_similarity == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------- dual threshold


def test_azure_run_uses_the_azure_threshold(tmp_path, monkeypatch):
    # Query and candidates deliberately ~0.6 apart: above 0.50, below 0.89.
    client = FakeAzureClient(vectors=[[1.0, 0.0], [0.6, 0.8], [0.6, 0.8]])
    retriever, _ = _retriever(tmp_path, monkeypatch, azure_client=client)

    candidates = retriever.retrieve(
        IncomingTicket(problem_description=PROBLEM, part_number=PART)
    )

    assert retriever.stats.threshold_used == AZURE_EMBEDDING_THRESHOLD == 0.50
    assert candidates, "0.6 clears the Azure gate"
    # The same score would have been rejected by the local threshold.
    assert retriever.stats.top_score < LOCAL_EMBEDDING_THRESHOLD


def test_fallback_run_uses_the_local_threshold(tmp_path, monkeypatch):
    client = FakeAzureClient(error=TimeoutError("timed out"))
    retriever, _ = _retriever(tmp_path, monkeypatch, azure_client=client)

    retriever.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))

    assert retriever.stats.threshold_used == LOCAL_EMBEDDING_THRESHOLD == 0.89


def test_an_explicit_override_beats_both(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0.0], [0.6, 0.8], [0.6, 0.8]])
    retriever, _ = _retriever(tmp_path, monkeypatch, azure_client=client)
    retriever.confidence_threshold = 0.95

    candidates = retriever.retrieve(
        IncomingTicket(problem_description=PROBLEM, part_number=PART)
    )

    assert retriever.stats.threshold_used == 0.95
    assert candidates == []


def test_stats_record_the_backend_for_the_status_sheet(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0, 0]] * 3)
    retriever, _ = _retriever(tmp_path, monkeypatch, azure_client=client)
    retriever.retrieve(IncomingTicket(problem_description=PROBLEM, part_number=PART))

    report = retriever.stats.as_dict()
    assert report["backend"] == "azure"
    assert report["backend_detail"] == "text-embedding-3-small"
    assert report["fallback_reason"] == ""
    assert report["threshold_used"] == 0.50
