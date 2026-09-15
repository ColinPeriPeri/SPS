"""Azure embeddings: the only encoder in the pipeline.

The local bge-small fallback is disabled. Its code is intact in
sps/embedding.py, but nothing in the retrievers reaches it, so the properties
that carry the risk here have changed shape:

* a failure ends the run instead of being absorbed -- silently answering from a
  different embedding space, under a different threshold, while still reporting
  success, is the outcome that is now impossible;
* missing credentials are distinguishable from a transient fault, because one
  is worth retrying and the other never will be;
* the Azure threshold is the one that applies, always.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("openpyxl")

from sps.config import AzureEmbeddingSettings  # noqa: E402
from sps.contracts import IncomingTicket  # noqa: E402
from sps.embedding import (  # noqa: E402
    AzureEmbedder,
    AzureEmbeddingError,
    AzureEmbeddingNotConfigured,
)
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


# ---------------------------------------------------------- the only encoder


def _retriever(tmp_path, monkeypatch, *, azure_client=None, azure_settings=CONFIGURED,
               rows=None):
    """Retriever whose Azure path is served by a fake."""
    if azure_client is not None:
        import sps.embedding as embedding_module

        real = embedding_module.AzureEmbedder
        monkeypatch.setattr(
            embedding_module,
            "AzureEmbedder",
            lambda settings, _real=real: _real(settings, client=azure_client),
        )

    return InMemoryRetriever(
        history_path=write_history(
            tmp_path / "h.xlsx", rows or [row("SPS-1"), row("SPS-2", minutes=1)]
        ),
        azure_settings=azure_settings,
    )


def _ticket():
    return IncomingTicket(problem_description=PROBLEM, part_number=PART)


def test_azure_answers_and_names_itself(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)

    retriever.retrieve(_ticket())

    assert retriever.stats.backend == AZURE_BACKEND
    assert retriever.stats.backend_detail == "text-embedding-3-small"


def test_query_and_candidates_go_in_one_request(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0, 0]] * 3)
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)

    retriever.retrieve(_ticket())

    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent[0] == PROBLEM          # query first, so its position is known
    assert len(sent) == 3              # query + two candidates


def test_a_transport_failure_ends_the_run(tmp_path, monkeypatch):
    """There is nothing to absorb it into any more. Previously this fell back
    to bge-small and the ticket still reported a result -- scored in a
    different embedding space, against a different threshold, with only the
    Embedding_Model column to say so."""
    client = FakeAzureClient(error=ConnectionError("connection reset by peer"))
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)

    with pytest.raises(AzureEmbeddingError, match="connection reset"):
        retriever.retrieve(_ticket())


def test_missing_credentials_are_a_distinct_error(tmp_path, monkeypatch):
    """A subclass, so `except AzureEmbeddingError` still catches it, but the
    resolver can tell it apart and exit 2 rather than asking a robot to retry
    its way to an API key."""
    retriever = _retriever(tmp_path, monkeypatch, azure_settings=AzureEmbeddingSettings())

    with pytest.raises(AzureEmbeddingNotConfigured) as info:
        retriever.retrieve(_ticket())

    assert issubclass(AzureEmbeddingNotConfigured, AzureEmbeddingError)
    assert "AZURE_EMBEDDING_ENDPOINT" in str(info.value)


def test_no_request_is_made_when_nothing_is_configured(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0, 0]] * 3)
    retriever = _retriever(
        tmp_path, monkeypatch, azure_client=client, azure_settings=AzureEmbeddingSettings()
    )

    with pytest.raises(AzureEmbeddingNotConfigured):
        retriever.retrieve(_ticket())

    assert client.calls == []


def test_a_short_response_ends_the_run(tmp_path, monkeypatch):
    """A partial batch would leave some candidates scored and some not, which
    ranks exactly as though the missing ones were poor matches."""
    client = FakeAzureClient(short=True)
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)

    with pytest.raises(AzureEmbeddingError, match="Expected"):
        retriever.retrieve(_ticket())


def test_a_ticket_with_no_history_never_reaches_azure(tmp_path, monkeypatch):
    """Order of work is cheapest-first: an unknown part costs a file scan and
    stops, without spending a request."""
    client = FakeAzureClient(vectors=[[1.0, 0, 0]] * 3)
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)

    assert retriever.retrieve(
        IncomingTicket(problem_description=PROBLEM, part_number="0099-99999")
    ) == []
    assert client.calls == []


# ------------------------------------------------------------- the threshold


def test_the_azure_threshold_is_the_one_that_applies(tmp_path, monkeypatch):
    # Query and candidates deliberately ~0.6 apart: above 0.50, below 0.89.
    client = FakeAzureClient(vectors=[[1.0, 0.0], [0.6, 0.8], [0.6, 0.8]])
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)

    candidates = retriever.retrieve(_ticket())

    assert retriever.stats.threshold_used == AZURE_EMBEDDING_THRESHOLD == 0.50
    assert candidates, "0.6 clears the Azure gate"
    # The same score would have been rejected by the local threshold, which is
    # precisely why a silent fallback was dangerous.
    assert retriever.stats.top_score < LOCAL_EMBEDDING_THRESHOLD


def test_an_explicit_override_beats_the_backend_default(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0.0], [0.6, 0.8], [0.6, 0.8]])
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)
    retriever.confidence_threshold = 0.95

    assert retriever.retrieve(_ticket()) == []
    assert retriever.stats.threshold_used == 0.95


def test_the_local_threshold_is_dormant_not_deleted():
    """It stays beside the engine so re-wiring the local model is a code change
    in one place, not a re-measurement."""
    assert LOCAL_EMBEDDING_THRESHOLD == 0.89
    fields = InMemoryRetriever.__dataclass_fields__
    assert fields["local_threshold"].default == 0.89
    # Reachable only through an injected encoder, which the pipeline never sets.
    assert "local_embedder_factory" not in fields


def test_an_injected_encoder_still_uses_the_local_threshold(tmp_path):
    """The seam the local model would be wired back through."""
    retriever = InMemoryRetriever(
        history_path=write_history(tmp_path / "h.xlsx", [row("SPS-1")]),
        embedder=TokenOverlapEmbedder(),
    )

    retriever.retrieve(_ticket())

    assert retriever.stats.backend == LOCAL_BACKEND
    assert retriever.stats.threshold_used == LOCAL_EMBEDDING_THRESHOLD


def test_stats_record_the_backend_for_the_status_sheet(tmp_path, monkeypatch):
    client = FakeAzureClient(vectors=[[1.0, 0, 0]] * 3)
    retriever = _retriever(tmp_path, monkeypatch, azure_client=client)
    retriever.retrieve(_ticket())

    report = retriever.stats.as_dict()
    assert report["backend"] == "azure"
    assert report["backend_detail"] == "text-embedding-3-small"
    assert report["threshold_used"] == 0.50
