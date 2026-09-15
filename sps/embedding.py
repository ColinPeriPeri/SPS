"""BAAI/bge-large-en-v1.5 embedding wrapper (local, CPU-only).

sentence-transformers/torch are imported lazily so the rest of the package --
and its tests -- import cleanly on a machine without the ML stack.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

from .config import BGE_QUERY_INSTRUCTION, EmbeddingSettings


@runtime_checkable
class Embedder(Protocol):
    """Minimal surface the pipeline needs; lets tests inject a fake."""

    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single incoming problem statement for retrieval."""


class BGEEmbedder:
    """sentence-transformers implementation pinned to CPU inference.

    Vectors are L2-normalized at encode time, which makes the vector DB's
    cosine similarity a plain dot product and keeps scores in [-1, 1].
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or EmbeddingSettings()
        self.dimension = self.settings.dimension
        self._model = None  # loaded on first use (~1.3 GB resident)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.settings.model_name,
                device=self.settings.device,
            )
            actual = self._model.get_sentence_embedding_dimension()
            if actual != self.dimension:
                raise ValueError(
                    f"{self.settings.model_name} produced {actual}-dim vectors, "
                    f"expected {self.dimension}"
                )
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        # convert_to_tensor, not convert_to_numpy: the torch->numpy bridge is an
        # ABI coupling that breaks outright ("Numpy is not available") whenever
        # the installed numpy major version differs from the one torch was built
        # against. The output is converted to plain lists for the vector DB
        # either way, so routing through numpy buys nothing and adds a
        # dependency-resolution failure mode to the hot path.
        vectors = self.model.encode(
            list(texts),
            batch_size=self.settings.encode_batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return vectors.detach().cpu().tolist()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Index side: the raw Problem_Description only.

        Per Component A.4 nothing is appended -- no metadata, no solution text,
        and (per BGE's own convention) no instruction prefix on passages.
        """
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Query side: optional BGE retrieval instruction prefix.

        The prefix is how bge-*-v1.5 was trained to encode queries and lifts
        recall measurably. It is a fixed encoding instruction, not record
        metadata, so Component A.4's "no appended metadata" rule is unaffected.
        Disable with SPS_USE_BGE_QUERY_INSTRUCTION=false to embed queries and
        passages identically.
        """
        prepared = (
            f"{BGE_QUERY_INSTRUCTION}{text}"
            if self.settings.use_query_instruction
            else text
        )
        return self._encode([prepared])[0]

    def unload(self) -> None:
        """Release model memory -- used by the indexer on shutdown."""
        self._model = None


class AzureEmbeddingError(RuntimeError):
    """The Azure embedding call failed.

    With the local encoder out of the pipeline there is nothing to fall back
    to, so this now ends the run. It stays a single base class because most
    callers only need "embedding did not happen".
    """


class AzureEmbeddingNotConfigured(AzureEmbeddingError):
    """Credentials or the deployment name are missing.

    A subclass, so `except AzureEmbeddingError` still catches it, but the
    resolver checks for it first and exits 2 rather than 1. A timeout is worth
    retrying; an absent API key is not, and a robot retrying one fifty times
    only delays the human who has to go and set it.
    """


class AzureEmbedder:
    """Azure OpenAI embeddings, used as the primary encoder.

    Two differences from the local BGE path that matter for correctness:

    * **No instruction prefix.** "Represent this sentence for searching relevant
      passages: " is a convention `bge-*-v1.5` was trained with. Azure's
      embedding models were not, so prepending it here would inject a constant
      meaningless string and shift every query vector for no benefit.
    * **Vectors are normalised defensively.** OpenAI returns unit-length
      embeddings today, but the ranking is a bare dot product that silently
      stops being a cosine if that ever changes. Normalising costs nothing at
      this size and makes the guarantee ours rather than the vendor's.
    """

    def __init__(self, settings, client=None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self):
        if self._client is None:
            # Configuration first, import second. Importing openai can fail on
            # its own -- a blocked native dependency, a broken install -- and
            # reporting that when the real problem is an unset API key sends
            # the reader somewhere entirely wrong.
            missing = self.settings.missing()
            if missing:
                raise AzureEmbeddingNotConfigured(
                    f"Azure embedding deployment is not configured: {', '.join(missing)}"
                )

            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.settings.endpoint,
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
                timeout=self.settings.request_timeout,
            )
        return self._client

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text in one request, in the order given.

        Any failure raises AzureEmbeddingError and the whole batch is discarded.
        Partial results are never returned: a half-embedded batch would leave
        some candidates scored and some not, which ranks as though the missing
        ones were simply poor matches.
        """
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.settings.deployment,
                input=list(texts),
            )
        except AzureEmbeddingError:
            raise
        except Exception as exc:
            # Network, auth, timeout, rate limit, bad deployment name -- all of
            # them mean the same thing here: this run cannot be embedded.
            raise AzureEmbeddingError(
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # The API documents order preservation, but the response also carries an
        # index per item. Sorting on it makes the guarantee explicit rather than
        # assumed -- a silently reordered batch would pair every candidate with
        # the wrong similarity score.
        items = sorted(response.data, key=lambda d: d.index)
        if len(items) != len(texts):
            raise AzureEmbeddingError(
                f"Expected {len(texts)} embeddings, received {len(items)}"
            )
        return [_unit(item.embedding) for item in items]


def _unit(vector: Sequence[float]) -> list[float]:
    """L2-normalise, so a dot product is a cosine."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]
