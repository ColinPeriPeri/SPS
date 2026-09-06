"""BAAI/bge-large-en-v1.5 embedding wrapper (local, CPU-only).

sentence-transformers/torch are imported lazily so the rest of the package --
and its tests -- import cleanly on a machine without the ML stack.
"""

from __future__ import annotations

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
