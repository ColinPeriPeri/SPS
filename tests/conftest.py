"""Test doubles for the embedder and the chat client.

Neither the ML stack nor Azure is required to exercise the resolver.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Sequence

import pytest


DIM = 128
_TOKEN = re.compile(r"[a-z0-9]+")


class TokenOverlapEmbedder:
    """Deterministic stand-in for BGE.

    Hashes tokens into a fixed-width space and L2-normalizes, so cosine
    similarity tracks token overlap: identical text scores 1.0, disjoint text
    scores 0.0. That makes retrieval assertions exact without a real model.
    """

    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall((text or "").lower()):
            vector[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passage_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)


class ScriptedChatClient:
    """Chat client that replays a fixed list of JSON responses.

    The loop alternates Actor / Judge calls, so a script reads as
    [actor_1, judge_1, actor_2, judge_2, ...].
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: Sequence[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedChatClient ran out of scripted responses")
        return self.responses.pop(0)

    async def complete_model(self, messages: Sequence[dict[str, str]], model):
        """Scripts stay plain JSON strings and are validated through the same
        pydantic models the real client is constrained by, so a script that the
        deployment could not have produced fails here too."""
        from sps.generation.llm import validate_json

        return validate_json(await self.complete(messages), model)

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def embedder() -> TokenOverlapEmbedder:
    return TokenOverlapEmbedder()


@pytest.fixture(autouse=True)
def isolated_0250_corpus(tmp_path_factory, monkeypatch):
    """Point Tier 2 at an empty folder for every test that does not set its own.

    Without this the default corpus path is resolved against the working
    directory, so a deployment that had actually loaded its 0250 documents
    would change the outcome of tests written about Tier 1 -- passing or
    failing depending on what was in a folder the test never mentions.
    """
    monkeypatch.setenv(
        "SPS_0250_DOCS_DIR", str(tmp_path_factory.mktemp("no_0250_docs"))
    )
