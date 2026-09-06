"""Configuration loading.

Regression guard: these dataclasses use slots=True, where a class attribute is
a slot descriptor rather than the field default. from_env() must therefore read
defaults from module constants, never off `cls`.
"""

from __future__ import annotations

import pytest

from sps.config import (
    CONFIDENCE_THRESHOLD,
    MAX_ATTEMPTS,
    MIN_QUERY_LENGTH,
    TOP_K,
    IndexingSettings,
    Settings,
)


def test_settings_load_from_a_bare_environment(monkeypatch):
    for key in (
        "SPS_EMBEDDING_MODEL", "SPS_EMBEDDING_DIM", "SPS_EMBEDDING_DEVICE",
        "SPS_EMBEDDING_ENCODE_BATCH", "SPS_INDEX_BATCH_SIZE", "SPS_SOURCE_TABLE",
        "SPS_WATERMARK_PATH", "SPS_QDRANT_URL", "SPS_COLLECTION",
        "AZURE_OPENAI_API_VERSION", "SPS_LLM_TIMEOUT", "SPS_TOP_K",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings.from_env()

    assert settings.embedding.model_name == "BAAI/bge-large-en-v1.5"
    assert settings.embedding.dimension == 1024
    assert isinstance(settings.embedding.device, str)
    assert isinstance(settings.embedding.encode_batch_size, int)
    assert isinstance(settings.indexing.batch_size, int)
    assert isinstance(settings.vector_store.url, str)
    assert isinstance(settings.llm.request_timeout, float)


def test_spec_constants_are_wired_through():
    settings = Settings()
    assert settings.retrieval.top_k == TOP_K == 15
    assert settings.retrieval.confidence_threshold == CONFIDENCE_THRESHOLD == 0.75
    assert settings.retrieval.min_query_length == MIN_QUERY_LENGTH == 10
    assert settings.llm.max_attempts == MAX_ATTEMPTS == 3


def test_inference_settings_are_deterministic_by_default():
    llm = Settings().llm
    assert llm.temperature == 0.0
    assert llm.top_p == 0.1


def test_environment_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("SPS_EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("SPS_TOP_K", "25")
    monkeypatch.setenv("SPS_INDEX_BATCH_SIZE", "500")
    monkeypatch.setenv("SPS_LLM_TEMPERATURE", "0.2")

    settings = Settings.from_env()
    assert settings.embedding.model_name == "custom/model"
    assert settings.retrieval.top_k == 25
    assert settings.indexing.batch_size == 500
    assert settings.llm.temperature == 0.2


def test_out_of_band_batch_size_from_env_is_rejected(monkeypatch):
    monkeypatch.setenv("SPS_INDEX_BATCH_SIZE", "10000")
    with pytest.raises(ValueError, match="RAM-safe"):
        IndexingSettings.from_env()


@pytest.mark.parametrize("raw,expected", [("true", True), ("false", False), ("1", True), ("0", False)])
def test_boolean_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("SPS_USE_BGE_QUERY_INSTRUCTION", raw)
    assert Settings.from_env().embedding.use_query_instruction is expected
