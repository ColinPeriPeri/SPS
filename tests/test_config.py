"""Configuration loading.

Regression guard: these dataclasses use slots=True, where a class attribute is
a slot descriptor rather than the field default. from_env() must therefore read
defaults from module constants, never off `cls`.
"""

from __future__ import annotations

import pytest

from sps.config import MAX_ATTEMPTS, EmbeddingSettings, LLMSettings


def test_settings_load_from_a_bare_environment(monkeypatch):
    for key in ("SPS_EMBEDDING_MODEL", "SPS_EMBEDDING_DIM", "SPS_EMBEDDING_DEVICE",
                "SPS_EMBEDDING_ENCODE_BATCH", "AZURE_OPENAI_API_VERSION", "SPS_LLM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)

    embedding = EmbeddingSettings.from_env()
    assert embedding.model_name == "BAAI/bge-small-en-v1.5"
    assert embedding.dimension == 384
    assert isinstance(embedding.device, str)
    assert isinstance(embedding.encode_batch_size, int)
    assert isinstance(LLMSettings.from_env().request_timeout, float)


def test_the_local_model_settings_survive_being_out_of_the_pipeline():
    """The bge-small defaults stay single-sourced in config even though nothing
    reaches them: the resolver no longer names a local model at all, so config
    is now the only place that does."""
    import scripts.run_resolver as resolver

    assert EmbeddingSettings().model_name == "BAAI/bge-small-en-v1.5"
    assert EmbeddingSettings().dimension == 384
    # The resolver used to re-declare both to build its fallback factory.
    assert not hasattr(resolver, "DEFAULT_MODEL")


def test_inference_settings_are_deterministic_by_default():
    llm = LLMSettings()
    assert llm.temperature == 0.0
    assert llm.top_p == 0.1
    assert llm.max_attempts == MAX_ATTEMPTS == 3


def test_environment_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("SPS_EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("SPS_EMBEDDING_DIM", "512")
    monkeypatch.setenv("SPS_LLM_TEMPERATURE", "0.2")

    assert EmbeddingSettings.from_env().model_name == "custom/model"
    assert EmbeddingSettings.from_env().dimension == 512
    assert LLMSettings.from_env().temperature == 0.2


@pytest.mark.parametrize("raw,expected", [("true", True), ("false", False), ("1", True), ("0", False)])
def test_boolean_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("SPS_USE_BGE_QUERY_INSTRUCTION", raw)
    assert EmbeddingSettings.from_env().use_query_instruction is expected


def test_credentials_are_never_defaulted():
    """A missing key must stay empty rather than acquire a placeholder."""
    llm = LLMSettings()
    assert llm.endpoint == "" and llm.api_key == "" and llm.deployment == ""
