"""Configuration loading.

Regression guard: these dataclasses use slots=True, where a class attribute is
a slot descriptor rather than the field default. from_env() must therefore read
defaults from module constants, never off `cls`.
"""

from __future__ import annotations

import os

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


# ------------------------------------------------------------- the .env load


def test_load_env_file_reads_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SPS_TEST_ONLY=from-the-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPS_TEST_ONLY", raising=False)

    from sps.config import load_env_file

    assert load_env_file() == tmp_path / ".env"
    assert os.environ["SPS_TEST_ONLY"] == "from-the-file"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """A machine-level setting must not be overridden by a stale checkout."""
    (tmp_path / ".env").write_text("SPS_TEST_ONLY=from-the-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPS_TEST_ONLY", "from-the-machine")

    from sps.config import load_env_file

    load_env_file()
    assert os.environ["SPS_TEST_ONLY"] == "from-the-machine"


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    """An absent .env is the normal case for a robot whose variables are set at
    machine level, so it returns None rather than raising."""
    monkeypatch.chdir(tmp_path)
    from sps.config import load_env_file

    # The project root is searched after the cwd, and a developer may well have
    # a .env there, so the contract is: never raise, and never name a path that
    # is not actually a file.
    found = load_env_file()
    assert found is None or found.exists()


@pytest.mark.parametrize("entry", ["run_resolver", "run_eval_batch", "verify_embedder"])
def test_every_entry_point_loads_the_env_file(entry, tmp_path, monkeypatch):
    """The bug this guards against: verify_embedder was written later than the
    others, did not know to call the resolver's private loader, and reported
    every credential missing on a machine whose .env was perfectly well filled
    in -- while the resolver read the same file without trouble.
    """
    import importlib

    import sps.config

    loaded = []
    monkeypatch.setattr(sps.config, "load_env_file", lambda *a, **k: loaded.append(entry))

    module = importlib.import_module(f"scripts.{entry}")
    absent = tmp_path / "absent.csv"
    empty = tmp_path / "empty"
    empty.mkdir()

    argv = {
        "run_resolver": ["--ticket-file", str(absent), "--history-file", str(absent),
                         "--output-dir", str(tmp_path / "out")],
        "run_eval_batch": ["--test-dir", str(empty), "--output-dir", str(tmp_path / "out")],
        "verify_embedder": [],
    }[entry]

    for name in ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY",
                 "AZURE_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)

    module.main(argv)
    assert loaded == [entry], f"scripts.{entry} did not load .env"
