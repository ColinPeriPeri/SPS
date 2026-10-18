"""Configuration loading.

Regression guard: these dataclasses use slots=True, where a class attribute is
a slot descriptor rather than the field default. from_env() must therefore read
defaults from module constants, never off `cls`.
"""

from __future__ import annotations

import os

import pytest

from sps import MAX_ATTEMPTS, LLMSettings


def test_settings_load_from_a_bare_environment(monkeypatch):
    for key in ("AZURE_OPENAI_API_VERSION", "SPS_LLM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)

    assert isinstance(LLMSettings.from_env().request_timeout, float)


def test_the_local_encoder_is_gone():
    """Removed rather than kept dormant: it needed torch to run at all, and its
    settings and thresholds were calibrations for a model nothing loads."""
    import sps

    for name in ("BGEEmbedder", "EmbeddingSettings", "BGE_QUERY_INSTRUCTION",
                 "LOCAL_EMBEDDING_THRESHOLD", "TIER2_LOCAL_THRESHOLD"):
        assert not hasattr(sps, name), name


def test_inference_settings_are_deterministic_by_default():
    llm = LLMSettings()
    assert llm.temperature == 0.0
    assert llm.top_p == 0.1
    assert llm.max_attempts == MAX_ATTEMPTS == 3


def test_environment_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("SPS_LLM_TEMPERATURE", "0.2")

    assert LLMSettings.from_env().temperature == 0.2


def test_credentials_are_never_defaulted():
    """A missing key must stay empty rather than acquire a placeholder."""
    llm = LLMSettings()
    assert llm.endpoint == "" and llm.api_key == "" and llm.deployment == ""


# ------------------------------------------------------------- the .env load


def test_load_env_file_reads_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SPS_TEST_ONLY=from-the-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPS_TEST_ONLY", raising=False)

    from sps import load_env_file

    assert load_env_file() == tmp_path / ".env"
    assert os.environ["SPS_TEST_ONLY"] == "from-the-file"


def test_the_project_env_file_is_found_from_any_directory(tmp_path, monkeypatch):
    """Run from a directory with no .env, and the project's own is still read.

    This is the fallback that broke when the package became one file: each
    module had located the project root by counting parents up from its own
    path, and moving the code a level up left the count pointing one directory
    too high. Nothing failed -- the file simply stopped being found, and every
    Azure credential read as missing.
    """
    import sps

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("SPS_TEST_ONLY=from-the-project\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.setattr(sps, "PROJECT_DIR", project)
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("SPS_TEST_ONLY", raising=False)

    assert sps.load_env_file() == project / ".env"
    assert os.environ["SPS_TEST_ONLY"] == "from-the-project"


def test_the_project_directory_is_where_sps_py_lives():
    """The fallback above is only right if PROJECT_DIR is. Pinned to the file's
    own location, so moving sps.py is the only thing that can move it."""
    from pathlib import Path

    import sps

    assert sps.PROJECT_DIR == Path(sps.__file__).resolve().parent
    assert (sps.PROJECT_DIR / "sps_prompts.py").exists()
    assert sps.env_file_candidates()[1] == sps.PROJECT_DIR / ".env"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """A machine-level setting must not be overridden by a stale checkout."""
    (tmp_path / ".env").write_text("SPS_TEST_ONLY=from-the-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPS_TEST_ONLY", "from-the-machine")

    from sps import load_env_file

    load_env_file()
    assert os.environ["SPS_TEST_ONLY"] == "from-the-machine"


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    """An absent .env is the normal case for a robot whose variables are set at
    machine level, so it returns None rather than raising."""
    monkeypatch.chdir(tmp_path)
    from sps import load_env_file

    # The project root is searched after the cwd, and a developer may well have
    # a .env there, so the contract is: never raise, and never name a path that
    # is not actually a file.
    found = load_env_file()
    assert found is None or found.exists()


@pytest.mark.parametrize("entry", ["resolve", "eval", "check"])
def test_every_entry_point_loads_the_env_file(entry, tmp_path, monkeypatch):
    """The bug this guards against: verify_embedder was written later than the
    others, did not know to call the resolver's private loader, and reported
    every credential missing on a machine whose .env was perfectly well filled
    in -- while the resolver read the same file without trouble.

    Driven through the real dispatcher, `sps.main([command, ...])`, so a
    subcommand that is wired up but skips the loader fails here too.
    """
    import sps

    loaded = []
    monkeypatch.setattr(sps, "load_env_file", lambda *a, **k: loaded.append(entry))

    absent = tmp_path / "absent.csv"
    empty = tmp_path / "empty"
    empty.mkdir()

    argv = {
        "resolve": ["--ticket-file", str(absent), "--history-file", str(absent),
                    "--output-dir", str(tmp_path / "out")],
        "eval": ["--test-dir", str(empty), "--output-dir", str(tmp_path / "out")],
        "check": [],
    }[entry]

    for name in ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY",
                 "AZURE_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)

    sps.main([entry, *argv])
    assert loaded == [entry], f"`sps.py {entry}` did not load .env"
