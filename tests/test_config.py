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


def _env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


# The rules of the file itself. It is read with the standard library to
# python-dotenv 1.0.1's rules, and each expected value below is what
# python-dotenv produced for the same text, compared side by side before it
# left the install.
@pytest.mark.parametrize("text, expected", [
    ("A=1\nexport B = two words  \n", {"A": "1", "B": "two words"}),
    ("A=abc # a comment\nB=abc#not-a-comment\n", {"A": "abc", "B": "abc#not-a-comment"}),
    ("A='kept # in quotes'\nB=\"tab\\there\"\nC='no \\t escape'\n",
     {"A": "kept # in quotes", "B": "tab\there", "C": "no \\t escape"}),
    ('A="first\nsecond"\nB=after\n', {"A": "first\nsecond", "B": "after"}),
    ("# a comment\n\nA\nB=\n", {"A": None, "B": ""}),
    ("A=1\nA=2\n", {"A": "2"}),
])
def test_env_file_rules(tmp_path, text, expected):
    from sps import read_env_file

    assert read_env_file(_env(tmp_path, text)) == expected


def test_a_real_environment_variable_wins_inside_an_expansion_too(tmp_path, monkeypatch):
    from sps import read_env_file

    monkeypatch.setenv("SPS_TEST_HOST", "machine")
    monkeypatch.delenv("SPS_TEST_UNSET", raising=False)
    path = _env(tmp_path, "SPS_TEST_HOST=file\nURL=https://${SPS_TEST_HOST}/x\n"
                          "FALLBACK=${SPS_TEST_UNSET:-default}\n")

    values = read_env_file(path)
    assert values["URL"] == "https://machine/x"
    assert values["FALLBACK"] == "default"


def test_a_byte_order_mark_does_not_hide_the_first_variable(tmp_path):
    """Notepad and PowerShell write one. python-dotenv kept it as part of the
    first name, so the first variable -- an endpoint, say -- was never set."""
    import codecs

    from sps import read_env_file

    path = tmp_path / ".env"
    path.write_bytes(codecs.BOM_UTF8 + b"AZURE_EMBEDDING_ENDPOINT=https://x/\nB=2\n")
    assert list(read_env_file(path)) == ["AZURE_EMBEDDING_ENDPOINT", "B"]


def test_a_utf16_file_is_read(tmp_path):
    """`echo KEY=value > .env` in Windows PowerShell 5 writes UTF-16, which
    python-dotenv could not decode at all."""
    from sps import read_env_file

    path = tmp_path / ".env"
    path.write_bytes("A=1\r\nB=2\r\n".encode("utf-16"))
    assert read_env_file(path) == {"A": "1", "B": "2"}


def test_a_line_that_cannot_be_parsed_is_named_and_the_rest_load(tmp_path, caplog):
    from sps import read_env_file

    with caplog.at_level("WARNING"):
        values = read_env_file(_env(tmp_path, "A=1\n=no key\nB=2\n"))

    assert values == {"A": "1", "B": "2"}
    assert "line 2" in caplog.text


def test_a_file_in_another_encoding_is_skipped_with_an_error(tmp_path, monkeypatch, caplog):
    """python-dotenv raised here, before status.xlsx existed, with an exit code
    the robot retries. Skipped, the run reports the settings as missing and
    exits 2 instead, which asks for the person who has to fix the file."""
    import sps

    (tmp_path / ".env").write_bytes("# café\nSPS_TEST_ONLY=x\n".encode("cp1252"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sps, "PROJECT_DIR", tmp_path / "elsewhere")
    monkeypatch.delenv("SPS_TEST_ONLY", raising=False)

    with caplog.at_level("ERROR"):
        assert sps.load_env_file() is None

    assert "SPS_TEST_ONLY" not in os.environ
    assert ".env was not read" in caplog.text and "UTF-8" in caplog.text


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


@pytest.mark.parametrize("entry", ["resolve", "bulk", "eval", "ingest", "check"])
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
        "bulk": ["--tickets", str(absent), "--history", str(absent)],
        "ingest": ["--history", str(absent)],
        "eval": ["--test-dir", str(empty), "--output-dir", str(tmp_path / "out")],
        "check": [],
    }[entry]

    for name in ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY",
                 "AZURE_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)

    sps.main([entry, *argv])
    assert loaded == [entry], f"`sps.py {entry}` did not load .env"


@pytest.mark.parametrize("command", ["resolve", "bulk", "eval", "ingest", "check"])
def test_each_command_names_itself_in_its_usage(command, monkeypatch, capsys):
    """The usage line is what an operator copies when a robot's call fails.
    After the merge into sps.py, three still named the `python -m scripts.*`
    module they came from, and `check` named no command at all."""
    import sps

    monkeypatch.setattr(sps, "load_env_file", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exit_:
        sps.main([command, "--help"])

    assert exit_.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: python sps.py {command} ")


def test_a_malformed_number_is_a_configuration_error_not_a_retry(tmp_path, monkeypatch):
    """A typo in .env used to surface as "Unhandled error", exit 1 -- which a
    robot retries until a person notices. Exit 2, naming the setting."""
    import sps
    from tests.test_resolver import read_sheet, row, write_history, write_ticket

    monkeypatch.setenv("SPS_INTENT_TOP_K", "five")
    out = tmp_path / "out"
    code = sps.resolve_main([
        "--ticket-file", str(write_ticket(tmp_path / "t.xlsx")),
        "--history-file", str(write_history(tmp_path / "h.xlsx", [row("SPS-1")])),
        "--output-dir", str(out),
    ])

    assert code == sps.EXIT_BAD_INPUT
    reason = read_sheet(out / "status.xlsx")[0]["Reason"]
    assert "SPS_INTENT_TOP_K='five' is not a whole number" in reason


def test_every_other_command_names_a_malformed_setting(tmp_path, monkeypatch, capsys):
    import sps
    from tests.test_resolver import row, write_csv_history

    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "d")
    monkeypatch.setenv("AZURE_EMBEDDING_TIMEOUT", "twenty")
    history = write_csv_history(tmp_path / "h.csv", [row("SPS-1")])

    assert sps.main(["ingest", "--history", str(history)]) == sps.EXIT_BAD_INPUT
    assert "AZURE_EMBEDDING_TIMEOUT='twenty' is not a number" in capsys.readouterr().err
