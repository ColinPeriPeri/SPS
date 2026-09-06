"""File-based integration: --output-file, and credentials staying off the CLI.

UiPath reads the output file and branches on the exit code, so both the file's
contents and the conditions under which it exists (or deliberately does not) are
part of the integration contract.
"""

from __future__ import annotations

import json

import pytest

import service.run_inference as cli
from sps.contracts import SOLUTION_NOT_FOUND, PipelineResult

CONTRACT_KEYS = {"AI_Recommendation", "Justification", "Confidence", "SPS_IDs_Referred"}

TICKET = '{"problem_description": "a weld seam cracked here"}'

GOOD = PipelineResult(
    ai_recommendation="1. Rework the weld seam.",
    justification="Drawn from SPS-100.",
    confidence="91%",
    sps_ids_referred=["SPS-100"],
)
GATED = PipelineResult(
    ai_recommendation=SOLUTION_NOT_FOUND,
    justification="Confidence below 82% threshold.",
    confidence="58%",
)
OUTAGE = PipelineResult(
    ai_recommendation=SOLUTION_NOT_FOUND,
    justification="AI service unavailable; escalate for manual review.",
    confidence="93%",
    infrastructure_failure=True,
)


class FakeStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePipeline:
    def __init__(self, results) -> None:
        self.results = list(results)

    async def process(self, ticket):
        return self.results.pop(0) if self.results else GATED


@pytest.fixture
def wire(monkeypatch):
    def _wire(results, explode=False):
        store = FakeStore()
        pipeline = FakePipeline(results)

        def fake_build(settings=None):
            if explode:
                raise RuntimeError("storage is locked by another process")
            return pipeline, store

        monkeypatch.setattr(cli, "build_pipeline", fake_build)
        return store

    return _wire


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- file contents


def test_output_file_holds_the_contract_as_json(wire, tmp_path):
    wire([GOOD])
    out = tmp_path / "result.txt"

    code = cli.main(["--payload", TICKET, "--output-file", str(out)])

    assert code == 0
    payload = read_json(out)
    assert set(payload) == CONTRACT_KEYS
    assert payload["AI_Recommendation"] == "1. Rework the weld seam."
    assert payload["SPS_IDs_Referred"] == ["SPS-100"]


def test_output_file_is_utf8_without_a_bom(wire, tmp_path):
    """RFC 8259 forbids a BOM on JSON; a stray U+FEFF breaks strict readers."""
    wire(
        [
            PipelineResult(
                ai_recommendation="1. Rework the seam — then re-inspect.",
                justification="Non-ASCII: é ü ≤ 100 µm",
                confidence="91%",
                sps_ids_referred=["SPS-100"],
            )
        ]
    )
    out = tmp_path / "result.txt"
    cli.main(["--payload", TICKET, "--output-file", str(out)])

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM must not be written"
    payload = json.loads(raw.decode("utf-8"))
    assert "≤" in payload["Justification"]
    assert "µm" in payload["Justification"]


def test_stdout_still_carries_the_contract_when_writing_a_file(wire, capsys, tmp_path):
    wire([GOOD])
    out = tmp_path / "result.txt"
    cli.main(["--payload", TICKET, "--output-file", str(out)])

    printed = json.loads(capsys.readouterr().out.strip())
    assert printed == read_json(out)


def test_batch_output_file_is_json_lines(wire, tmp_path):
    wire([GOOD, GATED, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps({"problem_description": f"weld seam defect {i} found here"})
            for i in range(3)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "results.txt"

    assert cli.main(["--batch-file", str(queue), "--output-file", str(out)]) == 0
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    assert all(set(line) == CONTRACT_KEYS for line in lines)


def test_pretty_is_refused_with_batch_mode():
    """Indented objects are not line-delimited: the file would be unparseable."""
    with pytest.raises(SystemExit):
        cli.parse_args(["--batch-file", "q.jsonl", "--pretty"])


# --------------------------------------------------------------- always written


@pytest.mark.parametrize(
    "results,explode,expected_code",
    [
        ([GOOD], False, 0),
        ([GATED], False, 0),
        ([OUTAGE], False, cli.EXIT_INFRASTRUCTURE),
        ([], True, cli.EXIT_INFRASTRUCTURE),
    ],
)
def test_output_file_written_on_every_pipeline_exit_path(
    wire, tmp_path, results, explode, expected_code
):
    """A file-based caller must never be left with no answer at all."""
    wire(results, explode=explode)
    out = tmp_path / "result.txt"

    assert cli.main(["--payload", TICKET, "--output-file", str(out)]) == expected_code
    assert set(read_json(out)) == CONTRACT_KEYS


def test_bad_payload_still_writes_a_file(wire, tmp_path):
    wire([GOOD])
    out = tmp_path / "result.txt"

    assert cli.main(["--payload", "not json at all", "--output-file", str(out)]) == (
        cli.EXIT_BAD_PAYLOAD
    )
    assert set(read_json(out)) == CONTRACT_KEYS


# --------------------------------------------------------------- stale safety


def test_stale_output_is_removed_before_work_starts(monkeypatch, tmp_path):
    """If the process is killed mid-run, the caller must find nothing rather
    than the previous run's result, which it would read as this answer."""
    out = tmp_path / "result.txt"
    out.write_text(json.dumps({"AI_Recommendation": "STALE FROM LAST RUN"}), encoding="utf-8")

    seen = {}

    def killed_mid_run(settings=None):
        seen["file_present_during_run"] = out.exists()
        raise KeyboardInterrupt("robot killed the process")

    monkeypatch.setattr(cli, "build_pipeline", killed_mid_run)

    with pytest.raises(KeyboardInterrupt):
        cli.main(["--payload", TICKET, "--output-file", str(out)])

    assert seen["file_present_during_run"] is False
    assert not out.exists()


def test_no_temp_files_are_left_behind(wire, tmp_path):
    """The atomic write must not litter the output directory."""
    wire([GOOD])
    out = tmp_path / "result.txt"
    cli.main(["--payload", TICKET, "--output-file", str(out)])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.txt"]


# --------------------------------------------------------------- write failures


def test_output_directory_is_created(wire, tmp_path):
    wire([GOOD])
    out = tmp_path / "nested" / "deeper" / "result.txt"

    assert cli.main(["--payload", TICKET, "--output-file", str(out)]) == 0
    assert out.exists()


def test_unwritable_output_path_is_an_infrastructure_fault(wire, tmp_path):
    """The answer exists but cannot be handed over -- a system fault, not a
    business outcome, even though the pipeline itself succeeded."""
    wire([GOOD])
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a directory", encoding="utf-8")

    code = cli.main(["--payload", TICKET, "--output-file", str(blocker / "result.txt")])
    assert code == cli.EXIT_INFRASTRUCTURE


# --------------------------------------------------------------- round trip


def test_file_in_file_out_round_trip(wire, tmp_path):
    """Exactly the shape scripts/run_inference.cmd uses."""
    wire([GOOD])
    ticket = tmp_path / "in.txt"
    ticket.write_text(
        json.dumps({"problem_description": "a weld seam cracked here", "part_number": "PN-1000"}),
        encoding="utf-8",
    )
    out = tmp_path / "out.txt"

    assert cli.main(["--payload-file", str(ticket), "--output-file", str(out)]) == 0
    assert read_json(out)["SPS_IDs_Referred"] == ["SPS-100"]


# --------------------------------------------------------------- credentials


def test_no_cli_flag_accepts_a_credential():
    """Command lines are visible in the Windows process list and in job logs, so
    credentials must have no CLI surface at all."""
    with pytest.raises(SystemExit):
        cli.parse_args(["--payload", "{}", "--api-key", "sk-secret"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--payload", "{}", "--azure-endpoint", "https://x"])


def test_credential_names_are_environment_only():
    assert cli.CREDENTIAL_VARS == (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    )


def test_missing_credentials_are_reported_by_name_not_value(wire, monkeypatch, caplog):
    for name in cli.CREDENTIAL_VARS:
        monkeypatch.setenv(name, "")
    wire([GATED])

    with caplog.at_level("WARNING"):
        cli.main(["--payload", TICKET])

    assert "AZURE_OPENAI_API_KEY" in caplog.text


def test_credentials_never_reach_stdout_or_the_output_file(wire, monkeypatch, capsys, tmp_path):
    secret = "sk-super-secret-value-do-not-leak"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", secret)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://internal.example.invalid/")
    wire([OUTAGE])
    out = tmp_path / "result.txt"

    cli.main(["--payload", TICKET, "--output-file", str(out)])

    assert secret not in out.read_text(encoding="utf-8")
    assert secret not in capsys.readouterr().out


# ------------------------------------------------- BOM on the INPUT side


def test_payload_file_with_a_utf8_bom_is_accepted(wire, tmp_path):
    """.NET writes UTF-8 with a BOM by default, so a payload produced by a
    UiPath Write Text File activity normally starts with one. Reading it as
    strict utf-8 fails with "Unexpected UTF-8 BOM" and every ticket would be
    rejected as an invalid payload."""
    wire([GOOD])
    ticket = tmp_path / "in.txt"
    ticket.write_text(TICKET, encoding="utf-8-sig")
    assert ticket.read_bytes().startswith(b"\xef\xbb\xbf")

    assert cli.main(["--payload-file", str(ticket)]) == 0


def test_payload_file_without_a_bom_still_works(wire, tmp_path):
    wire([GOOD])
    ticket = tmp_path / "in.txt"
    ticket.write_text(TICKET, encoding="utf-8")
    assert cli.main(["--payload-file", str(ticket)]) == 0


def test_batch_file_with_a_utf8_bom_is_accepted(wire, tmp_path):
    wire([GOOD, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        '{"problem_description": "weld seam cracked on the bracket"}\n'
        '{"problem_description": "carton label misprinted on packaging"}\n',
        encoding="utf-8-sig",
    )
    assert cli.main(["--batch-file", str(queue)]) == 0


def test_stdin_with_a_bom_is_accepted(wire, monkeypatch):
    import io

    wire([GOOD])
    monkeypatch.setattr("sys.stdin", io.StringIO(cli.BOM + TICKET))
    assert cli.main(["--stdin"]) == 0


def test_non_ascii_payload_round_trips(wire, tmp_path):
    """A supplier description with accents or micrometre symbols must survive."""
    wire([GOOD])
    ticket = tmp_path / "in.txt"
    ticket.write_text(
        json.dumps({"problem_description": "Fissure de soudure ≤ 50 µm sur la bride"}),
        encoding="utf-8-sig",
    )
    assert cli.main(["--payload-file", str(ticket)]) == 0
