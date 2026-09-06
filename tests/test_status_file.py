"""The status side-channel.

UiPath branches on this file, so its parseability, its agreement with the exit
code, and the order it is written relative to the data file are all part of the
integration contract.
"""

from __future__ import annotations

import json

import pytest

import service.run_inference as cli
from service.status_file import StatusReport, one_line, write_status_file
from sps.contracts import SOLUTION_NOT_FOUND, PipelineResult

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
    diagnostic="Confidence below 82% threshold.",
)
OUTAGE = PipelineResult(
    ai_recommendation=SOLUTION_NOT_FOUND,
    justification="AI service unavailable; escalate for manual review.",
    confidence="93%",
    infrastructure_failure=True,
    diagnostic="Generation failed: Missing Azure OpenAI configuration: AZURE_OPENAI_API_KEY",
)


class FakeStore:
    def close(self) -> None:
        pass


class FakePipeline:
    def __init__(self, results) -> None:
        self.results = list(results)

    async def process(self, ticket):
        return self.results.pop(0)


@pytest.fixture
def wire(monkeypatch):
    def _wire(results, explode=False):
        def build(settings=None):
            if explode:
                raise RuntimeError("storage is locked by another process")
            return FakePipeline(results), FakeStore()

        monkeypatch.setattr(cli, "build_pipeline", build)

    return _wire


def parse_status(path) -> dict[str, str]:
    """Parse the file the way a UiPath workflow would: split each line on ':'."""
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


# ------------------------------------------------------------------- format


def test_status_file_has_exactly_the_three_keys(wire, tmp_path, capsys):
    wire([GOOD])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])

    assert set(parse_status(status)) == {"STATUS", "EXIT_CODE", "REASON"}


def test_status_file_is_three_lines(wire, tmp_path, capsys):
    wire([GOOD])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])

    assert len(status.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_reason_is_always_one_line(wire, tmp_path, capsys):
    """A multi-line recommendation must not split REASON into extra records."""
    multiline = PipelineResult(
        ai_recommendation="1. Segregate.\n2. Rework.\n3. Re-inspect.",
        justification="Line one.\nLine two.\r\nLine three.",
        confidence="91%",
        sps_ids_referred=["SPS-100"],
    )
    wire([multiline])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])

    assert len(status.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert "\n" not in parse_status(status)["REASON"]


def test_long_reason_is_truncated():
    report = StatusReport(exit_code=1, reason="x" * 5000)
    line = [l for l in report.render().splitlines() if l.startswith("REASON")][0]
    assert len(line) < 600


def test_empty_reason_still_says_something():
    assert "No reason recorded." in StatusReport(exit_code=0, reason="").render()


def test_one_line_collapses_all_whitespace():
    assert one_line("a\n\n  b\t\tc\r\nd") == "a b c d"


def test_status_file_is_utf8_without_a_bom(tmp_path):
    path = tmp_path / "status.txt"
    write_status_file(path, StatusReport(exit_code=0, reason="Processed successfully."))
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


# -------------------------------------------------------- status vs exit code


@pytest.mark.parametrize(
    "results,explode,expected_code,expected_status",
    [
        ([GOOD], False, 0, "SUCCESS"),
        ([GATED], False, 0, "SUCCESS"),
        ([OUTAGE], False, 1, "FAILURE"),
        ([], True, 1, "FAILURE"),
    ],
)
def test_status_always_agrees_with_the_exit_code(
    wire, tmp_path, capsys, results, explode, expected_code, expected_status
):
    """If STATUS and EXIT_CODE could disagree, behaviour would depend on which
    one the workflow happened to read."""
    wire(results, explode=explode)
    status = tmp_path / "status.txt"

    code = cli.main(["--payload", TICKET, "--status-file", str(status)])
    fields = parse_status(status)

    assert code == expected_code
    assert fields["STATUS"] == expected_status
    assert fields["EXIT_CODE"] == str(expected_code)


def test_bad_payload_reports_exit_code_two(wire, tmp_path, capsys):
    wire([GOOD])
    status = tmp_path / "status.txt"

    assert cli.main(["--payload", "not json", "--status-file", str(status)]) == 2
    fields = parse_status(status)
    assert fields["STATUS"] == "FAILURE"
    assert fields["EXIT_CODE"] == "2"
    assert "Invalid input format" in fields["REASON"]


def test_status_is_only_ever_success_or_failure():
    assert StatusReport(exit_code=0, reason="x").status == "SUCCESS"
    for code in (1, 2, 137):
        assert StatusReport(exit_code=code, reason="x").status == "FAILURE"


# ------------------------------------------------------------------- reasons


def test_successful_run_says_so(wire, tmp_path, capsys):
    wire([GOOD])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])
    assert parse_status(status)["REASON"] == "Processed successfully."


def test_gated_ticket_explains_the_gate(wire, tmp_path, capsys):
    """A refusal is a SUCCESS -- the pipeline ran -- but REASON must make the
    outcome unmistakable to whoever opens the file."""
    wire([GATED])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])

    fields = parse_status(status)
    assert fields["STATUS"] == "SUCCESS"
    assert "Confidence below" in fields["REASON"]


def test_outage_reason_names_the_missing_credential_variable(wire, tmp_path, capsys):
    """The status file is for support, so it carries the specific diagnostic
    rather than the sanitized supplier-facing Justification."""
    wire([OUTAGE])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])

    reason = parse_status(status)["REASON"]
    assert "AZURE_OPENAI_API_KEY" in reason
    assert reason != OUTAGE.justification


def test_batch_reason_summarises_every_ticket(wire, tmp_path, capsys):
    wire([GOOD, GATED, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps({"problem_description": f"weld seam defect {i} found here"})
            for i in range(3)
        ),
        encoding="utf-8",
    )
    status = tmp_path / "status.txt"

    assert cli.main(["--batch-file", str(queue), "--status-file", str(status)]) == 0
    reason = parse_status(status)["REASON"]
    assert "3 tickets" in reason
    assert "2 with a recommendation" in reason


def test_batch_with_one_outage_is_a_failure(wire, tmp_path, capsys):
    wire([GOOD, OUTAGE, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps({"problem_description": f"weld seam defect {i} found here"})
            for i in range(3)
        ),
        encoding="utf-8",
    )
    status = tmp_path / "status.txt"

    assert cli.main(["--batch-file", str(queue), "--status-file", str(status)]) == 1
    fields = parse_status(status)
    assert fields["STATUS"] == "FAILURE"
    assert "failed on a dependency" in fields["REASON"]


def test_summarise_reason_handles_no_results():
    assert cli.summarise_reason([]) == "No tickets processed."


# ------------------------------------------------------------------ ordering


def test_status_is_written_after_the_output_file(wire, tmp_path, capsys, monkeypatch):
    """A SUCCESS status must never point at a data file that is not there yet."""
    order = []
    real_excel = cli.write_excel
    real_status = cli.write_status_file

    monkeypatch.setattr(
        cli, "write_excel", lambda p, c: (order.append("output"), real_excel(p, c))[1]
    )
    monkeypatch.setattr(
        cli, "write_status_file", lambda p, r: (order.append("status"), real_status(p, r))[1]
    )

    wire([GOOD])
    cli.main(
        ["--payload", TICKET,
         "--output-file", str(tmp_path / "out.xlsx"),
         "--status-file", str(tmp_path / "status.txt")]
    )

    assert order == ["output", "status"]


def test_unwritable_output_is_reported_in_the_status_file(wire, tmp_path, capsys):
    """The pipeline succeeded but the answer could not be handed over -- the
    status must say FAILURE, not SUCCESS."""
    wire([GOOD])
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    status = tmp_path / "status.txt"

    code = cli.main(
        ["--payload", TICKET, "--output-file", str(blocker / "out.xlsx"),
         "--status-file", str(status)]
    )

    assert code == cli.EXIT_INFRASTRUCTURE
    fields = parse_status(status)
    assert fields["STATUS"] == "FAILURE"
    assert "output file" in fields["REASON"].lower()


def test_stale_status_is_removed_before_work_starts(monkeypatch, tmp_path):
    status = tmp_path / "status.txt"
    write_status_file(status, StatusReport(exit_code=0, reason="STALE FROM LAST RUN"))
    seen = {}

    def killed_mid_run(settings=None):
        seen["present"] = status.exists()
        raise KeyboardInterrupt("robot killed the process")

    monkeypatch.setattr(cli, "build_pipeline", killed_mid_run)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--payload", TICKET, "--status-file", str(status)])

    assert seen["present"] is False
    assert not status.exists()


def test_status_file_leaves_no_temp_files(wire, tmp_path, capsys):
    wire([GOOD])
    status = tmp_path / "status.txt"
    cli.main(["--payload", TICKET, "--status-file", str(status)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.txt"]


def test_status_directory_is_created(wire, tmp_path, capsys):
    wire([GOOD])
    status = tmp_path / "nested" / "deeper" / "status.txt"
    assert cli.main(["--payload", TICKET, "--status-file", str(status)]) == 0
    assert status.exists()


def test_status_file_is_optional(wire, tmp_path, capsys):
    wire([GOOD])
    assert cli.main(["--payload", TICKET]) == 0


def test_credentials_never_leak_into_the_status_file(wire, monkeypatch, tmp_path, capsys):
    secret = "sk-super-secret-value-do-not-leak"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", secret)
    wire([OUTAGE])
    status = tmp_path / "status.txt"

    cli.main(["--payload", TICKET, "--status-file", str(status)])
    assert secret not in status.read_text(encoding="utf-8")
