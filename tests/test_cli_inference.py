"""The UiPath-facing CLI: stdout purity, exit codes, and lock release.

UiPath parses stdout and branches on the exit code, so both are part of the
integration contract and are pinned here.
"""

from __future__ import annotations

import json

import pytest

import service.run_inference as cli
from sps.contracts import SOLUTION_NOT_FOUND, PipelineResult

CONTRACT_KEYS = {"AI_Recommendation", "Justification", "Confidence", "SPS_IDs_Referred"}

GOOD = PipelineResult(
    ai_recommendation="1. Rework the weld seam.",
    justification="Drawn from SPS-100.",
    confidence="91%",
    sps_ids_referred=["SPS-100"],
)
GATED = PipelineResult(
    ai_recommendation=SOLUTION_NOT_FOUND,
    justification="Confidence below 75% threshold.",
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
        self.seen = []

    async def process(self, ticket):
        self.seen.append(ticket)
        return self.results.pop(0) if self.results else GATED


@pytest.fixture
def wire(monkeypatch):
    """Install a fake pipeline; returns the store so lock release can be checked."""

    def _wire(results, explode=False):
        store = FakeStore()
        pipeline = FakePipeline(results)

        def fake_build(settings=None):
            if explode:
                raise RuntimeError("storage is locked by another process")
            return pipeline, store

        monkeypatch.setattr(cli, "build_pipeline", fake_build)
        return store, pipeline

    return _wire


def read_stdout(capsys) -> list[dict]:
    out = capsys.readouterr().out.strip().splitlines()
    return [json.loads(line) for line in out if line.strip()]


# ------------------------------------------------------------------ stdout


def test_stdout_is_exactly_one_json_object(wire, capsys):
    wire([GOOD])
    code = cli.main(["--payload", json.dumps({"problem_description": "a weld seam cracked here"})])

    lines = read_stdout(capsys)
    assert code == 0
    assert len(lines) == 1
    assert set(lines[0]) == CONTRACT_KEYS
    assert lines[0]["AI_Recommendation"] == "1. Rework the weld seam."


def test_compact_json_by_default_and_pretty_on_request(wire, capsys):
    wire([GOOD])
    cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])
    assert len(capsys.readouterr().out.strip().splitlines()) == 1

    wire([GOOD])
    cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}', "--pretty"])
    assert len(capsys.readouterr().out.strip().splitlines()) > 1


def test_operational_flag_never_appears_in_the_contract(wire, capsys):
    wire([OUTAGE])
    cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])

    payload = read_stdout(capsys)[0]
    assert set(payload) == CONTRACT_KEYS
    assert "infrastructure_failure" not in payload


# ------------------------------------------------------------------ exit codes


def test_business_refusal_exits_zero(wire, capsys):
    """"Solution not found." is an expected outcome, not a system exception."""
    wire([GATED])
    code = cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])

    assert code == 0
    assert read_stdout(capsys)[0]["AI_Recommendation"] == SOLUTION_NOT_FOUND


def test_dependency_outage_exits_nonzero(wire, capsys):
    """Without this, an Azure outage would mark every queue item Successful."""
    wire([OUTAGE])
    code = cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])

    assert code == cli.EXIT_INFRASTRUCTURE
    # A contract is still emitted so the admin queue keeps a row.
    assert read_stdout(capsys)[0]["AI_Recommendation"] == SOLUTION_NOT_FOUND


def test_pipeline_that_cannot_start_still_emits_a_contract(wire, capsys):
    wire([], explode=True)
    code = cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])

    assert code == cli.EXIT_INFRASTRUCTURE
    payload = read_stdout(capsys)[0]
    assert set(payload) == CONTRACT_KEYS
    assert payload["AI_Recommendation"] == SOLUTION_NOT_FOUND


def test_malformed_json_exits_two(wire, capsys):
    wire([GOOD])
    code = cli.main(["--payload", "this is not json"])

    assert code == cli.EXIT_BAD_PAYLOAD
    assert set(read_stdout(capsys)[0]) == CONTRACT_KEYS


def test_non_object_payload_is_rejected(wire, capsys):
    wire([GOOD])
    assert cli.main(["--payload", "[1, 2, 3]"]) == cli.EXIT_BAD_PAYLOAD


def test_failure_contracts_are_never_truncated(wire, capsys):
    """Every emitted object must satisfy the schema, failure paths included."""
    wire([], explode=True)
    cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])
    payload = read_stdout(capsys)[0]
    assert payload["Confidence"].endswith("%")
    assert payload["SPS_IDs_Referred"] == []


# ------------------------------------------------------------------ input modes


def test_payload_file_mode(wire, capsys, tmp_path):
    wire([GOOD])
    ticket = tmp_path / "ticket.json"
    ticket.write_text(json.dumps({"problem_description": "a weld seam cracked here"}), "utf-8")

    assert cli.main(["--payload-file", str(ticket)]) == 0
    assert read_stdout(capsys)[0]["AI_Recommendation"] == "1. Rework the weld seam."


def test_stdin_mode(wire, capsys, monkeypatch):
    import io

    wire([GOOD])
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"problem_description": "a weld seam cracked here"}))
    )
    assert cli.main(["--stdin"]) == 0
    assert len(read_stdout(capsys)) == 1


def test_missing_payload_file_is_a_payload_error(wire, capsys, tmp_path):
    wire([GOOD])
    assert cli.main(["--payload-file", str(tmp_path / "nope.json")]) == cli.EXIT_BAD_PAYLOAD


def test_batch_mode_emits_one_object_per_line(wire, capsys, tmp_path):
    store, pipeline = wire([GOOD, GATED, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps({"problem_description": f"weld seam defect number {i} found"})
            for i in range(3)
        ),
        "utf-8",
    )

    code = cli.main(["--batch-file", str(queue)])
    lines = read_stdout(capsys)

    assert code == 0
    assert len(lines) == 3
    assert all(set(line) == CONTRACT_KEYS for line in lines)
    # One model load for the whole slice: the pipeline is built once.
    assert len(pipeline.seen) == 3


def test_batch_mode_blank_lines_are_skipped(wire, capsys, tmp_path):
    wire([GOOD, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        '{"problem_description": "weld seam cracked on the bracket"}\n'
        "\n"
        '{"problem_description": "carton label misprinted on packaging"}\n',
        "utf-8",
    )
    assert cli.main(["--batch-file", str(queue)]) == 0
    assert len(read_stdout(capsys)) == 2


def test_batch_reports_infrastructure_failure_if_any_ticket_hit_one(wire, capsys, tmp_path):
    wire([GATED, OUTAGE, GATED])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps({"problem_description": f"weld seam defect number {i} found"})
            for i in range(3)
        ),
        "utf-8",
    )
    assert cli.main(["--batch-file", str(queue)]) == cli.EXIT_INFRASTRUCTURE
    assert len(read_stdout(capsys)) == 3  # every ticket still reported


def test_input_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.parse_args(["--payload", "{}", "--stdin"])


def test_an_input_mode_is_required():
    with pytest.raises(SystemExit):
        cli.parse_args([])


# ------------------------------------------------------------------ lock release


def test_store_is_closed_on_success(wire, capsys):
    store, _ = wire([GOOD])
    cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])
    assert store.closed, "embedded-mode lock must be released for the next process"


def test_store_is_closed_even_when_the_pipeline_raises(wire, capsys, monkeypatch):
    store = FakeStore()

    class Exploding:
        async def process(self, ticket):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "build_pipeline", lambda settings=None: (Exploding(), store))
    code = cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}'])

    assert code == cli.EXIT_INFRASTRUCTURE
    assert store.closed, "a crash must not leave the embedded storage locked"


# ------------------------------------------------------------------ callable API


def test_callable_interface_returns_the_contract(wire):
    wire([GOOD])
    result = cli.run_inference({"problem_description": "a weld seam cracked here"})
    assert set(result) == CONTRACT_KEYS


def test_callable_batch_interface(wire):
    wire([GOOD, GATED])
    results = cli.run_batch(
        [
            {"problem_description": "weld seam cracked on the bracket"},
            {"problem_description": "carton label misprinted on packaging"},
        ]
    )
    assert len(results) == 2
    assert all(set(r) == CONTRACT_KEYS for r in results)
