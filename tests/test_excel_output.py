"""Excel handoff: schema validation, DataFrame shape, atomic workbook write.

UiPath reads this workbook with standard Workbook activities, so column order,
cell types and the atomicity of the write are all part of the contract.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import service.run_inference as cli  # noqa: E402
from service.excel_output import (  # noqa: E402
    COLUMNS,
    EXCEL_MAX_CELL,
    contracts_to_dataframe,
    is_excel_path,
    write_excel,
)
from sps.contracts import SOLUTION_NOT_FOUND, PipelineResult  # noqa: E402
from sps.schemas import SPSContract  # noqa: E402

CONTRACT = {
    "AI_Recommendation": "1. Segregate the lot.\n2. Rework the weld seam.",
    "Justification": "Drawn from SPS-100 and SPS-101, the same weld defect.",
    "Confidence": "91%",
    "SPS_IDs_Referred": ["SPS-100", "SPS-101"],
}

GOOD = PipelineResult(
    ai_recommendation=CONTRACT["AI_Recommendation"],
    justification=CONTRACT["Justification"],
    confidence=CONTRACT["Confidence"],
    sps_ids_referred=list(CONTRACT["SPS_IDs_Referred"]),
)
OUTAGE = PipelineResult(
    ai_recommendation=SOLUTION_NOT_FOUND,
    justification="AI service unavailable; escalate for manual review.",
    confidence="93%",
    infrastructure_failure=True,
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
    def _wire(results):
        monkeypatch.setattr(
            cli, "build_pipeline", lambda settings=None: (FakePipeline(results), FakeStore())
        )

    return _wire


def read_workbook(path):
    import pandas as pd

    return pd.read_excel(path, dtype=str).fillna("")


# ------------------------------------------------------------------- schema


def test_contract_model_accepts_the_four_keys():
    model = SPSContract.from_contract(CONTRACT)
    assert model.to_contract() == CONTRACT


def test_contract_model_rejects_an_unknown_key():
    with pytest.raises(Exception):
        SPSContract.from_contract({**CONTRACT, "Extra": "nope"})


def test_contract_model_rejects_a_missing_key():
    incomplete = {k: v for k, v in CONTRACT.items() if k != "Confidence"}
    with pytest.raises(Exception):
        SPSContract.from_contract(incomplete)


@pytest.mark.parametrize("bad", ["91", "ninety-one%", "", "0.91"])
def test_contract_model_rejects_a_malformed_confidence(bad):
    with pytest.raises(Exception):
        SPSContract.from_contract({**CONTRACT, "Confidence": bad})


@pytest.mark.parametrize("good", ["0%", "84%", "100%"])
def test_contract_model_accepts_wellformed_confidence(good):
    assert SPSContract.from_contract({**CONTRACT, "Confidence": good}).Confidence == good


# ---------------------------------------------------------------- dataframe


def test_dataframe_is_one_row_and_four_columns():
    frame = contracts_to_dataframe([CONTRACT])
    assert frame.shape == (1, 4)
    assert list(frame.columns) == list(COLUMNS)


def test_dataframe_column_order_matches_the_contract():
    assert COLUMNS == (
        "AI_Recommendation",
        "Justification",
        "Confidence",
        "SPS_IDs_Referred",
    )


def test_id_list_becomes_a_readable_string_not_a_python_repr():
    """A raw list would land in the cell as "['SPS-100', 'SPS-101']"."""
    value = contracts_to_dataframe([CONTRACT]).iloc[0]["SPS_IDs_Referred"]
    assert value == "SPS-100, SPS-101"
    assert "[" not in value and "'" not in value


def test_empty_id_list_becomes_an_empty_cell():
    payload = {**CONTRACT, "SPS_IDs_Referred": []}
    assert contracts_to_dataframe([payload]).iloc[0]["SPS_IDs_Referred"] == ""


def test_every_cell_is_text():
    """Confidence must not reach Excel as a number or a date."""
    frame = contracts_to_dataframe([CONTRACT])
    assert all(frame.dtypes == object)
    assert frame.iloc[0]["Confidence"] == "91%"


def test_batch_produces_one_row_per_ticket():
    frame = contracts_to_dataframe([CONTRACT, CONTRACT, CONTRACT])
    assert frame.shape == (3, 4)


def test_oversized_cell_is_truncated_rather_than_failing_the_handoff():
    huge = {**CONTRACT, "AI_Recommendation": "x" * (EXCEL_MAX_CELL + 5000)}
    cell = contracts_to_dataframe([huge]).iloc[0]["AI_Recommendation"]
    assert len(cell) <= EXCEL_MAX_CELL
    assert cell.endswith("truncated for Excel]")


# ------------------------------------------------------------------- writing


def test_write_excel_produces_a_readable_workbook(tmp_path):
    out = tmp_path / "result.xlsx"
    write_excel(out, [CONTRACT])

    frame = read_workbook(out)
    assert list(frame.columns) == list(COLUMNS)
    assert frame.shape == (1, 4)
    assert frame.iloc[0]["Confidence"] == "91%"
    assert frame.iloc[0]["SPS_IDs_Referred"] == "SPS-100, SPS-101"


def test_workbook_has_no_index_column(tmp_path):
    """to_excel(index=False): a leading unnamed column would shift every
    column reference in the UiPath workflow."""
    out = tmp_path / "result.xlsx"
    write_excel(out, [CONTRACT])

    from openpyxl import load_workbook

    sheet = load_workbook(out).active
    assert [c.value for c in sheet[1]] == list(COLUMNS)


def test_multiline_recommendation_survives_the_round_trip(tmp_path):
    out = tmp_path / "result.xlsx"
    write_excel(out, [CONTRACT])
    assert "\n" in read_workbook(out).iloc[0]["AI_Recommendation"]


def test_write_excel_creates_the_directory(tmp_path):
    out = tmp_path / "nested" / "deeper" / "result.xlsx"
    write_excel(out, [CONTRACT])
    assert out.exists()


def test_write_excel_leaves_no_temp_files(tmp_path):
    out = tmp_path / "result.xlsx"
    write_excel(out, [CONTRACT])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.xlsx"]


def test_write_excel_replaces_atomically(tmp_path):
    """A previous workbook stays intact and openable until the new one is
    complete -- a reader polling the path never catches a corrupt file."""
    out = tmp_path / "result.xlsx"
    write_excel(out, [CONTRACT])
    first = read_workbook(out).iloc[0]["Confidence"]

    write_excel(out, [{**CONTRACT, "Confidence": "58%"}])
    assert first == "91%"
    assert read_workbook(out).iloc[0]["Confidence"] == "58%"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.xlsx"]


def test_invalid_contract_never_produces_a_workbook(tmp_path):
    out = tmp_path / "result.xlsx"
    with pytest.raises(Exception):
        write_excel(out, [{**CONTRACT, "Confidence": "not-a-percentage"}])
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------ path dispatch


@pytest.mark.parametrize("name", ["out.xlsx", "OUT.XLSX", "out.xlsm", "a/b/c.xlsx"])
def test_excel_paths_are_detected(name):
    assert is_excel_path(name)


@pytest.mark.parametrize("name", ["out.txt", "out.json", "out", "out.xls", "out.csv"])
def test_non_excel_paths_are_not(name):
    assert not is_excel_path(name)


# --------------------------------------------------------------- end to end


def test_cli_writes_a_workbook_for_an_xlsx_path(wire, tmp_path, capsys):
    wire([GOOD])
    out = tmp_path / "result.xlsx"

    code = cli.main(
        ["--payload", '{"problem_description": "a weld seam cracked here"}',
         "--output-file", str(out)]
    )

    assert code == 0
    frame = read_workbook(out)
    assert frame.shape == (1, 4)
    assert frame.iloc[0]["SPS_IDs_Referred"] == "SPS-100, SPS-101"
    # stdout is still JSON, so the process stays debuggable by hand.
    assert json.loads(capsys.readouterr().out.strip())["Confidence"] == "91%"


def test_cli_still_writes_json_for_a_txt_path(wire, tmp_path, capsys):
    wire([GOOD])
    out = tmp_path / "result.txt"
    assert cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}',
                     "--output-file", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["SPS_IDs_Referred"] == [
        "SPS-100",
        "SPS-101",
    ]


def test_cli_writes_a_workbook_on_the_outage_path(wire, tmp_path, capsys):
    """The support team must still get a row when the dependency failed."""
    wire([OUTAGE])
    out = tmp_path / "result.xlsx"

    code = cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}',
                     "--output-file", str(out)])

    assert code == cli.EXIT_INFRASTRUCTURE
    frame = read_workbook(out)
    assert frame.iloc[0]["AI_Recommendation"] == SOLUTION_NOT_FOUND
    assert frame.iloc[0]["SPS_IDs_Referred"] == ""


def test_cli_writes_a_workbook_for_a_bad_payload(wire, tmp_path, capsys):
    wire([GOOD])
    out = tmp_path / "result.xlsx"

    assert cli.main(["--payload", "not json", "--output-file", str(out)]) == cli.EXIT_BAD_PAYLOAD
    assert read_workbook(out).shape == (1, 4)


def test_cli_batch_writes_one_row_per_ticket(wire, tmp_path, capsys):
    wire([GOOD, GOOD, GOOD])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps({"problem_description": f"weld seam defect {i} found here"})
            for i in range(3)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "results.xlsx"

    assert cli.main(["--batch-file", str(queue), "--output-file", str(out)]) == 0
    assert read_workbook(out).shape == (3, 4)


def test_stale_workbook_is_removed_before_work_starts(monkeypatch, tmp_path):
    out = tmp_path / "result.xlsx"
    write_excel(out, [CONTRACT])
    seen = {}

    def killed_mid_run(settings=None):
        seen["present"] = out.exists()
        raise KeyboardInterrupt("robot killed the process")

    monkeypatch.setattr(cli, "build_pipeline", killed_mid_run)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--payload", '{"problem_description": "a weld seam cracked here"}',
                  "--output-file", str(out)])

    assert seen["present"] is False
    assert not out.exists()
