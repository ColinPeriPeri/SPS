"""Format-agnostic reading, and the strict file-type gate.

The gate matters operationally: someone attaches the wrong file, and the run has
to say so clearly rather than failing somewhere deep in a header lookup.
"""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("openpyxl")
pytest.importorskip("pandas")

import scripts.run_resolver as resolver  # noqa: E402
from sps.file_reader import (  # noqa: E402
    EXCEL_SUFFIXES,
    SUPPORTED_SUFFIXES,
    FileReadError,
    UnsupportedFileType,
    is_blank,
    is_excel,
    iter_rows,
    read_header_and_rows,
    validate_file_type,
)
from tests.test_resolver import (  # noqa: E402
    HEADERS, NEAR_PROBLEM, PART, PROBLEM, read_sheet, row, write_csv_history,
    write_history, write_ticket,
)


@pytest.fixture(autouse=True)
def _stubbed_azure(azure_embeddings):
    """Azure is the only encoder now, so every test in this module embeds
    through the stub. A test that wants the unconfigured path deletes the
    variables itself."""


# Near-duplicates of the ticket rather than copies: through the stubbed Azure
# encoder identical text scores exactly 1.0, so a 0.99 gate would not gate.
ROWS = [row("SPS-1", problem=NEAR_PROBLEM),
        row("SPS-2", problem=NEAR_PROBLEM, minutes=1)]


# ---------------------------------------------------------------- the gate


@pytest.mark.parametrize("name", ["h.csv", "h.xlsx", "h.xlsm", "H.CSV", "H.XLSX"])
def test_supported_types_pass(tmp_path, name):
    path = tmp_path / name
    path.write_text("x", encoding="utf-8")
    assert validate_file_type(path) == path


@pytest.mark.parametrize(
    "name", ["h.pdf", "h.docx", "h.json", "h.txt", "h.tsv", "h.xls", "h.parquet", "h"]
)
def test_unsupported_types_are_rejected(tmp_path, name):
    path = tmp_path / name
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileType):
        validate_file_type(path)


def test_the_error_names_the_file_and_what_was_expected(tmp_path):
    path = tmp_path / "quarterly report.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileType) as info:
        validate_file_type(path, "History file")
    message = str(info.value)
    assert "History file" in message
    assert "quarterly report.pdf" in message
    assert ".csv" in message and ".xlsx" in message


def test_a_missing_extension_is_reported_as_such(tmp_path):
    path = tmp_path / "history"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileType, match="no extension"):
        validate_file_type(path)


def test_the_type_is_checked_before_the_file_is_opened(tmp_path):
    """A .pdf that does not even exist still fails on its type, not on I/O."""
    with pytest.raises(UnsupportedFileType):
        validate_file_type(tmp_path / "absent.pdf")


def test_xlsm_is_treated_as_excel():
    """A macro-enabled workbook is the same format; rejecting one would be a
    support ticket rather than a safety measure."""
    assert ".xlsm" in EXCEL_SUFFIXES
    assert is_excel("book.xlsm") and is_excel("book.xlsx")
    assert not is_excel("data.csv")


def test_only_csv_and_excel_are_supported():
    assert SUPPORTED_SUFFIXES == {".csv", ".xlsx", ".xlsm"}


# --------------------------------------------------------- format agnostic


def test_csv_and_xlsx_yield_the_same_rows(tmp_path):
    """Once loaded, nothing downstream can tell the formats apart."""
    from_csv = [list(r) for r in iter_rows(write_csv_history(tmp_path / "h.csv", ROWS))]
    from_xlsx = [[("" if c is None else str(c)) for c in r]
                 for r in iter_rows(write_history(tmp_path / "h.xlsx", ROWS))]

    assert from_csv[0] == HEADERS == from_xlsx[0]
    assert [r[0] for r in from_csv[1:]] == [r[0] for r in from_xlsx[1:]] == ["SPS-1", "SPS-2"]


def test_header_and_rows_are_split(tmp_path):
    header, rows = read_header_and_rows(write_csv_history(tmp_path / "h.csv", ROWS))
    assert list(header) == HEADERS
    assert len([r for r in rows if not is_blank(r)]) == 2


def test_an_empty_file_is_reported(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(FileReadError, match="empty"):
        read_header_and_rows(path)


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileReadError, match="not found"):
        list(iter_rows(tmp_path / "absent.csv"))


def test_a_csv_bom_is_tolerated(tmp_path):
    """.NET writes UTF-8 with a BOM, so a UiPath-produced file normally has one."""
    path = tmp_path / "h.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADERS)
        writer.writerows(ROWS)
    header, _ = read_header_and_rows(path)
    assert list(header)[0] == "SPS_ID"


def test_blank_rows_are_detectable():
    assert is_blank(None) and is_blank([None, None]) and is_blank(["", ""])
    assert not is_blank(["x", None])


# ------------------------------------------------------------ through the CLI


def _run(tmp_path, ticket_name="t.xlsx", history_name="h.xlsx", threshold="0.99"):
    out = tmp_path / "out"
    ticket = tmp_path / ticket_name
    history = tmp_path / history_name

    if ticket.suffix.lower() in {".xlsx", ".xlsm"}:
        write_ticket(ticket)
    else:
        ticket.write_text("not really a ticket", encoding="utf-8")

    if history.suffix.lower() in {".xlsx", ".xlsm"}:
        write_history(history, ROWS)
    elif history.suffix.lower() == ".csv":
        write_csv_history(history, ROWS)
    else:
        history.write_text("not really a history", encoding="utf-8")

    code = resolver.main([
        "--ticket-file", str(ticket),
        "--history-file", str(history),
        "--output-dir", str(out),
        "--threshold", threshold,
    ])
    return code, read_sheet(out / "status.xlsx").iloc[0]


def test_a_bad_history_type_is_invalid_input_and_exits_zero(tmp_path):
    """A wrong attachment is a business problem, not an I/O fault: the item is
    faulted rather than retried."""
    code, status = _run(tmp_path, history_name="h.pdf")

    assert code == resolver.EXIT_OK
    assert status["Status"] == "FAIL"
    assert status["Status_Code"] == "INVALID_INPUT"
    assert "h.pdf" in status["Reason"]


def test_a_bad_ticket_type_is_invalid_input_and_exits_zero(tmp_path):
    code, status = _run(tmp_path, ticket_name="t.docx")

    assert code == resolver.EXIT_OK
    assert status["Status_Code"] == "INVALID_INPUT"
    assert "t.docx" in status["Reason"]


def test_the_bad_type_aborts_before_any_embedding(tmp_path):
    """Nothing is encoded, so the status sheet names no model."""
    _, status = _run(tmp_path, history_name="h.json")
    assert status["Embedding_Model"] == ""


def test_a_missing_but_valid_type_still_exits_two(tmp_path):
    """Right format, absent file: genuine I/O trouble, so a human is alerted
    rather than the item being quietly faulted."""
    out = tmp_path / "out"
    write_ticket(tmp_path / "t.xlsx")
    code = resolver.main([
        "--ticket-file", str(tmp_path / "t.xlsx"),
        "--history-file", str(tmp_path / "absent.csv"),
        "--output-dir", str(out),
    ])
    assert code == resolver.EXIT_BAD_INPUT
    assert read_sheet(out / "status.xlsx").iloc[0]["Status_Code"] == "INVALID_INPUT"


@pytest.mark.parametrize("history_name", ["h.csv", "h.xlsx"])
def test_both_formats_reach_the_same_outcome(tmp_path, history_name):
    """Same history, two formats, identical result: the reader is the only
    thing that differed, and it is not allowed to matter."""
    code, status = _run(tmp_path, history_name=history_name)

    assert code == resolver.EXIT_OK
    # A threshold of 0.99 gates before the LLM, so this exercises read, filter
    # and embed without needing Azure.
    assert status["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"
    # Azure is the only encoder; the stub stands in for the deployment.
    assert status["Embedding_Model"].startswith("azure:")


def test_reason_carries_the_encoder_marker(tmp_path):
    _, status = _run(tmp_path, history_name="h.csv")
    assert status["Reason"].endswith("[Azure]")
