"""Excel handoff for the UiPath Performer.

The production support team reads the result as a workbook, so the delivered
artefact is an .xlsx: one row per ticket, four columns matching the Section 4
contract keys.

pandas, openpyxl and pydantic are all imported lazily, so importing this module
-- and therefore the CLI -- costs nothing and requires nothing. Only the Excel
path itself pulls them in.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Column order is fixed to the contract's key order so the support team always
# finds the same field in the same column.
COLUMNS = ("AI_Recommendation", "Justification", "Confidence", "SPS_IDs_Referred")

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}

# openpyxl refuses to write a cell longer than this, and Excel itself will not
# display more. Truncating with a visible marker beats failing the handoff.
EXCEL_MAX_CELL = 32767
_TRUNCATION_NOTE = " […truncated for Excel]"


def is_excel_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in EXCEL_SUFFIXES


def _fit_cell(value: str) -> str:
    if len(value) <= EXCEL_MAX_CELL:
        return value
    logger.warning("Truncating a %d-character cell to Excel's 32767 limit", len(value))
    return value[: EXCEL_MAX_CELL - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


def contracts_to_dataframe(contracts: Sequence[dict[str, Any]]):
    """Validate through the pydantic contract, then build the DataFrame.

    Validation happens here rather than at the writer so a malformed payload is
    caught before a workbook is produced from it.
    """
    import pandas as pd

    from sps.schemas import SPSContract

    rows = [SPSContract.from_contract(dict(c)).to_row() for c in contracts]
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    # Every cell is text. Without this, pandas would infer dtypes and a
    # Confidence of "84%" or an SPS_ID list that happens to hold one numeric-
    # looking id could reach Excel as a number.
    return frame.astype(str).map(_fit_cell)


def write_excel(path: Path | str, contracts: Sequence[dict[str, Any]]) -> None:
    """Write the workbook atomically.

    Same guarantee as the JSON path: the caller polling for this file sees
    either nothing or a complete, openable workbook -- never a partially
    written one, which Excel and UiPath both report as a corrupt file.
    """
    path = Path(path)
    frame = contracts_to_dataframe(contracts)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The temp file must live in the destination directory and carry an .xlsx
    # suffix: os.replace is only atomic within a filesystem, and pandas picks
    # its engine from the extension.
    handle, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".xlsx")
    os.close(handle)
    try:
        frame.to_excel(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Resolver workbooks: a status sheet written on every run, and a result sheet
# written only on success.
# --------------------------------------------------------------------------

STATUS_COLUMNS = ("Execution_Timestamp", "Status", "Status_Code", "Reason")
RESULT_COLUMNS = (
    "Part_Number",
    "AI_Recommendation",
    "Justification",
    "Confidence_Score",
    "Referenced_SPS_IDs",
)


def write_rows(path: Path | str, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write a sheet atomically, every cell as text.

    Text throughout for the same reason as the contract workbook: pandas would
    otherwise infer dtypes and a Confidence_Score of "84%" or a part number of
    "0012-43951" could reach Excel as a number or a date.
    """
    import pandas as pd

    path = Path(path)
    frame = pd.DataFrame(
        [{c: row.get(c, "") for c in columns} for row in rows], columns=list(columns)
    )
    frame = frame.astype(str).map(_fit_cell)

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".xlsx")
    os.close(handle)
    try:
        frame.to_excel(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
