"""Excel handoff for the UiPath Performer.

The resolver hands back two workbooks: a status sheet written on every run and
a result sheet written only on success.

pandas and openpyxl are imported lazily, so importing this module -- and
therefore the CLI -- costs nothing and requires nothing.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# openpyxl refuses to write a cell longer than this, and Excel itself will not
# display more. Truncating with a visible marker beats failing the handoff.
EXCEL_MAX_CELL = 32767
_TRUNCATION_NOTE = " […truncated for Excel]"


def _fit_cell(value: str) -> str:
    if len(value) <= EXCEL_MAX_CELL:
        return value
    logger.warning("Truncating a %d-character cell to Excel's 32767 limit", len(value))
    return value[: EXCEL_MAX_CELL - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


# --------------------------------------------------------------------------
# Resolver workbooks: a status sheet written on every run, and a result sheet
# written only on success.
# --------------------------------------------------------------------------

# Embedding_Model is appended LAST on purpose: a caller reading the first four
# columns positionally is unaffected by its arrival.
STATUS_COLUMNS = (
    "Execution_Timestamp",
    "Status",
    "Status_Code",
    "Reason",
    "Embedding_Model",
)
# Referenced_Sources replaces the former Referenced_SPS_IDs: with two tiers the
# column holds SPS IDs or document citations, and Resolution_Source says which.
# Both are a BREAKING change for a caller that reads this sheet by column name.
RESULT_COLUMNS = (
    "Part_Number",
    "AI_Recommendation",
    "Justification",
    "Confidence_Score",
    "Referenced_Sources",
    "Resolution_Source",
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
