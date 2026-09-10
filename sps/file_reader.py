"""Format-agnostic row reading for tickets and history.

One job: turn a `.csv` or `.xlsx` path into a stream of rows, header first, so
that everything downstream works on a list of cells and never needs to know
which format it came from. Column mapping, canonicalisation and filtering are
all format-blind as a result.

Both readers stream. CSV is a linear read; XLSX goes through openpyxl in
`read_only` mode, which iterates rows rather than materialising the sheet.
`pandas.read_excel` has no chunked mode and would pull the whole workbook into
memory, which is the spike that makes CSV preferable for a large history in the
first place. Measured at 300k rows on this hardware: about 1.5 s for CSV against
about 40 s for XLSX.

Stdlib only at import time; pandas and openpyxl are pulled in lazily by the
readers that need them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

CSV_SUFFIXES = frozenset({".csv"})

# .xlsm is the same format as .xlsx as far as openpyxl is concerned -- it is a
# workbook that happens to carry macros -- and business users hand those over
# routinely. Rejecting one would be a support ticket, not a safety measure.
EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm"})

SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES


class UnsupportedFileType(ValueError):
    """The path is not a format this system reads.

    Deliberately distinct from a missing or unreadable file: the wrong
    attachment is a business problem for whoever assembled the ticket, not an
    I/O fault worth retrying, and the caller maps the two to different outcomes.
    """


class FileReadError(ValueError):
    """The file is a supported format but could not be read."""


def validate_file_type(path: Path | str, label: str = "File") -> Path:
    """Reject anything that is not .csv or .xlsx before it is opened.

    Checked on the extension alone and checked first, so a `.pdf` or a `.docx`
    fails immediately rather than after a model load or a history scan.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        readable = ", ".join(sorted(SUPPORTED_SUFFIXES))
        got = suffix or "no extension"
        raise UnsupportedFileType(
            f"{label} {path.name!r} has an unsupported type ({got}). "
            f"Expected one of: {readable}."
        )
    return path


def is_excel(path: Path | str) -> bool:
    return Path(path).suffix.lower() in EXCEL_SUFFIXES


def iter_rows(path: Path | str, label: str = "File") -> Iterator[Sequence[Any]]:
    """Stream rows, header first, without materialising the file.

    Cells come back at their native type -- an Excel numeric cell yields a
    number, a CSV cell yields a string -- and callers coerce. That is the only
    format difference that survives this boundary, and it is handled once, in
    the cell coercion downstream.
    """
    path = validate_file_type(path, label)
    if not path.exists():
        raise FileReadError(f"{label} not found: {path}")

    if is_excel(path):
        yield from _iter_excel(path, label)
    else:
        yield from _iter_csv(path)


def _iter_csv(path: Path) -> Iterator[Sequence[Any]]:
    import csv

    # utf-8-sig, not utf-8: .NET writes UTF-8 with a BOM by default, so a file
    # produced by a UiPath Write Text File activity normally starts with one.
    # Strict utf-8 rejects it outright.
    with open(path, newline="", encoding="utf-8-sig") as handle:
        yield from csv.reader(handle)


def _iter_excel(path: Path, label: str) -> Iterator[Sequence[Any]]:
    from openpyxl import load_workbook

    try:
        # data_only returns computed values rather than formula strings, which
        # matters when a business user built the extract with formulas.
        book = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise FileReadError(f"{label} {path.name!r} could not be opened: {exc}") from exc
    try:
        sheet = book.active
        if sheet is None:
            raise FileReadError(f"{label} {path.name!r} has no active sheet.")
        yield from sheet.iter_rows(values_only=True)
    finally:
        book.close()


def read_header_and_rows(path: Path | str, label: str = "File"):
    """Split the stream into (header, remaining rows).

    Raises FileReadError on an empty file, which is otherwise indistinguishable
    from a file whose every row was filtered out.
    """
    rows = iter_rows(path, label)
    try:
        header = next(rows)
    except StopIteration:
        raise FileReadError(f"{label} {Path(path).name!r} is empty.") from None
    return header, rows


def is_blank(row: Sequence[Any] | None) -> bool:
    """Trailing blank rows are routine in hand-edited workbooks."""
    return row is None or all(cell in (None, "") for cell in row)
