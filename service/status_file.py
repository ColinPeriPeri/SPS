"""Status side-channel for the UiPath Performer.

A tiny key-value text file written next to the data file:

    STATUS: SUCCESS
    EXIT_CODE: 0
    REASON: Processed successfully

It exists so exception handling never has to infer anything: the workflow reads
three lines instead of parsing a workbook to work out what happened, and a
support engineer opening the file sees the cause in plain text.

STATUS is defined as SUCCESS if and only if EXIT_CODE is 0. The two can never
disagree -- if they could, the workflow's behaviour would depend on which one it
happened to read, which is exactly the class of bug this file is meant to remove.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")

# REASON is one line by construction. Recommendation and justification text is
# routinely multi-line, so it is collapsed rather than trusted.
MAX_REASON = 500

SUCCESS = "SUCCESS"
FAILURE = "FAILURE"


def one_line(text: str, limit: int = MAX_REASON) -> str:
    """Collapse to a single line and cap the length.

    A newline in REASON would split the record into an unparseable extra line,
    so this is a correctness requirement, not cosmetics.
    """
    flat = _WHITESPACE.sub(" ", (text or "")).strip()
    if not flat:
        return "No reason recorded."
    if len(flat) > limit:
        return flat[: limit - 1].rstrip() + "…"
    return flat


@dataclass(frozen=True, slots=True)
class StatusReport:
    exit_code: int
    reason: str

    @property
    def status(self) -> str:
        # Tied to the exit code by construction; see the module docstring.
        return SUCCESS if self.exit_code == 0 else FAILURE

    def render(self) -> str:
        return (
            f"STATUS: {self.status}\n"
            f"EXIT_CODE: {self.exit_code}\n"
            f"REASON: {one_line(self.reason)}\n"
        )


def write_status_file(path: Path | str, report: StatusReport) -> None:
    """Write the status atomically, UTF-8, no BOM.

    Atomic for the same reason the data file is: the caller may poll for this
    file, and a half-written record reads as a missing or malformed field.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(report.render())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
