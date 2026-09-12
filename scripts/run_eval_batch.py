"""Run many tickets in one process and write a single results workbook.

Built for threshold calibration, which needs the score distribution rather than
pass/fail: **every row carries the raw top cosine, including the cases the gate
rejected.** A run that only reported successes would show you the scores above
the threshold and hide exactly the ones you need to decide whether the threshold
is right.

    python -m scripts.run_eval_batch --test-dir test_cases
    python -m scripts.run_eval_batch --test-dir test_cases --history-file master.csv
    python -m scripts.run_eval_batch --run-list runs.csv --output-dir results

Case discovery, in order of precedence:

  --run-list      a .csv/.xlsx with columns Test_ID, Ticket_File, History_File
                  (History_File optional if --history-file is given). Paths are
                  resolved relative to the list's own directory.
  --test-dir      files named <id>_ticket.<ext> paired with <id>_history.<ext>.
                  A case with no history of its own uses --history-file, which
                  is the usual shape: fifty tickets against one extract.

One process for the whole batch, so a local-model cold start is paid once rather
than fifty times. One failing case never stops the run: it becomes a row saying
what went wrong.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sps.file_reader import SUPPORTED_SUFFIXES, UnsupportedFileType, validate_file_type

logger = logging.getLogger("sps.eval")

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_BAD_INPUT = 2

RESULTS_FILE = "eval_results.xlsx"

RESULT_COLUMNS = (
    "Test_ID",
    "Status",
    "Status_Code",
    "Reason",
    "Embedding_Model",
    "Confidence_Score",
    # Beyond the five requested, and all of them earn their place when the point
    # is calibration rather than a pass/fail report.
    "Threshold_Applied",
    "Cleared_Threshold",
    "Candidates_Considered",
    # Tier 2 is calibrated the same way and needs the same columns. They are
    # kept separate from the Tier-1 ones rather than reusing them: the two
    # tiers score in different ranges, so one pooled column could not be
    # calibrated against anything.
    "Resolution_Source",
    "Tier2_Score",
    "Tier2_Threshold",
    "Tier2_Chunks",
    "Tier2_Cache",
    "Duration_Seconds",
    "Ticket_File",
    "History_File",
)

TICKET_MARKER = "_ticket"
HISTORY_MARKER = "_history"


@dataclass(frozen=True, slots=True)
class Case:
    test_id: str
    ticket: Path
    # None, not Path(""), for "no history was found". Path("") is Path("."),
    # which is truthy and would sail past the guard in run_case to fail later
    # with a confusing message about "." having no extension.
    history: Path | None


@dataclass
class Row:
    """One line of eval_results.xlsx."""

    test_id: str
    status: str = "FAIL"
    status_code: str = ""
    reason: str = ""
    embedding_model: str = ""
    confidence_score: float | str = ""
    threshold_applied: float | str = ""
    cleared: str = ""
    candidates: int | str = ""
    resolution_source: str = ""
    tier2_score: float | str = ""
    tier2_threshold: float | str = ""
    tier2_chunks: int | str = ""
    tier2_cache: str = ""
    duration: float = 0.0
    ticket: str = ""
    history: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "Test_ID": self.test_id,
            "Status": self.status,
            "Status_Code": self.status_code,
            "Reason": self.reason,
            "Embedding_Model": self.embedding_model,
            "Confidence_Score": self.confidence_score,
            "Threshold_Applied": self.threshold_applied,
            "Cleared_Threshold": self.cleared,
            "Candidates_Considered": self.candidates,
            "Resolution_Source": self.resolution_source,
            "Tier2_Score": self.tier2_score,
            "Tier2_Threshold": self.tier2_threshold,
            "Tier2_Chunks": self.tier2_chunks,
            "Tier2_Cache": self.tier2_cache,
            "Duration_Seconds": round(self.duration, 2),
            "Ticket_File": self.ticket,
            "History_File": self.history,
        }


# --------------------------------------------------------------- discovery


def discover_from_dir(test_dir: Path, shared_history: Path | None) -> list[Case]:
    """Pair <id>_ticket.<ext> with <id>_history.<ext>.

    A ticket whose history is missing and with no --history-file to fall back on
    is left out here and reported as an error row, rather than silently skipped:
    a case that never ran is not the same as a case that failed.
    """
    tickets: dict[str, Path] = {}
    histories: dict[str, Path] = {}

    for path in sorted(test_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stem = path.stem
        lowered = stem.lower()
        if lowered.endswith(TICKET_MARKER):
            tickets[stem[: -len(TICKET_MARKER)]] = path
        elif lowered.endswith(HISTORY_MARKER):
            histories[stem[: -len(HISTORY_MARKER)]] = path

    cases = []
    for test_id, ticket in sorted(tickets.items()):
        history = histories.get(test_id) or shared_history
        cases.append(Case(test_id, ticket, Path(history) if history else None))

    orphan_histories = sorted(set(histories) - set(tickets))
    if orphan_histories:
        logger.warning(
            "%d history file(s) have no matching ticket and were ignored: %s",
            len(orphan_histories),
            ", ".join(orphan_histories),
        )
    return cases


def discover_from_list(run_list: Path, shared_history: Path | None) -> list[Case]:
    """Read Test_ID / Ticket_File / History_File from a run list.

    Paths are resolved relative to the list itself, so the whole test set stays
    portable between machines -- which matters when it is assembled on one
    laptop and run on another.
    """
    from sps.file_reader import read_header_and_rows, is_blank

    header, rows = read_header_and_rows(run_list, "Run list")
    index = {str(h).strip().casefold().replace(" ", "_"): i for i, h in enumerate(header) if h}
    for required in ("test_id", "ticket_file"):
        if required not in index:
            raise UnsupportedFileType(
                f"Run list must have a {required!r} column. Found: "
                + ", ".join(repr(str(h)) for h in header if str(h).strip())
            )

    base = run_list.parent
    cases = []
    for row in rows:
        if is_blank(row):
            continue

        def cell(name: str) -> str:
            position = index.get(name)
            if position is None or position >= len(row):
                return ""
            return str(row[position] or "").strip()

        history_cell = cell("history_file")
        history = (base / history_cell) if history_cell else shared_history
        cases.append(
            Case(
                test_id=cell("test_id") or f"row{len(cases) + 1}",
                ticket=base / cell("ticket_file"),
                history=Path(history) if history else None,
            )
        )
    return cases


# ------------------------------------------------------------------ running


def run_case(case: Case, work_dir: Path, threshold: float | None) -> Row:
    """Run one ticket through the real resolver path."""
    import scripts.run_resolver as resolver

    row = Row(
        test_id=case.test_id,
        ticket=case.ticket.name,
        history=case.history.name if case.history else "",
    )
    started = time.time()

    if case.history is None:
        row.status_code = "INVALID_INPUT"
        row.reason = (
            f"No history file for case {case.test_id!r}: expected "
            f"{case.test_id}{HISTORY_MARKER}.csv/.xlsx beside the ticket, "
            f"or pass --history-file."
        )
        row.duration = time.time() - started
        return row

    argv = [
        "--ticket-file", str(case.ticket),
        "--history-file", str(case.history),
        "--output-dir", str(work_dir / case.test_id),
    ]
    if threshold is not None:
        argv += ["--threshold", str(threshold)]

    try:
        outcome = resolver.resolve(resolver.parse_args(argv), work_dir / case.test_id)
    except Exception as exc:
        # One bad case must not end the batch.
        logger.error("case %s raised:\n%s", case.test_id, traceback.format_exc())
        row.status_code = "INFRASTRUCTURE_ERROR"
        row.reason = f"{type(exc).__name__}: {exc}"
        row.duration = time.time() - started
        return row

    row.status = "PASS" if outcome.code in resolver.SUCCESS_CODES else "FAIL"
    row.status_code = outcome.code
    row.reason = " ".join(outcome.reason.split())
    row.embedding_model = outcome.embedding_model
    row.duration = time.time() - started

    # Populated whenever anything was scored, which is the point: the rejected
    # cases are the ones that tell you whether the threshold is set correctly.
    if outcome.embedding_model:
        row.confidence_score = round(outcome.top_score, 4)
        row.threshold_applied = round(outcome.threshold_used, 4)
        row.cleared = "YES" if outcome.top_score >= outcome.threshold_used else "NO"
        row.candidates = outcome.candidates_considered

    row.resolution_source = outcome.resolution_source
    row.tier2_cache = outcome.tier2_cache_state
    # Populated only when Tier 2 actually ran, so a blank means "Tier 1
    # answered" rather than "Tier 2 scored zero".
    if outcome.tier2_cache_state and outcome.tier2_cache_state != "absent":
        row.tier2_score = round(outcome.tier2_top_score, 4)
        row.tier2_threshold = round(outcome.tier2_threshold, 4)
        row.tier2_chunks = outcome.tier2_chunks
    return row


def summarise(rows: list[Row]) -> str:
    """A console digest, so the distribution is visible without opening Excel."""
    from collections import Counter

    lines = [f"\n{len(rows)} case(s)"]
    for label, counts in (
        ("status", Counter(r.status_code or "ERROR" for r in rows)),
        ("encoder", Counter(r.embedding_model or "(none)" for r in rows)),
    ):
        lines.append(f"  by {label}:")
        for key, count in counts.most_common():
            lines.append(f"    {count:4d}  {key}")

    scored = sorted(float(r.confidence_score) for r in rows if r.confidence_score != "")
    if scored:
        def pct(fraction: float) -> float:
            return scored[min(int(fraction * (len(scored) - 1)), len(scored) - 1)]

        lines.append(f"  score distribution over {len(scored)} scored case(s):")
        lines.append(f"    min {scored[0]:.4f}   p25 {pct(0.25):.4f}   median {pct(0.5):.4f}")
        lines.append(f"    p75 {pct(0.75):.4f}   p90 {pct(0.90):.4f}   max {scored[-1]:.4f}")
        lines.append(
            "  Use these to set the threshold: it should sit above the scores of "
            "cases you judged wrong and below the ones you judged right."
        )

    # A threshold belongs to one embedding space. If Azure answered for some
    # cases and the local fallback for others, the percentiles above pool two
    # distributions and describe neither -- and nothing else in the run flags
    # it, because falling back is normal behaviour rather than an error.
    encoders = {r.embedding_model for r in rows if r.embedding_model}
    if len(encoders) > 1:
        lines.append(
            f"  WARNING: {len(encoders)} encoders ran in this batch ("
            + ", ".join(sorted(encoders))
            + "). The scores above are pooled across different embedding spaces, "
            "so the distribution describes neither. Get the encoder consistent "
            "and re-run before setting a threshold from it."
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_eval_batch",
        description="Run a directory of test tickets and write eval_results.xlsx.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-dir", help="Directory of <id>_ticket.* / <id>_history.* pairs")
    source.add_argument("--run-list", help="A .csv/.xlsx of Test_ID, Ticket_File, History_File")
    parser.add_argument(
        "--history-file",
        default=None,
        help="History used by any case without one of its own -- the usual shape, "
        "where many tickets share a single extract.",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Where eval_results.xlsx is written (default: cwd)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the gate for every case. Leave unset to use the per-encoder default; "
        "the raw score is reported either way, so the threshold does not affect calibration.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # The batch is the run that calibrates the Azure threshold, so it has to
    # see the same credentials the resolver would. Without this the whole batch
    # falls back to the local encoder and measures the wrong distribution --
    # quietly, because falling back is normal behaviour, not an error.
    from scripts.run_resolver import _load_dotenv

    _load_dotenv()

    output_dir = Path(args.output_dir)
    shared_history = Path(args.history_file) if args.history_file else None

    try:
        if shared_history is not None:
            validate_file_type(shared_history, "Shared history file")
        if args.run_list:
            cases = discover_from_list(validate_file_type(Path(args.run_list), "Run list"),
                                       shared_history)
        else:
            test_dir = Path(args.test_dir)
            if not test_dir.is_dir():
                print(f"Not a directory: {test_dir}", file=sys.stderr)
                return EXIT_BAD_INPUT
            cases = discover_from_dir(test_dir, shared_history)
    except (UnsupportedFileType, OSError, ValueError) as exc:
        print(f"Could not build the case list: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if not cases:
        print(
            "No cases found. Expected files named <id>_ticket.csv/.xlsx "
            "(optionally paired with <id>_history.csv/.xlsx).",
            file=sys.stderr,
        )
        return EXIT_BAD_INPUT

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_runs"

    rows: list[Row] = []
    started = time.time()
    for position, case in enumerate(cases, 1):
        print(f"[{position}/{len(cases)}] {case.test_id}", file=sys.stderr, flush=True)
        rows.append(run_case(case, work_dir, args.threshold))

    from service.excel_output import write_rows

    results_path = output_dir / RESULTS_FILE
    write_rows(results_path, RESULT_COLUMNS, [r.as_dict() for r in rows])

    print(summarise(rows), file=sys.stderr)
    print(f"\n{len(rows)} case(s) in {time.time() - started:.1f}s -> {results_path}",
          file=sys.stderr)

    # Non-zero only if something was genuinely broken. A case that legitimately
    # failed its gate is a result, not an error.
    broken = sum(1 for r in rows if r.status_code == "INFRASTRUCTURE_ERROR")
    if broken:
        print(f"{broken} case(s) hit an infrastructure error.", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
