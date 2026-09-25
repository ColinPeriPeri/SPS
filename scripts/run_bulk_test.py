"""Run a workbook of tickets through the real pipeline, one row at a time.

    python -m scripts.run_bulk_test --tickets tickets.xlsx --history history.csv

Reads a sheet where each row is a ticket, resolves every one, and writes a copy
carrying **every original column verbatim plus the result columns**. The input
is never modified: it stays a clean, re-runnable fixture, and two runs can be
diffed against each other.

This is the review tool. `run_eval_batch.py` is the calibration tool -- it takes
a directory of one-ticket-per-file cases and reports scores, not prose. Use this
one to read what the system actually recommended; use that one to decide where a
threshold belongs.

Nothing here reimplements the pipeline. Each row goes through the same
`resolve()` the UiPath wrapper calls, so a result in this sheet is the result
production would have produced for that ticket.

**The history file is re-scanned for every ticket**, because that is what
`resolve()` does per invocation and this deliberately does not work around it.
At 300k rows that is ~1.5 s per ticket as CSV and ~40 s as .xlsx -- so prefer a
CSV history here, and expect a large bulk run to be dominated by it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

# The same cell renderer the ticket contracts use. Shared rather than copied:
# it is what stops an Excel-sourced part number of 1243951.0 reaching the
# pipeline as "1243951.0", an id that matches nothing, and a second copy of
# that rule is a second place for the fix to be lost.
from sps.contracts import _clean as clean_cell
from sps.file_reader import SUPPORTED_SUFFIXES, UnsupportedFileType, is_blank, validate_file_type

logger = logging.getLogger("sps.bulk")

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_BAD_INPUT = 2

# Appended to the right of whatever the input already had. Order is
# reading order: the verdict, then what a reviewer judges, then the numbers
# that say how close a rejected row came.
RESULT_COLUMNS = (
    "Status",
    "Status_Code",
    "Reason",
    "AI_Recommendation",
    "Justification",
    "Confidence_Score",
    "Referenced_Sources",
    "Resolution_Source",
    "Tier1_Score",
    "Tier2_Score",
    # Why this row produced nothing, as one of a fixed set of values rather
    # than as prose. Reason says it in English already; this is the column you
    # can group by, which is what turns "your checks are too strict" from an
    # argument into a count.
    "No_Recommendation_Reason",
    # The single best precedent, gate or no gate, exactly as output.xlsx shows
    # it to the reviewer.
    "Closest_Matching_Solution",
    # What the reviewer should read before pasting, when the recommendation is
    # archive text sent unchanged.
    "Cascade_Warnings",
    # The historical solutions (or 0250 sections) the Actor was actually shown.
    # Put beside its recommendation because the commonest question about a bad
    # answer is "what was it looking at?", and answering it from the SPS IDs
    # alone means a lookup per row.
    "Matched_Solutions",
    "Embedding_Model",
    "Duration_Seconds",
)

# A row that was never attempted, so the output still lines up with the input.
NOT_RUN = "NOT RUN"

DEFAULT_STOP_AFTER = 5


def read_tickets(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (column names, rows) with every column preserved verbatim."""
    from sps.file_reader import read_header_and_rows

    header, raw_rows = read_header_and_rows(path, "Ticket workbook")
    columns = [str(h).strip() for h in header if str(h or "").strip()]
    if not columns:
        raise UnsupportedFileType(f"{path.name!r} has no column headings.")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if is_blank(raw):
            continue
        rows.append(
            {
                name: clean_cell(raw[i]) if i < len(raw) else ""
                for i, name in enumerate(columns)
            }
        )
    return columns, rows


def resolve_columns(original: Sequence[str]) -> dict[str, str]:
    """Map each result column to the name it will be written under.

    A ticket sheet that already has a `Status` column would otherwise produce
    two columns of that name, and whichever pandas kept would be anyone's
    guess. Collisions get an `_AI` suffix instead.
    """
    taken = {c.casefold() for c in original}
    mapping: dict[str, str] = {}
    for name in RESULT_COLUMNS:
        final = name if name.casefold() not in taken else f"{name}_AI"
        mapping[name] = final
        taken.add(final.casefold())
    return mapping


def write_ticket_file(row: dict[str, Any], columns: Sequence[str], path: Path) -> Path:
    """One row as a single-ticket CSV, which is what resolve() consumes.

    Every column goes in, not just the four the pipeline reads: whatever
    spelling of Problem_Description or Issue_Type this sheet uses, the ticket
    contract resolves it case-insensitively at the other end.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow([row.get(c, "") for c in columns])
    return path


def result_for(outcome, duration: float) -> dict[str, str]:
    """Flatten one ResolveOutcome into the appended columns."""
    import scripts.run_resolver as resolver

    result = outcome.result
    if result is None and outcome.exit_code == resolver.EXIT_OK:
        # The run concluded without a recommendation. This is the same row
        # main() writes into output.xlsx for that case, reused rather than
        # rebuilt so the bulk sheet and the per-ticket workbook cannot
        # disagree about what a refusal looks like.
        #
        # Deliberately not done for exit 1 or 2: no verdict was formed there,
        # and "Solution not found." would record one that never existed.
        result = resolver.unresolved_result(outcome)

    passed = outcome.code in resolver.SUCCESS_CODES
    return {
        "Status": "PASS" if passed else "FAIL",
        "Status_Code": outcome.code,
        "No_Recommendation_Reason": "" if passed else (outcome.stop_reason or outcome.code),
        "Closest_Matching_Solution": outcome.closest_match,
        "Cascade_Warnings": result.cascade_warnings if result is not None else "",
        "Reason": " ".join(str(outcome.reason).split()),
        "AI_Recommendation": result.ai_recommendation if result else "",
        "Justification": result.justification if result else "",
        "Confidence_Score": result.confidence if result else "",
        "Referenced_Sources": ", ".join(result.referenced_sources) if result else "",
        "Resolution_Source": result.resolution_source if result else "",
        # Blank rather than 0 when nothing was scored: an unknown part never
        # reaches the encoder, and 0 reads as a match that scored badly.
        "Tier1_Score": round(outcome.top_score, 4) if outcome.top_score else "",
        "Tier2_Score": round(outcome.tier2_top_score, 4) if outcome.tier2_top_score else "",
        "Matched_Solutions": "\n\n".join(outcome.evidence + outcome.tier2_evidence),
        "Embedding_Model": outcome.embedding_model,
        "Duration_Seconds": round(duration, 2),
    }


def blank_result(reason: str) -> dict[str, str]:
    """The appended columns for a row that was never attempted."""
    values = {name: "" for name in RESULT_COLUMNS}
    values["Status"] = NOT_RUN
    values["Reason"] = reason
    return values


def run_row(row: dict[str, Any], columns: Sequence[str], work: Path, args) -> dict[str, str]:
    """Resolve one ticket through the real pipeline."""
    import scripts.run_resolver as resolver

    started = time.time()
    ticket_file = write_ticket_file(row, columns, work / "ticket.csv")

    argv = [
        "--ticket-file", str(ticket_file),
        "--history-file", str(args.history),
        "--output-dir", str(work / "run"),
    ]
    if args.threshold is not None:
        argv += ["--threshold", str(args.threshold)]
    if args.tier2_threshold is not None:
        argv += ["--tier2-threshold", str(args.tier2_threshold)]
    if args.docs_dir:
        argv += ["--docs-dir", str(args.docs_dir)]
    if args.no_tier2:
        argv.append("--no-tier2")

    try:
        outcome = resolver.resolve(resolver.parse_args(argv), work / "run")
    except Exception as exc:
        # One bad row must not end a run that may have taken an hour.
        logger.error("row raised:\n%s", traceback.format_exc())
        values = blank_result(f"{type(exc).__name__}: {exc}")
        values["Status"] = "FAIL"
        values["Status_Code"] = "INFRASTRUCTURE_ERROR"
        values["Duration_Seconds"] = round(time.time() - started, 2)
        return values

    return result_for(outcome, time.time() - started)


def summarise(results: Sequence[dict[str, str]]) -> str:
    from collections import Counter

    lines = [f"\n{len(results)} row(s)"]
    counts = Counter(r.get("Status_Code") or r.get("Status") or "?" for r in results)
    lines.append("  by status:")
    for key, count in counts.most_common():
        lines.append(f"    {count:5d}  {key}")

    recommended = sum(1 for r in results if r.get("Status") == "PASS")
    lines.append(f"  {recommended} of {len(results)} produced a recommendation")

    # The breakdown that answers "why not more?". Status_Code above says how far
    # the pipeline got; this says what stopped it, which is a different question
    # and the one worth arguing from. GUARD/JUDGE rows mean our checks are the
    # binding constraint; the rest mean the archive is.
    reasons = Counter(
        r["No_Recommendation_Reason"]
        for r in results
        if str(r.get("No_Recommendation_Reason", "")) != ""
    )
    if reasons:
        blocked = sum(v for k, v in reasons.items() if k.startswith(("GATE_", "JUDGE_")))
        lines.append("  of those without one, why:")
        for key, count in reasons.most_common():
            lines.append(f"    {count:5d}  {key}")
        total = sum(reasons.values())
        lines.append(
            f"    -> {blocked} of {total} were stopped by a supplier-safety check; "
            f"{total - blocked} by the data."
        )

    scored = sorted(
        float(r["Tier1_Score"]) for r in results if str(r.get("Tier1_Score", "")) != ""
    )
    if scored:
        def pct(fraction: float) -> float:
            return scored[min(int(fraction * (len(scored) - 1)), len(scored) - 1)]

        lines.append(f"  Tier-1 score over {len(scored)} scored row(s):")
        lines.append(f"    min {scored[0]:.4f}   median {pct(0.5):.4f}   max {scored[-1]:.4f}")
        lines.append(
            "  Sort the sheet by Tier1_Score and read where the good answers stop: "
            "that is where the threshold belongs."
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_bulk_test",
        description="Resolve a workbook of tickets and write a copy with the results appended.",
    )
    parser.add_argument("--tickets", required=True, help="Sheet of tickets, one per row")
    parser.add_argument("--history", required=True, help="Historical records (.csv preferred)")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the results copy (default: <tickets>_results.xlsx beside the input). "
        "The input is never modified.",
    )
    parser.add_argument("--docs-dir", default=None, help="0250 standards folder for Tier 2")
    parser.add_argument("--no-tier2", action="store_true", help="History only")
    parser.add_argument("--threshold", type=float, default=None, help="Tier-1 gate override")
    parser.add_argument("--tier2-threshold", type=float, default=None, help="Tier-2 gate override")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Resolve only the first N rows. The rest still appear, marked NOT RUN, "
        "so the sheet stays aligned with the input.",
    )
    parser.add_argument(
        "--stop-after-errors", type=int, default=DEFAULT_STOP_AFTER,
        help=f"Give up after this many CONSECUTIVE infrastructure errors "
        f"(default {DEFAULT_STOP_AFTER}; 0 disables). Wrong credentials would "
        "otherwise burn one failing call per row for the whole sheet.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def default_output(tickets: Path) -> Path:
    return tickets.with_name(f"{tickets.stem}_results.xlsx")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    from scripts.run_resolver import _load_dotenv

    _load_dotenv()

    tickets_path = Path(args.tickets)
    history_path = Path(args.history)
    try:
        validate_file_type(tickets_path, "Ticket workbook")
        validate_file_type(history_path, "History file")
        for label, path in (("Ticket workbook", tickets_path), ("History file", history_path)):
            if not path.exists():
                print(f"{label} not found: {path}", file=sys.stderr)
                return EXIT_BAD_INPUT
        columns, rows = read_tickets(tickets_path)
    except (UnsupportedFileType, OSError, ValueError) as exc:
        print(f"Could not read the inputs: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if not rows:
        print(f"No ticket rows in {tickets_path}", file=sys.stderr)
        return EXIT_BAD_INPUT

    naming = resolve_columns(columns)
    clashed = [n for n, final in naming.items() if n != final]
    if clashed:
        print(
            f"{len(clashed)} result column(s) clash with the input and were suffixed "
            f"'_AI': {', '.join(clashed)}",
            file=sys.stderr,
        )

    output_path = Path(args.output) if args.output else default_output(tickets_path)
    started = time.time()
    results: list[dict[str, str]] = []
    consecutive = 0
    stopped = ""

    with tempfile.TemporaryDirectory(prefix="sps_bulk_") as tmp:
        work = Path(tmp)
        for position, row in enumerate(rows, 1):
            if stopped:
                results.append(blank_result(stopped))
                continue
            if args.limit is not None and position > args.limit:
                results.append(blank_result(f"Beyond --limit {args.limit}."))
                continue

            print(f"[{position}/{len(rows)}]", file=sys.stderr, flush=True)
            values = run_row(row, columns, work, args)
            results.append(values)

            if values.get("Status_Code") == "INFRASTRUCTURE_ERROR":
                consecutive += 1
                if args.stop_after_errors and consecutive >= args.stop_after_errors:
                    stopped = (
                        f"Stopped: {consecutive} consecutive infrastructure errors. "
                        "Check the Reason on the rows above."
                    )
                    print(f"\n{stopped}", file=sys.stderr)
            else:
                consecutive = 0

    # Original columns first, verbatim, then the appended ones -- so the sheet
    # reads as the input with answers alongside it.
    out_columns = list(columns) + [naming[n] for n in RESULT_COLUMNS]
    merged = [
        {**row, **{naming[n]: values.get(n, "") for n in RESULT_COLUMNS}}
        for row, values in zip(rows, results)
    ]

    from service.excel_output import write_rows

    try:
        write_rows(output_path, out_columns, merged)
    except Exception:
        logger.error("Could not write %s:\n%s", output_path, traceback.format_exc())
        return EXIT_INFRASTRUCTURE

    print(summarise(results), file=sys.stderr)
    print(
        f"\n{len(rows)} row(s) in {time.time() - started:.1f}s -> {output_path}",
        file=sys.stderr,
    )

    broken = sum(1 for r in results if r.get("Status_Code") == "INFRASTRUCTURE_ERROR")
    if broken:
        print(f"{broken} row(s) hit an infrastructure error.", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
