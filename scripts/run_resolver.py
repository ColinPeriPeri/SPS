"""Resolve one SPS ticket against a historical file. No vector database.

    python -m scripts.run_resolver --ticket-file ticket.xlsx \
        --history-file history.csv --output-dir .\\out

Writes two workbooks into --output-dir:

  status.xlsx   ALWAYS, including an early abort or an unhandled exception.
                Execution_Timestamp, Status (PASS/FAIL), Status_Code, Reason.
  output.xlsx   Only when Status is PASS.
                Part_Number, AI_Recommendation, Justification,
                Confidence_Score, Referenced_SPS_IDs.

Both are written atomically, and both are cleared before any work starts, so a
process killed outright leaves no stale result to be mistaken for this run's.

Exit codes are retained alongside the status sheet, since a caller can branch on
them without opening a workbook: 0 the run completed (PASS or a legitimate
FAIL), 1 an infrastructure fault, 2 the inputs could not be read.

Order of work is cheapest-first, so nothing expensive runs for a ticket that
cannot succeed: validate, then filter history by part, then cap, then embed,
then gate, then the LLM.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The model, its dimension and the threshold are all properties of the
# embedding space, defined once beside the code that owns them.
from sps.config import DEFAULT_DIMENSION, DEFAULT_MODEL_NAME as DEFAULT_MODEL
from sps.retrieval.in_memory import DEFAULT_CONFIDENCE_THRESHOLD as DEFAULT_THRESHOLD

logger = logging.getLogger("sps.resolver")

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_BAD_INPUT = 2

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

CODE_SUCCESS = "SUCCESS"
CODE_INVALID_INPUT = "INVALID_INPUT"
CODE_NO_MATCHES = "NO_MATCHES"
CODE_BELOW_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"
CODE_AUDIT_REJECTED = "LLM_AUDIT_REJECTED"
CODE_INFRASTRUCTURE = "INFRASTRUCTURE_ERROR"

STATUS_FILE = "status.xlsx"
OUTPUT_FILE = "output.xlsx"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_resolver",
        description="Resolve one SPS ticket against a historical file.",
        epilog=(
            "Azure credentials are read from the environment only "
            "(AZURE_OPENAI_ENDPOINT / _API_KEY / _DEPLOYMENT)."
        ),
    )
    parser.add_argument("--ticket-file", required=True, help="Single-ticket .xlsx or .csv")
    parser.add_argument("--history-file", required=True, help="Historical records .csv or .xlsx")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Where status.xlsx and output.xlsx are written (default: current directory)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Confidence threshold (default {DEFAULT_THRESHOLD}, or SPS_CONFIDENCE_THRESHOLD)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def read_ticket(path: Path) -> dict[str, Any]:
    """Read the first data row of a single-ticket file.

    Header keys are returned verbatim; the caller resolves spelling via
    IncomingTicket.from_dict, which already matches keys case-insensitively.
    """
    from sps.retrieval.in_memory import _iter_rows

    rows = _iter_rows(path)
    try:
        headers = next(rows)
    except StopIteration:
        raise ValueError(f"Ticket file is empty: {path}") from None

    for row in rows:
        if row is None or all(c in (None, "") for c in row):
            continue
        return {
            str(h).strip(): (row[i] if i < len(row) else "")
            for i, h in enumerate(headers)
            if str(h or "").strip()
        }
    raise ValueError(f"Ticket file has a header but no data row: {path}")


def _load_dotenv() -> None:
    """Best-effort .env load.

    A process launched by a UiPath robot does not necessarily inherit an
    interactive shell's environment, so the deployment's .env is read here.
    Real environment variables always win (`override=False`).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(output_dir: Path, code: str, reason: str) -> None:
    from service.excel_output import STATUS_COLUMNS, write_rows

    status = STATUS_PASS if code == CODE_SUCCESS else STATUS_FAIL
    write_rows(
        output_dir / STATUS_FILE,
        STATUS_COLUMNS,
        [
            {
                "Execution_Timestamp": _now(),
                "Status": status,
                "Status_Code": code,
                "Reason": " ".join(str(reason).split()),
            }
        ],
    )
    logger.info("status: %s / %s -- %s", status, code, reason)


def write_output(output_dir: Path, part_number: str, result, sps_ids) -> None:
    from service.excel_output import RESULT_COLUMNS, write_rows

    write_rows(
        output_dir / OUTPUT_FILE,
        RESULT_COLUMNS,
        [
            {
                "Part_Number": part_number,
                "AI_Recommendation": result.ai_recommendation,
                "Justification": result.justification,
                "Confidence_Score": result.confidence,
                "Referenced_SPS_IDs": ", ".join(sps_ids),
            }
        ],
    )


def resolve(args: argparse.Namespace, output_dir: Path) -> tuple[str, str, int]:
    """Run the pipeline. Returns (status_code, reason, exit_code)."""
    import asyncio
    import os

    from sps.config import LLMSettings
    from sps.contracts import IncomingTicket
    from sps.embedding import BGEEmbedder
    from sps.config import EmbeddingSettings
    from sps.generation import ActorCriticLoop, AzureOpenAIChatClient
    from sps.retrieval.in_memory import HistoryError, InMemoryRetriever
    from sps.validators import normalize_part_number, validate_ticket

    ticket_path = Path(args.ticket_file)
    history_path = Path(args.history_file)
    for label, path in (("Ticket", ticket_path), ("History", history_path)):
        if not path.exists():
            return CODE_INVALID_INPUT, f"{label} file not found: {path}", EXIT_BAD_INPUT

    try:
        raw = read_ticket(ticket_path)
    except (ValueError, OSError) as exc:
        return CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT

    ticket = IncomingTicket.from_dict(raw)
    part_number = normalize_part_number(ticket.part_number)

    # Gate before anything expensive: no file scan, no model load, no LLM.
    invalid = validate_ticket(ticket.part_number, ticket.problem_description)
    if invalid:
        return invalid.code, invalid.reason, EXIT_OK

    threshold = args.threshold
    if threshold is None:
        raw_threshold = os.environ.get("SPS_CONFIDENCE_THRESHOLD", "").strip()
        threshold = float(raw_threshold) if raw_threshold else DEFAULT_THRESHOLD

    embedder = BGEEmbedder(
        EmbeddingSettings(
            model_name=os.environ.get("SPS_EMBEDDING_MODEL", "").strip() or DEFAULT_MODEL,
            dimension=int(os.environ.get("SPS_EMBEDDING_DIM", "").strip() or DEFAULT_DIMENSION),
        )
    )
    retriever = InMemoryRetriever(
        embedder=embedder,
        history_path=history_path,
        confidence_threshold=threshold,
    )

    try:
        candidates = retriever.retrieve(ticket)
    except HistoryError as exc:
        return CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT

    stats = retriever.stats
    logger.info("retrieval stats: %s", stats.as_dict())

    if stats.usable == 0:
        return (
            CODE_NO_MATCHES,
            f"No usable history for part {part_number}: "
            f"{stats.part_matches} row(s) matched the part out of {stats.rows_scanned} scanned.",
            EXIT_OK,
        )
    if not candidates:
        return (
            CODE_BELOW_THRESHOLD,
            f"Best match {stats.top_score:.4f} is below the {threshold:.2f} threshold "
            f"across {stats.capped_to} candidate(s) for part {part_number}.",
            EXIT_OK,
        )

    loop = ActorCriticLoop(AzureOpenAIChatClient(LLMSettings.from_env()), LLMSettings.from_env())
    outcome = asyncio.run(loop.run(ticket, candidates))

    if not outcome.succeeded or outcome.draft is None:
        if outcome.infrastructure_failure:
            # A dependency outage is not a content rejection: it must not look
            # like a legitimate refusal, or a caller retries nothing.
            logger.error("dependency failure: %s", outcome.failure_reason)
            return CODE_INFRASTRUCTURE, outcome.failure_reason, EXIT_INFRASTRUCTURE
        return CODE_AUDIT_REJECTED, outcome.failure_reason, EXIT_OK

    from sps.output import success

    result = success(
        recommendation=outcome.draft.recommendation,
        justification=outcome.draft.justification
        or f"Synthesized from {len(candidates)} historical record(s) for part {part_number}.",
        top_score=stats.top_score,
        candidates=candidates,
    )
    write_output(output_dir, part_number, result, result.sps_ids_referred)
    return (
        CODE_SUCCESS,
        f"Resolved from {len(candidates)} record(s) at {result.confidence} confidence.",
        EXIT_OK,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    _load_dotenv()

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Clear both before any work: a process killed mid-run must leave no
        # stale workbook that reads as this run's answer.
        for name in (STATUS_FILE, OUTPUT_FILE):
            (output_dir / name).unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Cannot prepare output directory %s: %s", output_dir, exc)
        return EXIT_INFRASTRUCTURE

    try:
        code, reason, exit_code = resolve(args, output_dir)
    except Exception:
        # status.xlsx is written even here: an unhandled fault must still leave
        # the caller a row explaining why, not an empty directory.
        logger.error("Unhandled error:\n%s", traceback.format_exc())
        code, reason, exit_code = (
            CODE_INFRASTRUCTURE,
            "Unhandled error; see stderr for the traceback.",
            EXIT_INFRASTRUCTURE,
        )

    try:
        write_status(output_dir, code, reason)
    except Exception:
        logger.error("Could not write %s:\n%s", STATUS_FILE, traceback.format_exc())
        return EXIT_INFRASTRUCTURE
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
