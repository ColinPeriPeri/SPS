"""Resolve one SPS ticket against a historical file. No vector database.

    python -m scripts.run_resolver --ticket-file ticket.xlsx \
        --history-file history.csv --output-dir .\\out

Writes two workbooks into --output-dir:

  status.xlsx   ALWAYS, including an early abort or an unhandled exception.
                Execution_Timestamp, Status (PASS/FAIL), Status_Code, Reason,
                Embedding_Model. The last names the encoder that actually ran,
                so support can see how often the Azure fallback fires; it is
                blank when the run aborted before anything was encoded.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The model, its dimension and the threshold are all properties of the
# embedding space, defined once beside the code that owns them.
from sps.config import DEFAULT_DIMENSION, DEFAULT_MODEL_NAME as DEFAULT_MODEL
from sps.retrieval.in_memory import (
    AZURE_EMBEDDING_THRESHOLD,
    LOCAL_EMBEDDING_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD as DEFAULT_THRESHOLD,
)

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

    Format-agnostic: .csv and .xlsx both arrive here as a header row plus data
    rows, and the keys are returned verbatim. IncomingTicket.from_dict resolves
    spelling, matching keys case-insensitively.
    """
    from sps.file_reader import FileReadError, is_blank, read_header_and_rows

    headers, rows = read_header_and_rows(path, "Ticket file")
    for row in rows:
        if is_blank(row):
            continue
        return {
            str(h).strip(): (row[i] if i < len(row) else "")
            for i, h in enumerate(headers)
            if str(h or "").strip()
        }
    raise FileReadError(f"Ticket file {path.name!r} has a header but no data row.")


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


@dataclass(frozen=True, slots=True)
class ResolveOutcome:
    """What one run produced.

    A dataclass rather than a tuple because the batch evaluator needs the raw
    similarity alongside the outcome: calibrating a threshold means seeing the
    score distribution, including for the cases the threshold rejected.
    """

    code: str
    reason: str
    exit_code: int
    embedding_model: str = ""
    # The best cosine found, whether or not it cleared the gate. Zero means
    # nothing was scored -- no history for the part, or an abort before
    # embedding.
    top_score: float = 0.0
    threshold_used: float = 0.0
    candidates_considered: int = 0


def _reason_with_marker(reason: str, embedding_model: str) -> str:
    """Append [Azure] or [Local] so the encoder is visible in Reason itself."""
    flat = " ".join(str(reason).split())
    if not embedding_model:
        return flat
    marker = "[Azure]" if embedding_model.startswith("azure") else "[Local]"
    return f"{flat} {marker}"


def _describe_backend(stats) -> str:
    """One short token naming the encoder, for the status sheet.

    Support reads this to see how often the fallback is firing, so it names the
    model rather than just "azure" or "local".
    """
    if not stats.backend:
        return ""
    detail = stats.backend_detail or stats.backend
    return f"{stats.backend}:{detail}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(output_dir: Path, code: str, reason: str, embedding_model: str = "") -> None:
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
                # The marker is appended to Reason as well as carried in its
                # own column: a support engineer skimming the sheet, or a
                # caller reading only the first four columns, still sees which
                # encoder ran without having to know the column exists.
                "Reason": _reason_with_marker(reason, embedding_model),
                # Blank when the run aborted before embedding, which is itself
                # information: nothing was encoded.
                "Embedding_Model": embedding_model,
            }
        ],
    )
    logger.info("status: %s / %s [%s] -- %s", status, code, embedding_model or "none", reason)


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


def resolve(args: argparse.Namespace, output_dir: Path) -> ResolveOutcome:
    """Run the pipeline once and report what happened."""
    import asyncio
    import os

    from sps.config import LLMSettings
    from sps.contracts import IncomingTicket
    from sps.embedding import BGEEmbedder
    from sps.config import EmbeddingSettings
    from sps.generation import ActorCriticLoop, AzureOpenAIChatClient
    from sps.retrieval.in_memory import HistoryError, InMemoryRetriever
    from sps.validators import normalize_part_number, validate_ticket

    from sps.file_reader import FileReadError, UnsupportedFileType, validate_file_type

    ticket_path = Path(args.ticket_file)
    history_path = Path(args.history_file)

    # Extension check first, on both files, before anything is opened, any model
    # is loaded or any history is scanned. The wrong attachment is a business
    # problem for whoever assembled the ticket, not an I/O fault: it reports
    # INVALID_INPUT and exits 0, so the caller faults the item without retrying.
    for label, path in (("Ticket file", ticket_path), ("History file", history_path)):
        try:
            validate_file_type(path, label)
        except UnsupportedFileType as exc:
            return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_OK)

    # A file that is the right type but absent or unreadable is genuine I/O
    # trouble, and keeps exit 2 so a human is alerted rather than the item
    # being quietly faulted.
    for label, path in (("Ticket file", ticket_path), ("History file", history_path)):
        if not path.exists():
            return ResolveOutcome(
                CODE_INVALID_INPUT, f"{label} not found: {path}", EXIT_BAD_INPUT
            )

    try:
        raw = read_ticket(ticket_path)
    except (FileReadError, ValueError, OSError) as exc:
        return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT)

    ticket = IncomingTicket.from_dict(raw)
    part_number = normalize_part_number(ticket.part_number)

    # Gate before anything expensive: no file scan, no model load, no LLM.
    invalid = validate_ticket(ticket.part_number, ticket.problem_description)
    if invalid:
        return ResolveOutcome(invalid.code, invalid.reason, EXIT_OK)

    # An explicit --threshold (or SPS_CONFIDENCE_THRESHOLD) overrides both
    # per-model defaults; otherwise the engine picks the one belonging to
    # whichever encoder answered.
    threshold_override = args.threshold
    if threshold_override is None:
        raw_threshold = os.environ.get("SPS_CONFIDENCE_THRESHOLD", "").strip()
        threshold_override = float(raw_threshold) if raw_threshold else None

    def local_embedder() -> BGEEmbedder:
        """Built only if the Azure path fails. Constructing it is free; the
        ~15 s of torch import and weight loading happens on first encode, which
        is why nothing calls this on a successful primary run."""
        return BGEEmbedder(
            EmbeddingSettings(
                model_name=os.environ.get("SPS_EMBEDDING_MODEL", "").strip() or DEFAULT_MODEL,
                dimension=int(
                    os.environ.get("SPS_EMBEDDING_DIM", "").strip() or DEFAULT_DIMENSION
                ),
            )
        )

    def _threshold(name: str, fallback: float) -> float:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else fallback

    retriever = InMemoryRetriever(
        history_path=history_path,
        local_embedder_factory=local_embedder,
        confidence_threshold=threshold_override,
        azure_threshold=_threshold("AZURE_EMBEDDING_THRESHOLD", AZURE_EMBEDDING_THRESHOLD),
        local_threshold=_threshold("LOCAL_EMBEDDING_THRESHOLD", LOCAL_EMBEDDING_THRESHOLD),
    )

    try:
        candidates = retriever.retrieve(ticket)
    except HistoryError as exc:
        return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT)

    stats = retriever.stats
    logger.info("retrieval stats: %s", stats.as_dict())
    model = _describe_backend(stats)

    measured = dict(
        embedding_model=model,
        top_score=stats.top_score,
        threshold_used=stats.threshold_used,
        candidates_considered=stats.capped_to,
    )

    if stats.usable == 0:
        return ResolveOutcome(
            CODE_NO_MATCHES,
            f"No usable history for part {part_number}: "
            f"{stats.part_matches} row(s) matched the part out of {stats.rows_scanned} scanned.",
            EXIT_OK,
            **measured,
        )
    if not candidates:
        return ResolveOutcome(
            CODE_BELOW_THRESHOLD,
            f"Best match {stats.top_score:.4f} is below the "
            f"{stats.threshold_used:.2f} threshold across {stats.capped_to} candidate(s) "
            f"for part {part_number}.",
            EXIT_OK,
            **measured,
        )

    loop = ActorCriticLoop(AzureOpenAIChatClient(LLMSettings.from_env()), LLMSettings.from_env())
    outcome = asyncio.run(loop.run(ticket, candidates))

    if not outcome.succeeded or outcome.draft is None:
        if outcome.infrastructure_failure:
            # A dependency outage is not a content rejection: it must not look
            # like a legitimate refusal, or a caller retries nothing.
            logger.error("dependency failure: %s", outcome.failure_reason)
            return ResolveOutcome(
                CODE_INFRASTRUCTURE, outcome.failure_reason, EXIT_INFRASTRUCTURE, **measured
            )
        return ResolveOutcome(
            CODE_AUDIT_REJECTED, outcome.failure_reason, EXIT_OK, **measured
        )

    from sps.output import success

    result = success(
        recommendation=outcome.draft.recommendation,
        justification=outcome.draft.justification
        or f"Synthesized from {len(candidates)} historical record(s) for part {part_number}.",
        top_score=stats.top_score,
        candidates=candidates,
    )
    write_output(output_dir, part_number, result, result.sps_ids_referred)
    return ResolveOutcome(
        CODE_SUCCESS,
        f"Resolved from {len(candidates)} record(s) at {result.confidence} confidence.",
        EXIT_OK,
        **measured,
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
        outcome = resolve(args, output_dir)
    except Exception:
        # status.xlsx is written even here: an unhandled fault must still leave
        # the caller a row explaining why, not an empty directory.
        logger.error("Unhandled error:\n%s", traceback.format_exc())
        outcome = ResolveOutcome(
            CODE_INFRASTRUCTURE,
            "Unhandled error; see stderr for the traceback.",
            EXIT_INFRASTRUCTURE,
        )

    try:
        write_status(output_dir, outcome.code, outcome.reason, outcome.embedding_model)
    except Exception:
        logger.error("Could not write %s:\n%s", STATUS_FILE, traceback.format_exc())
        return EXIT_INFRASTRUCTURE
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
