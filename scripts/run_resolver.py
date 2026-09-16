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
                Confidence_Score, Referenced_Sources, Resolution_Source.

Both are written atomically, and both are cleared before any work starts, so a
process killed outright leaves no stale result to be mistaken for this run's.

Exit codes are retained alongside the status sheet, since a caller can branch on
them without opening a workbook: 0 the run completed (PASS or a legitimate
FAIL), 1 an infrastructure fault, 2 the inputs could not be read.

Two tiers. Tier 1 answers from the part's own historical SPS records. When
that produces nothing usable -- no matching part, nothing similar enough, or an
Actor that declines the precedent -- Tier 2 answers from the 0250 engineering
standards instead, and says so in Resolution_Source.

Order of work is cheapest-first, so nothing expensive runs for a ticket that
cannot succeed: validate, then filter history by part, then cap, then embed,
then gate, then the LLM. Tier 2 only runs after all of that has failed, which is
why it can afford to parse and embed a document corpus.
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

# A threshold is a property of the embedding space, defined once beside the
# code that owns it. The LOCAL_* pair is dormant while the local encoder is out
# of the pipeline, and is imported so the constants stay in one place.
from sps.retrieval.in_memory import (
    AZURE_EMBEDDING_THRESHOLD,
    LOCAL_EMBEDDING_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD as DEFAULT_THRESHOLD,
)

# Tier 2 has its own thresholds because a 300-word standards chunk and a
# one-sentence defect score far lower than two defect sentences do -- measured
# at 0.78 for a chunk that answers the ticket, against Tier 1's 0.89 gate.
from sps.retrieval.doc_cache import (
    DEFAULT_DOCS_DIR,
    TIER2_AZURE_THRESHOLD,
    TIER2_LOCAL_THRESHOLD,
)

logger = logging.getLogger("sps.resolver")

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_BAD_INPUT = 2

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

# The two tiers are delineated in the status code, so a caller can tell a
# precedent-backed answer from a standards-derived one without opening
# output.xlsx. Status itself stays PASS/FAIL for both, so a workflow branching
# on Status is unaffected.
CODE_SUCCESS_HISTORICAL = "SUCCESS_HISTORICAL"
CODE_SUCCESS_DOC = "SUCCESS_0250_DOC"
# When neither tier resolves the ticket, the code says WHY the pipeline ran out
# of options, because the robot routes on it: an unknown part goes to Master
# Data, a novel defect goes to a Reliability Engineer, and a draft the Judge
# refused goes to a human reviewer. One collapsed code would send all three to
# the same queue.
CODE_NO_MATCHES = "NO_MATCHES"
CODE_BELOW_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"
CODE_AUDIT_REJECTED = "LLM_AUDIT_REJECTED"
CODE_INVALID_INPUT = "INVALID_INPUT"
CODE_INFRASTRUCTURE = "INFRASTRUCTURE_ERROR"

# `Status == PASS` has one definition.
SUCCESS_CODES = frozenset({CODE_SUCCESS_HISTORICAL, CODE_SUCCESS_DOC})

# How far a tier got before it stopped. Ordered, because the reported code is
# the FURTHEST stage either tier reached: a run where history reached the Judge
# and was refused, while the standards had nothing to say, is an audit
# rejection -- routing it to Master Data as NO_MATCHES would be wrong, since
# the part was perfectly well known.
STAGE_NOTHING = 0    # nothing retrieved at all
STAGE_GATED = 1      # retrieved, nothing cleared the threshold
STAGE_REJECTED = 2   # cleared the threshold, the Actor or Judge refused it

STAGE_CODES = {
    STAGE_NOTHING: CODE_NO_MATCHES,
    STAGE_GATED: CODE_BELOW_THRESHOLD,
    STAGE_REJECTED: CODE_AUDIT_REJECTED,
}

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
        help=f"Tier-1 confidence threshold (default {AZURE_EMBEDDING_THRESHOLD} for "
        "Azure embeddings, or SPS_CONFIDENCE_THRESHOLD)",
    )
    parser.add_argument(
        "--tier2-threshold",
        type=float,
        default=None,
        help=f"Tier-2 gate (default {TIER2_LOCAL_THRESHOLD} local / "
        f"{TIER2_AZURE_THRESHOLD} azure, or SPS_TIER2_THRESHOLD). Separate from "
        "--threshold because the two tiers score in different ranges.",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="0250 standards folder for the Tier-2 fallback "
        f"(default {DEFAULT_DOCS_DIR}, or SPS_0250_DOCS_DIR). "
        "Tier 2 is skipped when the folder holds no .docx.",
    )
    parser.add_argument(
        "--no-tier2",
        action="store_true",
        help="Answer from history only, as before the 0250 fallback existed.",
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
    """Load the deployment's .env. Kept as a name here because the batch
    evaluator imports it and tests patch it; the logic lives in sps.config so
    every entry point reaches the same one."""
    from sps.config import load_env_file

    load_env_file()


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
    # Tier 2, reported separately rather than folded into the fields above: the
    # two tiers score in different ranges, so a single "top score" column that
    # sometimes meant one and sometimes the other could not be calibrated
    # against anything.
    tier2_top_score: float = 0.0
    tier2_threshold: float = 0.0
    tier2_chunks: int = 0
    tier2_cache_state: str = ""
    resolution_source: str = ""


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

    status = STATUS_PASS if code in SUCCESS_CODES else STATUS_FAIL
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


def write_output(output_dir: Path, part_number: str, result) -> None:
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
                # SPS IDs or document citations depending on the tier;
                # Resolution_Source alongside says which kind these are.
                "Referenced_Sources": ", ".join(result.referenced_sources),
                "Resolution_Source": result.resolution_source,
            }
        ],
    )


def resolve(args: argparse.Namespace, output_dir: Path) -> ResolveOutcome:
    """Run the pipeline once and report what happened."""
    import asyncio
    import os

    from sps.config import LLMSettings
    from sps.contracts import IncomingTicket
    from sps.embedding import AzureEmbeddingError, AzureEmbeddingNotConfigured
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

    # ---- LOCAL MODEL DISABLED ------------------------------------------
    # A `local_embedder()` factory was passed to both retrievers here, so an
    # Azure failure fell back to bge-small. Restoring it means re-adding that
    # factory and the `local_embedder_factory=` arguments below; the wrapper
    # itself is untouched in sps/embedding.py.
    # ---------------------------------------------------------------------

    def _threshold(name: str, fallback: float) -> float:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else fallback

    retriever = InMemoryRetriever(
        history_path=history_path,
        confidence_threshold=threshold_override,
        azure_threshold=_threshold("AZURE_EMBEDDING_THRESHOLD", AZURE_EMBEDDING_THRESHOLD),
        local_threshold=_threshold("LOCAL_EMBEDDING_THRESHOLD", LOCAL_EMBEDDING_THRESHOLD),
    )

    try:
        candidates = retriever.retrieve(ticket)
    except HistoryError as exc:
        return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT)
    except AzureEmbeddingNotConfigured as exc:
        # Exit 2, not 1. No number of retries produces an API key, and a robot
        # cycling through fifty of them only delays the human who has to go and
        # set one.
        logger.error("embedding is not configured: %s", exc)
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_BAD_INPUT)
    except AzureEmbeddingError as exc:
        # Transient: network, timeout, throttling, a bad response. Worth a
        # retry, and there is no longer a local encoder to absorb it.
        logger.error("embedding failed: %s", exc)
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_INFRASTRUCTURE)

    stats = retriever.stats
    logger.info("retrieval stats: %s", stats.as_dict())
    model = _describe_backend(stats)

    measured = dict(
        embedding_model=model,
        top_score=stats.top_score,
        threshold_used=stats.threshold_used,
        candidates_considered=stats.capped_to,
    )

    loop = ActorCriticLoop(AzureOpenAIChatClient(LLMSettings.from_env()), LLMSettings.from_env())

    # -- Tier 1: the part's own history ------------------------------------

    if stats.usable == 0:
        tier1_stage = STAGE_NOTHING
        tier1_detail = (
            f"No usable history for part {part_number}: {stats.part_matches} row(s) "
            f"matched the part out of {stats.rows_scanned} scanned."
        )
    elif not candidates:
        tier1_stage = STAGE_GATED
        tier1_detail = (
            f"Best historical match {stats.top_score:.4f} is below the "
            f"{stats.threshold_used:.2f} threshold across {stats.capped_to} candidate(s)."
        )
    else:
        outcome = asyncio.run(loop.run(ticket, candidates))

        if outcome.succeeded and outcome.draft is not None:
            from sps.output import success

            result = success(
                recommendation=outcome.draft.recommendation,
                justification=outcome.draft.justification
                or f"Synthesized from {len(candidates)} historical record(s) "
                f"for part {part_number}.",
                top_score=stats.top_score,
                candidates=candidates,
            )
            write_output(output_dir, part_number, result)
            return ResolveOutcome(
                CODE_SUCCESS_HISTORICAL,
                f"Resolved from {len(candidates)} historical record(s) at "
                f"{result.confidence} confidence.",
                EXIT_OK,
                resolution_source=result.resolution_source,
                **measured,
            )

        if outcome.infrastructure_failure:
            # A dependency outage is not a content rejection: it must not look
            # like a legitimate refusal, or a caller retries nothing. Tier 2
            # needs the same Azure deployment, so there is nothing to fall
            # back to -- trying it would only fail again, slower.
            logger.error("dependency failure: %s", outcome.failure_reason)
            return ResolveOutcome(
                CODE_INFRASTRUCTURE, outcome.failure_reason, EXIT_INFRASTRUCTURE, **measured
            )

        # Reaching the Actor at all means the retrieval maths was satisfied;
        # an abstention and a tripped circuit breaker are both the Judge's
        # side of the pipeline declining, not a retrieval shortfall.
        tier1_stage = STAGE_REJECTED
        tier1_detail = outcome.failure_reason

    logger.info("Tier 1 produced no resolution: %s", tier1_detail)

    # -- Tier 2: the 0250 engineering standards ----------------------------

    return _resolve_from_docs(
        args=args,
        ticket=ticket,
        part_number=part_number,
        output_dir=output_dir,
        loop=loop,
        tier1_detail=tier1_detail,
        tier1_stage=tier1_stage,
        measured=measured,
    )


def _resolve_from_docs(
    args: argparse.Namespace,
    ticket,
    part_number: str,
    output_dir: Path,
    loop,
    tier1_detail: str,
    tier1_stage: int,
    measured: dict,
) -> ResolveOutcome:
    """Tier 2. Reached only once Tier 1 has produced nothing usable."""
    import asyncio
    import os

    from sps.embedding import AzureEmbeddingError, AzureEmbeddingNotConfigured
    from sps.generation.actor_critic import documentation_grounding
    from sps.retrieval.doc_cache import DocRetriever

    def _no_resolution(extra: str, tier2_stage: int, **fields) -> ResolveOutcome:
        """Both tiers tried, neither resolved it.

        The code is the furthest stage either tier reached, so the robot can
        route on how the pipeline ran out of options. Reason carries the detail
        from both tiers, including each one's best score, because that is what
        tells a human whether this was a near miss worth a threshold change or
        a genuine absence of evidence.
        """
        return ResolveOutcome(
            STAGE_CODES[max(tier1_stage, tier2_stage)],
            f"{tier1_detail} {extra}".strip(),
            EXIT_OK,
            **measured,
            **fields,
        )

    if args.no_tier2:
        # Nothing was retrieved because nothing was looked for, so Tier 1's
        # stage stands on its own.
        return _no_resolution("Tier 2 disabled by --no-tier2.", STAGE_NOTHING)

    docs_dir = (
        args.docs_dir
        or os.environ.get("SPS_0250_DOCS_DIR", "").strip()
        or DEFAULT_DOCS_DIR
    )

    def _threshold(name: str, fallback: float) -> float:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else fallback

    # Deliberately NOT --threshold. That number is calibrated against Tier 1's
    # ticket-to-ticket distribution, where 0.89 is a normal gate; against
    # Tier 2's ticket-to-document distribution the same number rejects every
    # chunk. One override spanning both would be a single number standing for
    # two different embedding spaces, which is the mistake this codebase keeps
    # not making.
    tier2_override = args.tier2_threshold
    if tier2_override is None:
        raw = os.environ.get("SPS_TIER2_THRESHOLD", "").strip()
        tier2_override = float(raw) if raw else None

    docs = DocRetriever(
        docs_dir=docs_dir,
        confidence_threshold=tier2_override,
        azure_threshold=_threshold("TIER2_AZURE_THRESHOLD", TIER2_AZURE_THRESHOLD),
        local_threshold=_threshold("TIER2_LOCAL_THRESHOLD", TIER2_LOCAL_THRESHOLD),
    )

    if not docs.available:
        # The normal state of a deployment whose standards have not been loaded
        # yet. Not an error: the ticket reports exactly the Tier-1 outcome it
        # would have reported before Tier 2 existed.
        return _no_resolution(
            f"No 0250 documents found in {docs_dir}.", STAGE_NOTHING
        )

    try:
        chunks = docs.retrieve(ticket)
    except AzureEmbeddingNotConfigured as exc:
        logger.error("Tier 2 embedding is not configured: %s", exc)
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_BAD_INPUT, **measured)
    except AzureEmbeddingError as exc:
        # Tier 1 embedded successfully or the run would have ended already, so
        # Azure going down between the tiers is a genuine outage -- not a
        # corpus the robot should stop retrying.
        logger.error("Tier 2 embedding failed: %s", exc)
        return ResolveOutcome(
            CODE_INFRASTRUCTURE, str(exc), EXIT_INFRASTRUCTURE, **measured
        )
    except Exception as exc:
        # A corpus that cannot be PARSED is different: Tier 2 is a fallback, and
        # an unreadable document must not turn a legitimate Tier-1 "no
        # resolution" into a fault the robot retries forever.
        logger.error("Tier 2 failed; reporting the Tier-1 outcome: %s", exc)
        return _no_resolution(f"Tier 2 unavailable: {exc}", STAGE_NOTHING)

    tier2_stats = docs.stats
    logger.info("tier 2 stats: %s", tier2_stats.as_dict())

    # Tier 1 can stop before encoding anything -- an unknown part scans the
    # history, matches no row and never reaches the model. If Tier 2 then
    # encoded, the status sheet must name the encoder that actually ran:
    # support counts Azure fallbacks from this column, and a run where only
    # Tier 2 embedded would otherwise report no encoder at all.
    #
    # Rebinding rather than mutating, and _no_resolution reads `measured` from
    # this scope when it is called, so every later return picks this up.
    if not measured.get("embedding_model") and tier2_stats.backend:
        measured = dict(
            measured,
            embedding_model=f"{tier2_stats.backend}:{tier2_stats.backend_detail}",
        )

    tier2_measured = dict(
        tier2_top_score=tier2_stats.top_score,
        tier2_threshold=tier2_stats.threshold_used,
        tier2_chunks=tier2_stats.chunks,
        tier2_cache_state=tier2_stats.cache_state,
    )

    if not chunks:
        if not tier2_stats.chunks:
            # Documents were present but yielded no usable text at all -- every
            # one unparseable, or every section too short to embed. Nothing was
            # scored, so nothing was gated.
            return _no_resolution(
                f"No usable text in {tier2_stats.documents} 0250 document(s).",
                STAGE_NOTHING,
                **tier2_measured,
            )
        return _no_resolution(
            f"Best 0250 match {tier2_stats.top_score:.4f} is below the "
            f"{tier2_stats.threshold_used:.2f} threshold across "
            f"{tier2_stats.chunks} chunk(s).",
            STAGE_GATED,
            **tier2_measured,
        )

    outcome = asyncio.run(loop.run_grounded(ticket, documentation_grounding(chunks)))

    if not outcome.succeeded or outcome.draft is None:
        if outcome.infrastructure_failure:
            logger.error("dependency failure in Tier 2: %s", outcome.failure_reason)
            return ResolveOutcome(
                CODE_INFRASTRUCTURE,
                outcome.failure_reason,
                EXIT_INFRASTRUCTURE,
                **measured,
                **tier2_measured,
            )
        return _no_resolution(
            f"{outcome.failure_reason} Best 0250 match "
            f"{tier2_stats.top_score:.4f}.",
            STAGE_REJECTED,
            **tier2_measured,
        )

    from sps.output import success_from_docs

    result = success_from_docs(
        recommendation=outcome.draft.recommendation,
        justification=outcome.draft.justification
        or f"Derived from {len(chunks)} 0250 standard section(s).",
        top_score=tier2_stats.top_score,
        chunks=chunks,
    )
    write_output(output_dir, part_number, result)
    return ResolveOutcome(
        CODE_SUCCESS_DOC,
        f"{tier1_detail} Resolved instead from {len(chunks)} 0250 section(s) at "
        f"{result.confidence} confidence: {', '.join(result.referenced_sources)}.",
        EXIT_OK,
        resolution_source=result.resolution_source,
        **measured,
        **tier2_measured,
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
