"""Resolve one SPS ticket against a historical file. No vector database.

    python -m scripts.run_resolver --ticket-file ticket.xlsx \
        --history-file history.csv --output-dir .\\out

Writes two workbooks into --output-dir:

  status.xlsx   ALWAYS, including an early abort or an unhandled exception.
                Execution_Timestamp, Status (PASS/FAIL), Status_Code, Reason,
                Embedding_Model. The last names the encoder that actually ran,
                so support can see how often the Azure fallback fires; it is
                blank when the run aborted before anything was encoded.
  output.xlsx   Whenever the run REACHED A CONCLUSION, which includes
                concluding that neither tier had an answer. Part_Number,
                AI_Recommendation, Justification, Confidence_Score,
                Referenced_Sources, Resolution_Source. On a failure the
                recommendation is "Solution not found." and Resolution_Source
                is NONE -- a caller merging this into its own records needs one
                row per ticket, and a missing row cannot be told apart from a
                ticket that was never processed.

                NOT written for an infrastructure fault (exit 1 or 2): the run
                formed no verdict, and the robot is expected to retry it.

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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# A threshold is a property of the embedding space, defined once beside the
# code that owns it. The LOCAL_* pair is dormant while the local encoder is out
# of the pipeline, and is imported so the constants stay in one place.
from sps.retrieval.in_memory import (
    AZURE_EMBEDDING_THRESHOLD,
    CASCADE_LIMIT,
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
    # Everything main() needs to write output.xlsx, so that both workbooks are
    # produced in one place. Blank part number means the run stopped before the
    # ticket could be read.
    part_number: str = ""
    # The PipelineResult on a success; None when there is no recommendation, in
    # which case main() builds the "Solution not found." row.
    result: Any = None
    # The evidence actually shown to the Actor. Operational only -- it never
    # reaches output.xlsx, which is supplier-facing -- but it is what lets a
    # reviewer see WHY a recommendation says what it does, in the same glance
    # that shows what it said. Two fields rather than one because the tiers are
    # populated from different call frames and a single key would collide in
    # the shared **measured expansion.
    evidence: tuple[str, ...] = ()
    tier2_evidence: tuple[str, ...] = ()
    # The single best precedent, banner and all, ready to write. Unlike
    # `evidence` above this DOES reach output.xlsx -- a reviewer looking at a
    # refusal asked to see what we found, not merely that we found nothing.
    closest_match: str = ""
    # The scored source list, used on a success where closest_match is not.
    source_list: str = ""
    # The intent scorer's verdict, kept apart from `top_score` because they
    # measure different things: top_score is the cosine that shortlisted a
    # record, intent_score is the judgement that it asks the same question.
    # One number standing for both could not be calibrated against either.
    intent_score: float = 0.0
    # The scorer's one-line reading of what the ticket asks for. Recorded so a
    # wrong match can be traced to a wrong reading rather than being an
    # unexplained number.
    ticket_intent: str = ""
    # Which check ran out of road, as a value rather than as prose, so a batch
    # can be tallied by cause instead of read one row at a time.
    stop_reason: str = ""


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


# Long enough to tell boilerplate from a real disposition, short enough that
# fifteen candidates stay readable in one cell. The workbook writer truncates at
# Excel's hard limit anyway; this keeps it from getting that far.
EVIDENCE_CHARS = 500


def _evidence(label: str, text: str) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) > EVIDENCE_CHARS:
        flat = flat[: EVIDENCE_CHARS - 1] + "\u2026"
    return f"{label}: {flat}"


# DEA copies the recommendation into the SPS portal by hand. A field holding
# raw archive text therefore has to announce itself in the first few words, or
# it gets pasted too -- which is precisely how "ESW#20033465 is submitted for
# these issues" would reach a supplier. The banner is a control, not a caption.
RAW_BANNER = "[RAW HISTORY - not checked for supplier use]"
WEAK_BANNER = "[RAW HISTORY - WEAK MATCH, below the confidence gate]"


def _closest_match(records: Sequence[tuple[str, str, float]], gated: bool) -> str:
    """The closest precedents found, banner-prefixed, or '' if there were none.

    `records` is (label, text, score), best first. Several rather than one: a
    reviewer with no recommendation is reading these as options to answer the
    ticket from, and the top-scoring record is not always the one that happens
    to describe the fix.
    """
    if not records:
        return ""
    from sps.contracts import score_to_percent

    banner = WEAK_BANNER if gated else RAW_BANNER
    header = f"{banner} {len(records)} closest record(s):"
    blocks = [
        _evidence(f"{label} ({score_to_percent(score)}%)", text)
        for label, text, score in records
    ]
    return header + "\n\n" + "\n\n".join(blocks)


def _source_list(records: Sequence[tuple[str, float]]) -> str:
    """Where a recommendation came from, without reproducing it.

    On a success the answer is already written; repeating the archive text
    underneath it adds length without adding information. Worse, up to fifteen
    records reach the Actor and a recommendation may combine several, so
    showing one record's text would read as THE source and invite a reviewer to
    check a step against a record it did not come from.

    What is missing from `Referenced_Sources` is the per-record score -- it
    lists the ids and cannot say which matched at 94% and which at 71%, which
    is the one thing that says where to look first.
    """
    if not records:
        return ""
    from sps.contracts import score_to_percent

    listed = ", ".join(f"{label} ({score_to_percent(score)}%)" for label, score in records)
    return f"Synthesized from {len(records)} record(s): {listed}"


# Resolver-level stops, alongside the loop's own STOP_* values. These two
# happen before the Actor is ever called, so the loop cannot report them.
STOP_NO_HISTORY = "NO_HISTORY_FOR_PART"
STOP_BELOW_THRESHOLD = "BELOW_THRESHOLD"
# Records were found and read, and none of them asks the same question. A
# different finding from BELOW_THRESHOLD, which means nothing was similar
# enough to be worth reading -- and it routes differently, because this one
# says the archive has no answer rather than that retrieval missed.
STOP_INTENT_BELOW = "INTENT_BELOW_THRESHOLD"
STOP_NO_PART_NUMBER = "NO_PART_NUMBER"


def _business_justification(outcome: "ResolveOutcome") -> str:
    """Why there is no recommendation, for the person who reads the portal.

    The diagnostic prose -- scores, thresholds, row counts -- stays in
    status.xlsx's Reason, where the robot and support already read it. It was
    being shown to DEA as well, which is what "please say it in softer words"
    was about. The closest precedent leads, because a reviewer would rather see
    the near-miss first and the explanation second.
    """
    from sps.contracts import score_to_percent
    from sps.generation.actor_critic import (
        STOP_ACTOR_ABSTAINED,
        STOP_GATE_MISATTRIBUTED,
        STOP_GATE_UNTRANSFERABLE,
        STOP_JUDGE_REFUSED,
    )

    WHY = {
        STOP_BELOW_THRESHOLD: (
            "This is the closest past problem we found, but it is not similar "
            "enough to rely on. Not much historical data to infer the solution "
            "or recommendation."
        ),
        STOP_NO_PART_NUMBER: (
            "No part number was supplied, so the part's own history could not "
            "be searched."
        ),
        STOP_ACTOR_ABSTAINED: (
            "This is the closest past solution we found. We are not "
            "recommending it because it does not answer what this ticket asks."
        ),
        STOP_GATE_UNTRANSFERABLE: (
            "This is the closest past solution we found. We are not "
            "recommending it as it stands because it refers to an attachment, "
            "a prior conversation or a work request belonging to a different "
            "ticket, none of which the supplier can act on."
        ),
        STOP_GATE_MISATTRIBUTED: (
            "This is the closest past solution we found. We are not "
            "recommending it as it stands because it asks the supplier to "
            "perform an action only AMAT can perform."
        ),
        STOP_JUDGE_REFUSED: (
            "This is the closest past solution we found. We are not "
            "recommending it as it stands because it did not pass review."
        ),
    }

    if not outcome.closest_match:
        # No precedent to show, but there may still be a specific reason worth
        # giving. "Not much historical data" is true of an unknown part and
        # misleading about a ticket that arrived with no part number at all --
        # one is a gap in the archive, the other is a gap in the ticket, and
        # only the second is something the sender can fix.
        return WHY.get(
            outcome.stop_reason,
            "Not much historical data to infer the solution or recommendation.",
        )

    # The intent case leads with the number, because the number is the finding
    # -- "we read these records and none of them asks your question, here is
    # the nearest" is a different statement from "we found nothing". It is the
    # one entry that needs a value, so it is built rather than looked up.
    if outcome.stop_reason == STOP_INTENT_BELOW:
        lead = (
            f"Confidence is {score_to_percent(outcome.intent_score)}% hence we are not "
            "recommending the solution, but this is the closest match from "
            "historical data."
        )
        return lead + "\n\n" + outcome.closest_match

    why = WHY.get(
        outcome.stop_reason,
        "This is the closest past solution we found. We are not recommending "
        "it as it stands.",
    )
    # Precedent first, explanation second -- the order the reviewer asked for.
    return outcome.closest_match + "\n\n" + why


def unresolved_result(outcome: "ResolveOutcome"):
    """The result row for a run that concluded without a recommendation.

    A caller merging output.xlsx into its own records needs one row per ticket.
    Without this, "no historical data for this part" produced no row at all, and
    a ticket that found nothing was indistinguishable from a ticket that never
    ran -- which is exactly the case a reviewer most needs to see.
    """
    from sps.contracts import NO_RECOMMENDATION, SOURCE_NONE, PipelineResult, score_to_percent

    # Blank, not 0%, when nothing was ever scored. An unknown part and a near
    # miss at 42% are different findings, and "0%" reads as the second one.
    # Reason carries both tiers' scores in full either way.
    #
    # Tier 2's score is used when Tier 1 scored nothing: reporting a flat 0%
    # for a run where the standards matched at 0.41 described the wrong tier.
    # The intent score when there is one: on this path the number a reviewer
    # reads is the one that decided the outcome, and the cosine only
    # shortlisted. Falls back to the retrieval scores where no intent judgement
    # was ever made -- an unknown part, or a Tier-2-only run.
    score = outcome.intent_score or outcome.top_score or outcome.tier2_top_score
    confidence = f"{score_to_percent(score)}%" if score > 0 else ""
    return PipelineResult(
        ai_recommendation=NO_RECOMMENDATION,
        justification=_business_justification(outcome),
        confidence=confidence,
        referenced_sources=[],
        resolution_source=SOURCE_NONE,
        closest_matching_solution=outcome.closest_match,
    )


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
                "Closest_Matching_Solution": result.closest_matching_solution,
                "Cascade_Warnings": result.cascade_warnings,
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
    from sps.validators import (
        normalize_part_number,
        validate_part_number,
        validate_problem_description,
    )

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
    #
    # The two halves of the old validate_ticket() are now separated, because
    # they stop different amounts of the pipeline.
    #
    # A missing problem description is terminal. Both tiers match on that text
    # -- Tier 2 builds its query from it too -- so there is nothing to search
    # with on either side, and falling through would be searching for nothing.
    invalid = validate_problem_description(ticket.problem_description)
    if invalid:
        return ResolveOutcome(
            invalid.code, invalid.reason, EXIT_OK, part_number=part_number
        )

    # A missing part number stops Tier 1 only. Tier 1 filters history by exact
    # part, so there is nothing for it to retrieve; Tier 2 searches standards
    # by defect text and never looks at the part number.
    bad_part = validate_part_number(ticket.part_number)

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

    # Constructed before the part-number branch because Tier 2 needs the loop
    # on both paths. Constructing either opens no connection.
    #
    # The client is built separately rather than read back off the loop: the
    # intent scorer and the Tier-2 loop are two callers of one deployment, and
    # having one reach through the other for its transport made the dependency
    # invisible at the call site.
    chat_client = AzureOpenAIChatClient(LLMSettings.from_env())
    loop = ActorCriticLoop(chat_client, LLMSettings.from_env())

    if bad_part:
        logger.info("%s Tier 1 skipped; trying the standards.", bad_part.reason)
        return _resolve_from_docs(
            args=args,
            ticket=ticket,
            part_number=part_number,
            output_dir=output_dir,
            loop=loop,
            tier1_detail=bad_part.reason,
            tier1_stage=STAGE_NOTHING,
            tier1_stop=STOP_NO_PART_NUMBER,
            measured=dict(part_number=part_number),
        )

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
        part_number=part_number,
        # What the Actor was shown, or would have been. Present even on the
        # gated path, where seeing the near-miss text is the whole point.
        evidence=tuple(_evidence(c.sps_id, c.actual_solution) for c in candidates),
        # The closest rows, gate or no gate. `candidates` is empty on a gated
        # run, which is exactly when a reviewer most wants to see what was
        # almost good enough -- so this reads stats, not candidates.
        #
        # Only ever shown when there is NO recommendation. On a success the
        # source list below replaces it: see `_source_list`.
        closest_match=_closest_match(
            [(c.sps_id, c.actual_solution, c.composite_score) for c in stats.top_candidates],
            gated=not candidates,
        ),
        # Every record the Actor was given, with its score. All of them, not
        # the top three: a recommendation may draw on any of them, and naming
        # only some would misreport where the answer came from.
        source_list=_source_list([(c.sps_id, c.composite_score) for c in candidates]),
        embedding_model=model,
        top_score=stats.top_score,
        threshold_used=stats.threshold_used,
        candidates_considered=stats.capped_to,
    )

    # -- Tier 1: the part's own history ------------------------------------

    if stats.usable == 0:
        tier1_stage = STAGE_NOTHING
        tier1_stop = STOP_NO_HISTORY
        tier1_detail = (
            f"No usable history for part {part_number}: {stats.part_matches} row(s) "
            f"matched the part out of {stats.rows_scanned} scanned."
        )
    elif not candidates:
        tier1_stage = STAGE_GATED
        tier1_stop = STOP_BELOW_THRESHOLD
        tier1_detail = (
            f"Best historical match {stats.top_score:.4f} is below the "
            f"{stats.threshold_used:.2f} threshold across {stats.capped_to} candidate(s)."
        )
    else:
        from sps.config import IntentSettings
        from sps.generation import score_intent

        intent_settings = IntentSettings.from_env()
        assessment = asyncio.run(score_intent(chat_client, ticket, candidates))

        if assessment.failed:
            # Fails closed, and as an outage rather than a refusal. "Nothing
            # matched" is a business outcome the robot files; "we could not
            # tell" is something it should retry. Tier 2 shares the same
            # deployment, so there is nothing to fall back to.
            logger.error("dependency failure: %s", assessment.failure_reason)
            return ResolveOutcome(
                CODE_INFRASTRUCTURE, assessment.failure_reason, EXIT_INFRASTRUCTURE, **measured
            )

        best = assessment.best
        measured["ticket_intent"] = assessment.ticket_intent
        measured["intent_score"] = best.intent_match if best else 0.0
        # Re-derived from the intent scores. Retrieval ordered these by cosine,
        # which is precisely the ordering this design stopped trusting, so what
        # the reviewer is shown has to be the order that decided the outcome.
        measured["source_list"] = _source_list(
            [(s.candidate.sps_id, s.intent_match) for s in assessment.scored]
        )
        measured["closest_match"] = _closest_match(
            [
                (s.candidate.sps_id, s.candidate.actual_solution, s.intent_match)
                for s in assessment.scored[:CASCADE_LIMIT]
            ],
            gated=True,
        )

        if best is not None and best.intent_match >= intent_settings.threshold:
            from sps.output import cascade

            # Sent exactly as recorded. No strip, no renumbering, no
            # normalisation -- a reviewer can diff this cell against the source
            # record and expect a character-for-character match, and anything
            # tidier would break that.
            result = cascade(
                solution=best.candidate.actual_solution,
                candidate=best.candidate,
                intent_match=best.intent_match,
                reason=best.reason,
                source_list=measured["source_list"],
            )
            if result.cascade_warnings:
                logger.warning(
                    "Ticket %r cascading %s unchanged, and it %s",
                    ticket.sps_id, best.candidate.sps_id, result.cascade_warnings,
                )
            return ResolveOutcome(
                CODE_SUCCESS_HISTORICAL,
                f"Intent matched {best.candidate.sps_id} at {best.percent}% "
                f"across {len(assessment.scored)} record(s); its solution was "
                f"sent unchanged. Ticket intent: {assessment.ticket_intent}",
                EXIT_OK,
                resolution_source=result.resolution_source,
                result=result,
                **measured,
            )

        # Records were read and none asks the same question. STAGE_REJECTED,
        # not STAGE_GATED: retrieval did its job and a judgement was made on
        # the content, which is the same shape of outcome the Judge used to
        # produce and routes to the same place.
        tier1_stage = STAGE_REJECTED
        tier1_stop = STOP_INTENT_BELOW
        best_pct = best.percent if best else 0
        tier1_detail = (
            f"Best intent match {best_pct}% is below the "
            f"{int(intent_settings.threshold * 100)}% threshold across "
            f"{len(assessment.scored)} record(s). "
            f"Ticket intent: {assessment.ticket_intent}"
        )

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
        tier1_stop=tier1_stop,
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
    tier1_stop: str,
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
        # Merged rather than double-splatted: Tier 2 may supply a
        # closest_match when Tier 1 had none, and two **expansions carrying the
        # same key is a TypeError.
        merged = {**measured, **fields}
        # The furthest stage wins for the code, so the stop reason has to agree
        # with it -- reporting Tier 1's cause beside Tier 2's code would send a
        # reader looking in the wrong tier.
        merged.setdefault(
            "stop_reason", tier1_stop if tier1_stage >= tier2_stage else merged.get("stop_reason", "")
        )
        return ResolveOutcome(
            STAGE_CODES[max(tier1_stage, tier2_stage)],
            f"{tier1_detail} {extra}".strip(),
            EXIT_OK,
            **merged,
        )

    if args.no_tier2:
        # Nothing was retrieved because nothing was looked for, so Tier 1's
        # stage stands on its own.
        return _no_resolution("Tier 2 disabled by --no-tier2.", STAGE_NOTHING)

    # The 0250 standards are scoped to particular reason codes, so most
    # tickets never reach them.
    #
    # The two ways of not being allowed are reported differently on purpose.
    # An empty list disables Tier 2 for every ticket in the deployment, and if
    # that happened because a .env line was lost, the only place it would ever
    # show is this sentence. A code simply not being listed is routine.
    from sps.config import IntentSettings

    intent_settings = IntentSettings.from_env()
    reason_code = (ticket.problem_reason_code or "").strip()
    if not intent_settings.tier2_reason_codes:
        return _no_resolution(
            "Tier 2 skipped: no reason codes are configured "
            "(SPS_TIER2_REASON_CODES is empty).",
            STAGE_NOTHING,
        )
    if not intent_settings.tier2_allowed(reason_code):
        return _no_resolution(
            f"Tier 2 skipped: reason code {reason_code or '(blank)'!r} is not "
            f"one of the {len(intent_settings.tier2_reason_codes)} configured.",
            STAGE_NOTHING,
        )

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

    # Only when Tier 1 found nothing at all. A part's own history outranks a
    # general standard as the thing to show a reviewer, so a Tier-1 near-miss
    # is never displaced by a Tier-2 one.
    if tier2_stats.top_chunks and not measured.get("closest_match"):
        tier2_measured["closest_match"] = _closest_match(
            [(c.citation, c.chunk.text, c.score) for c in tier2_stats.top_chunks],
            gated=not chunks,
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
            stop_reason=STOP_BELOW_THRESHOLD,
            **tier2_measured,
        )

    tier2_measured["tier2_evidence"] = tuple(
        _evidence(c.citation, c.chunk.text) for c in chunks
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
            stop_reason=outcome.stop_reason,
            **tier2_measured,
        )

    from sps.output import success_from_docs

    result = success_from_docs(
        recommendation=outcome.draft.recommendation,
        justification=outcome.draft.justification
        or f"Derived from {len(chunks)} 0250 standard section(s).",
        top_score=tier2_stats.top_score,
        chunks=chunks,
        closest_matching_solution=_source_list([(c.citation, c.score) for c in chunks]),
    )
    return ResolveOutcome(
        CODE_SUCCESS_DOC,
        f"{tier1_detail} Resolved instead from {len(chunks)} 0250 section(s) at "
        f"{result.confidence} confidence: {', '.join(result.referenced_sources)}.",
        EXIT_OK,
        resolution_source=result.resolution_source,
        result=result,
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

    # A run that reached a conclusion always leaves a result row, even when the
    # conclusion is "no solution": the caller merges output.xlsx into its own
    # records, and a missing row cannot be told apart from a ticket that was
    # never processed.
    #
    # An infrastructure fault writes none. It reached no conclusion, the robot
    # is expected to retry it, and recording "Solution not found." for an Azure
    # outage would enter a verdict the pipeline never actually formed.
    if outcome.exit_code == EXIT_OK:
        try:
            write_output(
                output_dir,
                outcome.part_number,
                outcome.result if outcome.result is not None else unresolved_result(outcome),
            )
        except Exception:
            logger.error("Could not write %s:\n%s", OUTPUT_FILE, traceback.format_exc())
            # Leave no half-written workbook behind to be merged as a result.
            (output_dir / OUTPUT_FILE).unlink(missing_ok=True)
            outcome = replace(
                outcome,
                code=CODE_INFRASTRUCTURE,
                reason=f"Could not write {OUTPUT_FILE}; see stderr for the traceback.",
                exit_code=EXIT_INFRASTRUCTURE,
            )

    try:
        write_status(output_dir, outcome.code, outcome.reason, outcome.embedding_model)
    except Exception:
        logger.error("Could not write %s:\n%s", STATUS_FILE, traceback.format_exc())
        return EXIT_INFRASTRUCTURE
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
