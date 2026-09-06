"""Inference entry point for the UiPath Performer.

Components B and C exposed as a callable function and a CLI. No web framework,
no server, no database access -- UiPath owns all SQL, and integration is by file
plus exit code.

    python -m service.run_inference
        --payload-file in.txt --output-file out.xlsx --status-file status.txt

CONTRACT WITH THE CALLER
  --output-file  the Section 4 contract. An .xlsx/.xlsm path produces a
                 workbook with one row per ticket and the four contract columns;
                 any other path produces JSON text in UTF-8 with no BOM. Either
                 way the write is atomic (temp file + os.replace), so a reader
                 never sees a half-written or corrupt file.
  --status-file  three lines -- STATUS, EXIT_CODE, REASON -- written on every
                 exit path, and written LAST, only once the output file is
                 complete. So a SUCCESS status always means the data file beside
                 it is present and readable. STATUS is SUCCESS if and only if
                 EXIT_CODE is 0; the two can never disagree.
  stdout         the same JSON, always, whether or not --output-file is used.
  stderr         logs, diagnostics and fault traces. Never mixed into stdout.
  exit 0         the pipeline ran. This includes "Solution not found." -- a
                 refusal is an expected business outcome, not an error.
  exit 1         infrastructure fault (Azure unreachable, storage locked, bad
                 config, output file unwritable). A contract is still produced
                 so the admin queue keeps a row; the exit code is what should
                 raise a UiPath system exception.
  exit 2         the payload could not be parsed as JSON.

STALE-OUTPUT SAFETY
  An existing --output-file is deleted before any work begins. If this process
  is killed outright, the caller finds no file rather than a previous run's
  result, so a missing file and a non-zero exit both mean "do not trust this
  transaction".

CREDENTIALS
  Azure credentials are read only from the environment (AZURE_OPENAI_API_KEY,
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT), never from a CLI argument --
  arguments are visible in the Windows process list and in UiPath job logs. A
  .env file beside the project is loaded if present, and real environment
  variables always win over it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from service.excel_output import is_excel_path, write_excel
from service.status_file import StatusReport, write_status_file
from sps.config import Settings
from sps.contracts import SOLUTION_NOT_FOUND, PipelineResult
from sps.output import SERVICE_UNAVAILABLE_JUSTIFICATION
from sps.pipeline import SPSPipeline

logger = logging.getLogger("sps.inference")

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_BAD_PAYLOAD = 2

# Credentials are environment-only. Listed here so the CLI can report which are
# missing -- by NAME, never by value.
CREDENTIAL_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
)


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


def build_pipeline(settings: Settings | None = None):
    """Construct the production pipeline (BGE + Qdrant + Azure OpenAI).

    Returns (pipeline, store) -- the store is handed back so the caller can
    close it and release the embedded-mode directory lock.
    """
    settings = settings or Settings.from_env()
    from sps.vectorstore import QdrantVectorStore

    store = QdrantVectorStore(settings.vector_store)
    return SPSPipeline.build(store=store, settings=settings), store


def run_batch_results(
    payloads: Iterable[dict[str, Any]], settings: Settings | None = None
) -> list[PipelineResult]:
    """Score many tickets on one model load.

    The BGE weights cost several seconds to load; amortising that across a
    Performer's queue slice is worth far more than per-call isolation.
    """
    from sps.contracts import IncomingTicket

    tickets = [IncomingTicket.from_dict(p) for p in payloads]
    pipeline, store = build_pipeline(settings)

    async def _run() -> list[PipelineResult]:
        return [await pipeline.process(t) for t in tickets]

    try:
        return asyncio.run(_run())
    finally:
        store.close()


def run_inference_result(
    payload: dict[str, Any], settings: Settings | None = None
) -> PipelineResult:
    """Score one ticket, returning the full result object.

    Carries `infrastructure_failure` alongside the contract, which the CLI turns
    into an exit code.
    """
    return run_batch_results([payload], settings)[0]


def run_inference(payload: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    """Score one ticket. Returns the Section 4 contract dict.

    The callable interface: importable by any in-process host that would rather
    not shell out.
    """
    return run_inference_result(payload, settings).to_contract()


def run_batch(
    payloads: Iterable[dict[str, Any]], settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Score many tickets on one model load; returns contract dicts."""
    return [r.to_contract() for r in run_batch_results(payloads, settings)]


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------


def _failure_contract(reason: str) -> dict[str, Any]:
    """Emitted when the pipeline could not run at all."""
    return {
        "AI_Recommendation": SOLUTION_NOT_FOUND,
        "Justification": reason,
        "Confidence": "0%",
        "SPS_IDs_Referred": [],
    }


def summarise_reason(results: list[PipelineResult]) -> str:
    """One-line REASON for the status file.

    A dependency failure reports the specific `diagnostic` (which may name the
    missing credential variables) rather than the sanitized, supplier-facing
    Justification -- this file is for the support team, not the supplier.
    """
    if not results:
        return "No tickets processed."

    if len(results) == 1:
        result = results[0]
        if result.infrastructure_failure:
            return result.diagnostic or "Dependency failure; see stderr."
        if result.succeeded:
            return "Processed successfully."
        # A refusal is a legitimate outcome, and the Justification already
        # states it plainly ("Confidence below 82% threshold.").
        return result.justification

    failed = [r for r in results if r.infrastructure_failure]
    found = sum(1 for r in results if r.succeeded)
    summary = (
        f"Processed {len(results)} tickets: {found} with a recommendation, "
        f"{len(results) - found} without."
    )
    if failed:
        summary += f" {len(failed)} failed on a dependency: {failed[0].diagnostic}"
    return summary


def _parse_json(raw: str, source: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc


# Input is read as utf-8-sig, NOT utf-8. .NET writes UTF-8 with a BOM by
# default, so a payload produced by a UiPath Write Text File activity normally
# starts with one; plain utf-8 rejects it with "Unexpected UTF-8 BOM" and every
# ticket would fail as an invalid payload. utf-8-sig strips a BOM when present
# and is identical to utf-8 when it is not, so it is safe for every producer.
# (Output stays strict utf-8 with no BOM -- RFC 8259 forbids one on JSON.)
INPUT_ENCODING = "utf-8-sig"

# U+FEFF, written via chr() so no invisible character sits in this source file.
BOM = chr(0xFEFF)


def _read_payloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.batch_file:
        lines = Path(args.batch_file).read_text(encoding=INPUT_ENCODING).splitlines()
        return [
            _parse_json(line, f"{args.batch_file} line {n}")
            for n, line in enumerate(lines, 1)
            if line.strip()
        ]
    if args.payload_file:
        return [
            _parse_json(
                Path(args.payload_file).read_text(encoding=INPUT_ENCODING), args.payload_file
            )
        ]
    if args.stdin:
        # A piped stream carries no encoding declaration, so strip a leading
        # BOM by hand if the producer emitted one.
        return [_parse_json(sys.stdin.read().lstrip(BOM), "stdin")]
    return [_parse_json(args.payload, "--payload")]


def write_output_file(path: Path, text: str) -> None:
    """Write the contract atomically, UTF-8, no BOM.

    Atomic because the caller may poll for the file: temp file + os.replace
    means a reader sees either nothing or the complete contract, never a
    partial write. No BOM because RFC 8259 forbids one on JSON and a stray
    U+FEFF breaks strict deserializers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m service.run_inference",
        description="Score SPS ticket(s) and emit the JSON contract.",
        epilog=(
            "Azure credentials are read from the environment only "
            "(AZURE_OPENAI_ENDPOINT / _API_KEY / _DEPLOYMENT); there is "
            "deliberately no CLI flag for them."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--payload", help="Ticket as a JSON string")
    source.add_argument("--payload-file", help="Path to a file holding the ticket JSON")
    source.add_argument("--stdin", action="store_true", help="Read the ticket JSON from stdin")
    source.add_argument(
        "--batch-file",
        help="JSON Lines file, one ticket per line; emits one result per line",
    )
    parser.add_argument(
        "--output-file",
        help="Write the JSON contract here (atomically, UTF-8, no BOM). "
        "Also printed to stdout.",
    )
    parser.add_argument(
        "--status-file",
        help="Write STATUS / EXIT_CODE / REASON here on every exit path. "
        "Written only after the output file is complete, so a SUCCESS status "
        "always means the data file is readable.",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Indent the JSON (default: one compact line)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging on stderr")

    args = parser.parse_args(argv)
    if args.pretty and args.batch_file:
        # Indented objects are not line-delimited, so the output would not be
        # parseable as JSON Lines.
        parser.error("--pretty cannot be combined with --batch-file")
    return args


def _render(contracts: list[dict[str, Any]], pretty: bool) -> str:
    return "\n".join(
        json.dumps(c, indent=2 if pretty else None, ensure_ascii=False) for c in contracts
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Logs go to stderr so stdout stays parseable as pure JSON.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    _load_dotenv()

    output_path = Path(args.output_file) if args.output_file else None
    status_path = Path(args.status_file) if args.status_file else None

    # Clear both files BEFORE any work. Status goes first so there is never an
    # instant where a stale SUCCESS status points at an already-deleted data
    # file. If this process is killed outright the caller finds neither, and
    # "no status file" is unambiguous.
    for label, path in (("status file", status_path), ("output file", output_path)):
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("Cannot clear %s %s: %s", label, path, exc)
            return EXIT_INFRASTRUCTURE

    def finish(contracts: list[dict[str, Any]], code: int, reason: str) -> int:
        """Emit on every exit path, then return the caller's exit code."""
        text = _render(contracts, args.pretty)
        # stdout stays JSON whatever the file format, so the process is still
        # debuggable by hand and non-Excel callers keep working unchanged.
        print(text)

        if output_path is not None:
            try:
                if is_excel_path(output_path):
                    write_excel(output_path, contracts)
                else:
                    write_output_file(output_path, text + "\n")
            except Exception as exc:
                # The result exists but could not be handed over; that is an
                # infrastructure fault regardless of how the pipeline itself did.
                logger.error("Cannot write output file %s: %s", output_path, exc)
                code, reason = EXIT_INFRASTRUCTURE, f"Could not write the output file: {exc}"

        # Written LAST, and only once the data file is complete. That ordering is
        # the whole guarantee: a SUCCESS status always means the data file beside
        # it is present and readable.
        if status_path is not None:
            try:
                write_status_file(status_path, StatusReport(exit_code=code, reason=reason))
            except OSError as exc:
                logger.error("Cannot write status file %s: %s", status_path, exc)
                return EXIT_INFRASTRUCTURE
        return code

    try:
        payloads = _read_payloads(args)
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return finish(
            [_failure_contract("Ticket payload could not be read.")],
            EXIT_BAD_PAYLOAD,
            f"Invalid input format: {exc}",
        )

    for payload in payloads:
        if not isinstance(payload, dict):
            logger.error("Each ticket must be a JSON object, got %s", type(payload).__name__)
            return finish(
                [_failure_contract("Ticket payload was not a JSON object.")],
                EXIT_BAD_PAYLOAD,
                f"Invalid input format: expected a JSON object, got "
                f"{type(payload).__name__}.",
            )

    settings = Settings.from_env()
    logger.debug("vector store: %s", settings.vector_store.describe())
    missing = [name for name in CREDENTIAL_VARS if not os.environ.get(name, "").strip()]
    if missing:
        # Names only, never values. Not fatal here: retrieval may still gate the
        # ticket before any Azure call is made.
        logger.warning("Azure credentials absent from the environment: %s", ", ".join(missing))

    try:
        results = run_batch_results(payloads, settings)
    except Exception:
        # The pipeline could not even start (storage locked, bad config). Emit a
        # contract per ticket so nothing is silently dropped from the admin
        # queue, but signal the fault through the exit code.
        logger.exception("Inference could not run")
        missing_note = (
            f" Missing Azure credentials: {', '.join(missing)}."
            if missing
            else ""
        )
        return finish(
            [_failure_contract(SERVICE_UNAVAILABLE_JUSTIFICATION) for _ in payloads],
            EXIT_INFRASTRUCTURE,
            f"Pipeline could not start.{missing_note} See stderr for the traceback.",
        )

    # "Solution not found." is a valid business outcome and exits 0. A dependency
    # outage must NOT look like one: without this, an Azure failure would mark
    # every queue item Successful and quietly burn the whole queue.
    infrastructure = any(result.infrastructure_failure for result in results)
    if infrastructure:
        logger.error("At least one ticket failed on a dependency, not on content")

    return finish(
        [result.to_contract() for result in results],
        EXIT_INFRASTRUCTURE if infrastructure else EXIT_OK,
        summarise_reason(results),
    )


if __name__ == "__main__":
    sys.exit(main())
