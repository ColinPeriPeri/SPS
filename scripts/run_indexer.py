"""LEGACY PATH -- not on the resolver flow.

The primary entry point is now `scripts/run_resolver.py`, which filters a
history file by part number and embeds the survivors per ticket, so there is
no persistent index to build or maintain. This module is retained for
`service/run_inference.py` and for a future return to a persistent index;
nothing on the resolver path imports it.

Component A entry point -- the periodic (nightly/weekly) indexing job.

    python -m scripts.run_indexer                          # SQL delta
    python -m scripts.run_indexer --source-file dump.csv   # local file instead
    python -m scripts.run_indexer --source-file dump.xlsx
    python -m scripts.run_indexer --dry-run
    python -m scripts.run_indexer --reset-watermark        # full rebuild, deliberate

Exit code is non-zero on failure so the scheduler can alert.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from sps.config import Settings
from sps.embedding import BGEEmbedder
from sps.indexing import (
    FlatFileError,
    FlatFileRecordSource,
    IncrementalIndexer,
    SqlRecordSource,
    Watermark,
    WatermarkStore,
)
from sps.vectorstore import QdrantVectorStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPS incremental indexer")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the micro-batch size (must stay within 250-500).",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Read records from a local .csv or .xlsx instead of the SQL source. "
        "Bypasses SPS_SOURCE_DSN entirely and reads the whole file.",
    )
    parser.add_argument(
        "--reset-watermark",
        action="store_true",
        help="Clear the high-water mark and re-index the whole table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the pending delta without embedding or upserting.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("sps.indexer")

    try:
        settings = Settings.from_env()
        if args.batch_size is not None:
            from dataclasses import replace

            settings = replace(
                settings, indexing=replace(settings.indexing, batch_size=args.batch_size)
            )
    except ValueError as exc:
        # Misconfiguration, not a runtime fault: report it plainly, no traceback.
        log.error("Configuration error: %s", exc)
        return 2

    if not args.source_file and not settings.indexing.source_dsn:
        log.error(
            "No source configured: set SPS_SOURCE_DSN, or pass --source-file "
            "to read from a local .csv/.xlsx instead."
        )
        return 2

    log.info("Vector store: %s", settings.vector_store.describe())

    watermarks = WatermarkStore(settings.indexing.watermark_path)
    if args.reset_watermark:
        log.warning("Resetting the high-water mark: the next run is a FULL rebuild.")
        watermarks.write(Watermark.initial())

    try:
        if args.source_file:
            source = FlatFileRecordSource(args.source_file)
            log.info("Source: local file %s", args.source_file)
        else:
            source = SqlRecordSource(
                settings.indexing.source_dsn, settings.indexing.source_table
            )
            log.info("Source: SQL table %s", settings.indexing.source_table)
    except (FlatFileError, ValueError) as exc:
        log.error("Cannot open the record source: %s", exc)
        return 2

    if args.dry_run:
        mark = watermarks.read()
        pending = sum(1 for _ in source.fetch_since(mark, settings.indexing.batch_size))
        log.info(
            "Dry run: %d record(s) pending since %s",
            pending,
            mark.last_modified_date.isoformat(),
        )
        return 0

    embedder = BGEEmbedder(settings.embedding)
    store = QdrantVectorStore(settings.vector_store)

    try:
        report = IncrementalIndexer(
            source=source,
            embedder=embedder,
            store=store,
            settings=settings.indexing,
            watermark_store=watermarks,
        ).run()
    except Exception:
        # The watermark was committed per batch, so a rerun resumes where this
        # one stopped rather than repeating completed work.
        log.exception("Index run failed; rerun to resume from the last committed batch.")
        return 1
    finally:
        embedder.unload()
        # Release the embedded-mode directory lock so the next scheduled process
        # (the UiPath Performer) can open the same storage.
        store.close()

    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
