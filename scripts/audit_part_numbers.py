"""Audit -- and optionally repair -- part_number drift in the vector index.

The retrieval filter is an exact string match, and query-side part numbers are
canonicalised (`.strip().upper()`). Points written before that canonicalisation,
or from a source that spells part numbers inconsistently, carry a payload value
that no query will ever match. Such a record is not wrong, it is *invisible*:
retrieval reports "no history for this part" and the ticket is refused.

Repair does not require re-embedding. `part_number` is payload only -- the
vector encodes the cleansed Problem_Description and nothing else -- so the fix
is a `set_payload` on one field, which merges into the existing payload and
leaves the vector untouched. That is seconds against 300k points, rather than
hours of CPU re-encoding.

    python -m scripts.audit_part_numbers              # report only, changes nothing
    python -m scripts.audit_part_numbers --fix        # repair, then re-verify
    python -m scripts.audit_part_numbers --samples 25

A JSON report goes to stdout; logs go to stderr. Exit 0 on success, 1 on an
infrastructure fault, 2 on a configuration error -- the same convention as
`run_indexer`.

The same pass reports two data-quality figures that cost nothing extra:

* the **blank** part-number rate, since the retrieval filter makes history
  without a part number unreachable to any ticket that supplies one;
* the count of **purely numeric** part numbers, whose leading zeros Excel
  destroys in the sheet itself, before any reader sees the file.

Where the house format is alphanumeric and free of stray spaces, both
canonicalisation steps are no-ops and this run is a confirmation rather than a
repair -- a non-zero figure means an assumption about the source no longer holds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sps.config import Settings
from sps.contracts import normalize_part_number

logger = logging.getLogger("sps.audit")

EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_CONFIG = 2

SCROLL_PAGE = 1_000
# Qdrant takes a list of point IDs per set_payload call; cap it so one call
# cannot become unboundedly large on a heavily drifted collection.
WRITE_CHUNK = 1_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.audit_part_numbers",
        description=(
            "Report part_number values in the index that no query can match, "
            "and optionally rewrite them in place without re-embedding."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite drifted part_number payloads. Without this the run is read-only.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="How many distinct drift patterns to list in the report (default 10).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


@dataclass
class ScanResult:
    """What one pass over the payloads found."""

    total: int = 0
    blank: int = 0
    # Part numbers that are nothing but digits. These are the ones whose
    # leading zeros Excel destroys before any of this code sees the file,
    # so a non-zero count means an assumption about the source data no
    # longer holds and the extract needs checking.
    purely_numeric: int = 0
    drift: dict = field(default_factory=lambda: defaultdict(list))

    @property
    def affected(self) -> int:
        return sum(len(ids) for ids in self.drift.values())


def scan(client, collection: str) -> ScanResult:
    """One paged pass over the payloads. No vectors, no embedding."""
    result = ScanResult()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=["part_number"],
            with_vectors=False,
        )
        for point in points:
            result.total += 1
            stored = str((point.payload or {}).get("part_number", ""))
            if not stored.strip():
                # Blank is left alone: it is legitimately "no part number", not
                # a spelling of one, and rewriting it would invent data.
                result.blank += 1
                continue
            if stored.isdigit():
                result.purely_numeric += 1
            if stored != normalize_part_number(stored):
                result.drift[stored].append(point.id)
        if offset is None:
            return result


def repair(client, collection: str, drift: dict[str, list]) -> int:
    """Rewrite only the part_number field, in one call per distinct value.

    `set_payload` merges into the existing payload rather than replacing it, so
    every other field -- and the vector -- is untouched. Grouping by target
    value means a collection where 50k points share one misspelling costs one
    call, not 50k.
    """
    fixed = 0
    for stored, ids in sorted(drift.items()):
        canonical = normalize_part_number(stored)
        for start in range(0, len(ids), WRITE_CHUNK):
            chunk = ids[start : start + WRITE_CHUNK]
            client.set_payload(
                collection_name=collection,
                payload={"part_number": canonical},
                points=chunk,
                wait=True,
            )
            fixed += len(chunk)
        logger.info("rewrote %r -> %r on %d point(s)", stored, canonical, len(ids))
    return fixed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_CONFIG

    from sps.vectorstore import QdrantVectorStore

    store = QdrantVectorStore(settings.vector_store)
    collection = settings.vector_store.collection
    logger.info("Auditing %s collection %r", settings.vector_store.describe(), collection)

    try:
        client = store.client
        if not client.collection_exists(collection):
            logger.error(
                "Collection %r does not exist. Index something first, or check "
                "SPS_COLLECTION / SPS_QDRANT_PATH.",
                collection,
            )
            return EXIT_CONFIG

        found = scan(client, collection)
        total, drift, affected = found.total, found.drift, found.affected

        report = {
            "collection": collection,
            "store": settings.vector_store.describe(),
            "points_scanned": total,
            "blank_part_number": found.blank,
            "blank_pct": round(100 * found.blank / total, 2) if total else 0.0,
            "purely_numeric_part_numbers": found.purely_numeric,
            "non_canonical_points": affected,
            "non_canonical_pct": round(100 * affected / total, 2) if total else 0.0,
            "distinct_drift_values": len(drift),
            "examples": [
                {"stored": stored, "canonical": normalize_part_number(stored), "points": count}
                for stored, count in Counter(
                    {k: len(v) for k, v in drift.items()}
                ).most_common(max(args.samples, 0))
            ],
            "fixed": 0,
            "remaining": affected,
        }

        if affected and not args.fix:
            logger.warning(
                "%d point(s) across %d distinct value(s) are unreachable by an exact-match "
                "query. Re-run with --fix to rewrite them (payload only, no re-embedding).",
                affected,
                len(drift),
            )
        elif args.fix and affected:
            report["fixed"] = repair(client, collection, drift)
            # Re-scan rather than assume: cheap, and it proves the repair landed.
            report["remaining"] = scan(client, collection).affected
            if report["remaining"]:
                logger.error("%d point(s) still drifted after repair", report["remaining"])
            else:
                logger.info("All part_number payloads are canonical; vectors untouched.")
        elif not affected:
            logger.info("No drift: every populated part_number is already canonical.")

        if found.purely_numeric:
            # Excel stores numbers as doubles, so a purely numeric part number
            # loses its leading zeros in the sheet itself -- before any reader
            # sees it. Nothing here can recover them.
            logger.warning(
                "%d part number(s) are purely numeric. If the source is a "
                "spreadsheet, any leading zeros were lost before ingest; format "
                "that column as Text or supply CSV.",
                found.purely_numeric,
            )

        print(json.dumps(report, indent=2))
        return EXIT_OK

    except Exception:
        logger.exception("Audit failed")
        return EXIT_INFRASTRUCTURE
    finally:
        # Release the embedded-mode directory lock for the next scheduled process.
        store.close()


if __name__ == "__main__":
    sys.exit(main())
