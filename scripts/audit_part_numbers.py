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

The scan also reports the **blank** part-number rate, since the retrieval filter
makes history without a part number unreachable to any ticket that supplies one,
and that rate is worth knowing before an evaluation run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict

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


def scan(client, collection: str):
    """One paged pass over the payloads. No vectors, no embedding.

    Returns (total, blank, drift) where drift maps stored value -> point IDs.
    """
    drift: dict[str, list] = defaultdict(list)
    total = 0
    blank = 0
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
            total += 1
            stored = str((point.payload or {}).get("part_number", ""))
            if not stored.strip():
                blank += 1
                continue
            # Blank is left alone: it is legitimately "no part number", not a
            # spelling of one, and rewriting it would invent data.
            if stored != normalize_part_number(stored):
                drift[stored].append(point.id)
        if offset is None:
            return total, blank, drift


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

        total, blank, drift = scan(client, collection)
        affected = sum(len(ids) for ids in drift.values())

        report = {
            "collection": collection,
            "store": settings.vector_store.describe(),
            "points_scanned": total,
            "blank_part_number": blank,
            "blank_pct": round(100 * blank / total, 2) if total else 0.0,
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
            _, _, remaining_drift = scan(client, collection)
            report["remaining"] = sum(len(ids) for ids in remaining_drift.values())
            if report["remaining"]:
                logger.error("%d point(s) still drifted after repair", report["remaining"])
            else:
                logger.info("All part_number payloads are canonical; vectors untouched.")
        elif not affected:
            logger.info("No drift: every populated part_number is already canonical.")

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
