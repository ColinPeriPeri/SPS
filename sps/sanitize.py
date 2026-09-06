"""Component A.2 -- data sanitization and deduplication.

Pure functions over SourceRecord; no I/O, no model, no DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import MIN_TEXT_LENGTH
from .contracts import SourceRecord, content_hash, normalize_text

# content_hash / normalize_text live in contracts.py because SourceRecord.payload()
# now persists the hash; they are re-exported here so this module stays the single
# import site for sanitization concerns.
__all__ = [
    "SanitizeReport",
    "cleanse",
    "content_hash",
    "deduplicate",
    "is_usable",
    "normalize_text",
]


@dataclass(frozen=True, slots=True)
class SanitizeReport:
    """Per-batch counters, surfaced in the indexer log for ops visibility."""

    received: int = 0
    dropped_missing_id: int = 0
    dropped_short_problem: int = 0
    dropped_short_solution: int = 0
    dropped_duplicate: int = 0
    kept: int = 0

    def merge(self, other: "SanitizeReport") -> "SanitizeReport":
        return SanitizeReport(
            received=self.received + other.received,
            dropped_missing_id=self.dropped_missing_id + other.dropped_missing_id,
            dropped_short_problem=self.dropped_short_problem + other.dropped_short_problem,
            dropped_short_solution=self.dropped_short_solution + other.dropped_short_solution,
            dropped_duplicate=self.dropped_duplicate + other.dropped_duplicate,
            kept=self.kept + other.kept,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "received": self.received,
            "dropped_missing_id": self.dropped_missing_id,
            "dropped_short_problem": self.dropped_short_problem,
            "dropped_short_solution": self.dropped_short_solution,
            "dropped_duplicate": self.dropped_duplicate,
            "kept": self.kept,
        }


def is_usable(record: SourceRecord, min_length: int = MIN_TEXT_LENGTH) -> bool:
    """True when both texts survive the null/empty/too-short filter."""
    return (
        bool(record.sps_id)
        and len(normalize_text(record.problem_description)) >= min_length
        and len(normalize_text(record.actual_solution)) >= min_length
    )


def cleanse(records: Iterable[SourceRecord], min_length: int = MIN_TEXT_LENGTH):
    """Drop unusable rows, returning (kept, rejected_ids, report).

    Rejected IDs matter for incremental runs: a record that *used* to be valid
    and has since been blanked out must be evicted from the index, not merely
    skipped.
    """
    kept: list[SourceRecord] = []
    rejected: list[str] = []
    received = missing_id = short_problem = short_solution = 0

    for record in records:
        received += 1
        if not record.sps_id:
            missing_id += 1
            continue
        problem = normalize_text(record.problem_description)
        solution = normalize_text(record.actual_solution)
        if len(problem) < min_length:
            short_problem += 1
            rejected.append(record.sps_id)
            continue
        if len(solution) < min_length:
            short_solution += 1
            rejected.append(record.sps_id)
            continue
        # Persist the normalized text so what we embed is what we store.
        kept.append(
            SourceRecord(
                sps_id=record.sps_id,
                problem_description=problem,
                actual_solution=solution,
                part_number=record.part_number,
                part_description=record.part_description,
                item_status=record.item_status,
                problem_reason_code=record.problem_reason_code,
                issue_type=record.issue_type,
                last_modified_date=record.last_modified_date,
            )
        )

    report = SanitizeReport(
        received=received,
        dropped_missing_id=missing_id,
        dropped_short_problem=short_problem,
        dropped_short_solution=short_solution,
        kept=len(kept),
    )
    return kept, rejected, report


def deduplicate(records: Sequence[SourceRecord]):
    """Collapse identical problem-solution pairs, keeping the latest SPS_ID.

    "Latest" is ordered by (Last_Modified_Date, SPS_ID) so the winner is stable
    even when several duplicates share a timestamp. Returns
    (survivors, superseded_ids).
    """
    winners: dict[str, SourceRecord] = {}
    superseded: list[str] = []

    for record in records:
        key = content_hash(record.problem_description, record.actual_solution)
        incumbent = winners.get(key)
        if incumbent is None:
            winners[key] = record
            continue
        if record.sort_key() > incumbent.sort_key():
            superseded.append(incumbent.sps_id)
            winners[key] = record
        else:
            superseded.append(record.sps_id)

    return list(winners.values()), superseded
