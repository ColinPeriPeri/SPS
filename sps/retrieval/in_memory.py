"""In-memory retrieval: filter history by part number, then embed and rank.

No vector database. The part-number filter means semantic search only ever runs
against one part's history, which is small enough to embed on demand -- so the
index, the indexer schedule, the embedded-storage lock and payload drift all
stop existing.

The cost moves to load time: the history file is scanned once per ticket.
Measured on this hardware at 300k rows, that is about 0.8 s for CSV and about
40 s for XLSX, because openpyxl must inflate and parse XML per row while a CSV
is a linear read. Prefer CSV for a large history; the engine reads both.

Order of operations is deliberate -- each step is cheaper than the next, so the
expensive ones only run on tickets that survive:

    validate -> filter by part -> cap to the newest N -> embed -> rank -> gate
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..contracts import Candidate, IncomingTicket, SearchHit
from ..validators import normalize_part_number

logger = logging.getLogger(__name__)

# Columns the engine needs, and the spellings it accepts for each.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sps_id": ("SPS_ID", "SPSID", "Id"),
    "part_number": ("Part_Number", "Part Number", "PartNo"),
    "problem_description": ("Problem_Description", "Problem Description", "Problem"),
    "actual_solution": ("Solution_Text", "Actual_Solution", "Solution"),
    "issue_type": ("Issue_Type", "Issue Type"),
    "problem_reason_code": ("Problem_Reason_Code", "Reason_Code"),
    "part_description": ("Part_Description",),
    "item_status": ("Item_Status",),
    "last_modified_date": ("Last_Modified_Date", "Last Modified Date", "Modified"),
}
REQUIRED = ("sps_id", "part_number", "problem_description", "actual_solution")

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}

# Latency cap. Encoding is roughly linear in candidate count; at ~1 s per 300
# short texts with bge-small on CPU, this keeps the embedding step near a
# second even for a part with thousands of records.
MAX_CANDIDATES = 300

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class HistoryError(ValueError):
    """The history file cannot be used."""


@dataclass(frozen=True, slots=True)
class HistoryRow:
    sps_id: str
    part_number: str
    problem_description: str
    actual_solution: str
    issue_type: str = ""
    problem_reason_code: str = ""
    part_description: str = ""
    item_status: str = ""
    last_modified_date: datetime | None = None

    def sort_key(self) -> tuple[datetime, str]:
        """Recency ordering; undated rows sort oldest but stay comparable."""
        return (self.last_modified_date or _EPOCH, self.sps_id)

    def payload(self) -> dict[str, str]:
        return {
            "sps_id": self.sps_id,
            "actual_solution": self.actual_solution,
            "part_number": self.part_number,
            "part_description": self.part_description,
            "item_status": self.item_status,
            "problem_reason_code": self.problem_reason_code,
            "issue_type": self.issue_type,
        }


@dataclass
class RetrievalStats:
    """What the run actually did, for the status file and for latency work."""

    rows_scanned: int = 0
    part_matches: int = 0
    usable: int = 0
    capped_to: int = 0
    load_seconds: float = 0.0
    embed_seconds: float = 0.0
    qualified: int = 0
    top_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_scanned": self.rows_scanned,
            "part_matches": self.part_matches,
            "usable": self.usable,
            "capped_to": self.capped_to,
            "load_seconds": round(self.load_seconds, 3),
            "embed_seconds": round(self.embed_seconds, 3),
            "qualified": self.qualified,
            "top_score": round(self.top_score, 4),
        }


def _norm_header(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")


def build_header_map(headers: Sequence[Any]) -> dict[str, int]:
    """Resolve column positions, tolerating spelling and case."""
    lookup: dict[str, int] = {}
    for index, header in enumerate(headers):
        key = _norm_header(header)
        if key and key not in lookup:
            lookup[key] = index

    mapping: dict[str, int] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            index = lookup.get(_norm_header(alias))
            if index is not None:
                mapping[field_name] = index
                break

    missing = [f for f in REQUIRED if f not in mapping]
    if missing:
        wanted = ", ".join(f"{f} (e.g. {COLUMN_ALIASES[f][0]!r})" for f in missing)
        found = ", ".join(repr(str(h)) for h in headers if str(h).strip())
        raise HistoryError(f"History file is missing required column(s): {wanted}. Found: {found}")
    return mapping


def _cell(row: Sequence[Any], mapping: dict[str, int], field_name: str) -> Any:
    index = mapping.get(field_name)
    if index is None or index >= len(row):
        return ""
    return row[index]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _parse_date(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iter_rows(path: Path) -> Iterator[Sequence[Any]]:
    """Stream raw rows, header first, without materialising the file."""
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        import csv

        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, newline="", encoding="utf-8-sig") as handle:
            yield from csv.reader(handle, delimiter=delimiter)
    elif suffix in EXCEL_SUFFIXES:
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            yield from book.active.iter_rows(values_only=True)
        finally:
            book.close()
    else:
        raise HistoryError(
            f"Unsupported history file type {suffix!r}; expected one of "
            f"{', '.join(sorted(CSV_SUFFIXES | EXCEL_SUFFIXES))}"
        )


def load_matching_history(
    path: Path | str,
    part_number: str,
    min_text_length: int = 15,
    stats: RetrievalStats | None = None,
) -> list[HistoryRow]:
    """Scan the history once, keeping only usable rows for this part.

    Filtering happens during the scan rather than after it, so memory stays flat
    regardless of how large the file is: only the matching rows are retained.
    """
    path = Path(path)
    if not path.exists():
        raise HistoryError(f"History file not found: {path}")

    stats = stats or RetrievalStats()
    wanted = normalize_part_number(part_number)
    started = time.time()
    mapping: dict[str, int] | None = None
    kept: list[HistoryRow] = []

    for row in _iter_rows(path):
        if mapping is None:
            mapping = build_header_map(list(row))
            continue
        if row is None or all(c in (None, "") for c in row):
            continue
        stats.rows_scanned += 1

        # Dual check: the stored part number is canonicalised the same way the
        # ticket's was, so drift in the file cannot hide a genuine match.
        if normalize_part_number(_cell(row, mapping, "part_number")) != wanted:
            continue
        stats.part_matches += 1

        problem = _text(_cell(row, mapping, "problem_description"))
        solution = _text(_cell(row, mapping, "actual_solution"))
        sps_id = _text(_cell(row, mapping, "sps_id"))
        if not sps_id or len(problem) < min_text_length or len(solution) < min_text_length:
            continue

        kept.append(
            HistoryRow(
                sps_id=sps_id,
                part_number=wanted,
                problem_description=problem,
                actual_solution=solution,
                issue_type=_text(_cell(row, mapping, "issue_type")),
                problem_reason_code=_text(_cell(row, mapping, "problem_reason_code")),
                part_description=_text(_cell(row, mapping, "part_description")),
                item_status=_text(_cell(row, mapping, "item_status")),
                last_modified_date=_parse_date(_cell(row, mapping, "last_modified_date")),
            )
        )

    if mapping is None:
        raise HistoryError(f"History file is empty: {path}")

    stats.usable = len(kept)
    stats.load_seconds = time.time() - started
    logger.info(
        "Scanned %d row(s) in %.2fs; %d matched part %r, %d usable",
        stats.rows_scanned,
        stats.load_seconds,
        stats.part_matches,
        wanted,
        stats.usable,
    )
    return kept


def cap_to_newest(rows: list[HistoryRow], limit: int = MAX_CANDIDATES) -> list[HistoryRow]:
    """Keep the newest `limit` rows.

    Recency is the right thing to drop on: an old fix for a part is more likely
    to have been superseded than a recent one, and encoding cost is linear in
    what survives.
    """
    if len(rows) <= limit:
        return rows
    ordered = sorted(rows, key=HistoryRow.sort_key, reverse=True)[:limit]
    logger.info("Capped %d candidate(s) to the %d most recent", len(rows), limit)
    return ordered


@dataclass
class InMemoryRetriever:
    """Filter, embed and rank one part's history per ticket."""

    embedder: Any
    history_path: Path | str
    confidence_threshold: float = 0.82
    max_candidates: int = MAX_CANDIDATES
    max_context_records: int = 15
    min_text_length: int = 15
    stats: RetrievalStats = field(default_factory=RetrievalStats)

    def retrieve(self, ticket: IncomingTicket) -> list[Candidate]:
        """Return qualified candidates, best first. Empty if none qualify.

        Raises HistoryError if the file is unusable; the caller distinguishes
        NO_MATCHES (nothing for this part) from BELOW_CONFIDENCE_THRESHOLD
        (matches found, none similar enough) by inspecting `stats`.
        """
        rows = load_matching_history(
            self.history_path,
            ticket.part_number,
            min_text_length=self.min_text_length,
            stats=self.stats,
        )
        if not rows:
            return []

        rows = cap_to_newest(rows, self.max_candidates)
        self.stats.capped_to = len(rows)

        started = time.time()
        # One batch for the candidates, one call for the query: the query needs
        # the BGE instruction prefix and the passages must not have it, so they
        # cannot share a single encode call.
        matrix = self.embedder.embed_passages([r.problem_description for r in rows])
        query = self.embedder.embed_query(ticket.problem_description.strip())
        self.stats.embed_seconds = time.time() - started

        import numpy as np

        candidates_matrix = np.asarray(matrix, dtype=np.float32)
        query_vector = np.asarray(query, dtype=np.float32)
        # Vectors are L2-normalised at encode time, so the dot product is the
        # cosine similarity and the whole ranking is one matrix multiply.
        similarities = candidates_matrix @ query_vector

        scored = [
            _to_candidate(row, float(sim)) for row, sim in zip(rows, similarities)
        ]
        scored.sort(key=lambda c: (c.composite_score, c.cosine_similarity, c.sps_id), reverse=True)

        self.stats.top_score = scored[0].composite_score if scored else 0.0
        qualified = [c for c in scored if c.composite_score >= self.confidence_threshold]
        self.stats.qualified = len(qualified)

        logger.info(
            "Embedded %d candidate(s) in %.2fs; top %.4f, %d at or above %.2f",
            len(rows),
            self.stats.embed_seconds,
            self.stats.top_score,
            len(qualified),
            self.confidence_threshold,
        )
        return qualified[: self.max_context_records]


def _to_candidate(row: HistoryRow, similarity: float) -> Candidate:
    """Score is the cosine alone.

    No metadata boosting: part number is already an exact filter, and the other
    boosts existed to discriminate within a mixed-part result set that no longer
    occurs.
    """
    bounded = min(max(similarity, 0.0), 1.0)
    return Candidate(
        sps_id=row.sps_id,
        actual_solution=row.actual_solution,
        part_number=row.part_number,
        part_description=row.part_description,
        item_status=row.item_status,
        problem_reason_code=row.problem_reason_code,
        issue_type=row.issue_type,
        cosine_similarity=bounded,
        composite_score=bounded,
    )


def to_search_hits(rows: Sequence[HistoryRow], similarities: Sequence[float]) -> list[SearchHit]:
    """Adapter for anything still expecting the vector-store shape."""
    return [
        SearchHit(payload=row.payload(), cosine_similarity=float(sim))
        for row, sim in zip(rows, similarities)
    ]
