"""Data contracts shared across the SPS pipeline.

Pure stdlib: importable without torch / qdrant / openai installed so the
scoring, gating and loop-control logic stays unit-testable on any machine.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

# Sentinel string the contract requires for every non-success path.
SOLUTION_NOT_FOUND = "Solution not found."

# Metadata payload keys, in the order fixed by the spec.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "sps_id",
    "content_hash",
    "actual_solution",
    "part_number",
    "part_description",
    "item_status",
    "problem_reason_code",
    "issue_type",
)


PAIR_SEPARATOR = chr(31)  # ASCII unit separator

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace and trim.

    Applied before length checks so a field of spaces or newlines cannot pass
    the minimum-length gate, and before hashing so cosmetic reformatting does
    not read as a distinct record.
    """
    return _WHITESPACE.sub(" ", (text or "")).strip()


def content_hash(problem: str, solution: str) -> str:
    """Stable SHA-256 identity of a problem-solution *pair*.

    Case- and whitespace-insensitive. Persisted in the vector payload so
    duplicates can be detected across indexing runs, not just within one.
    """
    # Joined on the ASCII unit separator, which cannot occur in SPS free text,
    # so ("ab", "c") and ("a", "bc") cannot hash alike.
    problem_key = normalize_text(problem).casefold()
    solution_key = normalize_text(solution).casefold()
    basis = problem_key + PAIR_SEPARATOR + solution_key
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def normalize_part_number(value: Any) -> str:
    """Canonical form of a part number: trimmed and upper-cased.

    Applied on both sides -- at ingest, so the payload is written canonically,
    and on the incoming ticket -- so the vector search's exact-match filter
    cannot miss on casing alone and report NO_MATCHES for a part that is
    genuinely in the index.
    """
    return _clean(value).upper()


def _lookup(data: dict[str, Any], field: str) -> Any:
    """Fetch `field` from a payload, ignoring key casing and separators.

    Every other interface in this system names fields the SQL way
    (`Problem_Description`, `Part_Number`), so a caller hand-writing a ticket
    naturally uses that form. Reading only the lower-case spelling silently
    yields an empty ticket: the description is judged invalid, the part number
    is dropped so the search filter fails open, and the whole thing exits 0
    looking like a legitimate refusal.
    """
    if field in data:
        return data[field]
    wanted = field.replace("_", "")
    for key, value in data.items():
        if str(key).strip().casefold().replace("_", "").replace(" ", "") == wanted:
            return value
    return None


def _clean(value: Any) -> str:
    """Coerce any source value to a trimmed string; None/NaN become ''."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One row read from the source relational database (Component A input)."""

    sps_id: str
    problem_description: str
    actual_solution: str
    part_number: str = ""
    part_description: str = ""
    item_status: str = ""
    problem_reason_code: str = ""
    issue_type: str = ""
    last_modified_date: datetime | None = None

    def __post_init__(self) -> None:
        # Frozen dataclass: normalise in place so every construction path -- SQL,
        # flat file, cleanse() rebuilds, tests -- writes the same canonical form.
        canonical = normalize_part_number(self.part_number)
        if canonical != self.part_number:
            object.__setattr__(self, "part_number", canonical)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SourceRecord":
        """Build from a DB row mapping, tolerating NULLs and stray whitespace."""
        lmd = row.get("last_modified_date")
        if isinstance(lmd, str):
            lmd = datetime.fromisoformat(lmd)
        return cls(
            sps_id=_clean(row.get("sps_id")),
            problem_description=_clean(row.get("problem_description")),
            actual_solution=_clean(row.get("actual_solution")),
            part_number=_clean(row.get("part_number")),
            part_description=_clean(row.get("part_description")),
            item_status=_clean(row.get("item_status")),
            problem_reason_code=_clean(row.get("problem_reason_code")),
            issue_type=_clean(row.get("issue_type")),
            last_modified_date=lmd,
        )

    def content_digest(self) -> str:
        """SHA-256 of this record's sanitized problem-solution pair."""
        return content_hash(self.problem_description, self.actual_solution)

    def payload(self) -> dict[str, str]:
        """Metadata payload upserted alongside the vector."""
        return {
            "sps_id": self.sps_id,
            "content_hash": self.content_digest(),
            "actual_solution": self.actual_solution,
            "part_number": self.part_number,
            "part_description": self.part_description,
            "item_status": self.item_status,
            "problem_reason_code": self.problem_reason_code,
            "issue_type": self.issue_type,
        }

    def sort_key(self) -> tuple[datetime, str]:
        """Recency ordering used to pick the survivor among duplicate pairs."""
        return (self.last_modified_date or datetime.min, self.sps_id)


@dataclass(frozen=True, slots=True)
class IncomingTicket:
    """A supplier-submitted problem sheet awaiting an AI recommendation."""

    problem_description: str
    sps_id: str = ""
    part_number: str = ""
    part_description: str = ""
    item_status: str = ""
    problem_reason_code: str = ""
    issue_type: str = ""

    def __post_init__(self) -> None:
        canonical = normalize_part_number(self.part_number)
        if canonical != self.part_number:
            object.__setattr__(self, "part_number", canonical)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IncomingTicket":
        """Build from a ticket payload, accepting either key spelling.

        `Problem_Description` and `problem_description` both work; see _lookup.
        """
        return cls(
            problem_description=_clean(_lookup(data, "problem_description")),
            sps_id=_clean(_lookup(data, "sps_id")),
            part_number=_clean(_lookup(data, "part_number")),
            part_description=_clean(_lookup(data, "part_description")),
            item_status=_clean(_lookup(data, "item_status")),
            problem_reason_code=_clean(_lookup(data, "problem_reason_code")),
            issue_type=_clean(_lookup(data, "issue_type")),
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Raw vector-DB result: payload plus cosine similarity."""

    payload: dict[str, str]
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieved historical record after metadata boosting (Component B.3)."""

    sps_id: str
    actual_solution: str
    part_number: str
    part_description: str
    item_status: str
    problem_reason_code: str
    issue_type: str
    cosine_similarity: float
    composite_score: float
    applied_boosts: tuple[str, ...] = ()

    @property
    def confidence_percent(self) -> int:
        return score_to_percent(self.composite_score)


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """A vector plus its metadata payload, ready to upsert."""

    sps_id: str
    vector: Sequence[float]
    payload: dict[str, str]


def score_to_percent(score: float) -> int:
    """Format a 0..1 score as an integer percent.

    Truncates rather than rounds: a system that gates at 75% must never report
    "75%" for a candidate it rejected at 0.7499, and understating confidence is
    the safe direction for a compliance-reviewed recommendation.
    """
    bounded = min(max(score, 0.0), 1.0)
    return int(bounded * 100)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The final, admin-facing output (Section 4 contract)."""

    ai_recommendation: str
    justification: str
    confidence: str
    sps_ids_referred: list[str] = field(default_factory=list)
    # Operational signals only -- deliberately NOT part of to_contract(), which
    # must stay exactly the four keys of the Section 4 schema.
    #
    # infrastructure_failure lets a caller set a process exit code / raise a
    # system exception on a dependency outage. diagnostic carries the specific
    # cause for the ops status file, which may say "Missing Azure credentials"
    # where the supplier-facing Justification must stay generic.
    infrastructure_failure: bool = False
    diagnostic: str = ""

    def to_contract(self) -> dict[str, Any]:
        """Serialise to the exact JSON schema the service must emit."""
        return {
            "AI_Recommendation": self.ai_recommendation,
            "Justification": self.justification,
            "Confidence": self.confidence,
            "SPS_IDs_Referred": list(self.sps_ids_referred),
        }

    @property
    def succeeded(self) -> bool:
        return self.ai_recommendation != SOLUTION_NOT_FOUND
