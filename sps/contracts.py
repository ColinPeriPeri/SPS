"""Data contracts shared across the resolver.

Pure stdlib apart from `validators`, so the ticket and result shapes stay
importable without the ML stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .validators import normalize_part_number

# Two strings, deliberately separate, because they serve different readers.
#
# SOLUTION_NOT_FOUND is a PROTOCOL TOKEN between us and the model. The Actor is
# instructed to emit it verbatim and `Draft.is_abstention` matches on it. It is
# not business copy and should not be reworded to suit a reader who never sees
# it -- doing so silently changes what the model is asked to produce.
#
# NO_RECOMMENDATION is the only one a person reads. It reaches
# AI_Recommendation, which DEA copies into the SPS portal, so it is phrased for
# a supplier rather than for an engineer reading a log.
SOLUTION_NOT_FOUND = "Solution not found."
NO_RECOMMENDATION = "No recommendation available."


def _clean(value: Any) -> str:
    """Coerce any source value to a trimmed string; None/NaN become ''.

    Whole floats render without the fractional part. Excel holds every number as
    a double, so a numeric-looking identifier can arrive as 1001.0 and would
    otherwise become "1001.0": an id that matches nothing and cites nothing.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _lookup(data: dict[str, Any], field_name: str) -> Any:
    """Fetch a field from a ticket payload, ignoring key casing and separators.

    Every other interface names fields the SQL way (`Problem_Description`,
    `Part_Number`), so a caller writing a ticket by hand naturally uses that
    form. Reading only the lower-case spelling yields an empty ticket: the
    description is judged invalid and the part number is dropped, while the run
    still completes and looks like a legitimate refusal.
    """
    if field_name in data:
        return data[field_name]
    wanted = field_name.replace("_", "")
    for key, value in data.items():
        if str(key).strip().casefold().replace("_", "").replace(" ", "") == wanted:
            return value
    return None


@dataclass(frozen=True, slots=True)
class IncomingTicket:
    """A supplier-submitted problem sheet awaiting a recommendation."""

    problem_description: str
    sps_id: str = ""
    part_number: str = ""
    part_description: str = ""
    item_status: str = ""
    problem_reason_code: str = ""
    issue_type: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass: canonicalise in place so every construction path
        # yields the same part-number form the history filter matches on.
        canonical = normalize_part_number(self.part_number)
        if canonical != self.part_number:
            object.__setattr__(self, "part_number", canonical)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IncomingTicket":
        """Build from a ticket payload, accepting either key spelling."""
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
class Candidate:
    """A historical record retrieved for the incoming ticket."""

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


def score_to_percent(score: float) -> int:
    """Format a 0..1 score as an integer percent.

    Truncates rather than rounds: a system that gates at 89% must never report
    "89%" for a candidate it rejected at 0.8899, and understating confidence is
    the safe direction for a compliance-reviewed recommendation.
    """
    return int(min(max(score, 0.0), 1.0) * 100)


# Which evidence a recommendation was built from. Written verbatim into
# output.xlsx, so a human reviewer can weigh a precedent-backed answer
# differently from one derived from a written standard.
SOURCE_HISTORICAL = "HISTORICAL_DATA"
SOURCE_DOCUMENTATION = "0250_DOCUMENTATION"
# The run concluded, and the conclusion was that neither tier had an answer.
# Distinct from a blank cell, which would read as "this field was not filled in".
SOURCE_NONE = "NONE"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The recommendation handed to the output writer."""

    ai_recommendation: str
    justification: str
    confidence: str
    # SPS IDs when the answer came from history, document citations when it
    # came from the standards. One column either way: a reviewer reading
    # output.xlsx wants "what backs this", and Resolution_Source already says
    # which kind of thing they are looking at.
    referenced_sources: list[str] = field(default_factory=list)
    resolution_source: str = SOURCE_HISTORICAL
    # The best precedent we found, verbatim, whether or not we recommended it.
    # DEA asked to see this even on a refusal: a near-miss they can judge for
    # themselves beats a bare "no". It is raw archive text that has passed none
    # of the supplier-facing checks, which is why it carries its own banner and
    # its own column rather than being blended into the recommendation.
    closest_matching_solution: str = ""
    # Operational signals only, never written to output.xlsx. They let the
    # caller separate a dependency outage, which is worth retrying, from a
    # legitimate refusal, which is not.
    infrastructure_failure: bool = False
    diagnostic: str = ""

    @property
    def succeeded(self) -> bool:
        # Both strings, not just the one we write. The sentinel should never
        # reach this field now that NO_RECOMMENDATION exists, but if some path
        # regresses and lets it through, a refusal must not read as a success.
        return self.ai_recommendation not in (NO_RECOMMENDATION, SOLUTION_NOT_FOUND)
