"""Input gatekeeping: canonical part numbers, and strict ticket validation.

The retrieval engine filters history by exact part-number match, so a part
number that differs only by invisible characters is not a near miss -- it is a
total miss, and the ticket is refused as though no history existed. Enterprise
extracts routinely carry non-breaking spaces from web forms and zero-width
characters from copy-paste, neither of which is visible to the person who
pasted them.

Stdlib only, so this module imports without the ML stack.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters that carry no meaning in an identifier but break exact matching.
# Unicode category Cf ("format") covers the zero-width family -- ZWSP U+200B,
# ZWNJ U+200C, ZWJ U+200D, word joiner U+2060, BOM U+FEFF -- and the soft
# hyphen U+00AD, which renders as nothing but is not whitespace.
_INVISIBLE = {"­"}

# Every Unicode space separator, not just ASCII: U+00A0 non-breaking space is
# the usual culprit from a browser form, and str.strip() does remove it, but
# interior ones survive and must go too.
_WHITESPACE = re.compile(r"\s+", re.UNICODE)

# Structural delimiters an engineering part number legitimately contains. These
# are preserved: 0012-43951, 0012/43951, 0012_43951 and 0012.43951 are four
# different identifiers, not four spellings of one.
STRUCTURAL = frozenset("-/_.")

MIN_QUERY_LENGTH = 10


def strip_invisible(value: str) -> str:
    """Remove format characters and soft hyphens, wherever they appear."""
    return "".join(
        ch for ch in value if ch not in _INVISIBLE and unicodedata.category(ch) != "Cf"
    )


# A value made only of printable ASCII with no spaces is already canonical bar
# its case, so the expensive path can be skipped. This matters: the history scan
# canonicalises every row's part number, and per-character unicodedata lookups
# over 300k rows cost more than reading the file.
_ALREADY_CLEAN = re.compile(r"^[\x21-\x7E]+$")


def normalize_part_number(value: object) -> str:
    """Canonical form of a part number.

    Trims surrounding whitespace of every kind, removes invisible and
    zero-width characters, collapses interior whitespace away entirely (a part
    number has no internal spaces), preserves structural delimiters, and
    upper-cases.

    Whole floats are rendered without a decimal: Excel stores every number as a
    double, so a numeric-looking part number can arrive as 1243951.0 and would
    otherwise canonicalise to "1243951.0", matching nothing.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            value = int(value)
    text = str(value)
    # Fast path for the overwhelmingly common case: nothing to strip.
    if _ALREADY_CLEAN.match(text):
        return text.upper()
    text = strip_invisible(text)
    # Interior whitespace is removed rather than collapsed to a single space:
    # "0012 - 43951" and "0012-43951" are the same identifier badly typed.
    text = _WHITESPACE.sub("", text)
    return text.upper()


@dataclass(frozen=True, slots=True)
class ValidationError:
    """Why a ticket cannot be processed, in terms fit for a status file."""

    code: str
    reason: str


def validate_part_number(value: object) -> ValidationError | None:
    """Reject a part number that cannot be used as an exact-match key.

    Malformed means: nothing left after canonicalisation, or nothing but
    structural delimiters. A value of "---" or "  " carries no identity, and
    filtering on it would silently match nothing while looking like a real
    query.
    """
    canonical = normalize_part_number(value)
    if not canonical:
        return ValidationError("INVALID_INPUT", "Part number is missing or blank.")
    if all(ch in STRUCTURAL for ch in canonical):
        return ValidationError(
            "INVALID_INPUT",
            f"Part number {canonical!r} contains no identifying characters.",
        )
    return None


def validate_problem_description(value: object, minimum: int = MIN_QUERY_LENGTH):
    """Reject a problem statement too short to retrieve on."""
    text = _WHITESPACE.sub(" ", strip_invisible(str(value or ""))).strip()
    if not text:
        return ValidationError("INVALID_INPUT", "Problem description is missing or blank.")
    if len(text) < minimum:
        return ValidationError(
            "INVALID_INPUT",
            f"Problem description is {len(text)} characters; at least {minimum} required.",
        )
    return None


def validate_ticket(part_number: object, problem_description: object) -> ValidationError | None:
    """Gate a ticket before any embedding or LLM call is made.

    Part number is checked first: it is the cheaper check and the one that
    determines whether there is any history to search at all.
    """
    return validate_part_number(part_number) or validate_problem_description(
        problem_description
    )
