"""Catch facts that are true of the historical record but false about the ticket.

The Judge checks *provenance* -- did this text come from the source? -- and it
is very good at that. What it cannot reliably catch is the class of statement
that passes provenance and is still wrong:

    "ESW#20033465 is submitted for these issues."

That sentence is in the historical solution verbatim, so it is not a
hallucination. It is also false about the ticket now being answered: the work
request was raised for a different issue, months ago, and the supplier reading
it can chase a number that has nothing to do with them. The same applies to
"see the attachment" when no attachment exists, and "per discussed" when no
discussion has taken place.

This module is the deterministic half of the defence. It runs before the Judge
because it is local and free while the Judge is a paid network call, and because
a regex does not talk itself round: an LLM asked "is this in the source?" and
then "but should it still be here?" tends to answer the first question twice.

Stdlib only, like `sps.validators`, so it stays testable in milliseconds.
"""

from __future__ import annotations

import re

# Internal tracking systems whose identifiers must never reach a supplier. Site
# specific, so extend this rather than the patterns below. Matched with or
# without a separator, because the house style varies: ESW#20033465, CAR-1234,
# "NCR 5566".
KNOWN_TRACKING_PREFIXES = (
    "ESW",   # engineering support work request
    "CAR",   # corrective action request
    "SCAR",  # supplier corrective action request
    "NCR",   # non-conformance report
    "MRB",   # material review board
    "8D",
)

_PREFIX_ALTERNATION = "|".join(re.escape(p) for p in KNOWN_TRACKING_PREFIXES)

# (category, pattern). The category names the reason in the critique handed back
# to the Actor, so it can be told what kind of thing to remove rather than only
# which characters.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "internal tracking identifier",
        re.compile(rf"\b(?:{_PREFIX_ALTERNATION})\s*[#:\-]?\s*\d{{3,}}\b", re.IGNORECASE),
    ),
    (
        # Any LETTER-led token followed by # and digits. Deliberately requires
        # the letters: a part number is digit-led (0012-43951) and must never
        # trip this.
        "tracking identifier",
        re.compile(r"\b[A-Za-z]{2,6}\s*#\s*\d{3,}\b"),
    ),
    (
        # "attach" the verb is fine -- "attach photos to your response" asks the
        # supplier to do something they can do. "attachment" and "attached" name
        # a document that travelled with the original record and is not here.
        "reference to an attachment",
        re.compile(r"\b(?:attachments?|attached|enclosed|enclosure)\b", re.IGNORECASE),
    ),
    (
        "reference to something not included",
        re.compile(r"\bsee\s+(?:the\s+)?(?:below|above|enclosed)\b", re.IGNORECASE),
    ),
    (
        "reference to a prior conversation",
        re.compile(
            r"\b(?:as|per)\s+(?:discussed|agreed|advised|requested)\b|"
            r"\bper\s+(?:our|the)\s+(?:call|discussion|conversation|meeting|email|phone)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lot, batch or purchase-order number",
        re.compile(
            r"\b(?:lot|batch|purchase\s+order|p\.?o\.?)\s*(?:#|no\.?|number)?\s*\d{3,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        # A calendar date in a recommendation is nearly always the historical
        # record's date. Durations ("within 30 days") are untouched.
        "a specific date",
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ),
)


def untransferable_references(text: str) -> list[str]:
    """Fragments in `text` that belong to the source record, not this ticket.

    Returns a list of human-readable findings, empty when the text is clean.
    Each finding names the category and quotes the offending fragment, because
    it is handed to the Actor as a critique and "remove the attachment
    reference" is actionable where "failed check 3" is not.

    Named individuals are deliberately absent: no pattern distinguishes a
    person's name from a material or a process, so that one is left to the
    Judge, where a wrong answer costs a retry rather than a false rejection.
    """
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            fragment = " ".join(match.group(0).split())
            # Deduplicated on the fragment alone, not the category. A known
            # prefix also matches the generic LETTERS#DIGITS rule, and listing
            # ESW#20033465 twice under two headings makes the critique read as
            # two problems when there is one. _PATTERNS is ordered specific
            # first, so the better description is the one that survives.
            key = fragment.casefold()
            if key in seen:
                continue
            seen.add(key)
            findings.append(f"{category}: {fragment!r}")
    return findings


def critique_for(findings: list[str]) -> str:
    """Turn findings into an instruction the Actor can act on."""
    listed = "; ".join(findings)
    return (
        f"The draft carries {len(findings)} reference(s) that exist only in the "
        f"historical record and are not true of this ticket: {listed}. "
        "Remove each one. State the action without the reference where the step "
        "is still meaningful on its own, and drop the step entirely where it is "
        "not. Being present in the historical solutions does not make these "
        "transferable -- they describe a different ticket. If removing them "
        "leaves no action the supplier could actually perform, answer "
        "'Solution not found.' instead."
    )
