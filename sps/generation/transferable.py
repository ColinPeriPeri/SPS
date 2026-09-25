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

# The abstention token the Actor must emit. Imported rather than spelled out so
# the critique cannot drift from what the prompts ask for. `contracts` is itself
# stdlib-only, so this keeps the module's no-dependency property.
from ..contracts import SOLUTION_NOT_FOUND

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


# Instruments only the customer can issue. A supplier may well REQUEST one --
# the live tickets show exactly that -- but cannot grant one, so a recommendation
# that tells them to is instructing them to do the customer's job.
#
# Site-specific. Extend this rather than the pattern below.
INTERNAL_INSTRUMENTS = (
    "ESW",       # engineering specification waiver
    "waiver",
    "deviation",
    "MRB",       # material review board disposition
)

# Base forms only, and that is the whole trick. A base-form verb is what makes
# an imperative, so it separates the instruction ("Issue an ESW") from the
# outcome ("the ESW is approved", "an ESW will be issued"), which is legitimate
# and must survive. "request" is deliberately absent: requesting is the
# supplier's own action.
INTERNAL_ACTION_VERBS = (
    "issue",
    "approve",
    "waive",
    "authorise",
    "authorize",
    "grant",
)

# Verbs that are the customer's prerogative whatever the object. "Waive the
# requirement" names no instrument and is still the customer deciding. Kept
# deliberately small: "approve" and "issue" are NOT here, because a supplier can
# legitimately approve their own rework or issue replacement parts, and those
# need an instrument to disambiguate.
ALWAYS_CUSTOMER_VERBS = ("waive",)

_MISATTRIBUTED = re.compile(
    rf"\b({'|'.join(INTERNAL_ACTION_VERBS)})\s+(?:an?\s+|the\s+)?"
    rf"({'|'.join(re.escape(i) for i in INTERNAL_INSTRUMENTS)})\b",
    re.IGNORECASE,
)

# The bare-verb rule only fires at the start of a step, because that is what
# makes it an imperative. Real history contains "we will waive it if the result
# is ok" -- the customer stating an outcome the supplier can expect, which is
# useful text and must survive. Only "1. Waive the requirement." is an
# instruction, and only position tells the two apart.
_MISATTRIBUTED_IMPERATIVE = re.compile(
    rf"(?:^|\n)\s*(?:\d+[.)]\s*)?(?:please\s+)?"
    rf"({'|'.join(ALWAYS_CUSTOMER_VERBS)})\b",
    re.IGNORECASE,
)


def misattributed_actions(text: str) -> list[str]:
    """Instructions telling the supplier to perform the customer's action.

    The Actor's constraints and the Judge's second check both forbid this
    already, in prose. Neither fired on "1. Issue an ESW." because nothing in
    the system knows who issues an ESW: the historical Solution_Text is an
    internal engineer's own to-do note, carrying no actor, and the Judge has no
    domain knowledge to supply one. This supplies it, as data.

    Returns findings naming the fragment, empty when the text is clean.
    """
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for pattern in (_MISATTRIBUTED, _MISATTRIBUTED_IMPERATIVE):
        for match in pattern.finditer(text):
            fragment = " ".join(match.group(0).split())
            key = fragment.casefold()
            if key in seen:
                continue
            seen.add(key)
            findings.append(f"instruction to perform an internal action: {fragment!r}")
    return findings


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


def critique_for(findings: list[str], misattributed: list[str] | None = None) -> str:
    """Turn findings into an instruction the Actor can act on.

    Both kinds arrive in one critique rather than two, so a draft carrying both
    problems is rewritten once instead of burning two of its three attempts.
    """
    parts: list[str] = []

    if findings:
        parts.append(
            f"The draft carries {len(findings)} reference(s) that exist only in "
            f"the historical record and are not true of this ticket: "
            f"{'; '.join(findings)}. "
            "Remove each one. State the action without the reference where the "
            "step is still meaningful on its own, and drop the step entirely "
            "where it is not. Being present in the historical solutions does "
            "not make these transferable -- they describe a different ticket."
        )

    if misattributed:
        parts.append(
            f"The draft tells the supplier to perform {len(misattributed)} "
            f"action(s) that only the customer can perform: "
            f"{'; '.join(misattributed)}. "
            "The supplier may request these; they cannot grant them. Rewrite "
            "each as an outcome the supplier awaits -- 'An ESW has been "
            "requested; do not ship until it is approved' -- or drop the step. "
            "The historical text is an internal engineer's own note, so it "
            "reads as an instruction to themselves, not to the supplier."
        )

    parts.append(
        "If what remains is no action the supplier could actually perform, "
        f"answer '{SOLUTION_NOT_FOUND}' instead."
    )
    return " ".join(parts)
