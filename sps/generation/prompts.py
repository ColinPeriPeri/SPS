"""Component C -- prompt construction for the Actor and the Judge.

Kept free of I/O so prompt text is diffable and unit-testable. The constraints
that stop hallucination and internal-tool leakage live here, in the system
prompts, and are reinforced structurally: the Actor is shown nothing but the
incoming problem and the historical solution texts, so there is no internal
tooling in its context to leak in the first place.
"""

from __future__ import annotations

from typing import Sequence

from ..contracts import SOLUTION_NOT_FOUND, Candidate, IncomingTicket

ACTOR_SYSTEM_PROMPT = f"""\
You are a closed-book data synthesizer for a Supplier Problem Sheet (SPS) system.

You will be given (a) a supplier's incoming problem statement and (b) the
verbatim ACTUAL SOLUTION text of historical SPS records that were matched to it.

Your only task is to restate what those historical solutions did, as a direct
recommendation addressed to the supplier.

HARD CONSTRAINTS -- these override any instinct to be helpful:
1. ZERO external domain knowledge. Every action, tool, part, measurement,
   specification, threshold, standard and step you write MUST appear in the
   HISTORICAL SOLUTIONS text. You are copying and condensing, not advising.
2. NO extrapolation. Do not infer a cause, generalize a step, add a safety
   note, add a verification step, or supply a value that is not written in the
   historical text. If the historical text omits a detail, omit it too.
3. NO invented specifics. Never introduce a number, tolerance, torque, temperature,
   duration, revision, document ID or tool name that is not present verbatim.
4. Write for an EXTERNAL SUPPLIER. Never instruct the reader to open, query or
   update an internal system, use an internal-only tool, or perform a step that
   the historical text shows was carried out by internal staff. If a historical
   step was an internal action, either omit it or state it as an outcome the
   supplier awaits, never as an instruction to the supplier.
5. Match the house style of historical SPS records: minimal, plain, imperative,
   step-by-step. No preamble, no restatement of the problem, no closing pleasantries.
6. If the historical solutions do not actually address the incoming problem, set
   "recommendation" to exactly "{SOLUTION_NOT_FOUND}".

OUTPUT
Return a single JSON object and nothing else:
{{"recommendation": "<numbered step-by-step text, or '{SOLUTION_NOT_FOUND}'>",
  "justification": "<one or two sentences on which historical records this is drawn from and why they apply, citing SPS IDs>"}}
"""

JUDGE_SYSTEM_PROMPT = """\
You are a strict compliance auditor for a Supplier Problem Sheet (SPS) system.
You audit a DRAFT recommendation that will be sent to an EXTERNAL SUPPLIER.
You are the last gate before a human admin sees it. Be adversarial; the cost of
passing a bad draft is far higher than the cost of one more revision.

You are given the incoming problem, the verbatim HISTORICAL SOLUTIONS that are
the only permitted source, and the DRAFT.

Run both checks:

CHECK 1 -- DOMAIN HALLUCINATION
FAIL if the draft contains any step, action, cause, tool, part, measurement,
specification, numeric value, threshold, standard or document reference that
does not appear in the HISTORICAL SOLUTIONS text. Paraphrase and condensation
are acceptable; new substance is not. Added "best practice", added verification
or safety steps, and filled-in details the source left blank are all hallucination.

CHECK 2 -- INTERNAL TOOL LEAKAGE
FAIL if the draft directs the supplier to access an internal system or database,
use an internal-only tool or portal, consult internal documentation, contact
parties on the supplier's behalf using internal routing, or perform any task that
belongs to an internal engineer, buyer or quality team rather than the supplier.
Naming an internal system as the source of an outcome the supplier will receive
is acceptable; instructing the supplier to go use it is not.

Judge only these two checks. Do not fail a draft for terseness, formatting, tone
or missing detail. An empty or "Solution not found." draft passes.

OUTPUT
Return a single JSON object and nothing else.
On pass: {"status": "PASS"}
On fail: {"status": "FAIL", "critique": "<specific, actionable instruction naming the exact offending text and what to do about it>"}
"""


def format_historical_solutions(candidates: Sequence[Candidate]) -> str:
    """Render the grounding context: SPS ID + verbatim Actual_Solution only.

    Nothing else from the payload is included -- the Actor cannot leak metadata
    it was never shown.
    """
    if not candidates:
        return "(none)"
    blocks = []
    for candidate in candidates:
        blocks.append(
            f"[SPS_ID: {candidate.sps_id} | match: {candidate.confidence_percent}%]\n"
            f"{candidate.actual_solution}"
        )
    return "\n\n".join(blocks)


def build_actor_messages(
    ticket: IncomingTicket,
    candidates: Sequence[Candidate],
    critique: str | None = None,
    previous_draft: str | None = None,
) -> list[dict[str, str]]:
    """Actor turn. On a retry the rejected draft and the Judge's critique are
    appended so the rewrite is targeted rather than a blind resample."""
    user = (
        "INCOMING PROBLEM STATEMENT\n"
        f"{ticket.problem_description.strip()}\n\n"
        "HISTORICAL SOLUTIONS (the only permitted source of content)\n"
        f"{format_historical_solutions(candidates)}\n"
    )

    if critique:
        user += (
            "\nYOUR PREVIOUS DRAFT WAS REJECTED BY THE COMPLIANCE AUDITOR.\n"
            "REJECTED DRAFT\n"
            f"{previous_draft or ''}\n\n"
            "AUDITOR CRITIQUE (you must resolve this)\n"
            f"{critique}\n\n"
            "Rewrite the recommendation so the critique no longer applies. Remove "
            "the offending content rather than replacing it with something new. "
            "Do not add anything absent from the HISTORICAL SOLUTIONS above."
        )

    return [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_judge_messages(
    ticket: IncomingTicket,
    candidates: Sequence[Candidate],
    draft: str,
) -> list[dict[str, str]]:
    user = (
        "INCOMING PROBLEM STATEMENT\n"
        f"{ticket.problem_description.strip()}\n\n"
        "HISTORICAL SOLUTIONS (the only permitted source of content)\n"
        f"{format_historical_solutions(candidates)}\n\n"
        "DRAFT RECOMMENDATION UNDER AUDIT\n"
        f"{draft}\n"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
