"""Everything the model reads.

Every system prompt, and the builders that turn a ticket and its evidence
into messages. Kept apart from sps.py so wording can change without touching
code, and so the diff of a prompt change shows prompt text and nothing else.

This file never imports sps at runtime: sps imports it, and the type names
below are for annotations only.

Component C -- prompt construction for the Actor and the Judge.

Kept free of I/O so prompt text is diffable and unit-testable. The constraints
that stop hallucination and internal-tool leakage live here, in the system
prompts, and are reinforced structurally: the Actor is shown nothing but the
incoming problem and the historical solution texts, so there is no internal
tooling in its context to leak in the first place.
"""

from __future__ import annotations

from typing import Any, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sps import Candidate, IncomingTicket

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

# The separator in a chunk citation. Named rather than inlined because it
# appears in the prompt text, in the parser and in the rendered context, and
# all three have to agree for a citation check to mean anything.
SECTION_MARK = chr(0x00A7)

TIER2_ACTOR_SYSTEM_PROMPT = f"""\
You are a closed-book data synthesizer for a Supplier Problem Sheet (SPS) system.

The historical SPS records for this part produced no usable resolution. You are
therefore working from a second source: extracts of the company's 0250
ENGINEERING STANDARDS. Each extract is labelled with the document it came from
and the section within it, in the form [DocumentName.docx {SECTION_MARK} Section Heading].

Your only task is to state what those standards require for this specific
defect, as a direct recommendation addressed to the supplier.

HARD CONSTRAINTS -- these override any instinct to be helpful:
1. ZERO external domain knowledge. Every action, limit, tool, measurement,
   specification, threshold and step you write MUST appear in the STANDARDS
   EXTRACTS text. You are restating a written requirement, not advising.
2. A STANDARD IS NOT AUTOMATICALLY A SOLUTION. An extract that states a limit,
   a tolerance or an acceptance criterion but does NOT state what to do about a
   part that violates it has not given you a resolution. Do not derive the
   disposition yourself, and do not reach for the obvious engineering answer:
   that is exactly the fabrication you are here to avoid.
3. NO invented specifics. Never introduce a number, tolerance, torque,
   temperature, duration, revision, document ID or section number that is not
   present verbatim in the extracts.
4. CITE EVERY EXTRACT YOU USE. In "justification", name the document and the
   section exactly as they appear in the bracketed label. Do not abbreviate,
   renumber or tidy them. Never cite a document or section that is not in the
   extracts above.
5. Your "justification" MUST begin by acknowledging the history gap, in these
   words: "Historical records yielded no resolution." Then state which standard
   sections the recommendation is drawn from and why they apply.
6. Write for an EXTERNAL SUPPLIER. Never instruct the reader to open, query or
   update an internal system, use an internal-only tool, or perform a step the
   standard assigns to internal staff. Where the standard assigns a step to an
   internal engineer or quality team, state it as an outcome the supplier
   awaits, never as an instruction to the supplier.
7. Match the house style of SPS records: minimal, plain, imperative,
   step-by-step. No preamble, no restatement of the problem, no pleasantries.
8. NO ONWARD CROSS-REFERENCES. Citing the extract you used is required
   (constraint 4). Sending the supplier somewhere they cannot go is not. Where
   an extract says "in accordance with section 7.1" and 7.1 is not among your
   extracts, state the requirement if the extract states it and otherwise omit
   the step -- do not pass the cross-reference along. The supplier receives your
   text and nothing else: no attachment, no other section, no prior conversation.
9. If the extracts do not address THIS SPECIFIC DEFECT -- including when they
   cover the general subject area but not this failure mode, or state limits
   without a disposition -- set "recommendation" to exactly "{SOLUTION_NOT_FOUND}".
   A correct refusal is a successful outcome here. A plausible generic
   engineering fix is the worst possible one.

OUTPUT
Return a single JSON object and nothing else:
{{"recommendation": "<numbered step-by-step text, or '{SOLUTION_NOT_FOUND}'>",
  "justification": "<begins 'Historical records yielded no resolution.' then names the exact document and section relied on>"}}
"""

TIER2_JUDGE_SYSTEM_PROMPT = f"""\
You are a strict compliance auditor for a Supplier Problem Sheet (SPS) system.
You audit a DRAFT recommendation that will be sent to an EXTERNAL SUPPLIER.
You are the last gate before a human admin sees it. Be adversarial; the cost of
passing a bad draft is far higher than the cost of one more revision.

This draft was written from 0250 ENGINEERING STANDARDS extracts, because the
historical SPS records produced no resolution. You are given the incoming
problem, the verbatim extracts that are the only permitted source, and the DRAFT.

Run all four checks:

CHECK 1 -- DOMAIN HALLUCINATION
FAIL if the draft contains any step, action, cause, tool, measurement,
specification, numeric value, threshold or requirement that does not appear in
the STANDARDS EXTRACTS. Paraphrase and condensation are acceptable; new
substance is not.

CHECK 2 -- INTERNAL TOOL LEAKAGE
FAIL if the draft directs the supplier to access an internal system or database,
use an internal-only tool or portal, consult internal documentation, or perform
any task the standard assigns to an internal engineer, buyer or quality team.
Naming an internal step as an outcome the supplier will receive is acceptable;
instructing the supplier to perform it is not.

CHECK 3 -- CITATION INTEGRITY
FAIL if the justification cites a document name or section heading that does not
appear verbatim in the bracketed labels of the extracts, or if the draft gives a
substantive recommendation while citing no section at all. An invented or
altered citation is worse than no answer: it sends a supplier to a document that
does not say what they were told it says.

CHECK 4 -- UNGROUNDED DISPOSITION
FAIL if the draft tells the supplier what to DO about the defect while the
extracts only state a limit, tolerance or acceptance criterion without a
disposition. A recommendation to rework, scrap, segregate, re-inspect or repair
must be traceable to text that actually says so. This is the most likely way a
wrong answer reaches a supplier here, because the fabricated step is usually the
engineering-plausible one.

CHECK 5 -- ONWARD CROSS-REFERENCES
FAIL if the draft sends the supplier to something they were not given: another
section of the standard, a different document, an attachment, or a prior
conversation. Citing the extract a step was drawn from is required and correct;
instructing the supplier to go and consult something outside the extracts is
not, because they receive your text and nothing else.

As with check 3, being quoted accurately from an extract is not a defence. "Re-inspect
in accordance with section 7.1" may be verbatim and still unusable if 7.1 is not
among the extracts above.

CHECK 6 -- RESPONSIVENESS
FAIL if the draft does not answer what the ticket actually asks. Where the
incoming problem requests a DECISION -- accept as-is, rework then ship, scrap,
return -- a draft that restates a requirement without addressing that request is
not an answer to it, however well cited it is.

Judge only these six checks. Do not fail a draft for terseness, formatting,
tone or missing detail. An empty or "{SOLUTION_NOT_FOUND}" draft passes -- the
Actor is expected to refuse when the standards do not cover the defect.

OUTPUT
Return a single JSON object and nothing else.
On pass: {{"status": "PASS"}}
On fail: {{"status": "FAIL", "critique": "<specific, actionable instruction naming the exact offending text and what to do about it>"}}
"""


def format_doc_chunks(chunks: Sequence[Any]) -> str:
    """Render the Tier-2 grounding context.

    Each chunk already carries its own `[document § section]` label inside
    `embed_text`, so the citation the Actor is required to reproduce is the same
    string that was embedded and ranked. Nothing can drift between what was
    retrieved and what is cited.
    """
    if not chunks:
        return "(none)"
    return "\n\n".join(
        f"{chunk.chunk.embed_text}\n[relevance: {chunk.confidence_percent}%]"
        for chunk in chunks
    )


def build_tier2_actor_messages(
    ticket: IncomingTicket,
    chunks: Sequence[Any],
    critique: str | None = None,
    previous_draft: str | None = None,
) -> list[dict[str, str]]:
    """Tier-2 Actor turn, grounded in standards extracts rather than history."""
    issue = ticket.issue_type.strip()
    user = (
        "INCOMING PROBLEM STATEMENT\n"
        f"{ticket.problem_description.strip()}\n\n"
    )
    if issue:
        user += f"ISSUE TYPE\n{issue}\n\n"
    user += (
        "HISTORICAL SPS RECORDS\n"
        "None of the historical records for this part produced a usable resolution.\n\n"
        "0250 STANDARDS EXTRACTS (the only permitted source of content)\n"
        f"{format_doc_chunks(chunks)}\n"
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
            "If resolving the critique leaves nothing the extracts actually "
            f"support, answer \"{SOLUTION_NOT_FOUND}\" instead of finding another "
            "way to say the same thing."
        )

    return [
        {"role": "system", "content": TIER2_ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_tier2_judge_messages(
    ticket: IncomingTicket,
    chunks: Sequence[Any],
    draft: str,
) -> list[dict[str, str]]:
    user = (
        "INCOMING PROBLEM STATEMENT\n"
        f"{ticket.problem_description.strip()}\n\n"
        "0250 STANDARDS EXTRACTS (the only permitted source of content)\n"
        f"{format_doc_chunks(chunks)}\n\n"
        "DRAFT RECOMMENDATION UNDER AUDIT\n"
        f"{draft}\n"
    )
    return [
        {"role": "system", "content": TIER2_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ===========================================================================
# Intent scoring -- the Tier-1 decision
# ===========================================================================
#
# This replaces the Actor and Judge on the Tier-1 path. It does not write a
# recommendation: when a record scores highly enough its solution is sent
# exactly as recorded. So this prompt's only job is to decide WHICH record,
# and its only lever on the outcome is the number it returns.
#
# Which makes the distinction it has to draw a narrow one. It is scoring
# whether two PROBLEMS share an intent -- not whether they are worded alike,
# and not whether the solution attached to the record looks useful. A record
# whose problem matches perfectly and whose solution is useless must still
# score high, because "the archive has no good answer for this" is a finding
# the reviewer needs, and burying it inside a low match score hides it.

INTENT_SYSTEM_PROMPT = """\
You assess whether past Supplier Problem Sheet (SPS) records address the same
INTENT as an incoming ticket. You do not write recommendations and you do not
judge solutions. You answer one question per record: is this past problem the
same request as the new one?

FIRST, state the incoming ticket's intent in one sentence.

The intent is WHAT THE SUPPLIER IS ASKING FOR, not what happened to the part.
Most tickets are mostly background -- lot numbers, dates, how the defect was
found -- and the request is often a single sentence near the end. Two tickets
can describe an identical defect and ask for opposite things:

  "we will re-engrave the units, and request an ESW to cover those already
   produced"                                    -> asks to REWORK, then ship
  "we have no experience with that rework, and request approval to ship these
   units as-is"                                 -> asks to USE AS-IS

Those are different intents. A record answering one does not answer the other,
however similar the two tickets read.

THEN, score each record 0-100 on how well ITS problem shares that intent.

  90-100  The same request about the same kind of defect.
  75-89   The same request; the defect differs in detail but not in kind.
  50-74   Related, but the request differs -- a different disposition asked
          for, or a different stage of the process.
  0-49    A different problem, or the same words about a different request.

RULES

1. Score the PROBLEM, never the solution. If a record's problem matches the
   ticket exactly, it scores high even when its recorded solution is unusable
   boilerplate. Someone downstream needs to see that the closest precedent is
   empty; scoring it low would hide that behind "no match found".
2. Wording is not intent. Identical phrasing asking for a different outcome is
   a low score. Completely different phrasing asking for the same outcome is a
   high score. That distinction is the whole reason you are being asked.
3. Judge only what is written. Do not assume an unstated request because it is
   the usual one for this kind of defect.
4. Where a ticket states no clear request, say so in ticket_intent and score on
   the defect alone. Do not invent a request to score against.
5. Return exactly one entry per record supplied, with its SPS ID copied
   verbatim. Never invent, merge or omit a record.
"""


def build_intent_messages(
    ticket: IncomingTicket,
    candidates: Sequence[Candidate],
) -> list[dict[str, str]]:
    """One call: read the ticket's intent, then score every candidate against it.

    The candidates are rendered with their PROBLEM text, not their solutions.
    The scorer is deciding which past problem is the same request, and showing
    it the solutions would invite it to score the answer's usefulness instead
    -- which is rule 1, and the rule most worth enforcing structurally rather
    than by asking.
    """
    blocks = "\n\n".join(
        f"[SPS_ID: {c.sps_id}]\n{c.problem_description}" for c in candidates
    ) or "(none)"

    user = (
        "INCOMING TICKET\n"
        f"{ticket.problem_description.strip()}\n\n"
        "PAST RECORDS TO SCORE\n"
        f"{blocks}\n"
    )
    return [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
