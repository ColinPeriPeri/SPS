"""Pydantic schemas.

Two distinct jobs, deliberately kept apart:

* `ActorDraft` / `JudgeVerdict` are the **LLM output schemas**. They are handed
  to Azure OpenAI as a strict `response_format`, so the model cannot return a
  shape the pipeline then has to defend against.

Only two of the delivered fields are the LLM's to produce. Confidence is the
measured cosine and the referenced SPS IDs are the records the retriever
actually passed to the Actor. If the LLM were asked to emit them it could state
a confidence it never computed and cite records it was never shown -- precisely
the fabrication the grounding architecture exists to prevent. So the model is
constrained to the two fields it genuinely authors, and the resolver supplies
the rest from measured values.

This module imports pydantic; `sps.contracts` deliberately does not, so the
scoring, gating and loop-control logic stays importable with no dependencies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import SOLUTION_NOT_FOUND


# ---------------------------------------------------------------- LLM schemas


class ActorDraft(BaseModel):
    """What the Actor is allowed to return.

    Note what is absent: no confidence, no SPS IDs, no metadata. The Actor
    authors prose grounded in the historical solutions it was shown, nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    recommendation: str = Field(
        description=(
            "Numbered, minimal, step-by-step recommendation for the supplier, "
            "drawn only from the historical solutions provided. "
            f"Exactly '{SOLUTION_NOT_FOUND}' if they do not address the problem."
        )
    )
    justification: str = Field(
        description=(
            "One or two sentences on which historical records this is drawn from "
            "and why they apply, citing their SPS IDs."
        )
    )


class JudgeVerdict(BaseModel):
    """What the Judge is allowed to return."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["PASS", "FAIL"] = Field(
        description="PASS if the draft clears both compliance checks, else FAIL."
    )
    critique: str = Field(
        default="",
        description=(
            "Empty when status is PASS. When FAIL, a specific actionable "
            "instruction naming the exact offending text and what to do about it."
        ),
    )
