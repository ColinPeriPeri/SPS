"""Pydantic schemas.

Two distinct jobs, deliberately kept apart:

* `ActorDraft` / `JudgeVerdict` are the **LLM output schemas**. They are handed
  to Azure OpenAI as a strict `response_format`, so the model cannot return a
  shape the pipeline then has to defend against.

* `SPSContract` is the **final delivered schema** -- the four keys of Section 4.
  It is assembled by the pipeline and validated here before handoff.

Why these are not the same model
--------------------------------
Only two of the four contract fields are the LLM's to produce.
`Confidence` is the retrieval composite score, and `SPS_IDs_Referred` is the set
of records the retriever actually passed to the Actor. If the LLM were asked to
emit them, it could state a confidence it did not compute and cite SPS IDs it
was never shown -- precisely the fabrication the grounding architecture exists to
prevent. So the LLM is constrained to the two fields it genuinely authors, and
the pipeline supplies the other two from measured values.

This module imports pydantic; `sps.contracts` deliberately does not, so the
scoring, gating and loop-control logic stays importable with no dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIDENCE_PATTERN = re.compile(r"^\d{1,3}%$")


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
            "Exactly 'Solution not found.' if they do not address the problem."
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


# ------------------------------------------------------------ delivered schema


class SPSContract(BaseModel):
    """The Section 4 output contract.

    Field names are the delivered keys verbatim, so `model_dump()` is the
    payload -- no aliasing layer to drift out of step with the spec.
    """

    model_config = ConfigDict(extra="forbid")

    AI_Recommendation: str
    Justification: str
    Confidence: str
    SPS_IDs_Referred: list[str] = Field(default_factory=list)

    @field_validator("Confidence")
    @classmethod
    def _confidence_is_a_percentage(cls, value: str) -> str:
        if not CONFIDENCE_PATTERN.match(value):
            raise ValueError(f"Confidence must look like '84%', got {value!r}")
        return value

    @classmethod
    def from_contract(cls, payload: dict[str, Any]) -> "SPSContract":
        return cls.model_validate(payload)

    def to_contract(self) -> dict[str, Any]:
        return self.model_dump()

    def to_row(self) -> dict[str, Any]:
        """Flatten to one spreadsheet row.

        `SPS_IDs_Referred` is a list, which has no faithful representation in a
        single cell. It is joined on ", " -- readable for the support team and
        trivially split by UiPath -- rather than written as a Python repr, which
        is what a DataFrame would otherwise produce.
        """
        return {
            "AI_Recommendation": self.AI_Recommendation,
            "Justification": self.Justification,
            "Confidence": self.Confidence,
            "SPS_IDs_Referred": ", ".join(self.SPS_IDs_Referred),
        }
