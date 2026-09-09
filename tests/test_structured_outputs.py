"""Strict structured outputs on the Azure client.

The deployment is constrained by a pydantic schema rather than merely asked for
one, with a graceful fallback for deployments that cannot do json_schema.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from sps.config import LLMSettings  # noqa: E402
from sps.generation.llm import AzureOpenAIChatClient, LLMError, validate_json  # noqa: E402
from sps.schemas import ActorDraft, JudgeVerdict  # noqa: E402


class _Message:
    def __init__(self, parsed=None, refusal=None, content=None):
        self.parsed = parsed
        self.refusal = refusal
        self.content = content


class _Completion:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


class FakeAzure:
    """Mimics the shape of the openai SDK surface the client actually touches."""

    def __init__(self, parse_result=None, parse_error=None, json_content=None):
        self.parse_calls = []
        self.create_calls = []
        outer = self

        class _Parse:
            async def parse(self, **kwargs):
                outer.parse_calls.append(kwargs)
                if parse_error is not None:
                    raise parse_error
                return _Completion(parse_result)

        class _Create:
            async def create(self, **kwargs):
                outer.create_calls.append(kwargs)
                return _Completion(_Message(content=json_content))

        self.beta = type("B", (), {"chat": type("C", (), {"completions": _Parse()})()})()
        self.chat = type("C", (), {"completions": _Create()})()


def client(fake) -> AzureOpenAIChatClient:
    return AzureOpenAIChatClient(
        settings=LLMSettings(
            endpoint="https://x.openai.azure.com/",
            api_key="k",
            deployment="gpt-4o",
            temperature=0.0,
            top_p=0.1,
        ),
        client=fake,
    )


MESSAGES = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------- schema passing


async def test_the_pydantic_model_is_passed_as_response_format():
    draft = ActorDraft(recommendation="1. Rework the seam.", justification="From SPS-100.")
    fake = FakeAzure(parse_result=_Message(parsed=draft))

    result = await client(fake).complete_model(MESSAGES, ActorDraft)

    assert result is draft
    assert fake.parse_calls[0]["response_format"] is ActorDraft
    assert fake.create_calls == []  # the fallback was not used


async def test_inference_settings_are_applied_to_the_structured_call():
    fake = FakeAzure(parse_result=_Message(parsed=JudgeVerdict(status="PASS")))
    await client(fake).complete_model(MESSAGES, JudgeVerdict)

    call = fake.parse_calls[0]
    assert call["temperature"] == 0.0
    assert call["top_p"] == 0.1
    assert call["model"] == "gpt-4o"


async def test_a_refusal_is_an_error_not_a_silent_empty_draft():
    fake = FakeAzure(parse_result=_Message(parsed=None, refusal="I cannot help with that"))
    with pytest.raises(LLMError, match="refused"):
        await client(fake).complete_model(MESSAGES, ActorDraft)


async def test_missing_parsed_content_is_an_error():
    fake = FakeAzure(parse_result=_Message(parsed=None))
    with pytest.raises(LLMError, match="no parsed content"):
        await client(fake).complete_model(MESSAGES, ActorDraft)


# ----------------------------------------------------------------- fallback


async def test_unsupported_schema_falls_back_to_json_mode():
    fake = FakeAzure(
        parse_error=TypeError("response_format json_schema is not supported"),
        json_content='{"status": "FAIL", "critique": "invented a torque value"}',
    )
    verdict = await client(fake).complete_model(MESSAGES, JudgeVerdict)

    assert verdict.status == "FAIL"
    assert verdict.critique == "invented a torque value"
    assert fake.create_calls[0]["response_format"] == {"type": "json_object"}


async def test_fallback_is_remembered_and_not_retried_every_call():
    fake = FakeAzure(
        parse_error=TypeError("json_schema not supported"),
        json_content='{"status": "PASS", "critique": ""}',
    )
    engine = client(fake)
    await engine.complete_model(MESSAGES, JudgeVerdict)
    await engine.complete_model(MESSAGES, JudgeVerdict)

    assert len(fake.parse_calls) == 1   # only the first call probes
    assert len(fake.create_calls) == 2


async def test_a_real_outage_is_not_mistaken_for_a_capability_gap():
    fake = FakeAzure(parse_error=ConnectionError("connection reset by peer"))
    with pytest.raises(LLMError, match="request failed"):
        await client(fake).complete_model(MESSAGES, ActorDraft)
    assert fake.create_calls == []  # no pointless fallback attempt


async def test_fallback_output_is_still_schema_validated():
    fake = FakeAzure(
        parse_error=TypeError("json_schema not supported"),
        json_content='{"status": "MAYBE"}',
    )
    with pytest.raises(LLMError, match="did not match"):
        await client(fake).complete_model(MESSAGES, JudgeVerdict)


# ------------------------------------------------------------------ schemas


def test_actor_schema_carries_only_the_two_fields_the_model_authors():
    """Confidence and SPS_IDs_Referred are measured, not generated. If the LLM
    could emit them it could state a confidence it never computed and cite
    records it was never shown."""
    assert set(ActorDraft.model_fields) == {"recommendation", "justification"}
    assert "Confidence" not in ActorDraft.model_fields
    assert "SPS_IDs_Referred" not in ActorDraft.model_fields


def test_judge_schema_constrains_status_to_two_values():
    assert JudgeVerdict(status="PASS").critique == ""
    with pytest.raises(Exception):
        JudgeVerdict(status="MAYBE")


def test_actor_schema_forbids_extra_fields():
    with pytest.raises(Exception):
        ActorDraft(recommendation="x", justification="y", confidence="99%")


def test_validate_json_turns_a_schema_violation_into_llm_error():
    with pytest.raises(LLMError):
        validate_json('{"recommendation": "x"}', ActorDraft)
