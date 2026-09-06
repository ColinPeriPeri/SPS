"""Azure OpenAI chat client.

Deterministic by construction: temperature=0.0 and top_p=0.1 are applied on
every call so the Actor and the Judge are reproducible for audit.

Output shape is enforced server-side. Each call passes a pydantic model as a
strict `response_format`, so the deployment is constrained to the schema rather
than merely asked for it. Deployments or API versions without structured-output
support fall back to JSON mode plus client-side pydantic validation, which
yields the same guarantee one round-trip later.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from ..config import LLMSettings

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# Imported lazily inside functions so this module stays importable without
# pydantic for the dependency-free core tests.
ModelT = TypeVar("ModelT")


class LLMError(RuntimeError):
    """Raised when the model call fails or returns unusable content."""


@runtime_checkable
class ChatClient(Protocol):
    async def complete_model(self, messages: Sequence[dict[str, str]], model: type) -> Any:
        """Return an instance of `model` built from the deployment's response."""


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a model response that is supposed to be a single JSON object.

    Falls back to extracting the outermost brace-delimited block, which covers
    the occasional deployment that wraps JSON in a code fence.
    """
    text = (raw or "").strip()
    if not text:
        raise LLMError("Model returned an empty response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise LLMError(f"Model response was not JSON: {text[:200]!r}") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model response was not JSON: {text[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def validate_json(raw: str, model: type[ModelT]) -> ModelT:
    """Parse text into a pydantic model, turning any failure into LLMError."""
    from pydantic import ValidationError

    payload = parse_json_response(raw)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"Response did not match {model.__name__}: {exc}") from exc


class AzureOpenAIChatClient:
    """Async Azure OpenAI client using strict structured outputs."""

    def __init__(self, settings: LLMSettings | None = None, client=None) -> None:
        self.settings = settings or LLMSettings()
        self._client = client
        # Flipped once a deployment rejects json_schema, so the fallback is
        # taken directly on subsequent calls instead of paying a failed
        # round-trip every time.
        self._structured_outputs_supported = True

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncAzureOpenAI

            missing = [
                name
                for name, value in (
                    ("AZURE_OPENAI_ENDPOINT", self.settings.endpoint),
                    ("AZURE_OPENAI_API_KEY", self.settings.api_key),
                    ("AZURE_OPENAI_DEPLOYMENT", self.settings.deployment),
                )
                if not value
            ]
            if missing:
                # Names only. Never the values -- this text reaches the ops log.
                raise LLMError(f"Missing Azure OpenAI configuration: {', '.join(missing)}")

            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.settings.endpoint,
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
                timeout=self.settings.request_timeout,
            )
        return self._client

    def _common(self) -> dict[str, Any]:
        return {
            "model": self.settings.deployment,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }

    async def complete_model(
        self, messages: Sequence[dict[str, str]], model: type[ModelT]
    ) -> ModelT:
        """Constrain the deployment to `model` and return a validated instance."""
        if self._structured_outputs_supported:
            try:
                completion = await self.client.beta.chat.completions.parse(
                    messages=list(messages),
                    response_format=model,
                    **self._common(),
                )
            except Exception as exc:
                if not _is_unsupported_schema_error(exc):
                    raise LLMError(f"Azure OpenAI request failed: {exc}") from exc
                logger.warning(
                    "Deployment %r rejected structured outputs (%s); falling back to "
                    "JSON mode with client-side validation. Set "
                    "AZURE_OPENAI_API_VERSION>=2024-08-01-preview on a gpt-4o "
                    "2024-08-06 or later deployment to enable them.",
                    self.settings.deployment,
                    type(exc).__name__,
                )
                self._structured_outputs_supported = False
            else:
                choice = completion.choices[0].message
                refusal = getattr(choice, "refusal", None)
                if refusal:
                    raise LLMError(f"Model refused the request: {refusal}")
                if choice.parsed is None:
                    raise LLMError("Structured output returned no parsed content")
                return choice.parsed

        return validate_json(await self._complete_json_mode(messages), model)

    async def _complete_json_mode(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            response = await self.client.chat.completions.create(
                messages=list(messages),
                response_format={"type": "json_object"},
                **self._common(),
            )
        except Exception as exc:
            raise LLMError(f"Azure OpenAI request failed: {exc}") from exc
        if not response.choices:
            raise LLMError("Azure OpenAI returned no choices")
        return response.choices[0].message.content or ""


def _is_unsupported_schema_error(exc: Exception) -> bool:
    """True when the failure is 'this deployment cannot do json_schema'.

    Distinguished from a real outage so a capability gap degrades gracefully
    while a genuine fault still surfaces as an infrastructure failure.
    """
    if isinstance(exc, (TypeError, NotImplementedError)):
        return True
    # A capability gap always names the feature; a timeout, reset connection or
    # 5xx never does. That is the whole distinction -- keep it legible.
    text = str(exc).lower()
    markers = ("json_schema", "response_format", "structured output")
    return any(marker in text for marker in markers)
