"""Environment-driven configuration.

The embedding and inference settings, in one auditable place. Stdlib only, so
this imports without the ML stack.

The confidence threshold deliberately lives elsewhere -- beside the retrieval
engine in `sps/retrieval/in_memory.py` -- because it is a property of the
embedding model's scoring distribution, not of the process configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MAX_ATTEMPTS = 3  # 1 initial draft + 2 refinement retries

# BGE-v1.5 retrieval convention: queries carry an instruction prefix, indexed
# passages do not. This is a model-level encoding convention, not metadata
# concatenation -- the embedded text is still only the Problem_Description.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Defaults are module constants rather than class attributes: with
# @dataclass(slots=True) the class attribute is a slot descriptor, not the
# default value, so from_env() cannot read defaults off `cls`.
# bge-small: 384 dimensions, roughly ten times faster than bge-large on CPU
# (about 1 s versus 11 s to encode 300 short texts), which is what makes
# embedding at query time viable at all.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384
DEFAULT_DEVICE = "cpu"
DEFAULT_ENCODE_BATCH = 16
# 2024-10-21 is the first GA version with json_schema structured outputs;
# 2024-06-01 silently lacks them and would take the JSON-mode fallback.
DEFAULT_API_VERSION = "2024-10-21"
DEFAULT_REQUEST_TIMEOUT = 60.0

# Azure embeddings are a separate deployment from the chat model, with its own
# endpoint and key. Kept apart so one can be configured, rotated or fail
# without touching the other.
DEFAULT_EMBEDDING_API_VERSION = "2024-10-21"
DEFAULT_EMBEDDING_TIMEOUT = 20.0


def load_env_file(start=None):
    """Best-effort .env load. Returns the Path that was read, or None.

    It lives here, next to the settings that read the variables, rather than
    inside one entry point -- `verify_embedder` was written later, did not know
    to call the resolver's private copy, and spent a migration reporting every
    credential as missing while the resolver read the same file perfectly well.

    A process launched by a UiPath robot does not necessarily inherit an
    interactive shell's environment, which is why the file is read at all. Real
    environment variables always win (`override=False`), so a machine-level
    setting is never overridden by a stale checkout.
    """
    from pathlib import Path

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    candidates = [
        Path(start) / ".env" if start else Path.cwd() / ".env",
        # The project root, so running from anywhere still finds the
        # deployment's file.
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def env_file_candidates():
    """Where load_env_file would look, for error messages that can be acted on."""
    from pathlib import Path

    return [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_list(key: str) -> tuple[str, ...]:
    """A comma-separated list, trimmed, casefolded, blanks dropped.

    Casefolded on the way in so every comparison downstream is already
    case-insensitive and no caller has to remember to do it.
    """
    return tuple(part.strip().casefold() for part in _env(key).split(",") if part.strip())


# How closely the retrieved precedent must address the ticket's intent before
# its solution is sent verbatim. Process configuration rather than a property
# of the embedding space -- it gates an LLM's judgement, not a cosine, which is
# why it lives here and the retrieval thresholds do not.
#
# PROVISIONAL, and for a reason worth remembering: an LLM's "82%" is not a
# calibrated probability. It ranks candidates against each other reliably
# enough; as an absolute gate it means whatever the model decides it means.
# Measure with scripts/run_eval_batch.py before treating 0.75 as settled.
DEFAULT_INTENT_THRESHOLD = 0.75

# How many candidates the intent scorer is shown. The retrieval gate below it
# is deliberately permissive, so this is what actually bounds the call.
DEFAULT_INTENT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class IntentSettings:
    """The Tier-1 decision: which precedent is close enough to send as-is."""

    threshold: float = DEFAULT_INTENT_THRESHOLD
    top_k: int = DEFAULT_INTENT_TOP_K
    # Problem_Reason_Code values for which Tier 2 may run. Empty means Tier 2
    # never runs: the 0250 standards are scoped to specific reason codes, and
    # an unset list is read as "none configured" rather than "all of them".
    #
    # That makes a lost .env line disable Tier 2 silently, so the resolver
    # reports the two cases -- nothing configured, versus this code not listed
    # -- in different words. Only the first is a deployment fault.
    tier2_reason_codes: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "IntentSettings":
        return cls(
            threshold=_env_float("SPS_INTENT_THRESHOLD", DEFAULT_INTENT_THRESHOLD),
            top_k=_env_int("SPS_INTENT_TOP_K", DEFAULT_INTENT_TOP_K),
            tier2_reason_codes=_env_list("SPS_TIER2_REASON_CODES"),
        )

    def tier2_allowed(self, reason_code: str) -> bool:
        return bool(self.tier2_reason_codes) and (
            str(reason_code or "").strip().casefold() in self.tier2_reason_codes
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model_name: str = DEFAULT_MODEL_NAME
    dimension: int = DEFAULT_DIMENSION
    device: str = DEFAULT_DEVICE
    encode_batch_size: int = DEFAULT_ENCODE_BATCH  # inner batch per forward pass
    use_query_instruction: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            model_name=_env("SPS_EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
            dimension=_env_int("SPS_EMBEDDING_DIM", DEFAULT_DIMENSION),
            device=_env("SPS_EMBEDDING_DEVICE", DEFAULT_DEVICE),
            encode_batch_size=_env_int("SPS_EMBEDDING_ENCODE_BATCH", DEFAULT_ENCODE_BATCH),
            use_query_instruction=_env_bool("SPS_USE_BGE_QUERY_INSTRUCTION", True),
        )


@dataclass(frozen=True, slots=True)
class LLMSettings:
    endpoint: str = ""
    api_key: str = ""
    api_version: str = DEFAULT_API_VERSION
    deployment: str = ""
    temperature: float = 0.0
    top_p: float = 0.1
    max_attempts: int = MAX_ATTEMPTS
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_API_KEY"),
            api_version=_env("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
            deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            temperature=_env_float("SPS_LLM_TEMPERATURE", 0.0),
            top_p=_env_float("SPS_LLM_TOP_P", 0.1),
            max_attempts=_env_int("SPS_MAX_ATTEMPTS", MAX_ATTEMPTS),
            request_timeout=_env_float("SPS_LLM_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
        )


@dataclass(frozen=True, slots=True)
class AzureEmbeddingSettings:
    """The Azure embedding deployment used as the primary encoder.

    Separate from LLMSettings: the embedding deployment is its own resource,
    and the resolver must be able to lose it (bad key, throttling, network)
    and fall back to the local model without the chat path being affected.

    The timeout is deliberately shorter than the chat timeout. A slow embedding
    call has a working alternative one exception away, so waiting a full minute
    to discover that costs more than it saves.
    """

    endpoint: str = ""
    api_key: str = ""
    deployment: str = ""
    api_version: str = DEFAULT_EMBEDDING_API_VERSION
    request_timeout: float = DEFAULT_EMBEDDING_TIMEOUT

    @property
    def configured(self) -> bool:
        """All three are needed; a partial configuration is a misconfiguration."""
        return bool(self.endpoint and self.api_key and self.deployment)

    def missing(self) -> list[str]:
        """Which variables are absent, by NAME -- never a value."""
        return [
            name
            for name, value in (
                ("AZURE_EMBEDDING_ENDPOINT", self.endpoint),
                ("AZURE_EMBEDDING_API_KEY", self.api_key),
                ("AZURE_EMBEDDING_DEPLOYMENT", self.deployment),
            )
            if not value
        ]

    @classmethod
    def from_env(cls) -> "AzureEmbeddingSettings":
        return cls(
            endpoint=_env("AZURE_EMBEDDING_ENDPOINT"),
            api_key=_env("AZURE_EMBEDDING_API_KEY"),
            deployment=_env("AZURE_EMBEDDING_DEPLOYMENT"),
            api_version=_env("AZURE_EMBEDDING_API_VERSION", DEFAULT_EMBEDDING_API_VERSION),
            request_timeout=_env_float("AZURE_EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT),
        )
