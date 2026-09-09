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
