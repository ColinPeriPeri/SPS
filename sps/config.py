"""Environment-driven configuration.

Every tunable named in the spec lives here so the operational envelope
(batch size, threshold, top-k, inference settings) is auditable in one place.
Stdlib only -- no pydantic import, so tests run without the ML stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --- Fixed by the spec; changing these changes the documented behaviour. -----
TOP_K = 15
CONFIDENCE_THRESHOLD = 0.75
MIN_QUERY_LENGTH = 10
MIN_TEXT_LENGTH = 15
MAX_ATTEMPTS = 3  # 1 initial draft + 2 refinement retries

# Component A must micro-batch inside this band to stay clear of OOM on the
# 10-12 GB CPU host.
MIN_INDEX_BATCH = 250
MAX_INDEX_BATCH = 500

# BGE-v1.5 retrieval convention: queries carry an instruction prefix, indexed
# passages do not. This is a model-level encoding convention, not metadata
# concatenation -- the embedded text is still only the Problem_Description.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Defaults are module constants rather than class attributes: with
# @dataclass(slots=True) the class attribute is a slot descriptor, not the
# default value, so from_env() cannot read defaults off `cls`.
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_DIMENSION = 1024
DEFAULT_DEVICE = "cpu"
DEFAULT_ENCODE_BATCH = 16
DEFAULT_INDEX_BATCH = 400
DEFAULT_SOURCE_TABLE = "dbo.SupplierProblemSheet"
DEFAULT_WATERMARK_PATH = "./state/watermark.json"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "sps_problems"
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
class IndexingSettings:
    batch_size: int = DEFAULT_INDEX_BATCH
    source_dsn: str = ""
    source_table: str = DEFAULT_SOURCE_TABLE
    watermark_path: str = DEFAULT_WATERMARK_PATH
    min_text_length: int = MIN_TEXT_LENGTH

    def __post_init__(self) -> None:
        if not (MIN_INDEX_BATCH <= self.batch_size <= MAX_INDEX_BATCH):
            raise ValueError(
                f"SPS_INDEX_BATCH_SIZE must be within "
                f"[{MIN_INDEX_BATCH}, {MAX_INDEX_BATCH}] to stay RAM-safe on the "
                f"CPU host; got {self.batch_size}"
            )

    @classmethod
    def from_env(cls) -> "IndexingSettings":
        return cls(
            batch_size=_env_int("SPS_INDEX_BATCH_SIZE", DEFAULT_INDEX_BATCH),
            source_dsn=_env("SPS_SOURCE_DSN"),
            source_table=_env("SPS_SOURCE_TABLE", DEFAULT_SOURCE_TABLE),
            watermark_path=_env("SPS_WATERMARK_PATH", DEFAULT_WATERMARK_PATH),
            min_text_length=_env_int("SPS_MIN_TEXT_LENGTH", MIN_TEXT_LENGTH),
        )


@dataclass(frozen=True, slots=True)
class VectorStoreSettings:
    """Qdrant connection.

    `path` selects embedded mode -- Qdrant runs in-process against a local
    directory, with no Docker, no service and no network. It takes an exclusive
    lock on that directory, so exactly one process may hold it at a time; the
    UiPath schedule guarantees that by pausing the Performer queue while the
    indexer runs. When `path` is set it wins over `url`.
    """

    url: str = DEFAULT_QDRANT_URL
    api_key: str = ""
    collection: str = DEFAULT_COLLECTION
    path: str = ""

    @property
    def embedded(self) -> bool:
        return bool(self.path)

    def describe(self) -> str:
        return f"embedded:{self.path}" if self.embedded else f"server:{self.url}"

    @classmethod
    def from_env(cls) -> "VectorStoreSettings":
        return cls(
            url=_env("SPS_QDRANT_URL", DEFAULT_QDRANT_URL),
            api_key=_env("SPS_QDRANT_API_KEY"),
            collection=_env("SPS_COLLECTION", DEFAULT_COLLECTION),
            path=_env("SPS_QDRANT_PATH"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    top_k: int = TOP_K
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    min_query_length: int = MIN_QUERY_LENGTH
    # Cap on how many >=threshold records are handed to the Actor. Defaults to
    # top_k, i.e. every qualifying candidate; lower it only to trim LLM cost.
    max_context_records: int = TOP_K

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            top_k=_env_int("SPS_TOP_K", TOP_K),
            confidence_threshold=_env_float("SPS_CONFIDENCE_THRESHOLD", CONFIDENCE_THRESHOLD),
            min_query_length=_env_int("SPS_MIN_QUERY_LENGTH", MIN_QUERY_LENGTH),
            max_context_records=_env_int("SPS_MAX_CONTEXT_RECORDS", TOP_K),
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
class Settings:
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    indexing: IndexingSettings = field(default_factory=IndexingSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embedding=EmbeddingSettings.from_env(),
            indexing=IndexingSettings.from_env(),
            vector_store=VectorStoreSettings.from_env(),
            retrieval=RetrievalSettings.from_env(),
            llm=LLMSettings.from_env(),
        )
