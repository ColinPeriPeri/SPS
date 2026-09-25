"""Tier 2 -- the 0250 standards corpus: cache, then search.

Tier 1 searches one part's history, which is small and different every run, so
it is embedded on demand. Tier 2 searches the *same* standards corpus on every
ticket, so embedding it per ticket would pay the same cost forever. It is
embedded once and cached on disk, keyed by a hash of the documents themselves:
edit a document and the next run rebuilds, touch nothing and the next run loads
vectors in milliseconds.

Two caches, never one. A threshold and a vector both belong to a single
embedding space, so `0250_cache_azure.npz` and `0250_cache_local.npz` are
separate files and a run reads whichever matches the encoder that actually
answered. Mixing them would compute cosines between unrelated spaces and return
numbers that look like similarities.

Only the azure cache is written at present: the local encoder is out of the
pipeline, so `0250_cache_local.npz` is reachable only through an injected
embedder. The filename stays reserved rather than removed, so a corpus embedded
by a re-wired local model can never land in the Azure file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..contracts import IncomingTicket
from ..embedding import Embedder
from .docx_parser import DocChunk, DocParseError, iter_doc_files, parse_docx

logger = logging.getLogger(__name__)

# Where the standards live. Relative to the working directory, so the UiPath
# Performer's project folder is the root, same as every other path the robot
# passes.
DEFAULT_DOCS_DIR = Path("data/0250_docs")

CACHE_FILENAMES = {
    "azure": "0250_cache_azure.npz",
    "local": "0250_cache_local.npz",
}

AZURE_BACKEND = "azure"
LOCAL_BACKEND = "local"

# How many chunks reach the model. The spec's 3-5: enough that a defect
# described in one section and dispositioned in the next is still covered,
# few enough that the Actor cannot quietly blend four unrelated standards.
TIER2_TOP_K = 5

# Tier-2 thresholds are NOT Tier-1's, and the gap is not a detail.
#
# Measured with bge-small on a representative 0250 section: the enriched query
# against a chunk that genuinely answers it scores 0.7800, an adjacent section
# of the same standard 0.6681, and an unrelated standard 0.5237. Tier 1's local
# gate is 0.89 -- reusing it would reject the correct chunk on every ticket and
# the tier would never fire once.
#
# The distribution is lower because the comparison is different in kind: Tier 1
# matches one short defect sentence against another, while Tier 2 matches a
# defect sentence against 300 words of standards prose that answers it without
# resembling it.
#
# 0.62 sits below the adjacent-section score and well above the unrelated one.
# It is deliberately permissive: Tier 2 only runs when Tier 1 has already
# failed, and the Actor's grounding check is the gate that actually decides
# whether a chunk answers the defect. Measured on one document -- widen the
# eval set before treating it as settled.
TIER2_LOCAL_THRESHOLD = 0.62

# PROVISIONAL, and doubly so: derived from Tier 1's Azure threshold, which is
# itself unmeasured, scaled by the local Tier-1:Tier-2 ratio (0.62/0.89). Run
# the batch evaluator against the real deployment before trusting it.
TIER2_AZURE_THRESHOLD = 0.35

CACHE_FORMAT = 2


class DocCacheError(RuntimeError):
    """The corpus could not be prepared."""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk that survived the Tier-2 gate."""

    chunk: DocChunk
    score: float

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def confidence_percent(self) -> int:
        from ..contracts import score_to_percent

        return score_to_percent(self.score)


@dataclass
class Tier2Stats:
    """What Tier 2 did, for the status sheet and for latency work."""

    documents: int = 0
    chunks: int = 0
    # "hit" | "rebuilt" | "absent" -- absent means there was nothing to search.
    cache_state: str = "absent"
    backend: str = ""
    backend_detail: str = ""
    fallback_reason: str = ""
    threshold_used: float = 0.0
    top_score: float = 0.0
    qualified: int = 0
    build_seconds: float = 0.0
    query_seconds: float = 0.0
    # The best-scoring chunk, gate or no gate. Tier 1's `best_candidate` twin,
    # and kept out of `as_dict()` for the same reason.
    best_chunk: "ScoredChunk | None" = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "cache_state": self.cache_state,
            "backend": self.backend,
            "backend_detail": self.backend_detail,
            "fallback_reason": self.fallback_reason,
            "threshold_used": self.threshold_used,
            "top_score": round(self.top_score, 4),
            "qualified": self.qualified,
            "build_seconds": round(self.build_seconds, 3),
            "query_seconds": round(self.query_seconds, 3),
        }


# ------------------------------------------------------------------ hashing


def corpus_hash(paths: Sequence[Path]) -> str:
    """SHA-256 over the whole corpus: names, sizes and bytes.

    Names are hashed as well as contents so that renaming a document -- which
    changes every citation it produces -- invalidates the cache, even though not
    one byte of its text moved.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        data = path.read_bytes()
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(data)
    return digest.hexdigest()


# -------------------------------------------------------------- cache files


def _cache_path(cache_dir: Path, backend: str) -> Path:
    return Path(cache_dir) / CACHE_FILENAMES[backend]


def load_cache(path: Path, expected_hash: str, expected_model: str):
    """Return (chunks, vectors) if the cache is valid, else None.

    Validated on the corpus hash *and* the model identity. The two cache files
    separate Azure from local, but a deployment swapped from
    text-embedding-3-large to -small keeps the same filename while producing a
    different space -- so the model that wrote the vectors is recorded and
    checked.
    """
    import numpy as np

    if not path.exists():
        return None
    try:
        # allow_pickle stays off: a cache file is data, and a pickle loader
        # would execute whatever it was handed.
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta.get("format") != CACHE_FORMAT:
                logger.info("%s is an older cache format; rebuilding", path.name)
                return None
            if meta.get("corpus_hash") != expected_hash:
                logger.info("%s is stale (documents changed); rebuilding", path.name)
                return None
            if meta.get("model") != expected_model:
                logger.info(
                    "%s was written by %r, now running %r; rebuilding",
                    path.name, meta.get("model"), expected_model,
                )
                return None
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
            documents = archive["documents"]
            sections = archive["sections"]
            texts = archive["texts"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        # A truncated or hand-edited cache is a rebuild, not a failure: the
        # documents on disk are always the source of truth.
        logger.warning("Ignoring unreadable cache %s: %s", path.name, exc)
        return None

    if not (len(documents) == len(sections) == len(texts) == len(vectors)):
        logger.warning("Ignoring inconsistent cache %s", path.name)
        return None

    chunks = [
        DocChunk(document=str(d), section=str(s), text=str(t))
        for d, s, t in zip(documents, sections, texts)
    ]
    return chunks, vectors


def save_cache(path: Path, chunks: Sequence[DocChunk], vectors, meta: dict[str, Any]) -> None:
    """Write atomically, so a killed process never leaves a half cache."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(
            tmp_path,
            vectors=np.asarray(vectors, dtype=np.float32),
            documents=np.array([c.document for c in chunks]),
            sections=np.array([c.section for c in chunks]),
            texts=np.array([c.text for c in chunks]),
            meta=np.array(json.dumps(meta)),
        )
        # mkstemp made the file; savez_compressed appends .npz when the target
        # has no suffix, but ours does, so the path is used verbatim.
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------ the tier


def build_query(ticket: IncomingTicket) -> str:
    """The enriched Tier-2 query.

    Issue_Type is prepended because a standards corpus is organised by issue
    class -- welding, packaging, plating -- while a defect sentence often names
    only the symptom. It costs a little raw similarity (0.8191 -> 0.7800 on the
    measured example, since the header tokens are not defect language) and buys
    discrimination between two standards that describe the same symptom under
    different issue classes.
    """
    issue = ticket.issue_type.strip()
    defect = ticket.problem_description.strip()
    if not issue:
        return f"Defect: {defect}"
    return f"Issue Type: {issue} | Defect: {defect}"


@dataclass
class DocRetriever:
    """Search the 0250 corpus for chunks that address the ticket."""

    docs_dir: Path | str = DEFAULT_DOCS_DIR
    # Caches live beside the documents by default: one folder to copy to a new
    # machine, one folder to clear when something looks wrong.
    cache_dir: Path | str | None = None
    azure_settings: Any = None
    azure_threshold: float = TIER2_AZURE_THRESHOLD
    local_threshold: float = TIER2_LOCAL_THRESHOLD
    confidence_threshold: float | None = None
    top_k: int = TIER2_TOP_K
    # Injection seam for tests, and where a locally-hosted encoder would be
    # supplied if one is wired back in. The pipeline never sets it.
    #
    # `force_backend` used to live here, to stop Tier 2 re-probing Azure after
    # Tier 1 had already fallen back. With no fallback there is nothing to
    # re-probe: Tier 1 either embedded through Azure, in which case Tier 2 will
    # too, or the run ended before Tier 2 was reached.
    embedder: Embedder | None = None
    stats: Tier2Stats = field(default_factory=Tier2Stats)

    @property
    def documents(self) -> list[Path]:
        """The corpus, listed once per instance.

        Memoised because the listing warns about any legacy .doc it finds, and
        `available` plus `retrieve` would otherwise emit that warning twice for
        every ticket.
        """
        cached = getattr(self, "_documents", None)
        if cached is None:
            cached = iter_doc_files(self.docs_dir)
            self._documents = cached
        return cached

    @property
    def available(self) -> bool:
        """False when there is nothing to search.

        A deployment with no 0250 folder is the normal case until the documents
        are loaded, so its absence is not an error -- Tier 2 simply does not run
        and the ticket reports the Tier-1 outcome it would always have reported.
        """
        return bool(self.documents)

    def retrieve(self, ticket: IncomingTicket) -> list[ScoredChunk]:
        """Top-k chunks above the Tier-2 gate, best first."""
        import numpy as np

        paths = self.documents
        self.stats.documents = len(paths)
        if not paths:
            self.stats.cache_state = "absent"
            return []

        query = build_query(ticket)
        started = time.time()
        prepared = self._prepare(paths, query)
        if prepared is None:
            return []
        chunks, matrix, query_vector = prepared
        self.stats.query_seconds = time.time() - started

        threshold = self.confidence_threshold
        if threshold is None:
            threshold = (
                self.azure_threshold
                if self.stats.backend == AZURE_BACKEND
                else self.local_threshold
            )
        self.stats.threshold_used = threshold

        similarities = matrix @ query_vector
        order = np.argsort(-similarities)

        scored = [
            ScoredChunk(chunk=chunks[i], score=float(min(max(similarities[i], 0.0), 1.0)))
            for i in order[: max(self.top_k, 1)]
        ]
        self.stats.top_score = scored[0].score if scored else 0.0
        self.stats.best_chunk = scored[0] if scored else None
        qualified = [s for s in scored if s.score >= threshold]
        self.stats.qualified = len(qualified)

        logger.info(
            "Tier 2: %d chunk(s) from %d document(s) via %s; top %.4f, %d at or above %.2f",
            len(chunks), len(paths), self.stats.backend,
            self.stats.top_score, len(qualified), threshold,
        )
        return qualified

    # -- corpus preparation ------------------------------------------------

    def _prepare(self, paths: list[Path], query: str):
        """Return (chunks, matrix, query_vector), or None if unusable.

        The query is encoded first, deliberately. It is one short text, so it
        is the cheapest possible probe of whether Azure is answering -- and its
        answer decides which cache file is the right one to read.
        """
        import numpy as np

        digest = corpus_hash(paths)
        cache_dir = Path(self.cache_dir) if self.cache_dir else Path(self.docs_dir)

        backend, model, query_vector = self._encode_query(query)

        cached = load_cache(_cache_path(cache_dir, backend), digest, model)
        if cached is not None:
            chunks, matrix = cached
            self.stats.cache_state = "hit"
            self.stats.chunks = len(chunks)
            logger.info("Tier 2 cache hit: %d chunk(s) loaded from disk", len(chunks))
            return chunks, matrix, np.asarray(query_vector, dtype=np.float32)

        chunks = self._parse_all(paths)
        if not chunks:
            self.stats.cache_state = "absent"
            return None

        started = time.time()
        vectors = self._encode_passages([c.embed_text for c in chunks], backend)
        matrix = np.asarray(vectors, dtype=np.float32)
        self.stats.build_seconds = time.time() - started
        self.stats.cache_state = "rebuilt"
        self.stats.chunks = len(chunks)

        try:
            save_cache(
                _cache_path(cache_dir, backend),
                chunks,
                matrix,
                {
                    "format": CACHE_FORMAT,
                    "corpus_hash": digest,
                    "backend": backend,
                    "model": model,
                    "chunks": len(chunks),
                    "documents": [p.name for p in paths],
                    "dimension": int(matrix.shape[1]) if matrix.size else 0,
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            logger.info(
                "Tier 2 cache rebuilt: %d chunk(s) from %d document(s) in %.1fs",
                len(chunks), len(paths), self.stats.build_seconds,
            )
        except OSError as exc:
            # A read-only deployment still works, it just re-embeds every run.
            logger.warning("Could not write the Tier-2 cache: %s", exc)

        return chunks, matrix, np.asarray(query_vector, dtype=np.float32)

    def _parse_all(self, paths: list[Path]) -> list[DocChunk]:
        """Parse every document. One bad file does not lose the corpus."""
        chunks: list[DocChunk] = []
        for path in paths:
            try:
                chunks.extend(parse_docx(path))
            except DocParseError as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
        if not chunks:
            logger.warning(
                "No usable text in %d document(s) under %s", len(paths), self.docs_dir
            )
        return chunks

    # -- encoding ----------------------------------------------------------

    def _encode_query(self, query: str):
        """Returns (backend, model identity, vector).

        Raises AzureEmbeddingError if the encoder is unavailable.
        """
        if self.embedder is not None:
            # ---- LOCAL MODEL DISABLED -------------------------------------
            # The Azure-primary / local-fallback branch used to live here. To
            # restore it: re-add `local_embedder_factory` to this dataclass and
            # fall through to it when the Azure call below raises. The model
            # wrapper is untouched in sps/embedding.py, as are the TIER2_LOCAL_*
            # thresholds above.
            # ---------------------------------------------------------------
            detail = getattr(
                getattr(self.embedder, "settings", None),
                "model_name",
                type(self.embedder).__name__,
            )
            self.stats.backend = LOCAL_BACKEND
            self.stats.backend_detail = detail
            return LOCAL_BACKEND, f"local:{detail}", self.embedder.embed_query(query)

        settings, vectors = self._embed_azure([query])
        self.stats.backend = AZURE_BACKEND
        self.stats.backend_detail = settings.deployment
        return AZURE_BACKEND, f"azure:{settings.deployment}", vectors[0]

    def _encode_passages(self, texts: list[str], backend: str):
        """Encode the corpus with the backend the query already used.

        Never re-decides. A corpus encoded by a different model than the query
        would be ranked by a dot product between two unrelated spaces -- a
        number between -1 and 1 that is not a similarity.
        """
        if backend == AZURE_BACKEND:
            return self._embed_azure(texts)[1]
        # Whatever encoded the query encodes the corpus. Building a second
        # embedder here is how a 128-dim query ends up multiplied against a
        # 384-dim matrix.
        return self.embedder.embed_passages(texts)

    def _embed_azure(self, texts: list[str]):
        """Returns (settings, vectors). Raises rather than falling back."""
        from ..config import AzureEmbeddingSettings
        from ..embedding import AzureEmbedder, AzureEmbeddingNotConfigured

        settings = self.azure_settings or AzureEmbeddingSettings.from_env()
        if not settings.configured:
            raise AzureEmbeddingNotConfigured(
                "Azure embeddings are not configured: " + ", ".join(settings.missing())
            )
        return settings, AzureEmbedder(settings).embed_batch(texts)
