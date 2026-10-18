"""Supplier Problem Sheet (SPS) recommendation pipeline.

    python sps.py resolve --ticket-file T --history-file H --output-dir D
    python sps.py bulk    --tickets SHEET --history H [--output OUT]
    python sps.py eval    --test-dir DIR [--history-file H] [--output-dir D]
    python sps.py ingest  --history H
    python sps.py check   [--probe [PAIRS]] [--out XLSX]

Every prompt and message builder lives in sps_prompts.py, so the model's
wording can change without touching this file.

Sections run in the order a ticket passes through them:
  Validation and part-number canonicalisation
  Data contracts
  Configuration
  Reading .csv and .xlsx
  LLM response schemas
  Embedding
  Tier 2: reading the 0250 .docx standards
  Tier 2: the standards vector cache
  Tier 1: history retrieval
  The Azure OpenAI chat client
  The deterministic gates
  Tier 1: intent scoring
  The Actor/Judge loop
  Building results
  Writing workbooks
  Command: resolve
  Command: bulk
  Command: eval
  Command: ingest
  Command: check
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import logging
import math
import os
import posixpath
import re
import sqlite3
import sys
import tempfile
import time
import traceback
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Protocol, Sequence, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from sps_prompts import (
    build_intent_messages,
    build_tier2_actor_messages,
    build_tier2_judge_messages,
    SOLUTION_NOT_FOUND,
)

logger = logging.getLogger("sps")

# The directory holding this file. Every path the code finds by itself --
# the project .env, the shipped sample pairs -- is relative to it. Computed
# once here because when this was a package each module walked up its own
# number of parents, and merging them into one file left every one of those
# counts silently wrong by a level.
PROJECT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Validation and part-number canonicalisation
# =============================================================================
#
# Input gatekeeping: canonical part numbers, and strict ticket validation.
#
# The retrieval engine selects history by exact part number, or by the exact
# commodity code taken from it, so a part number that differs only by invisible
# characters is not a near miss -- it is a total miss, and the ticket is
# refused as though no history existed. Enterprise
# extracts routinely carry non-breaking spaces from web forms and zero-width
# characters from copy-paste, neither of which is visible to the person who
# pasted them.
#
# Stdlib only, so this module imports without the ML stack.

# Characters that carry no meaning in an identifier but break exact matching.
# Unicode category Cf ("format") covers the zero-width family -- ZWSP U+200B,
# ZWNJ U+200C, ZWJ U+200D, word joiner U+2060, BOM U+FEFF -- and the soft
# hyphen U+00AD, which renders as nothing but is not whitespace.
_INVISIBLE = {"­"}

# Every Unicode space separator, not just ASCII: U+00A0 non-breaking space is
# the usual culprit from a browser form, and str.strip() does remove it, but
# interior ones survive and must go too.
_WHITESPACE = re.compile(r"\s+", re.UNICODE)

# Structural delimiters an engineering part number legitimately contains. These
# are preserved: 0012-43951, 0012/43951, 0012_43951 and 0012.43951 are four
# different identifiers, not four spellings of one.
STRUCTURAL = frozenset("-/_.")

MIN_QUERY_LENGTH = 10


def strip_invisible(value: str) -> str:
    """Remove format characters and soft hyphens, wherever they appear."""
    return "".join(
        ch for ch in value if ch not in _INVISIBLE and unicodedata.category(ch) != "Cf"
    )


# A value made only of printable ASCII with no spaces is already canonical bar
# its case, so the expensive path can be skipped. This matters: the history scan
# canonicalises every row's part number, and per-character unicodedata lookups
# over 300k rows cost more than reading the file.
_ALREADY_CLEAN = re.compile(r"^[\x21-\x7E]+$")


def normalize_part_number(value: object) -> str:
    """Canonical form of a part number.

    Trims surrounding whitespace of every kind, removes invisible and
    zero-width characters, collapses interior whitespace away entirely (a part
    number has no internal spaces), preserves structural delimiters, and
    upper-cases.

    Whole floats are rendered without a decimal: Excel stores every number as a
    double, so a numeric-looking part number can arrive as 1243951.0 and would
    otherwise canonicalise to "1243951.0", matching nothing.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            value = int(value)
    text = str(value)
    # Fast path for the overwhelmingly common case: nothing to strip.
    if _ALREADY_CLEAN.match(text):
        return text.upper()
    text = strip_invisible(text)
    # Interior whitespace is removed rather than collapsed to a single space:
    # "0012 - 43951" and "0012-43951" are the same identifier badly typed.
    text = _WHITESPACE.sub("", text)
    return text.upper()


@dataclass(frozen=True, slots=True)
class ValidationError:
    """Why a ticket cannot be processed, in terms fit for a status file."""

    code: str
    reason: str


def validate_part_number(value: object) -> ValidationError | None:
    """Reject a part number that cannot be used as an exact-match key.

    Malformed means: nothing left after canonicalisation, or nothing but
    structural delimiters. A value of "---" or "  " carries no identity, and
    filtering on it would silently match nothing while looking like a real
    query.
    """
    canonical = normalize_part_number(value)
    if not canonical:
        return ValidationError("INVALID_INPUT", "Part number is missing or blank.")
    if all(ch in STRUCTURAL for ch in canonical):
        return ValidationError(
            "INVALID_INPUT",
            f"Part number {canonical!r} contains no identifying characters.",
        )
    return None


def validate_problem_description(value: object, minimum: int = MIN_QUERY_LENGTH):
    """Reject a problem statement too short to retrieve on."""
    text = _WHITESPACE.sub(" ", strip_invisible(str(value or ""))).strip()
    if not text:
        return ValidationError("INVALID_INPUT", "Problem description is missing or blank.")
    if len(text) < minimum:
        return ValidationError(
            "INVALID_INPUT",
            f"Problem description is {len(text)} characters; at least {minimum} required.",
        )
    return None


def validate_ticket(part_number: object, problem_description: object) -> ValidationError | None:
    """Gate a ticket before any embedding or LLM call is made.

    Part number is checked first: it is the cheaper check and the one that
    determines whether there is any history to search at all.
    """
    return validate_part_number(part_number) or validate_problem_description(
        problem_description
    )


# =============================================================================
# Data contracts
# =============================================================================
#
# Data contracts shared across the resolver.
#
# Pure stdlib apart from `validators`, so the ticket and result shapes stay
# importable without the ML stack.

NO_RECOMMENDATION = "No recommendation available."


def _clean(value: Any) -> str:
    """Coerce any source value to a trimmed string; None/NaN become ''.

    Whole floats render without the fractional part. Excel holds every number as
    a double, so a numeric-looking identifier can arrive as 1001.0 and would
    otherwise become "1001.0": an id that matches nothing and cites nothing.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _lookup(data: dict[str, Any], field_name: str) -> Any:
    """Fetch a field from a ticket payload, ignoring key casing and separators.

    Every other interface names fields the SQL way (`Problem_Description`,
    `Part_Number`), so a caller writing a ticket by hand naturally uses that
    form. Reading only the lower-case spelling yields an empty ticket: the
    description is judged invalid and the part number is dropped, while the run
    still completes and looks like a legitimate refusal.
    """
    if field_name in data:
        return data[field_name]
    wanted = field_name.replace("_", "")
    for key, value in data.items():
        if str(key).strip().casefold().replace("_", "").replace(" ", "") == wanted:
            return value
    return None


@dataclass(frozen=True, slots=True)
class IncomingTicket:
    """A supplier-submitted problem sheet awaiting a recommendation."""

    problem_description: str
    sps_id: str = ""
    part_number: str = ""
    part_description: str = ""
    item_status: str = ""
    problem_reason_code: str = ""
    issue_type: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass: canonicalise in place so every construction path
        # yields the same part-number form the history filter matches on.
        canonical = normalize_part_number(self.part_number)
        if canonical != self.part_number:
            object.__setattr__(self, "part_number", canonical)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IncomingTicket":
        """Build from a ticket payload, accepting either key spelling."""
        return cls(
            problem_description=_clean(_lookup(data, "problem_description")),
            sps_id=_clean(_lookup(data, "sps_id")),
            part_number=_clean(_lookup(data, "part_number")),
            part_description=_clean(_lookup(data, "part_description")),
            item_status=_clean(_lookup(data, "item_status")),
            problem_reason_code=_clean(_lookup(data, "problem_reason_code")),
            issue_type=_clean(_lookup(data, "issue_type")),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A historical record retrieved for the incoming ticket."""

    sps_id: str
    actual_solution: str
    part_number: str
    part_description: str
    item_status: str
    problem_reason_code: str
    issue_type: str
    cosine_similarity: float
    composite_score: float
    applied_boosts: tuple[str, ...] = ()
    # The record's own problem text, which the intent scorer compares against
    # the ticket. Defaulted rather than required because it was added later and
    # nothing that existed before needs it.
    #
    # Deliberately NOT shown to the Actor or the Judge: those are grounded in
    # solutions and a past problem statement is context they could quote from.
    # `tests/test_component_c_actor_critic.py` pins that boundary.
    problem_description: str = ""
    # True when the record is the ticket's own part, False for a sibling part
    # in its commodity family. A tiebreak between equal intent scores, and a
    # line in the justification -- never a boost to any score.
    exact_part: bool = True

    @property
    def confidence_percent(self) -> int:
        return score_to_percent(self.composite_score)


def score_to_percent(score: float) -> int:
    """Format a 0..1 score as an integer percent.

    Truncates rather than rounds: a system that gates at 89% must never report
    "89%" for a candidate it rejected at 0.8899, and understating confidence is
    the safe direction for a compliance-reviewed recommendation.
    """
    return int(min(max(score, 0.0), 1.0) * 100)


# Which evidence a recommendation was built from. Written verbatim into
# output.xlsx, so a human reviewer can weigh a precedent-backed answer
# differently from one derived from a written standard.
SOURCE_HISTORICAL = "HISTORICAL_DATA"
SOURCE_DOCUMENTATION = "0250_DOCUMENTATION"
# The run concluded, and the conclusion was that neither tier had an answer.
# Distinct from a blank cell, which would read as "this field was not filled in".
SOURCE_NONE = "NONE"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The recommendation handed to the output writer."""

    ai_recommendation: str
    justification: str
    confidence: str
    # SPS IDs when the answer came from history, document citations when it
    # came from the standards. One column either way: a reviewer reading
    # output.xlsx wants "what backs this", and Resolution_Source already says
    # which kind of thing they are looking at.
    referenced_sources: list[str] = field(default_factory=list)
    resolution_source: str = SOURCE_HISTORICAL
    # The best precedent we found, verbatim, whether or not we recommended it.
    # DEA asked to see this even on a refusal: a near-miss they can judge for
    # themselves beats a bare "no". It is raw archive text that has passed none
    # of the supplier-facing checks, which is why it carries its own banner and
    # its own column rather than being blended into the recommendation.
    closest_matching_solution: str = ""
    # What a reviewer should read before pasting the recommendation, when the
    # recommendation is archive text sent unchanged. Advisory: it names a risk
    # and blocks nothing, because on this path nothing is allowed to block.
    cascade_warnings: str = ""
    # Operational signals only, never written to output.xlsx. They let the
    # caller separate a dependency outage, which is worth retrying, from a
    # legitimate refusal, which is not.
    infrastructure_failure: bool = False
    diagnostic: str = ""

    @property
    def succeeded(self) -> bool:
        # Both strings, not just the one we write. The sentinel should never
        # reach this field now that NO_RECOMMENDATION exists, but if some path
        # regresses and lets it through, a refusal must not read as a success.
        return self.ai_recommendation not in (NO_RECOMMENDATION, SOLUTION_NOT_FOUND)


# =============================================================================
# Configuration
# =============================================================================
#
# Environment-driven configuration.
#
# The embedding and inference settings, in one auditable place.
#
# The confidence threshold deliberately lives elsewhere -- with the retriever,
# under "Tier 1: history retrieval" -- because it is a property of the
# embedding model's scoring distribution, not of the process configuration.

MAX_ATTEMPTS = 3  # 1 initial draft + 2 refinement retries

# Defaults are module constants rather than class attributes: with
# @dataclass(slots=True) the class attribute is a slot descriptor, not the
# default value, so from_env() cannot read defaults off `cls`.
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
    inside one entry point -- the old `verify_embedder` script was written
    later, did not know to call the resolver's private copy, and spent a
    migration reporting every credential as missing while the resolver read the
    same file perfectly well.

    A process launched by a UiPath robot does not necessarily inherit an
    interactive shell's environment, which is why the file is read at all. Real
    environment variables always win, so a machine-level setting is never
    overridden by a stale checkout.
    """
    candidates = [
        Path(start) / ".env" if start else Path.cwd() / ".env",
        # The project root, so running from anywhere still finds the
        # deployment's file.
        PROJECT_DIR / ".env",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            values = read_env_file(candidate)
        except UnicodeDecodeError as exc:
            # Logged and skipped rather than raised: raised, it stopped the run
            # before status.xlsx existed, with an exit code the robot retries.
            # Skipped, the run reports the missing settings and exits 2 -- a
            # person is needed either way, and now the sheet says so.
            logger.error(
                "%s was not read: it is neither UTF-8 nor UTF-16 text (%s). "
                "Save it as UTF-8.", candidate, exc,
            )
            return None
        for name, value in values.items():
            if value is not None and name not in os.environ:
                os.environ[name] = value
        return candidate
    return None


def env_file_candidates():
    """Where load_env_file would look, for error messages that can be acted on."""
    return [Path.cwd() / ".env", PROJECT_DIR / ".env"]


# ---- Reading the .env file -------------------------------------------------
#
# python-dotenv 1.0.1's rules, which read this file before, reproduced with its
# own patterns: KEY=value lines, an optional `export`, '#' comments (one after a
# bare value needs a space before it), single or double quotes -- which may span
# lines, and in double quotes \n, \t and friends are escapes -- and ${VAR} or
# ${VAR:-default} expansion, where a real environment variable beats a value set
# earlier in the file. A KEY with no '=' sets nothing. A line that cannot be
# parsed is skipped with a warning naming it, and the rest still load.
#
# Three deliberate differences, all where python-dotenv failed:
#
#   * A byte-order mark is not read as part of the first key. Notepad and
#     PowerShell both write one, and python-dotenv kept it, invisibly, at the
#     front of the first variable's name -- leaving the real one unset.
#   * A UTF-16 file -- what `echo KEY=value > .env` writes in Windows
#     PowerShell 5 -- is read, where python-dotenv stopped the run with a
#     UnicodeDecodeError.
#   * A file in any other encoding is skipped with an error naming it (see
#     `load_env_file`), where python-dotenv stopped the run the same way.

_ENV_LEADING_SPACE = re.compile(r"\s*")
_ENV_EXPORT = re.compile(r"(?:export[^\S\r\n]+)?")
_ENV_QUOTED_KEY = re.compile(r"'([^']+)'")
_ENV_KEY = re.compile(r"([^=\#\s]+)")
_ENV_INLINE_SPACE = re.compile(r"[^\S\r\n]*")
_ENV_EQUALS = re.compile(r"(=[^\S\r\n]*)")
_ENV_SINGLE_QUOTED = re.compile(r"'((?:\\'|[^'])*)'")
_ENV_DOUBLE_QUOTED = re.compile(r'"((?:\\"|[^"])*)"')
_ENV_BARE_VALUE = re.compile(r"([^\r\n]*)")
_ENV_COMMENT = re.compile(r"(?:[^\S\r\n]*#[^\r\n]*)?")
_ENV_END_OF_LINE = re.compile(r"[^\S\r\n]*(?:\r\n|\n|\r|$)")
_ENV_REST_OF_LINE = re.compile(r"[^\r\n]*(?:\r|\n|\r\n)?")
_ENV_SINGLE_ESCAPES = re.compile(r"\\[\\']")
_ENV_DOUBLE_ESCAPES = re.compile(r"\\[\\'\"abfnrtv]")
_ENV_VARIABLE = re.compile(r"\$\{(?P<name>[^\}:]*)(?::-(?P<default>[^\}]*))?\}")


class _EnvSyntax(Exception):
    """A statement in a .env file that does not parse."""


def _unescape(pattern: re.Pattern[str], value: str) -> str:
    return pattern.sub(lambda m: codecs.decode(m.group(0), "unicode-escape"), value)


def parse_env(text: str) -> tuple[list[tuple[str, str | None]], list[int]]:
    """(key, value) pairs in file order, and the lines that could not be parsed.

    The value is None for a key with no '='. No expansion here; see
    `read_env_file`.
    """
    pairs: list[tuple[str, str | None]] = []
    bad_lines: list[int] = []
    at = 0

    def take(pattern: re.Pattern[str]) -> tuple[str, ...]:
        nonlocal at
        match = pattern.match(text, at)
        if match is None:
            raise _EnvSyntax
        at = match.end()
        return match.groups()

    while at < len(text):
        take(_ENV_LEADING_SPACE)
        if at >= len(text):
            break
        statement = at
        try:
            take(_ENV_EXPORT)
            if text.startswith("#", at):
                key = None
            elif text.startswith("'", at):
                (key,) = take(_ENV_QUOTED_KEY)
            else:
                (key,) = take(_ENV_KEY)
            take(_ENV_INLINE_SPACE)
            value = None
            if text.startswith("=", at):
                take(_ENV_EQUALS)
                if text.startswith("'", at):
                    value = _unescape(_ENV_SINGLE_ESCAPES, take(_ENV_SINGLE_QUOTED)[0])
                elif text.startswith('"', at):
                    value = _unescape(_ENV_DOUBLE_ESCAPES, take(_ENV_DOUBLE_QUOTED)[0])
                elif at >= len(text) or text[at] in "\r\n":
                    value = ""
                else:
                    (raw,) = take(_ENV_BARE_VALUE)
                    value = re.sub(r"\s+#.*", "", raw).rstrip()
            take(_ENV_COMMENT)
            take(_ENV_END_OF_LINE)
        except _EnvSyntax:
            take(_ENV_REST_OF_LINE)
            bad_lines.append(text.count("\n", 0, statement) + 1)
            continue
        if key is not None:
            pairs.append((key, value))
    return pairs, bad_lines


def read_env_file(path: Path) -> dict[str, str | None]:
    """A .env file's variables, expanded, in file order; later keys win."""
    raw = Path(path).read_bytes()
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    # As text mode would have read it: every line ending becomes "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    pairs, bad_lines = parse_env(text)
    for line in bad_lines:
        logger.warning("%s: line %d could not be parsed and was skipped", path, line)

    values: dict[str, str | None] = {}
    for name, value in pairs:
        if value is not None:
            # The real environment beats an earlier value from the file.
            scope = {**values, **os.environ}
            value = _ENV_VARIABLE.sub(
                lambda m: scope.get(m["name"], m["default"] or "") or "", value
            )
        values[name] = value
    return values


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class SettingError(ValueError):
    """A setting that is present but unusable. A person has to fix it, so it
    exits 2 -- never 1, which a robot would retry until someone noticed."""


def _env_number(key: str, cast, default):
    raw = _env(key)
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        kind = "a whole number" if cast is int else "a number"
        raise SettingError(
            f"{key}={raw!r} is not {kind}. Fix it in .env or the machine's environment."
        ) from None


def _env_int(key: str, default: int) -> int:
    return _env_number(key, int, default)


def _env_float(key: str, default: float | None) -> float | None:
    return _env_number(key, float, default)




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
# Judge it on labelled cases with `python sps.py eval` -- its Status column
# shows which it passed -- before treating 0.75 as settled.
DEFAULT_INTENT_THRESHOLD = 0.75

# How many candidates the intent scorer is shown -- the retriever's shortlist.
# It must be at least MAX_SOLUTIONS for that many to be possible.
DEFAULT_INTENT_TOP_K = 5

# How many solutions one ticket can receive: the best that clear the intent
# gate, up to this many, each sent verbatim on its own row of output.xlsx.
# Fewer clear it, fewer are sent; none, and the refusal is unchanged.
MAX_SOLUTIONS = 3


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
    without the chat path being affected.

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


# =============================================================================
# Reading .csv and .xlsx
# =============================================================================
#
# Format-agnostic row reading for tickets and history.
#
# One job: turn a `.csv` or `.xlsx` path into a stream of rows, header first, so
# that everything downstream works on a list of cells and never needs to know
# which format it came from. Column mapping, canonicalisation and filtering are
# all format-blind as a result.
#
# Both readers stream. CSV is a linear read; XLSX goes through openpyxl in
# `read_only` mode, which iterates rows rather than materialising the sheet.
# Streamed or not, XLSX is far slower to read -- measured at 300k rows on this
# hardware, about 1.5 s for CSV against about 40 s for XLSX -- which is why CSV
# is the format to prefer for a large history.
#
# openpyxl is imported by the XLSX reader itself, so reading a CSV never loads
# it.

CSV_SUFFIXES = frozenset({".csv"})

# .xlsm is the same format as .xlsx as far as openpyxl is concerned -- it is a
# workbook that happens to carry macros -- and business users hand those over
# routinely. Rejecting one would be a support ticket, not a safety measure.
EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm"})

SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES


class UnsupportedFileType(ValueError):
    """The path is not a format this system reads.

    Deliberately distinct from a missing or unreadable file: the wrong
    attachment is a business problem for whoever assembled the ticket, not an
    I/O fault worth retrying, and the caller maps the two to different outcomes.
    """


class FileReadError(ValueError):
    """The file is a supported format but could not be read."""


def validate_file_type(path: Path | str, label: str = "File") -> Path:
    """Reject anything that is not .csv or .xlsx before it is opened.

    Checked on the extension alone and checked first, so a `.pdf` or a `.docx`
    fails immediately rather than after a model load or a history scan.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        readable = ", ".join(sorted(SUPPORTED_SUFFIXES))
        got = suffix or "no extension"
        raise UnsupportedFileType(
            f"{label} {path.name!r} has an unsupported type ({got}). "
            f"Expected one of: {readable}."
        )
    return path


def is_excel(path: Path | str) -> bool:
    return Path(path).suffix.lower() in EXCEL_SUFFIXES


def iter_rows(path: Path | str, label: str = "File") -> Iterator[Sequence[Any]]:
    """Stream rows, header first, without materialising the file.

    Cells come back at their native type -- an Excel numeric cell yields a
    number, a CSV cell yields a string -- and callers coerce. That is the only
    format difference that survives this boundary, and it is handled once, in
    the cell coercion downstream.
    """
    path = validate_file_type(path, label)
    if not path.exists():
        raise FileReadError(f"{label} not found: {path}")

    if is_excel(path):
        yield from _iter_excel(path, label)
    else:
        yield from _iter_csv(path)


def _iter_csv(path: Path) -> Iterator[Sequence[Any]]:
    import csv

    # utf-8-sig, not utf-8: .NET writes UTF-8 with a BOM by default, so a file
    # produced by a UiPath Write Text File activity normally starts with one.
    # Strict utf-8 rejects it outright.
    with open(path, newline="", encoding="utf-8-sig") as handle:
        yield from csv.reader(handle)


def _iter_excel(path: Path, label: str) -> Iterator[Sequence[Any]]:
    from openpyxl import load_workbook

    try:
        # data_only returns computed values rather than formula strings, which
        # matters when a business user built the extract with formulas.
        book = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise FileReadError(f"{label} {path.name!r} could not be opened: {exc}") from exc
    try:
        sheet = book.active
        if sheet is None:
            raise FileReadError(f"{label} {path.name!r} has no active sheet.")
        yield from sheet.iter_rows(values_only=True)
    finally:
        book.close()


def read_header_and_rows(path: Path | str, label: str = "File"):
    """Split the stream into (header, remaining rows).

    Raises FileReadError on an empty file, which is otherwise indistinguishable
    from a file whose every row was filtered out.
    """
    rows = iter_rows(path, label)
    try:
        header = next(rows)
    except StopIteration:
        raise FileReadError(f"{label} {Path(path).name!r} is empty.") from None
    return header, rows


def is_blank(row: Sequence[Any] | None) -> bool:
    """Trailing blank rows are routine in hand-edited workbooks."""
    return row is None or all(cell in (None, "") for cell in row)


# =============================================================================
# LLM response schemas
# =============================================================================
#
# Pydantic schemas.
#
# Two distinct jobs, deliberately kept apart:
#
# * `ActorDraft` / `JudgeVerdict` are the **LLM output schemas**. They are handed
#   to Azure OpenAI as a strict `response_format`, so the model cannot return a
#   shape the pipeline then has to defend against.
#
# Only two of the delivered fields are the LLM's to produce. Confidence is the
# measured cosine and the referenced SPS IDs are the records the retriever
# actually passed to the Actor. If the LLM were asked to emit them it could state
# a confidence it never computed and cite records it was never shown -- precisely
# the fabrication the grounding architecture exists to prevent. So the model is
# constrained to the two fields it genuinely authors, and the resolver supplies
# the rest from measured values.
#
# This module imports pydantic; `sps.contracts` deliberately does not, so the
# scoring, gating and loop-control logic stays importable with no dependencies.

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


class IntentScore(BaseModel):
    """One candidate's intent match, as the scorer judged it."""

    model_config = ConfigDict(extra="forbid")

    sps_id: str = Field(description="The SPS ID of the record being scored, copied exactly.")
    intent_match: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100. How well this record's PROBLEM addresses the same intent as "
            "the incoming ticket -- not how similarly it is worded, and not how "
            "good its solution is."
        ),
    )
    reason: str = Field(
        description=(
            "One short sentence on why this intent does or does not match. "
            "Read by a human deciding whether to trust the score."
        )
    )


class IntentAssessment(BaseModel):
    """The whole Tier-1 decision, in one response.

    The ticket's intent is stated FIRST and the scores second, on purpose. It
    makes each score conditional on a reading of the ticket that is written
    down, so a wrong match can be traced to a wrong reading instead of being an
    unexplained number. It is also why this is one call rather than two.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_intent: str = Field(
        description=(
            "What the supplier is actually asking for, in one sentence -- the "
            "request, not the background narrative. Two tickets can describe "
            "the same defect and ask for different things."
        )
    )
    scores: list[IntentScore] = Field(
        description="One entry per historical record supplied, in any order."
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


# =============================================================================
# Embedding
# =============================================================================
#
# The Azure embedding client, and the Embedder protocol through which tests
# inject a deterministic stand-in.

@runtime_checkable
class Embedder(Protocol):
    """Minimal surface the pipeline needs; lets tests inject a fake."""

    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single incoming problem statement for retrieval."""


class AzureEmbeddingError(RuntimeError):
    """The Azure embedding call failed.

    There is no other encoder to fall back to, so this ends the run. It stays a single base class because most
    callers only need "embedding did not happen".
    """


class AzureEmbeddingNotConfigured(AzureEmbeddingError):
    """Credentials or the deployment name are missing.

    A subclass, so `except AzureEmbeddingError` still catches it, but the
    resolver checks for it first and exits 2 rather than 1. A timeout is worth
    retrying; an absent API key is not, and a robot retrying one fifty times
    only delays the human who has to go and set it.
    """


# Azure's documented ceiling on inputs per embeddings request.
AZURE_EMBEDDING_BATCH_LIMIT = 2048
# Retries per request, made by the SDK with exponential backoff, on a connection
# error, a timeout, a 429 or a 5xx. Stated rather than left to the SDK default
# so a version bump cannot change it unnoticed. Our own loop around it would
# multiply the waits for nothing.
AZURE_EMBEDDING_RETRIES = 2


class AzureEmbedder:
    """Azure OpenAI embeddings, the only encoder.

    Two properties that matter for correctness:

    * **No instruction prefix.** Some open models (bge-*-v1.5, which this
      pipeline once used) are trained to expect one on queries. Azure's are
      not, so prepending one would shift every query vector for no benefit.
    * **Vectors are normalised defensively.** OpenAI returns unit-length
      embeddings today, but the ranking is a bare dot product that silently
      stops being a cosine if that ever changes. Normalising costs nothing at
      this size and makes the guarantee ours rather than the vendor's.
    """

    def __init__(self, settings, client=None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self):
        if self._client is None:
            # Configuration first, import second. Importing openai can fail on
            # its own -- a blocked native dependency, a broken install -- and
            # reporting that when the real problem is an unset API key sends
            # the reader somewhere entirely wrong.
            missing = self.settings.missing()
            if missing:
                raise AzureEmbeddingNotConfigured(
                    f"Azure embedding deployment is not configured: {', '.join(missing)}"
                )

            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.settings.endpoint,
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
                timeout=self.settings.request_timeout,
                max_retries=AZURE_EMBEDDING_RETRIES,
            )
        return self._client

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in the order given, in as few requests as allowed.

        Azure takes at most AZURE_EMBEDDING_BATCH_LIMIT inputs per request, so
        a longer list -- an ingest of a large export -- goes in consecutive
        chunks and comes back as one list in the original order. Each request
        is retried by the SDK on a connection error, timeout, throttle or
        server error, with backoff (`max_retries`).

        Any failure raises AzureEmbeddingError and the whole batch is discarded.
        Partial results are never returned: a half-embedded batch would leave
        some candidates scored and some not, which ranks as though the missing
        ones were simply poor matches.
        """
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), AZURE_EMBEDDING_BATCH_LIMIT):
            chunk = list(texts[start:start + AZURE_EMBEDDING_BATCH_LIMIT])
            vectors.extend(self._embed_request(chunk))
            if len(texts) > AZURE_EMBEDDING_BATCH_LIMIT:
                logger.info("Embedded %d of %d text(s)", len(vectors), len(texts))
        return vectors

    def _embed_request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.settings.deployment,
                input=list(texts),
            )
        except AzureEmbeddingError:
            raise
        except Exception as exc:
            # Network, auth, timeout, rate limit, bad deployment name -- all of
            # them mean the same thing here: this run cannot be embedded.
            raise AzureEmbeddingError(
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # The API documents order preservation, but the response also carries an
        # index per item. Sorting on it makes the guarantee explicit rather than
        # assumed -- a silently reordered batch would pair every candidate with
        # the wrong similarity score.
        items = sorted(response.data, key=lambda d: d.index)
        if len(items) != len(texts):
            raise AzureEmbeddingError(
                f"Expected {len(texts)} embeddings, received {len(items)}"
            )
        return [_unit(item.embedding) for item in items]


def _unit(vector: Sequence[float]) -> list[float]:
    """L2-normalise, so a dot product is a cosine."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


# =============================================================================
# Tier 2: reading the 0250 .docx standards
# =============================================================================
#
# Parse 0250 engineering standards into citable, embeddable chunks.
#
# Two things make a chunk useful to a grounded LLM, and both are done here:
#
# * It is **section-bounded**. A chunk never spans a heading, so the citation
#   attached to it is actually true. A chunk that straddled 4.2 and 4.3 could be
#   cited as either and would be wrong half the time.
# * It **carries its own provenance**. The document name and section trail are
#   prepended to the text itself, so the citation travels with the content into
#   the model's context rather than living in a side-channel the model cannot see.

DOCX_SUFFIX = ".docx"
LEGACY_DOC_SUFFIX = ".doc"

# 250-350 words per the standard's own guidance. Sized originally for the
# since-removed bge-small's 512-token window (382 tokens measured for 350
# words with the citation prefix). Azure's input limit is far larger; the size
# stays because a chunk of about one section keeps each citation pointing at
# what the supplier is actually being sent to.
TARGET_CHUNK_WORDS = 300
MAX_CHUNK_WORDS = 350

# Below this a "chunk" is a heading with nothing under it, or a stray caption.
# Embedding it produces a vector that matches everything weakly and nothing
# well, so it is dropped rather than ranked.
MIN_USABLE_WORDS = 12

# Word's heading styles. style_id is checked as well as name because a
# localized Word install reports a localized style.name ("Titre 1", "Uberschrift
# 1") while the style_id stays English -- and a global supplier base means
# documents do arrive from localized installs.
_HEADING_NAME = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)
_HEADING_ID = re.compile(r"^heading(\d+)$", re.IGNORECASE)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class DocParseError(RuntimeError):
    """A .docx could not be read. Never raised for an unsupported type."""


@dataclass(frozen=True, slots=True)
class DocChunk:
    """One citable passage of a standards document."""

    document: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        """What the LLM is required to reproduce when it uses this chunk."""
        if self.section:
            return f"{self.document} § {self.section}"
        return self.document

    @property
    def embed_text(self) -> str:
        """Text as embedded and as shown to the model.

        The citation is part of the embedded string rather than metadata beside
        it. That costs a little similarity -- the header tokens are not defect
        language -- and buys the guarantee that a model looking at a chunk can
        always see where it came from.
        """
        return f"[{self.citation}] \n{self.text}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _heading_level(name: str, style_id: str) -> int | None:
    """Word heading level for a paragraph style, or None for body text."""
    for value, pattern in ((name, _HEADING_NAME), (style_id, _HEADING_ID)):
        match = pattern.match(str(value or "").strip())
        if match:
            return int(match.group(1))
    # "Title" heads the document but is not numbered.
    if str(style_id or "").strip().casefold() == "title":
        return 0
    return None


def _clean_doc_text(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


# ---- Reading the .docx itself ---------------------------------------------
#
# A .docx is a zip of XML, and this reads it with the standard library. It used
# to go through python-docx, whose lxml dependency ships a native DLL that
# Windows Application Control blocks on the deployment machines.
#
# It began as python-docx 1.1.2's rules reproduced exactly, proven document by
# document against python-docx before that was dropped. It now reads what Word
# itself shows, which python-docx did not:
#
#   * Text inside a content control (w:sdt) -- including whole sections, which
#     documents built from a template are often wrapped in -- inside a smart
#     tag, custom XML markup or a simple field, and at every depth.
#   * A tracked insertion not yet accepted, and not a tracked deletion: the
#     document as Word's default view shows it. A document carrying either is
#     named in a warning, because its text is not settled.
#   * A table nested in a cell, flattened into that cell's text.
#
# Text boxes and headers and footers are still not read.
#
# A change to these rules changes chunk text, so DOC_READER_VERSION is part of
# the Tier-2 cache's key: bump it, and every cache rebuilds.
#
# Where python-docx raised instead of reading:
#
#   * A vertically merged cell with no cell above it to continue. python-docx
#     raised mid-document, past the guard that skips one unreadable file, and
#     took every document down with it. Here it is an empty cell.
#   * A merge running down more rows than Python's recursion limit.
#   * A macro-enabled document or a template saved with a .docx name, which
#     was refused for its content type. Its text is read like any other.
#   * An attribute value outside its schema -- a w:default of "yes", a
#     w:pStyle with no value -- is read leniently rather than refused.
#   * Whatever does still fail, a non-numeric w:gridSpan say, costs only its
#     own document: the walk runs inside the guard in `parse_docx`.

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REL_OFFICE_DOCUMENT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
_REL_STYLES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"


def _break_text(element) -> str:
    # A text-wrapping break is a newline; page and column breaks are layout.
    return "\n" if element.get(_W + "type", "textWrapping") == "textWrapping" else ""


# What each run child contributes. Anything not listed -- deleted text, field
# codes, drawings -- contributes nothing.
_RUN_TEXT = {
    _W + "t": lambda element: element.text or "",
    _W + "tab": lambda element: "\t",
    _W + "ptab": lambda element: "\t",
    _W + "cr": lambda element: "\n",
    _W + "br": _break_text,
    _W + "noBreakHyphen": lambda element: "-",
}


def _run_text(run) -> str:
    return "".join(_RUN_TEXT[child.tag](child) for child in run if child.tag in _RUN_TEXT)


# Containers whose runs are part of the paragraph as Word shows it. Anything
# else is skipped, which is what keeps out deleted text (w:del), the old
# position of moved text (w:moveFrom) and a content control's properties.
_TRANSPARENT = {_W + tag for tag in (
    "hyperlink", "smartTag", "customXml", "fldSimple", "ins", "moveTo", "dir", "bdo",
    "sdt", "sdtContent",
)}


def _runs(element):
    """Every run in document order, through containers Word shows."""
    for child in element:
        if child.tag == _W + "r":
            yield child
        elif child.tag in _TRANSPARENT:
            yield from _runs(child)


def _paragraph_text(paragraph) -> str:
    return "".join(_run_text(run) for run in _runs(paragraph))


def _blocks(element):
    """Paragraphs and tables in document order, through content controls and
    custom XML, at any depth. Nothing else in a body or a cell holds text."""
    for child in element:
        if child.tag in (_W + "p", _W + "tbl"):
            yield child
        elif child.tag in (_W + "sdt", _W + "sdtContent", _W + "customXml"):
            yield from _blocks(child)


def _paragraph_style_id(paragraph) -> str | None:
    style = paragraph.find(f"{_W}pPr/{_W}pStyle")
    return None if style is None else style.get(_W + "val")


class _Styles:
    """Paragraph-style lookup, resolved the way python-docx resolves it.

    A paragraph with no style id, or one naming a missing style or a style of
    another type, takes the default paragraph style; with several defaults the
    last one wins, as the spec says. A style with no w:type counts as no type
    at all, so it neither matches nor serves as the default.
    """

    def __init__(self, root) -> None:
        self.by_id: dict[str, tuple[str | None, str]] = {}
        self.default: tuple[str, str] | None = None
        if root is None:
            # No styles part. python-docx substituted a template that defines
            # Normal and nothing else, so no paragraph in such a file is a
            # heading, whatever its pStyle says.
            self.default = ("Normal", "Normal")
            return
        for style in root.findall(_W + "style"):
            style_type = style.get(_W + "type")
            style_id = style.get(_W + "styleId", "")
            name_element = style.find(_W + "name")
            name = "" if name_element is None else name_element.get(_W + "val", "")
            self.by_id.setdefault(style_id, (style_type, name))
            if style_type == "paragraph" and style.get(_W + "default") in ("1", "true", "on"):
                self.default = (name, style_id)

    def resolve(self, style_id: str | None) -> tuple[str, str] | None:
        """(name, style_id) of the paragraph's effective style, or None."""
        if style_id:
            found = self.by_id.get(style_id)
            if found is not None and found[0] == "paragraph":
                return found[1], style_id
        return self.default


def _grid_span(cell) -> int:
    span = cell.find(f"{_W}tcPr/{_W}gridSpan")
    return 1 if span is None else int(span.get(_W + "val", "1"))


def _vertical_merge(cell) -> str | None:
    merge = cell.find(f"{_W}tcPr/{_W}vMerge")
    # Present without a value means "continue", per the schema default.
    return None if merge is None else merge.get(_W + "val", "continue")


def _grid_before(row) -> int:
    before = row.find(f"{_W}trPr/{_W}gridBefore")
    return 0 if before is None else int(before.get(_W + "val", "0"))


def _cell_text(cell) -> str:
    """The cell's paragraphs, and any table nested in it, one row per line."""
    lines = []
    for block in _blocks(cell):
        if block.tag == _W + "p":
            lines.append(_paragraph_text(block))
        else:
            lines.append(_table_text(_table_rows(block)))
    return "\n".join(lines)


def _row_cells(row):
    """A row's cells, including those a content control wraps."""
    for child in row:
        if child.tag == _W + "tc":
            yield child
        elif child.tag in (_W + "sdt", _W + "sdtContent", _W + "customXml"):
            yield from _row_cells(child)


def _table_rows_of(table):
    for child in table:
        if child.tag == _W + "tr":
            yield child
        elif child.tag in (_W + "sdt", _W + "sdtContent", _W + "customXml"):
            yield from _table_rows_of(child)


def _table_rows(table) -> list[list[str]]:
    """Each row's cell texts, one entry per grid column the row covers.

    A horizontally merged cell repeats once per column it spans; a vertically
    merged continuation reads as whatever the cell starting at the same grid
    column in the row above read as. Both repeats collapse again in
    `_table_text`.

    Row by row rather than by recursing upward, as python-docx did, so a merge
    down a long table is not limited by Python's recursion depth.
    """
    rows: list[list[str]] = []
    above: dict[int, list[str]] = {}  # grid column -> what the cell starting there read as
    for row in _table_rows_of(table):
        texts_here: list[str] = []
        starts: dict[int, list[str]] = {}
        at = _grid_before(row)
        for cell in _row_cells(row):
            span = _grid_span(cell)
            if _vertical_merge(cell) == "continue":
                texts = above.get(at, [""] * span)
            else:
                texts = [_cell_text(cell)] * span
            starts.setdefault(at, texts)  # the first cell to start at a column is the one found
            texts_here.extend(texts)
            at += span
        rows.append(texts_here)
        above = starts
    return rows


def _iter_blocks(body, styles: _Styles) -> Iterator[tuple[str, object]]:
    """Yield ("paragraph", (text, style)) / ("table", rows) in document order.

    `document.paragraphs` in python-docx silently omitted every table, and in a
    standards document the tables are where the numbers live -- porosity
    limits, torque values, acceptance criteria. Walking the body in order is
    the only way to get both, in the order a reader would meet them.
    """
    for child in _blocks(body):
        if child.tag == _W + "p":
            yield "paragraph", (_paragraph_text(child), styles.resolve(_paragraph_style_id(child)))
        else:
            yield "table", _table_rows(child)


def _table_text(rows: list[list[str]]) -> str:
    """Flatten a table to one line per row.

    Row-wise with a separator keeps a limits table readable as
    "Porosity | 2 percent max", which is the shape the answer is actually in.
    """
    lines = []
    for cells in rows:
        # Merged cells repeat their text across the span; collapse the repeats.
        deduped: list[str] = []
        for cell in (_clean_doc_text(c) for c in cells):
            if cell and (not deduped or deduped[-1] != cell):
                deduped.append(cell)
        if deduped:
            lines.append(" | ".join(deduped))
    return "\n".join(lines)


def _relationship_target(package, rels_name: str, base: str, rel_type: str) -> str | None:
    if rels_name not in package.namelist():
        return None
    for rel in ET.fromstring(package.read(rels_name)):
        if rel.get("Type") == rel_type and rel.get("TargetMode") != "External":
            target = rel.get("Target", "")
            if target.startswith("/"):
                return target.lstrip("/")
            return posixpath.normpath(posixpath.join(base, target))
    return None


def _read_docx(path: Path):
    """The document body and the styles root, found through the package's own
    relationships rather than assumed paths."""
    with zipfile.ZipFile(path) as package:
        main = _relationship_target(package, "_rels/.rels", "", _REL_OFFICE_DOCUMENT)
        main = main or "word/document.xml"
        document = ET.fromstring(package.read(main))
        base = posixpath.dirname(main)
        rels = posixpath.join(base, "_rels", posixpath.basename(main) + ".rels")
        styles_name = _relationship_target(package, rels, base, _REL_STYLES)
        styles = None
        if styles_name and styles_name in package.namelist():
            styles = ET.fromstring(package.read(styles_name))

    body = document.find(_W + "body")
    if body is None:
        raise ValueError("the document has no body")
    if next(body.iter(_W + "ins"), None) is not None or next(body.iter(_W + "del"), None) is not None:
        logger.warning(
            "%s has tracked changes not yet accepted; read as Word shows them, "
            "insertions in and deletions out. Accept or reject them to settle the text.",
            path.name,
        )
    return body, styles


def _split_long(text: str, limit: int = MAX_CHUNK_WORDS) -> list[str]:
    """Break an over-long passage on sentence boundaries.

    A single paragraph can exceed the cap on its own. Splitting mid-sentence
    would hand the model a fragment ending in "shall not exceed", so the break
    goes at a sentence end and only falls back to a hard word split for prose
    that has no sentence ends at all.
    """
    words = text.split()
    if len(words) <= limit:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_END.split(text):
        length = len(sentence.split())
        if count + length > limit and current:
            parts.append(" ".join(current))
            current, count = [], 0
        if length > limit:
            # One sentence longer than the cap: split it on words.
            chunk_words = sentence.split()
            for start in range(0, len(chunk_words), limit):
                parts.append(" ".join(chunk_words[start:start + limit]))
            continue
        current.append(sentence)
        count += length
    if current:
        parts.append(" ".join(current))
    return [p for p in parts if p.strip()]


class _Accumulator:
    """Collects body text under one heading and flushes it as chunks."""

    def __init__(self, document_name: str) -> None:
        self.document_name = document_name
        self.section = ""
        self.buffer: list[str] = []
        self.words = 0
        self.chunks: list[DocChunk] = []

    def set_section(self, trail: Sequence[str]) -> None:
        """A heading always ends the previous chunk: a chunk that spanned two
        sections could not be cited truthfully."""
        self.flush()
        # The deepest heading only. The levels above it are almost always the
        # document's own title restated -- "0250 Weld Standards > 4.2 Weld Seam
        # Cracking" -- which the filename already carries, so the trail would
        # spend context tokens repeating what the citation says two words
        # earlier.
        headings = [t for t in trail if t]
        self.section = headings[-1] if headings else ""

    def add(self, text: str) -> None:
        text = _clean_doc_text(text)
        if not text:
            return
        for part in _split_long(text):
            length = len(part.split())
            if self.words and self.words + length > MAX_CHUNK_WORDS:
                self.flush()
            self.buffer.append(part)
            self.words += length
            if self.words >= TARGET_CHUNK_WORDS:
                self.flush()

    def add_table(self, lines: Sequence[str]) -> None:
        """A table, one row per line, never split inside a row.

        Each row is added as its own part, so it keeps its own line in the
        chunk -- run together, "2 percent max" and the next row's "Porosity"
        read as one claim. A table too long for one chunk continues in the
        next with its first row repeated, so the columns still have names.
        """
        rows = [line for line in (_clean_doc_text(x) for x in lines) if line]
        for position, line in enumerate(rows):
            if position:
                if self.words and self.words + len(line.split()) > MAX_CHUNK_WORDS:
                    self.flush()
                if not self.buffer:
                    self.add(rows[0])
            self.add(line)

    def flush(self) -> None:
        if not self.buffer:
            return
        text = "\n".join(self.buffer).strip()
        self.buffer, self.words = [], 0
        if len(text.split()) < MIN_USABLE_WORDS:
            return
        self.chunks.append(
            DocChunk(document=self.document_name, section=self.section, text=text)
        )


def parse_docx(path: Path | str) -> list[DocChunk]:
    """Parse one .docx into section-bounded chunks."""
    path = Path(path)
    try:
        body, styles_root = _read_docx(path)
        # Walked inside the guard, not after it: malformed markup surfaces
        # while walking, and an error escaping from here would make Tier 2
        # unavailable for the whole run rather than lose this one document.
        blocks = list(_iter_blocks(body, _Styles(styles_root)))
    except Exception as exc:
        # A corrupt file, a password-protected one, a .doc renamed to .docx, a
        # table whose markup does not add up -- each fails differently, and the
        # caller only needs "this one is unusable".
        raise DocParseError(f"Could not read {path.name!r}: {exc}") from exc

    accumulator = _Accumulator(path.name)
    trail: list[str] = []

    for kind, block in blocks:
        if kind == "table":
            accumulator.add_table(_table_text(block).split("\n"))
            continue

        text, style = block
        level = _heading_level(*style) if style else None
        text = _clean_doc_text(text)
        if level is None:
            accumulator.add(text)
            continue
        if not text:
            continue
        # Maintain the heading trail: a level-2 heading replaces any deeper
        # heading but keeps its level-1 parent.
        del trail[level:]
        while len(trail) < level:
            trail.append("")
        trail.append(text)
        accumulator.set_section(trail)

    accumulator.flush()
    logger.info("Parsed %s into %d chunk(s)", path.name, len(accumulator.chunks))
    return accumulator.chunks


def iter_doc_files(directory: Path | str) -> list[Path]:
    """The .docx files in a directory, sorted, with legacy .doc refused.

    A .doc is not a zip container, so the reader here cannot open it. The only
    way to read one on Windows is to drive Word through COM automation, which
    on a headless robot blocks on a modal dialog and hangs the run rather than
    failing it -- so the file is named in a warning and skipped.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    legacy = sorted(
        p.name for p in directory.iterdir()
        if p.is_file() and p.suffix.casefold() == LEGACY_DOC_SUFFIX
    )
    if legacy:
        logger.warning(
            "Ignoring %d legacy .doc file(s); convert them to .docx to include them: %s",
            len(legacy),
            ", ".join(legacy),
        )

    return sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.casefold() == DOCX_SUFFIX
        # Word writes ~$name.docx lock files beside open documents. They are
        # not readable as documents and their presence is an accident of
        # someone having the file open.
        and not p.name.startswith("~$")
    )


# =============================================================================
# Tier 2: the standards vector cache
# =============================================================================
#
# Tier 2 -- the 0250 standards corpus: cache, then search.
#
# Tier 1's history lives in its own store (see "The history store"). Tier 2
# searches the *same* standards corpus on every ticket, so embedding it per
# ticket would pay the same cost forever. It is embedded once and cached on
# disk, keyed by a hash of the documents themselves:
# edit a document and the next run rebuilds, touch nothing and the next run loads
# vectors in milliseconds.
#
# Two caches, never one. A threshold and a vector both belong to a single
# embedding space, so `0250_cache_azure.npz` and `0250_cache_injected.npz` are
# separate files and a run reads whichever matches the encoder that actually
# answered. Mixing them would compute cosines between unrelated spaces and return
# numbers that look like similarities.
#
# Only the azure cache is written in production. An encoder passed through the
# `embedder=` seam writes `0250_cache_injected.npz` instead, so vectors from a
# stand-in can never be read back as though they shared Azure's space.

# Where the standards live. Relative to the working directory, so the UiPath
# Performer's project folder is the root, same as every other path the robot
# passes.
DEFAULT_DOCS_DIR = Path("data/0250_docs")

CACHE_FILENAMES = {
    "azure": "0250_cache_azure.npz",
    "injected": "0250_cache_injected.npz",
}


# How many chunks reach the model. The spec's 3-5: enough that a defect
# described in one section and dispositioned in the next is still covered,
# few enough that the Actor cannot quietly blend four unrelated standards.
TIER2_TOP_K = 5


# Tier 2 has its own threshold, not Tier 1's, and the gap is not a detail.
#
# The comparison differs in kind. Tier 1 matches one short defect sentence
# against another; Tier 2 matches a defect sentence against 300 words of
# standards prose that answers it without resembling it, so its scores run
# structurally lower. Measured with the since-removed bge-small encoder: a
# chunk that genuinely answered the ticket scored 0.78, an adjacent section
# 0.67 and an unrelated standard 0.52, against a Tier-1 gate of 0.89 that
# would have rejected the right chunk on every ticket.
#
# PROVISIONAL, and doubly so: never measured on Azure, and derived from Tier
# 1's former Azure gate of 0.50 scaled by that bge-small Tier-1:Tier-2 ratio
# (0.62/0.89). Run the batch evaluator against the real deployment before
# trusting it.
TIER2_AZURE_THRESHOLD = 0.35

CACHE_FORMAT = 2
# The .docx reading rules the cached chunks were made with. Bump it whenever
# `parse_docx` or what it calls would produce different chunk text: a cache from
# older rules is then rebuilt rather than served.
DOC_READER_VERSION = 2


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
    threshold_used: float = 0.0
    top_score: float = 0.0
    qualified: int = 0
    build_seconds: float = 0.0
    query_seconds: float = 0.0
    # The best-scoring chunks, gate or no gate. Tier 1's `top_candidates`
    # twin, and kept out of `as_dict()` for the same reason.
    top_chunks: tuple["ScoredChunk", ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "cache_state": self.cache_state,
            "backend": self.backend,
            "backend_detail": self.backend_detail,
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
            if meta.get("reader") != DOC_READER_VERSION:
                logger.info("%s was read under older .docx rules; rebuilding", path.name)
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
    confidence_threshold: float | None = None
    top_k: int = TIER2_TOP_K
    # Injection seam for tests. The pipeline never sets it.
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
            # One embedding space in production, so one default gate. An
            # injected encoder has no measured threshold of its own; a caller
            # that needs a different one passes confidence_threshold.
            threshold = self.azure_threshold
        self.stats.threshold_used = threshold

        similarities = matrix @ query_vector
        order = np.argsort(-similarities)

        scored = [
            ScoredChunk(chunk=chunks[i], score=float(min(max(similarities[i], 0.0), 1.0)))
            for i in order[: max(self.top_k, 1)]
        ]
        self.stats.top_score = scored[0].score if scored else 0.0
        self.stats.top_chunks = tuple(scored[:CASCADE_LIMIT])
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
                    "reader": DOC_READER_VERSION,
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
            # An explicitly supplied encoder: the seam tests use. The pipeline
            # never sets it.
            detail = getattr(
                getattr(self.embedder, "settings", None),
                "model_name",
                type(self.embedder).__name__,
            )
            self.stats.backend = INJECTED_BACKEND
            self.stats.backend_detail = detail
            return INJECTED_BACKEND, f"injected:{detail}", self.embedder.embed_query(query)

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

        settings = self.azure_settings or AzureEmbeddingSettings.from_env()
        if not settings.configured:
            raise AzureEmbeddingNotConfigured(
                "Azure embeddings are not configured: " + ", ".join(settings.missing())
            )
        return settings, AzureEmbedder(settings).embed_batch(texts)


# =============================================================================
# Tier 1: history retrieval
# =============================================================================
#
# A ticket is matched against the history of its part's COMMODITY FAMILY:
# every part whose commodity code -- the part number up to its first hyphen --
# shares a group with the ticket's, per the commodity groups file. With no
# groups file, an unmapped code, or a part number with no hyphen, it falls back
# to the part's own history, exactly as before families existed, and the Reason
# says which and why.
#
# A family can hold 30,000 records or more, so history lives in a local store
# (below) rather than being re-read and re-embedded per ticket. The store
# mirrors the history file: an unchanged file is not read at all, a changed one
# is read once, and a record's problem text is embedded once and kept. Ranking
# is still exact -- every record in scope is scored, in NumPy; nothing is
# approximated, so the true best match is never missed.
#
#     validate -> sync store -> fetch scope -> embed what is new -> rank -> gate -> shortlist

# Columns the engine needs, and the spellings it accepts for each.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sps_id": ("SPS_ID", "SPSID", "Id"),
    "part_number": ("Part_Number", "Part Number", "PartNo"),
    "problem_description": ("Problem_Description", "Problem Description", "Problem"),
    "actual_solution": ("Solution_Text", "Actual_Solution", "Solution"),
    "issue_type": ("Issue_Type", "Issue Type"),
    "problem_reason_code": ("Problem_Reason_Code", "Reason_Code"),
    "part_description": ("Part_Description",),
    "item_status": ("Item_Status",),
    "last_modified_date": ("Last_Modified_Date", "Last Modified Date", "Modified"),
}
REQUIRED = ("sps_id", "part_number", "problem_description", "actual_solution")

# How many records the reviewer is shown when there is no recommendation.
# Three is a judgement, not a measurement: enough that a near-miss is not
# hidden behind a marginally better one, few enough that the cell stays
# readable in Excel.
CASCADE_LIMIT = 3

# A threshold is a property of one embedding space and does not survive a change
# of model -- swap the Azure deployment and this number must be re-measured.
#
# Azure: a RECALL filter, not a decision.
#
# This number used to decide whether a ticket got an answer. It no longer does
# -- the intent scorer decides -- so it was lowered from 0.50, and the reason
# matters more than the value. Two gates in series, a cosine at 0.50 and an
# intent score at 0.75, means the cosine quietly does half the deciding: a
# record whose intent matches perfectly but whose wording scores 0.49 is
# discarded before the scorer ever sees it, and nothing reports that it
# happened. Wording similarity is exactly the judgement this design moved away
# from making.
#
# So it is set to admit generously and let the scorer reject. Too low costs a
# slightly longer prompt; too high costs answers, silently.
AZURE_EMBEDDING_THRESHOLD = 0.30

AZURE_BACKEND = "azure"
# Anything passed through the `embedder=` seam. Its own label, and its own
# Tier-2 cache file, so vectors from a stand-in encoder are never mistaken
# for Azure's.
INJECTED_BACKEND = "injected"


class HistoryError(ValueError):
    """The history file cannot be used.

    Raised for a structural problem -- a missing column, an empty sheet. An
    unsupported *file type* raises UnsupportedFileType instead, because the
    caller reports that differently.
    """


@dataclass(frozen=True, slots=True)
class HistoryRow:
    sps_id: str
    part_number: str
    problem_description: str
    actual_solution: str
    issue_type: str = ""
    problem_reason_code: str = ""
    part_description: str = ""
    item_status: str = ""
    last_modified_date: datetime | None = None
    # Derived from part_number when the history is read, so the store can
    # select a whole family by an index rather than by parsing every row.
    commodity_code: str = ""

    def payload(self) -> dict[str, str]:
        return {
            "sps_id": self.sps_id,
            "actual_solution": self.actual_solution,
            "part_number": self.part_number,
            "part_description": self.part_description,
            "item_status": self.item_status,
            "problem_reason_code": self.problem_reason_code,
            "issue_type": self.issue_type,
        }


@dataclass
class RetrievalStats:
    """What the run actually did, for the status file and for latency work."""

    rows_scanned: int = 0
    part_matches: int = 0
    usable: int = 0
    # Every usable record in scope is ranked: there is no cap. The cap that was
    # here dropped by recency to bound encoding cost, which the store removed,
    # and at family scale it would silently have discarded the oldest record
    # even when it was the one that matched.
    considered: int = 0
    load_seconds: float = 0.0
    embed_seconds: float = 0.0
    # Problem texts embedded during this run because the store had not seen
    # them. Zero once `ingest` has run against the current export.
    embedded_now: int = 0
    qualified: int = 0
    top_score: float = 0.0
    # Which encoder produced the vectors.
    backend: str = ""
    backend_detail: str = ""
    threshold_used: float = 0.0
    # Which history was searched -- a commodity family or the part alone -- in
    # one sentence for the Reason, and where the store lives.
    scope_note: str = ""
    family: str = ""
    store: str = ""
    # The best-scoring candidates, whether or not they cleared the gate.
    #
    # `retrieve()` returns only what qualified, so on a gated run the near-miss
    # text was discarded and a reviewer asking "what was the closest thing you
    # found?" had no answer anywhere. Several rather than one: a reviewer who
    # now has to answer the ticket by hand is reading these as options, and the
    # second-best is not obviously the wrong one to reach for. Kept off
    # `as_dict()` on purpose: that feeds the log line and the status sheet, and
    # neither wants 500 characters of solution text.
    top_candidates: tuple["Candidate", ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_scanned": self.rows_scanned,
            "part_matches": self.part_matches,
            "usable": self.usable,
            "considered": self.considered,
            "load_seconds": round(self.load_seconds, 3),
            "embed_seconds": round(self.embed_seconds, 3),
            "embedded_now": self.embedded_now,
            "qualified": self.qualified,
            "top_score": round(self.top_score, 4),
            "backend": self.backend,
            "backend_detail": self.backend_detail,
            "threshold_used": self.threshold_used,
            "family": self.family,
            "store": self.store,
        }


def _norm_header(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")


def _map_columns(
    headers: Sequence[Any],
    aliases: dict[str, tuple[str, ...]],
    required: Sequence[str],
    what: str,
) -> dict[str, int]:
    """Resolve column positions, tolerating spelling and case."""
    lookup: dict[str, int] = {}
    for index, header in enumerate(headers):
        key = _norm_header(header)
        if key and key not in lookup:
            lookup[key] = index

    mapping: dict[str, int] = {}
    for field_name, spellings in aliases.items():
        for alias in spellings:
            index = lookup.get(_norm_header(alias))
            if index is not None:
                mapping[field_name] = index
                break

    missing = [f for f in required if f not in mapping]
    if missing:
        wanted = ", ".join(f"{f} (e.g. {aliases[f][0]!r})" for f in missing)
        found = ", ".join(repr(str(h)) for h in headers if str(h).strip())
        raise HistoryError(f"{what} is missing required column(s): {wanted}. Found: {found}")
    return mapping


def build_header_map(headers: Sequence[Any]) -> dict[str, int]:
    """Resolve the history file's column positions."""
    return _map_columns(headers, COLUMN_ALIASES, REQUIRED, "History file")


def _cell(row: Sequence[Any], mapping: dict[str, int], field_name: str) -> Any:
    index = mapping.get(field_name)
    if index is None or index >= len(row):
        return ""
    return row[index]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _parse_date(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---- Commodity families ------------------------------------------------------
#
# The groups file is two columns, one row per commodity code:
#
#     Commodity_Code,Group
#     0011,FASTENERS
#     0022,FASTENERS
#
# Groups are symmetric -- every code in a group reaches every other -- and a
# group name is matched ignoring case, so "Fasteners" and "FASTENERS" are one
# family rather than two. A code listed under two groups is reported and
# treated as unmapped: guessing which family it belongs to would decide which
# history a ticket is answered from, and that is not ours to guess.

COMMODITY_GROUPS_ENV = "SPS_COMMODITY_GROUPS"
COMMODITY_ALIASES: dict[str, tuple[str, ...]] = {
    "commodity_code": ("Commodity_Code", "Commodity Code", "Commodity"),
    "group": ("Group", "Commodity_Group", "Family"),
}


def commodity_code(part_number: object) -> str:
    """The part number up to its first hyphen, or '' if it has none.

    `0022-43951` -> `0022`. Taken from the canonical form, so it comes out the
    same whatever spacing or case the part arrived in. The part number itself
    is still compared whole wherever it is compared: the code is derived from
    it, not substituted for it.
    """
    head, hyphen, _ = normalize_part_number(part_number).partition("-")
    return head if hyphen else ""


class CommodityGroupsError(ValueError):
    """The commodity groups file cannot be used."""


@dataclass(frozen=True)
class CommodityGroups:
    """Which commodity codes are searched together."""

    source: str
    # group key (the name casefolded) -> its codes, conflicted ones excluded
    members: dict[str, tuple[str, ...]]
    names: dict[str, str]
    code_group: dict[str, str]
    # code -> every group name it appears under, for codes listed in more than one
    conflicts: dict[str, tuple[str, ...]]

    def family(self, code: str) -> tuple[str, tuple[str, ...]] | None:
        """(group name, every code in the group), or None if the code is unmapped."""
        key = self.code_group.get(code)
        if key is None:
            return None
        return self.names[key], self.members[key]


_GROUPS_CACHE: dict[tuple[str, int, int], CommodityGroups] = {}


def load_commodity_groups(path: Path | str) -> CommodityGroups:
    """Read a groups file, .csv or .xlsx. Cached for as long as the file is unchanged."""
    path = Path(path)
    if not path.is_file():
        raise CommodityGroupsError(f"{path} not found")
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _GROUPS_CACHE:
        return _GROUPS_CACHE[key]

    mapping: dict[str, int] | None = None
    names: dict[str, str] = {}
    codes_of: dict[str, set[str]] = {}
    groups_of: dict[str, set[str]] = {}
    numeric: list[int] = []
    try:
        for number, row in enumerate(iter_rows(path, "Commodity groups file"), 1):
            if mapping is None:
                mapping = _map_columns(row, COMMODITY_ALIASES, ("commodity_code", "group"),
                                       "Commodity groups file")
                continue
            if is_blank(row):
                continue
            raw = _cell(row, mapping, "commodity_code")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                numeric.append(number)
            code = normalize_part_number(_text(raw))
            name = _text(_cell(row, mapping, "group"))
            if not code or not name:
                continue
            group = name.casefold()
            names.setdefault(group, name)
            codes_of.setdefault(group, set()).add(code)
            groups_of.setdefault(code, set()).add(group)
    except (FileReadError, HistoryError) as exc:
        raise CommodityGroupsError(str(exc)) from exc
    if mapping is None:
        raise CommodityGroupsError(f"{path.name} is empty")

    if numeric:
        # Excel keeps a number, not the text typed: 0011 entered into a General
        # cell is stored as 11, and nothing here can know it ever had zeros.
        logger.warning(
            "%s: the commodity code on row(s) %s is stored as a number, so any "
            "leading zeros are already gone. Format the column as Text, or use CSV.",
            path.name, ", ".join(map(str, numeric[:10])),
        )
    conflicts = {
        code: tuple(sorted(names[g] for g in groups))
        for code, groups in groups_of.items() if len(groups) > 1
    }
    for code, listed in sorted(conflicts.items()):
        logger.error(
            "%s: commodity code %s is listed under %s; it is treated as unmapped "
            "until the file gives it one group.", path.name, code, " and ".join(listed),
        )
    code_group = {code: next(iter(gs)) for code, gs in groups_of.items() if len(gs) == 1}
    members = {
        group: tuple(sorted(c for c in codes if c in code_group))
        for group, codes in codes_of.items()
    }
    groups = CommodityGroups(path.name, members, names, code_group, conflicts)
    _GROUPS_CACHE[key] = groups
    return groups


@dataclass(frozen=True)
class MatchScope:
    """Which history a ticket is matched against, and why, in one sentence."""

    part_number: str
    # Empty means the part's own history only.
    codes: tuple[str, ...] = ()
    group: str = ""
    note: str = ""

    @property
    def is_family(self) -> bool:
        return bool(self.codes)


def match_scope(part_number: str, groups_file: str | Path | None) -> MatchScope:
    """The family to search, or the part alone with the reason it is alone."""
    part_number = normalize_part_number(part_number)

    def alone(why: str) -> MatchScope:
        return MatchScope(part_number, note=f"Matched on the part number only: {why}.")

    if not groups_file:
        return alone(f"no commodity groups file is configured ({COMMODITY_GROUPS_ENV})")
    code = commodity_code(part_number)
    if not code:
        return alone(f"{part_number} has no hyphen, so no commodity code")
    try:
        groups = load_commodity_groups(groups_file)
    except CommodityGroupsError as exc:
        logger.error("Commodity groups unusable: %s", exc)
        return alone(f"the commodity groups file could not be used ({exc})")
    if code in groups.conflicts:
        listed = " and ".join(groups.conflicts[code])
        return alone(f"commodity code {code} is listed under {listed} in {groups.source}")
    family = groups.family(code)
    if family is None:
        return alone(f"commodity code {code} is not in {groups.source}")
    name, codes = family
    return MatchScope(
        part_number, codes=codes, group=name,
        note=f"Matched across commodity family {name} ({', '.join(codes)}).",
    )


# ---- Reading the history file ------------------------------------------------


def read_history(path: Path | str) -> list[HistoryRow]:
    """Every non-blank row of the history file, canonicalised, in file order.

    Nothing is filtered here except blank rows: whether a record is usable is
    decided when it is fetched (see `is_usable`), so the store holds the file as
    it is and the Reason can say how many records a part has before saying how
    many could be used.
    """
    path = Path(path)
    # The reader's own failures are re-raised as HistoryError so callers have
    # one exception to catch for "the history is unusable". UnsupportedFileType
    # is deliberately not caught: the caller reports a wrong file type
    # differently, and the resolver rejects it before reaching here.
    try:
        rows = _scan(path)
    except FileReadError as exc:
        raise HistoryError(str(exc)) from exc
    if rows is None:
        raise HistoryError(f"History file is empty: {path}")
    return rows


def _scan(path: Path) -> list[HistoryRow] | None:
    """The rows, or None if not even a header was found."""
    mapping: dict[str, int] | None = None
    kept: list[HistoryRow] = []
    for row in iter_rows(path, "History file"):
        if mapping is None:
            mapping = build_header_map(list(row))
            continue
        if is_blank(row):
            continue
        # Canonicalised the same way the ticket's part number is, so drift in
        # the file -- a stray space, a zero-width character -- cannot hide a
        # genuine match.
        part = normalize_part_number(_cell(row, mapping, "part_number"))
        kept.append(
            HistoryRow(
                sps_id=_text(_cell(row, mapping, "sps_id")),
                part_number=part,
                problem_description=_text(_cell(row, mapping, "problem_description")),
                actual_solution=_text(_cell(row, mapping, "actual_solution")),
                issue_type=_text(_cell(row, mapping, "issue_type")),
                problem_reason_code=_text(_cell(row, mapping, "problem_reason_code")),
                part_description=_text(_cell(row, mapping, "part_description")),
                item_status=_text(_cell(row, mapping, "item_status")),
                last_modified_date=_parse_date(_cell(row, mapping, "last_modified_date")),
                commodity_code=commodity_code(part),
            )
        )
    return kept if mapping is not None else None


def is_usable(row: HistoryRow, min_text_length: int = 15) -> bool:
    """A record with an id and enough problem and solution text to be worth ranking."""
    return bool(row.sps_id) and (
        len(row.problem_description) >= min_text_length
        and len(row.actual_solution) >= min_text_length
    )


# ---- The history store -------------------------------------------------------
#
# SQLite, from the standard library: no server, no extra package, nothing native
# beyond what Python ships. One database per history file, holding the file's
# rows and one vector per distinct problem text, keyed by the text's hash and
# the embedding deployment that produced it -- so an edited solution costs no
# embedding, a repeated problem text is embedded once, and vectors from two
# deployments can never be compared.
#
#   * An unchanged file -- same path, size and modification time -- is not read
#     at all. A changed one is read once and its rows replaced wholesale, so a
#     record deleted from the export leaves the store with it.
#   * A record is embedded the first time a ticket needs it, or ahead of time
#     by `python sps.py ingest`. Either way, once.
#   * Vectors are float32, which is what the API returns: storing anything
#     narrower would change scores, and at 30k records the top few are often
#     within a ten-thousandth of each other.
#   * WAL mode, so robots reading the store never wait for one writing it.
#     LOCAL DISK ONLY: SQLite's locking is unreliable over a network share, so a
#     store that would land on one is put under %LOCALAPPDATA% instead.
#
# `HistoryStore` is the seam: a Qdrant server, say, would implement the same two
# methods. Ranking stays in the retriever, identical whichever store answers.

HISTORY_STORE_ENV = "SPS_HISTORY_STORE"
HISTORY_STORE_DIR_ENV = "SPS_HISTORY_STORE_DIR"
# Bump when what a stored row holds, or how it is derived, changes: an existing
# store is then rebuilt rather than read under rules it was not written with.
HISTORY_STORE_SCHEMA = 1
STORE_SUFFIX = ".sps-store.sqlite"
# A writer can hold the store while it embeds a family's new records; a second
# robot waits this long for it before reporting a transient failure.
STORE_BUSY_SECONDS = 600


class HistoryStoreError(RuntimeError):
    """The history store could not be read or written."""


class HistoryStoreConfigError(HistoryStoreError):
    """The store is configured wrongly: a person has to change a setting."""


@dataclass(frozen=True)
class SyncResult:
    rows: int
    reread: bool
    seconds: float


@dataclass
class Fetched:
    """One scope's usable records, their vectors, and the query's."""

    rows: list[HistoryRow]
    matrix: Any  # (len(rows), dims) float32
    query: Any   # (dims,) float32, or None when nothing needed encoding
    matched: int
    embedded: int
    embed_seconds: float


@dataclass(frozen=True)
class Encoder:
    """One embedding space: what to call it, and how to enter it.

    `identity` keys the stored vectors. `both` encodes a query and passages
    together -- one request to Azure, where it was one request before the
    store existed -- and `passages` encodes passages alone, for ingest.
    """

    identity: str
    backend: str
    detail: str
    both: Callable[[str, list[str]], tuple[Sequence[float], list[Sequence[float]]]]
    passages: Callable[[list[str]], list[Sequence[float]]]


def azure_encoder(settings: "AzureEmbeddingSettings") -> Encoder:
    def both(query: str, passages: list[str]):
        if not settings.configured:
            raise AzureEmbeddingNotConfigured(
                "Azure embeddings are not configured: " + ", ".join(settings.missing())
            )
        vectors = AzureEmbedder(settings).embed_batch([query] + passages)
        return vectors[0], vectors[1:]

    def passages_only(passages: list[str]):
        if not settings.configured:
            raise AzureEmbeddingNotConfigured(
                "Azure embeddings are not configured: " + ", ".join(settings.missing())
            )
        return AzureEmbedder(settings).embed_batch(passages)

    return Encoder(f"azure:{settings.deployment}", AZURE_BACKEND, settings.deployment,
                   both, passages_only)


def injected_encoder(embedder: "Embedder") -> Encoder:
    """The test seam: the Embedder protocol, query and passages encoded apart."""
    name = type(embedder).__name__

    def both(query: str, passages: list[str]):
        vectors = embedder.embed_passages(passages) if passages else []
        return embedder.embed_query(query), vectors

    return Encoder(f"injected:{name}", INJECTED_BACKEND, name, both,
                   lambda passages: embedder.embed_passages(passages))


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@runtime_checkable
class HistoryStore(Protocol):
    location: str

    def sync(self, history_path: Path) -> SyncResult: ...

    def fetch(self, scope: MatchScope, encoder: Encoder, query: str,
              min_text_length: int = 15) -> Fetched: ...


class _VectorStoreBase:
    """What both stores share: turning a scope's rows into a ranked-ready matrix."""

    location = ""

    def _rows_and_vectors(self, scope: MatchScope, model: str) -> list[tuple[HistoryRow, bytes | None]]:
        """The scope's rows in file order, each with its stored vector or None."""
        raise NotImplementedError

    def all_rows(self) -> list[HistoryRow]:
        raise NotImplementedError

    def close(self) -> None:
        """Release the store. A no-op unless it holds a connection."""

    def _known(self, model: str, keys: set[str]) -> dict[str, bytes]:
        raise NotImplementedError

    def _remember(self, model: str, items: list[tuple[str, bytes]]) -> None:
        raise NotImplementedError

    def fetch(self, scope: MatchScope, encoder: Encoder, query: str,
              min_text_length: int = 15) -> Fetched:
        import numpy as np

        in_scope = self._rows_and_vectors(scope, encoder.identity)
        usable = [(r, blob) for r, blob in in_scope if is_usable(r, min_text_length)]
        if not usable:
            return Fetched([], np.zeros((0, 0), np.float32), None, len(in_scope), 0, 0.0)

        rows = [r for r, _ in usable]
        keys = [text_key(r.problem_description) for r in rows]
        known: dict[str, bytes] = {}
        missing: dict[str, str] = {}
        for key, (row, blob) in zip(keys, usable):
            if blob is not None:
                known[key] = blob
            elif key not in known:
                missing.setdefault(key, row.problem_description)
        for key in [k for k in missing if k in known]:
            del missing[key]

        started = time.time()
        query_vector, new = encoder.both(query, list(missing.values()))
        seconds = time.time() - started
        query_vector = np.asarray(query_vector, dtype=np.float32)
        dims = query_vector.shape[0]

        fresh = [(key, np.asarray(v, dtype=np.float32).tobytes()) for key, v in zip(missing, new)]
        if fresh:
            self._remember(encoder.identity, fresh)
            known.update(fresh)

        matrix = np.empty((len(rows), dims), dtype=np.float32)
        for i, key in enumerate(keys):
            vector = np.frombuffer(known[key], dtype=np.float32)
            if vector.shape[0] != dims:
                raise HistoryStoreError(
                    f"A stored vector has {vector.shape[0]} dimensions and the query "
                    f"{dims}: the deployment {encoder.detail!r} has changed shape. "
                    f"Delete {self.location} and run ingest again."
                )
            matrix[i] = vector
        return Fetched(rows, matrix, query_vector, len(in_scope), len(fresh), seconds)

    def missing_texts(self, encoder: Encoder,
                      min_text_length: int = 15) -> tuple[dict[str, str], int]:
        """(every usable problem text not yet embedded in this space, how many
        distinct usable texts there are in all), for ingest."""
        rows = [r for r in self.all_rows() if is_usable(r, min_text_length)]
        texts = {text_key(r.problem_description): r.problem_description for r in rows}
        known = self._known(encoder.identity, set(texts))
        return {k: t for k, t in texts.items() if k not in known}, len(texts)

    def remember(self, encoder: Encoder, keys: list[str], vectors) -> None:
        import numpy as np

        self._remember(encoder.identity, [
            (k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in zip(keys, vectors)
        ])


class MemoryHistoryStore(_VectorStoreBase):
    """No persistence: the file is read and records are embedded every run.

    What `SPS_HISTORY_STORE=memory` selects, and what the resolver falls back to
    when the SQLite store cannot be created -- a read-only folder, say. It
    answers exactly as the SQLite store does, only without remembering.
    """

    def __init__(self, location: str = "memory") -> None:
        self.location = location
        self.rows: list[HistoryRow] = []
        self.vectors: dict[tuple[str, str], bytes] = {}

    def sync(self, history_path: Path) -> SyncResult:
        started = time.time()
        self.rows = read_history(history_path)
        return SyncResult(len(self.rows), True, time.time() - started)

    def all_rows(self) -> list[HistoryRow]:
        return list(self.rows)

    def _rows_in(self, scope: MatchScope) -> list[HistoryRow]:
        if scope.is_family:
            codes = set(scope.codes)
            return [r for r in self.rows if r.commodity_code in codes]
        return [r for r in self.rows if r.part_number == scope.part_number]

    def _rows_and_vectors(self, scope: MatchScope, model: str):
        return [(r, self.vectors.get((model, text_key(r.problem_description))))
                for r in self._rows_in(scope)]

    def _known(self, model: str, keys: set[str]) -> dict[str, bytes]:
        return {k: self.vectors[(model, k)] for k in keys if (model, k) in self.vectors}

    def _remember(self, model: str, items: list[tuple[str, bytes]]) -> None:
        for key, blob in items:
            self.vectors[(model, key)] = blob


_ROW_FIELDS = (
    "sps_id", "part_number", "commodity_code", "problem_description", "actual_solution",
    "issue_type", "problem_reason_code", "part_description", "item_status",
)


class SqliteHistoryStore(_VectorStoreBase):
    """The persistent store. See the section comment for what it keeps and why."""

    def __init__(self, location: Path) -> None:
        self.location = str(location)
        Path(location).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.location, timeout=STORE_BUSY_SECONDS,
                                   isolation_level=None)
        # 64 KB pages, set before anything is written: a 3,072-dimension vector
        # is 12 KB, and at the default 4 KB every one spans a chain of
        # overflow pages. Measured at 30k vectors, reads were several times
        # faster this way.
        self._db.execute("PRAGMA page_size=65536")
        if self._db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._prepare()

    def close(self) -> None:
        self._db.close()

    _OBJECTS = {"meta", "rows", "vectors", "rows_by_code", "rows_by_part", "vectors_by_key"}

    def _ready(self) -> bool:
        """Every table and index present, under this schema. Reads only."""
        names = {name for (name,) in self._db.execute("SELECT name FROM sqlite_master")}
        return self._OBJECTS <= names and self._meta("schema") == str(HISTORY_STORE_SCHEMA)

    def _prepare(self) -> None:
        # A prepared store is opened with reads alone. Even a no-op
        # CREATE ... IF NOT EXISTS takes the write lock, and every ticket would
        # then queue behind any robot syncing an export or storing vectors --
        # measured: a ticket that should have taken milliseconds waited out a
        # six-second write elsewhere.
        if self._ready():
            return
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        try:
            if self._ready():  # another process prepared it while this one waited
                db.execute("COMMIT")
                return
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            found = self._meta("schema")
            if found not in (None, str(HISTORY_STORE_SCHEMA)):
                # Written under other rules: rebuilt rather than misread.
                logger.warning("History store %s is schema %s; rebuilding as %s",
                               self.location, found, HISTORY_STORE_SCHEMA)
                db.execute("DROP TABLE IF EXISTS rows")
                db.execute("DROP TABLE IF EXISTS vectors")
                db.execute("DELETE FROM meta")
            db.execute(
                "CREATE TABLE IF NOT EXISTS rows (ordinal INTEGER PRIMARY KEY, "
                + ", ".join(f"{f} TEXT NOT NULL" for f in _ROW_FIELDS)
                + ", last_modified TEXT, text_key TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS rows_by_code ON rows (commodity_code)")
            db.execute("CREATE INDEX IF NOT EXISTS rows_by_part ON rows (part_number)")
            # A rowid table with a separate index, not a WITHOUT ROWID one keyed
            # on the hash: keyed that way, vectors are stored in hash order --
            # which is random -- and reading a family touched pages all over
            # the file. Stored in the order they were written, they are read
            # nearly in sequence.
            db.execute(
                "CREATE TABLE IF NOT EXISTS vectors (model TEXT NOT NULL, "
                "text_key TEXT NOT NULL, vec BLOB NOT NULL)"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS vectors_by_key ON vectors (model, text_key)"
            )
            db.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
                       (str(HISTORY_STORE_SCHEMA),))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def _meta(self, key: str) -> str | None:
        try:
            found = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        except sqlite3.OperationalError:  # no meta table: a new, empty file
            return None
        return found[0] if found else None

    @staticmethod
    def _signature(history_path: Path) -> str:
        stat = history_path.stat()
        return json.dumps([str(history_path.resolve()), stat.st_size, stat.st_mtime_ns])

    def rows_count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM rows").fetchone()[0]

    def sync(self, history_path: Path) -> SyncResult:
        started = time.time()
        history_path = Path(history_path)
        if not history_path.is_file():
            raise HistoryError(f"History file not found: {history_path}")
        signature = self._signature(history_path)
        if self._meta("source") == signature:
            return SyncResult(self.rows_count(), False, time.time() - started)

        # Read outside the write lock: a large .xlsx takes long enough that
        # holding the store for it would stall every other robot.
        rows = read_history(history_path)
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        try:
            if self._meta("source") == signature:
                # Another process synced the same export while this one read it.
                db.execute("COMMIT")
                return SyncResult(self.rows_count(), False, time.time() - started)
            db.execute("DELETE FROM rows")
            db.executemany(
                "INSERT INTO rows VALUES (?" + ", ?" * (len(_ROW_FIELDS) + 2) + ")",
                (
                    (i, *(getattr(r, f) for f in _ROW_FIELDS),
                     r.last_modified_date.isoformat() if r.last_modified_date else None,
                     text_key(r.problem_description))
                    for i, r in enumerate(rows)
                ),
            )
            # Vectors whose text no record carries any more.
            db.execute("DELETE FROM vectors WHERE text_key NOT IN (SELECT text_key FROM rows)")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('source', ?)", (signature,))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        logger.info("History store %s synced: %d row(s) in %.2fs",
                    self.location, len(rows), time.time() - started)
        return SyncResult(len(rows), True, time.time() - started)

    _COLUMNS = ", ".join(_ROW_FIELDS) + ", last_modified"

    def all_rows(self) -> list[HistoryRow]:
        return self._materialise(
            self._db.execute(f"SELECT {self._COLUMNS} FROM rows ORDER BY ordinal")
        )

    def _scope_sql(self, scope: MatchScope) -> tuple[str, tuple]:
        if scope.is_family:
            return f"r.commodity_code IN ({', '.join('?' * len(scope.codes))})", tuple(scope.codes)
        return "r.part_number = ?", (scope.part_number,)

    def _rows_and_vectors(self, scope: MatchScope, model: str):
        where, params = self._scope_sql(scope)
        columns = ", ".join(f"r.{c}" for c in (*_ROW_FIELDS, "last_modified"))
        # Ordered here rather than in SQL: ORDER BY made SQLite sort every
        # result row, 12 KB vector and all, where this sorts references.
        found = self._db.execute(
            f"SELECT r.ordinal, {columns}, v.vec FROM rows r LEFT JOIN vectors v "
            f"ON v.model = ? AND v.text_key = r.text_key WHERE {where}",
            (model, *params),
        ).fetchall()
        found.sort(key=lambda r: r[0])
        rows = self._materialise(r[1:-1] for r in found)
        return [(row, r[-1]) for row, r in zip(rows, found)]

    @staticmethod
    def _materialise(cursor) -> list[HistoryRow]:
        return [
            HistoryRow(
                sps_id=r[0], part_number=r[1], commodity_code=r[2],
                problem_description=r[3], actual_solution=r[4], issue_type=r[5],
                problem_reason_code=r[6], part_description=r[7], item_status=r[8],
                last_modified_date=_parse_date(r[9]),
            )
            for r in cursor
        ]

    def _known(self, model: str, keys: set[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        ordered = list(keys)
        # Batched below SQLite's bound-parameter limit.
        for start in range(0, len(ordered), 900):
            batch = ordered[start:start + 900]
            marks = ", ".join("?" * len(batch))
            for key, blob in self._db.execute(
                f"SELECT text_key, vec FROM vectors WHERE model = ? AND text_key IN ({marks})",
                (model, *batch),
            ):
                found[key] = blob
        return found

    def _remember(self, model: str, items: list[tuple[str, bytes]]) -> None:
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        try:
            db.executemany(
                "INSERT OR REPLACE INTO vectors (model, text_key, vec) VALUES (?, ?, ?)",
                ((model, key, blob) for key, blob in items),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def drop_other_models(self, keep: str) -> int:
        """Vectors from any other deployment, removed by ingest."""
        cursor = self._db.execute("DELETE FROM vectors WHERE model != ?", (keep,))
        return cursor.rowcount


def _is_network_path(path: Path) -> bool:
    """A UNC path, or a drive letter Windows reports as remote."""
    text = str(path)
    if text.startswith(("\\\\", "//")):
        return True
    if sys.platform != "win32" or not path.drive:
        return False
    try:
        import ctypes

        return ctypes.windll.kernel32.GetDriveTypeW(path.drive + "\\") == 4  # DRIVE_REMOTE
    except Exception:  # pragma: no cover - platform dependent
        return False


def _local_store_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "SPS" / "history-store"


def history_store_path(history_path: Path | str) -> Path:
    """Where the store for this history file lives.

    Beside the file unless SPS_HISTORY_STORE_DIR says otherwise -- so `ingest`
    and every robot reading the same export share one store -- and never on a
    network share.
    """
    history_path = Path(history_path).resolve()
    folder = Path(os.environ.get(HISTORY_STORE_DIR_ENV, "").strip() or history_path.parent)
    if _is_network_path(folder.resolve() if folder.exists() else folder):
        local = _local_store_dir()
        logger.warning(
            "%s is on a network share, where SQLite's locking cannot be trusted; "
            "keeping the history store in %s instead. Set %s to a local folder "
            "to choose where.", folder, local, HISTORY_STORE_DIR_ENV,
        )
        folder = local
    tag = hashlib.sha1(str(history_path).casefold().encode("utf-8")).hexdigest()[:8]
    return folder / f"{history_path.name}.{tag}{STORE_SUFFIX}"


def open_history_store(history_path: Path | str) -> HistoryStore:
    """The store the environment selects, falling back to memory if SQLite cannot be used."""
    kind = os.environ.get(HISTORY_STORE_ENV, "").strip().casefold() or "sqlite"
    if kind == "memory":
        return MemoryHistoryStore()
    if kind != "sqlite":
        raise HistoryStoreConfigError(
            f"{HISTORY_STORE_ENV} must be 'sqlite' or 'memory', not {kind!r}"
        )
    location = history_store_path(history_path)
    try:
        return SqliteHistoryStore(location)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    # A read-only folder, a locked file, a full disk. The local folder next,
    # because the memory store below embeds a whole family on every ticket --
    # tolerable for a part, ruinous for 30,000 records.
    local = _local_store_dir() / location.name
    if local != location:
        try:
            store = SqliteHistoryStore(local)
        except Exception:
            pass
        else:
            logger.warning(
                "History store %s cannot be used (%s); using %s instead. Set %s to a "
                "writable local folder so the robots and ingest share one store.",
                location, failure, local, HISTORY_STORE_DIR_ENV,
            )
            return store
    # Still answerable -- just without remembering anything.
    logger.warning(
        "History store %s cannot be used (%s); reading the history without one this "
        "run, which embeds the ticket's whole family each time. Set %s to a writable "
        "local folder.", location, failure, HISTORY_STORE_DIR_ENV,
    )
    return MemoryHistoryStore(location=f"memory (could not open {location})")


# ---- Retrieval ---------------------------------------------------------------


@dataclass
class HistoryRetriever:
    """Rank a ticket's family -- or its part alone -- and shortlist for the scorer.

    Azure embeddings are the only encoder, and an Azure failure ends the run
    rather than falling back to anything. A silent fallback would change which
    threshold applied and which embedding space the scores lived in, inside a
    run that still reported success.
    """

    history_path: Path | str
    azure_settings: Any = None
    azure_threshold: float = AZURE_EMBEDDING_THRESHOLD
    # An explicit override wins over both. Used by --threshold, which is a
    # deliberate operator instruction and should not be second-guessed by the
    # backend that happened to answer.
    confidence_threshold: float | None = None
    # How many candidates leave the retriever: the intent scorer's shortlist,
    # set from SPS_INTENT_TOP_K. The scorer reads every one of them in a single
    # prompt and has to tell them apart, so a shorter list is a sharper
    # judgement -- and it must hold at least as many as the solutions wanted.
    max_context_records: int = 5
    min_text_length: int = 15
    # The groups file; None matches on the part number alone.
    commodity_file: str | Path | None = None
    stats: RetrievalStats = field(default_factory=RetrievalStats)
    # Injection seams for tests. The pipeline sets neither.
    embedder: Embedder | None = None
    store: HistoryStore | None = None

    def encoder(self) -> Encoder:
        if self.embedder is not None:
            return injected_encoder(self.embedder)
        settings = self.azure_settings or AzureEmbeddingSettings.from_env()
        return azure_encoder(settings)

    def retrieve(self, ticket: IncomingTicket) -> list[Candidate]:
        """Return qualified candidates, best first. Empty if none qualify.

        Raises HistoryError if the file is unusable. The caller separates
        NO_MATCHES (nothing in scope) from BELOW_CONFIDENCE_THRESHOLD (records
        found, none similar enough) by reading `stats`.
        """
        if self.store is not None:
            return self._retrieve(ticket, self.store)
        store = open_history_store(self.history_path)
        try:
            return self._retrieve(ticket, store)
        finally:
            store.close()

    def _retrieve(self, ticket: IncomingTicket, store) -> list[Candidate]:
        started = time.time()
        part = normalize_part_number(ticket.part_number)
        scope = match_scope(part, self.commodity_file)
        self.stats.scope_note = scope.note
        self.stats.family = scope.group

        self.stats.store = store.location
        synced = store.sync(Path(self.history_path))
        self.stats.rows_scanned = synced.rows

        encoder = self.encoder()
        fetched = store.fetch(scope, encoder, ticket.problem_description.strip(),
                              self.min_text_length)
        self.stats.part_matches = fetched.matched
        self.stats.usable = len(fetched.rows)
        self.stats.considered = len(fetched.rows)
        self.stats.embedded_now = fetched.embedded
        self.stats.embed_seconds = fetched.embed_seconds
        self.stats.load_seconds = time.time() - started - fetched.embed_seconds
        logger.info(
            "%s %d of %d stored row(s) in scope, %d usable; %d newly embedded",
            scope.note, fetched.matched, synced.rows, len(fetched.rows), fetched.embedded,
        )
        if not fetched.rows:
            return []
        self.stats.backend = encoder.backend
        self.stats.backend_detail = encoder.detail

        threshold = self.confidence_threshold
        if threshold is None:
            # One embedding space in production, so one default gate. An
            # injected encoder has no measured threshold of its own; a caller
            # that needs a different one passes confidence_threshold.
            threshold = self.azure_threshold
        self.stats.threshold_used = threshold

        # Both sides are L2-normalised, so the dot product is the cosine and the
        # whole ranking is one matrix multiply.
        import numpy as np

        scores = np.clip(fetched.matrix @ fetched.query, 0.0, 1.0)
        exact = [r.part_number == part for r in fetched.rows]
        # Best first; at an equal score the ticket's own part ahead of a
        # sibling's, then by SPS ID so the order never depends on the file's.
        order = sorted(
            range(len(fetched.rows)),
            key=lambda i: (float(scores[i]), exact[i], fetched.rows[i].sps_id),
            reverse=True,
        )
        # One entry per SPS ID. A family can carry the same record under two
        # part numbers, and the scorer's answers are matched back by SPS ID --
        # two entries with one id would make that match ambiguous.
        seen: set[str] = set()
        ranked: list[int] = []
        for i in order:
            if fetched.rows[i].sps_id not in seen:
                seen.add(fetched.rows[i].sps_id)
                ranked.append(i)

        def candidate(i: int) -> Candidate:
            return _to_candidate(fetched.rows[i], float(scores[i]), exact[i])

        self.stats.top_score = float(scores[ranked[0]])
        # Captured before the gate, for the same reason top_score is: the rows
        # we rejected are exactly the ones a reviewer wants to see.
        self.stats.top_candidates = tuple(candidate(i) for i in ranked[:CASCADE_LIMIT])
        qualified = [i for i in ranked if scores[i] >= threshold]
        self.stats.qualified = len(qualified)

        logger.info(
            "Ranked %d candidate(s) via %s; top %.4f, %d at or above %.2f",
            len(ranked), self.stats.backend, self.stats.top_score, len(qualified), threshold,
        )
        return [candidate(i) for i in qualified[: max(self.max_context_records, 1)]]


def _to_candidate(row: HistoryRow, similarity: float, exact_part: bool = True) -> Candidate:
    """Score is the cosine alone.

    No metadata boosting. Whether the record is the ticket's own part or a
    sibling in its family is carried alongside -- it breaks a tie between equal
    intent scores, and is never added to a score.
    """
    bounded = min(max(similarity, 0.0), 1.0)
    return Candidate(
        sps_id=row.sps_id,
        actual_solution=row.actual_solution,
        problem_description=row.problem_description,
        part_number=row.part_number,
        part_description=row.part_description,
        item_status=row.item_status,
        problem_reason_code=row.problem_reason_code,
        issue_type=row.issue_type,
        cosine_similarity=bounded,
        composite_score=bounded,
        exact_part=exact_part,
    )

# =============================================================================
# The Azure OpenAI chat client
# =============================================================================
#
# Azure OpenAI chat client.
#
# Deterministic by construction: temperature=0.0 and top_p=0.1 are applied on
# every call so the Actor and the Judge are reproducible for audit.
#
# Output shape is enforced server-side. Each call passes a pydantic model as a
# strict `response_format`, so the deployment is constrained to the schema rather
# than merely asked for it. Deployments or API versions without structured-output
# support fall back to JSON mode plus client-side pydantic validation, which
# yields the same guarantee one round-trip later.

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


# =============================================================================
# The deterministic gates
# =============================================================================
#
# Catch facts that are true of the historical record but false about the ticket.
#
# The Judge checks *provenance* -- did this text come from the source? -- and it
# is very good at that. What it cannot reliably catch is the class of statement
# that passes provenance and is still wrong:
#
#     "ESW#20033465 is submitted for these issues."
#
# That sentence is in the historical solution verbatim, so it is not a
# hallucination. It is also false about the ticket now being answered: the work
# request was raised for a different issue, months ago, and the supplier reading
# it can chase a number that has nothing to do with them. The same applies to
# "see the attachment" when no attachment exists, and "per discussed" when no
# discussion has taken place.
#
# This module is the deterministic half of the defence. It runs before the Judge
# because it is local and free while the Judge is a paid network call, and because
# a regex does not talk itself round: an LLM asked "is this in the source?" and
# then "but should it still be here?" tends to answer the first question twice.
#
# Stdlib only, like `sps.validators`, so it stays testable in milliseconds.

# The abstention token the Actor must emit. Imported rather than spelled out so
# the critique cannot drift from what the prompts ask for. `contracts` is itself
# stdlib-only, so this keeps the module's no-dependency property.

# Internal tracking systems whose identifiers must never reach a supplier. Site
# specific, so extend this rather than the patterns below. Matched with or
# without a separator, because the house style varies: ESW#20033465, CAR-1234,
# "NCR 5566".
KNOWN_TRACKING_PREFIXES = (
    "ESW",   # engineering support work request
    "CAR",   # corrective action request
    "SCAR",  # supplier corrective action request
    "NCR",   # non-conformance report
    "MRB",   # material review board
    "8D",
)

_PREFIX_ALTERNATION = "|".join(re.escape(p) for p in KNOWN_TRACKING_PREFIXES)

# (category, pattern). The category names the reason in the critique handed back
# to the Actor, so it can be told what kind of thing to remove rather than only
# which characters.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "internal tracking identifier",
        re.compile(rf"\b(?:{_PREFIX_ALTERNATION})\s*[#:\-]?\s*\d{{3,}}\b", re.IGNORECASE),
    ),
    (
        # Any LETTER-led token followed by # and digits. Deliberately requires
        # the letters: a part number is digit-led (0012-43951) and must never
        # trip this.
        "tracking identifier",
        re.compile(r"\b[A-Za-z]{2,6}\s*#\s*\d{3,}\b"),
    ),
    (
        # "attach" the verb is fine -- "attach photos to your response" asks the
        # supplier to do something they can do. "attachment" and "attached" name
        # a document that travelled with the original record and is not here.
        "reference to an attachment",
        re.compile(r"\b(?:attachments?|attached|enclosed|enclosure)\b", re.IGNORECASE),
    ),
    (
        "reference to something not included",
        re.compile(r"\bsee\s+(?:the\s+)?(?:below|above|enclosed)\b", re.IGNORECASE),
    ),
    (
        "reference to a prior conversation",
        re.compile(
            r"\b(?:as|per)\s+(?:discussed|agreed|advised|requested)\b|"
            r"\bper\s+(?:our|the)\s+(?:call|discussion|conversation|meeting|email|phone)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lot, batch or purchase-order number",
        re.compile(
            r"\b(?:lot|batch|purchase\s+order|p\.?o\.?)\s*(?:#|no\.?|number)?\s*\d{3,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        # A calendar date in a recommendation is nearly always the historical
        # record's date. Durations ("within 30 days") are untouched.
        "a specific date",
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ),
)


# Instruments only the customer can issue. A supplier may well REQUEST one --
# the live tickets show exactly that -- but cannot grant one, so a recommendation
# that tells them to is instructing them to do the customer's job.
#
# Site-specific. Extend this rather than the pattern below.
INTERNAL_INSTRUMENTS = (
    "ESW",       # engineering specification waiver
    "waiver",
    "deviation",
    "MRB",       # material review board disposition
)

# Base forms only, and that is the whole trick. A base-form verb is what makes
# an imperative, so it separates the instruction ("Issue an ESW") from the
# outcome ("the ESW is approved", "an ESW will be issued"), which is legitimate
# and must survive. "request" is deliberately absent: requesting is the
# supplier's own action.
INTERNAL_ACTION_VERBS = (
    "issue",
    "approve",
    "waive",
    "authorise",
    "authorize",
    "grant",
)

# Verbs that are the customer's prerogative whatever the object. "Waive the
# requirement" names no instrument and is still the customer deciding. Kept
# deliberately small: "approve" and "issue" are NOT here, because a supplier can
# legitimately approve their own rework or issue replacement parts, and those
# need an instrument to disambiguate.
ALWAYS_CUSTOMER_VERBS = ("waive",)

_MISATTRIBUTED = re.compile(
    rf"\b({'|'.join(INTERNAL_ACTION_VERBS)})\s+(?:an?\s+|the\s+)?"
    rf"({'|'.join(re.escape(i) for i in INTERNAL_INSTRUMENTS)})\b",
    re.IGNORECASE,
)

# The bare-verb rule only fires at the start of a step, because that is what
# makes it an imperative. Real history contains "we will waive it if the result
# is ok" -- the customer stating an outcome the supplier can expect, which is
# useful text and must survive. Only "1. Waive the requirement." is an
# instruction, and only position tells the two apart.
_MISATTRIBUTED_IMPERATIVE = re.compile(
    rf"(?:^|\n)\s*(?:\d+[.)]\s*)?(?:please\s+)?"
    rf"({'|'.join(ALWAYS_CUSTOMER_VERBS)})\b",
    re.IGNORECASE,
)


def misattributed_actions(text: str) -> list[str]:
    """Instructions telling the supplier to perform the customer's action.

    The Actor's constraints and the Judge's second check both forbid this
    already, in prose. Neither fired on "1. Issue an ESW." because nothing in
    the system knows who issues an ESW: the historical Solution_Text is an
    internal engineer's own to-do note, carrying no actor, and the Judge has no
    domain knowledge to supply one. This supplies it, as data.

    Returns findings naming the fragment, empty when the text is clean.
    """
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for pattern in (_MISATTRIBUTED, _MISATTRIBUTED_IMPERATIVE):
        for match in pattern.finditer(text):
            fragment = " ".join(match.group(0).split())
            key = fragment.casefold()
            if key in seen:
                continue
            seen.add(key)
            findings.append(f"instruction to perform an internal action: {fragment!r}")
    return findings


def untransferable_references(text: str) -> list[str]:
    """Fragments in `text` that belong to the source record, not this ticket.

    Returns a list of human-readable findings, empty when the text is clean.
    Each finding names the category and quotes the offending fragment, because
    it is handed to the Actor as a critique and "remove the attachment
    reference" is actionable where "failed check 3" is not.

    Named individuals are deliberately absent: no pattern distinguishes a
    person's name from a material or a process, so that one is left to the
    Judge, where a wrong answer costs a retry rather than a false rejection.
    """
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            fragment = " ".join(match.group(0).split())
            # Deduplicated on the fragment alone, not the category. A known
            # prefix also matches the generic LETTERS#DIGITS rule, and listing
            # ESW#20033465 twice under two headings makes the critique read as
            # two problems when there is one. _PATTERNS is ordered specific
            # first, so the better description is the one that survives.
            key = fragment.casefold()
            if key in seen:
                continue
            seen.add(key)
            findings.append(f"{category}: {fragment!r}")
    return findings


def critique_for(findings: list[str], misattributed: list[str] | None = None) -> str:
    """Turn findings into an instruction the Actor can act on.

    Both kinds arrive in one critique rather than two, so a draft carrying both
    problems is rewritten once instead of burning two of its three attempts.
    """
    parts: list[str] = []

    if findings:
        parts.append(
            f"The draft carries {len(findings)} reference(s) that exist only in "
            f"the historical record and are not true of this ticket: "
            f"{'; '.join(findings)}. "
            "Remove each one. State the action without the reference where the "
            "step is still meaningful on its own, and drop the step entirely "
            "where it is not. Being present in the historical solutions does "
            "not make these transferable -- they describe a different ticket."
        )

    if misattributed:
        parts.append(
            f"The draft tells the supplier to perform {len(misattributed)} "
            f"action(s) that only the customer can perform: "
            f"{'; '.join(misattributed)}. "
            "The supplier may request these; they cannot grant them. Rewrite "
            "each as an outcome the supplier awaits -- 'An ESW has been "
            "requested; do not ship until it is approved' -- or drop the step. "
            "The historical text is an internal engineer's own note, so it "
            "reads as an instruction to themselves, not to the supplier."
        )

    parts.append(
        "If what remains is no action the supplier could actually perform, "
        f"answer '{SOLUTION_NOT_FOUND}' instead."
    )
    return " ".join(parts)


# =============================================================================
# Tier 1: intent scoring
# =============================================================================
#
# Tier 1's decision: which past record is asking the same thing?
#
# This is what replaced the Actor and Judge on the historical path. The
# difference is worth stating plainly, because the whole safety posture of Tier 1
# turns on it: the Actor *wrote* a recommendation and the Judge audited it, so a
# bad retrieval could still be caught downstream. Nothing writes anything here.
# A record scores, and if it scores highly enough its solution is sent exactly as
# it was recorded. The score is the only decision in the path.
#
# Two consequences follow from that, and both are deliberate.
#
# The score is an LLM's, so it is not a calibrated probability. It ranks five
# candidates against each other well; read as an absolute gate it means whatever
# the model decides it means. `SPS_INTENT_THRESHOLD` is provisional until
# measured.
#
# And the scorer is shown problems, never solutions. It is deciding which past
# problem is the same request -- showing it the answers would invite it to score
# their usefulness instead, and a record whose problem matches perfectly but
# whose solution is boilerplate must still score high. "The closest precedent is
# empty" is a finding a reviewer needs; folding it into a low match score would
# hide it behind "nothing matched".

@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    intent_match: float  # 0..1, to match every other score in the codebase
    reason: str

    @property
    def percent(self) -> int:

        return score_to_percent(self.intent_match)


@dataclass(frozen=True, slots=True)
class IntentOutcome:
    """What the scorer concluded, or why it could not."""

    ticket_intent: str = ""
    scored: tuple[ScoredCandidate, ...] = ()
    failure_reason: str = ""

    @property
    def best(self) -> ScoredCandidate | None:
        return self.scored[0] if self.scored else None

    @property
    def failed(self) -> bool:
        return bool(self.failure_reason)


async def score_intent(
    client: ChatClient,
    ticket: IncomingTicket,
    candidates: Sequence[Candidate],
) -> IntentOutcome:
    """Score every candidate against the ticket's intent, best first.

    Fails closed. An unreachable or malformed scorer returns a failure rather
    than an empty score list, because the two mean opposite things: "nothing
    matched" is a legitimate business outcome the robot files, and "we could
    not tell" is an outage it should retry. Collapsing them would quietly
    convert every outage into a refusal.
    """
    if not candidates:
        return IntentOutcome()


    try:
        assessment = await client.complete_model(
            build_intent_messages(ticket, candidates), IntentAssessment
        )
    except LLMError as exc:
        logger.warning("Ticket %r: intent scoring failed: %s", ticket.sps_id, exc)
        return IntentOutcome(failure_reason=f"Intent scoring failed: {exc}")

    by_id = {c.sps_id: c for c in candidates}
    scored: list[ScoredCandidate] = []
    for entry in assessment.scores:
        candidate = by_id.get(entry.sps_id)
        if candidate is None:
            # A score for a record we never supplied. The id is the only thing
            # tying a score to a solution, so a hallucinated one cannot be
            # matched to anything and is dropped rather than guessed at.
            logger.warning(
                "Ticket %r: scorer returned unknown SPS ID %r; ignoring",
                ticket.sps_id, entry.sps_id,
            )
            continue
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                intent_match=min(max(entry.intent_match, 0), 100) / 100.0,
                reason=entry.reason.strip(),
            )
        )

    if not scored:
        # Every entry was unusable, or none came back. Not the same as a low
        # score: we have no judgement at all, so nothing may be cascaded.
        return IntentOutcome(
            ticket_intent=assessment.ticket_intent.strip(),
            failure_reason="Intent scoring returned no usable scores.",
        )

    # Ties broken by whether the record is the ticket's own part, then by the
    # retrieval score, then by SPS ID. The scorer works in whole percents, so
    # ties are common -- five candidates and a coarse scale -- and without a
    # tiebreak the cascaded solution would depend on dict ordering. The own
    # part wins only a tie: a sibling part that scores higher is still sent
    # first.
    scored.sort(
        key=lambda s: (
            s.intent_match, s.candidate.exact_part, s.candidate.composite_score,
            s.candidate.sps_id,
        ),
        reverse=True,
    )

    logger.info(
        "Ticket %r intent %r; best %s at %d%%",
        ticket.sps_id,
        assessment.ticket_intent.strip()[:80],
        scored[0].candidate.sps_id,
        scored[0].percent,
    )
    return IntentOutcome(
        ticket_intent=assessment.ticket_intent.strip(),
        scored=tuple(scored),
    )


# =============================================================================
# The Actor/Judge loop
# =============================================================================
#
# Component C -- the Actor / Judge generate-audit-refine loop.
#
# Circuit breaker: MAX_ATTEMPTS total passes (1 draft + 2 refinements). A draft
# that still fails the Judge on the final attempt is discarded outright; the
# pipeline falls back to "Solution not found." rather than shipping unverified
# text to a supplier.

TIER_HISTORICAL = "historical"
TIER_DOCUMENTATION = "0250"


@dataclass(frozen=True, slots=True)
class Draft:
    recommendation: str
    justification: str

    @property
    def is_abstention(self) -> bool:
        """The Actor may decline when the precedent does not fit the problem."""
        return self.recommendation.strip().rstrip(".").casefold() == (
            SOLUTION_NOT_FOUND.rstrip(".").casefold()
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    critique: str = ""


@dataclass(frozen=True, slots=True)
class Grounding:
    """The evidence one loop run is allowed to draw on, and how to render it.

    Both tiers share one loop: the same circuit breaker, the same fail-closed
    Judge, the same abstention handling. Only the evidence and its prompts
    differ, so those are the parameters rather than a forked implementation --
    a second copy of the loop would be a second place for the retry limit to
    drift.
    """

    items: Sequence[Any]
    actor_messages: Callable[..., list[dict[str, str]]]
    judge_messages: Callable[..., list[dict[str, str]]]
    # What to report when the Actor declines. Tier 1 and Tier 2 decline for
    # different reasons and the status sheet should say which.
    abstention_reason: str
    tier: str = TIER_HISTORICAL


def documentation_grounding(chunks: Sequence[Any]) -> Grounding:
    return Grounding(
        items=chunks,
        actor_messages=build_tier2_actor_messages,
        judge_messages=build_tier2_judge_messages,
        abstention_reason=(
            "The 0250 standards retrieved do not address this defect."
        ),
        tier=TIER_DOCUMENTATION,
    )


# Why the loop stopped, as a value rather than as prose.
#
# `failure_reason` is written for a person and reads differently for every
# cause, so counting causes across a batch meant pattern-matching English. The
# business question behind that count -- "are your checks the reason we get so
# few recommendations, or is it the archive?" -- deserves an answer that is
# tallied rather than argued, and these are what make it tallyable.
STOP_ACTOR_ABSTAINED = "ACTOR_ABSTAINED"
STOP_GATE_UNTRANSFERABLE = "GATE_UNTRANSFERABLE"
STOP_GATE_MISATTRIBUTED = "GATE_MISATTRIBUTED"
STOP_JUDGE_REFUSED = "JUDGE_REFUSED"
STOP_INFRASTRUCTURE = "INFRASTRUCTURE"


def _sanitised(draft: Draft, sps_id: str) -> Draft:
    """Drop a justification that carries the record's own baggage.

    Everything above audits `recommendation`; nothing audited `justification`,
    and that field is written to the workbook a reviewer reads. So a spotless
    recommendation could ship beside "Based on SPS-100, for which ESW#20033465
    was submitted; see the attachment" -- text the very same gate rejects when
    it is handed it.

    Blanked rather than rewritten, because both tiers already substitute a
    generated sentence for an empty justification. Blanking therefore needs no
    new code path, no second model call and no retry.

    And blanked rather than treated as a rejection, because the costs are not
    symmetric. The recommendation is already clean by this point; throwing it
    away over its rationale would spend a retry, and often the whole ticket, to
    fix prose that is context for the reviewer rather than the text sent on.
    """
    findings = untransferable_references(draft.justification) + misattributed_actions(
        draft.justification
    )
    if not findings:
        return draft

    logger.warning(
        "Ticket %r: justification discarded, it carried %s",
        sps_id,
        "; ".join(findings),
    )
    return replace(draft, justification="")


@dataclass(slots=True)
class LoopOutcome:
    """Result plus the audit trail an admin reviewer may need to see."""

    draft: Draft | None
    attempts: int
    critiques: list[str] = field(default_factory=list)
    failure_reason: str = ""
    # One of the STOP_* values above, or "" on success.
    stop_reason: str = ""
    # True when the loop stopped because a dependency was unavailable rather
    # than because the content failed review. The caller needs this to tell an
    # Azure outage apart from a legitimate refusal.
    infrastructure_failure: bool = False
    # Which evidence produced this outcome. The status sheet and the result
    # workbook both report it, so a reviewer can see whether a recommendation
    # came from precedent or from a standard.
    tier: str = TIER_HISTORICAL

    @property
    def succeeded(self) -> bool:
        return self.draft is not None


class ActorCriticLoop:
    def __init__(
        self,
        client: ChatClient,
        settings: LLMSettings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or LLMSettings()

    async def run_grounded(
        self,
        ticket: IncomingTicket,
        grounding: Grounding,
    ) -> LoopOutcome:
        """Generate, audit and refine against whatever evidence is supplied."""
        max_attempts = max(1, self.settings.max_attempts)
        critique: str | None = None
        previous_draft: str | None = None
        critiques: list[str] = []
        # What rejected the most recent draft. The circuit breaker reports it,
        # because "failed the audit three times" does not say whether a regex
        # or the Judge was doing the failing, and those lead to different fixes.
        last_stop = ""

        for attempt in range(1, max_attempts + 1):
            try:
                draft = await self._act(ticket, grounding, critique, previous_draft)
            except LLMError as exc:
                logger.warning("Ticket %r attempt %d: Actor failed: %s", ticket.sps_id, attempt, exc)
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=f"Generation failed: {exc}",
                    stop_reason=STOP_INFRASTRUCTURE,
                    infrastructure_failure=True,
                    tier=grounding.tier,
                )

            if draft.is_abstention:
                # The Actor judged the precedent inapplicable. Retrying would
                # only pressure it into inventing something.
                logger.info("Ticket %r attempt %d: Actor abstained", ticket.sps_id, attempt)
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=grounding.abstention_reason,
                    stop_reason=STOP_ACTOR_ABSTAINED,
                    tier=grounding.tier,
                )

            # Deterministic gate, deliberately BEFORE the Judge. It is local and
            # free where the Judge is a paid network call, and unlike the Judge
            # it cannot talk itself round: CHECK 1 has just taught the model
            # that anything drawn from the source is acceptable, which is
            # precisely why "ESW#20033465 is submitted for these issues" passed
            # an audit it should have failed. A rewrite is fed back through the
            # same critique path the Judge uses, so the circuit breaker still
            # bounds the retries.
            leaks = untransferable_references(draft.recommendation)
            misattributed = misattributed_actions(draft.recommendation)
            if leaks or misattributed:
                logger.info(
                    "Ticket %r attempt %d rejected locally: %s",
                    ticket.sps_id, attempt, "; ".join(leaks + misattributed),
                )
                # One critique covering both, so a draft with both problems is
                # rewritten once rather than burning two of its three attempts.
                feedback = critique_for(leaks, misattributed)
                # Leaks win when both fired. Arbitrary but fixed, so the tally
                # is stable; `failure_reason` still names every fragment found.
                last_stop = (
                    STOP_GATE_UNTRANSFERABLE if leaks else STOP_GATE_MISATTRIBUTED
                )
                critiques.append(feedback)
                critique = feedback
                previous_draft = draft.recommendation
                continue

            try:
                verdict = await self._judge(ticket, grounding, draft.recommendation)
            except LLMError as exc:
                logger.warning("Ticket %r attempt %d: Judge failed: %s", ticket.sps_id, attempt, exc)
                # An unverified draft is never shipped: a Judge outage fails closed.
                return LoopOutcome(
                    draft=None,
                    attempts=attempt,
                    critiques=critiques,
                    failure_reason=f"Compliance audit unavailable: {exc}",
                    stop_reason=STOP_INFRASTRUCTURE,
                    infrastructure_failure=True,
                    tier=grounding.tier,
                )

            if verdict.passed:
                logger.info("Ticket %r: draft passed audit on attempt %d", ticket.sps_id, attempt)
                return LoopOutcome(
                    draft=_sanitised(draft, ticket.sps_id),
                    attempts=attempt,
                    critiques=critiques,
                    tier=grounding.tier,
                )

            logger.info(
                "Ticket %r attempt %d rejected: %s", ticket.sps_id, attempt, verdict.critique
            )
            last_stop = STOP_JUDGE_REFUSED
            critiques.append(verdict.critique)
            critique = verdict.critique
            previous_draft = draft.recommendation

        logger.warning(
            "Ticket %r: circuit breaker tripped after %d attempts", ticket.sps_id, max_attempts
        )
        return LoopOutcome(
            draft=None,
            attempts=max_attempts,
            critiques=critiques,
            failure_reason=(
                f"Draft failed the compliance audit on all {max_attempts} attempts."
                + (f" Last critique: {critiques[-1]}" if critiques else "")
            ),
            stop_reason=last_stop or STOP_JUDGE_REFUSED,
            tier=grounding.tier,
        )

    async def _act(
        self,
        ticket: IncomingTicket,
        grounding: Grounding,
        critique: str | None,
        previous_draft: str | None,
    ) -> Draft:

        messages = grounding.actor_messages(
            ticket, grounding.items, critique, previous_draft
        )
        drafted = await self.client.complete_model(messages, ActorDraft)
        recommendation = drafted.recommendation.strip()
        if not recommendation:
            raise LLMError("Actor returned an empty recommendation")
        return Draft(
            recommendation=recommendation,
            justification=drafted.justification.strip(),
        )

    async def _judge(
        self,
        ticket: IncomingTicket,
        grounding: Grounding,
        draft: str,
    ) -> Verdict:

        messages = grounding.judge_messages(ticket, grounding.items, draft)
        # The Literal["PASS", "FAIL"] on JudgeVerdict means an unrecognised
        # status fails validation and raises rather than being read as a pass.
        verdict = await self.client.complete_model(messages, JudgeVerdict)
        if verdict.status == "PASS":
            return Verdict(passed=True)
        return Verdict(
            passed=False,
            critique=verdict.critique.strip()
            or "The draft was rejected without a stated reason.",
        )


# =============================================================================
# Building results
# =============================================================================
#
# Builders for the resolved recommendation.
#
# One per tier. Both produce the same `PipelineResult`, so the writer downstream
# has a single shape to render; what differs is where `referenced_sources` comes
# from and what `resolution_source` says about it.

def _dedupe(values: Sequence[str]) -> list[str]:
    """Preserve order, drop repeats and blanks.

    Order is retrieval order, which is descending relevance -- so the first
    citation a reviewer reads is the strongest one.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            kept.append(value)
    return kept


def cascade_warnings(text: str) -> str:
    """What a reviewer should read before pasting this, or '' if nothing.

    The same two scanners that BLOCK a draft in Tier 2 only ANNOTATE here, and
    the asymmetry is the point rather than an oversight. Tier 2 text is written
    by a model, so rejecting it costs a retry. Tier 1 text is the archive's own
    and is sent exactly as recorded -- rejecting it would mean refusing a
    precedent the business asked to be sent, so this names the risk and leaves
    the decision where it now sits, with the reviewer.
    """

    findings = untransferable_references(text) + misattributed_actions(text)
    return "Contains: " + "; ".join(findings) if findings else ""


def cascade(
    solution: str,
    candidate: Candidate,
    intent_match: float,
    reason: str,
    source_list: str = "",
) -> PipelineResult:
    """Tier 1: a past solution passed through unchanged.

    `solution` is written to the sheet character for character -- no strip, no
    renumbering, no whitespace collapse. That is the requirement, and it is
    also what makes the output auditable: a reviewer can diff the cell against
    the source record and expect an exact match.
    """
    return PipelineResult(
        ai_recommendation=solution,
        justification=(
            f"Confidence is {score_to_percent(intent_match)}% against {candidate.sps_id}, "
            f"whose recorded solution is sent unchanged."
            + ("" if candidate.exact_part else
               f" It was recorded for {candidate.part_number}, another part in the same "
               f"commodity family.")
            + f" {reason}"
        ).strip(),
        confidence=f"{score_to_percent(intent_match)}%",
        # One record, because one record's solution was sent. Naming the other
        # four would say the answer came from all of them.
        referenced_sources=[candidate.sps_id],
        resolution_source=SOURCE_HISTORICAL,
        closest_matching_solution=source_list,
        cascade_warnings=cascade_warnings(solution),
    )


def _with_precedent(justification: str, precedent: str) -> str:
    """Put the source beneath the rationale, in the field a reviewer reads.

    The precedent also has its own column, but a reviewer checking whether a
    recommendation really follows from the record should not have to hold two
    columns in their head to do it. So it appears in both, and this is the one
    they are already looking at.

    The rationale leads here, unlike on a refusal where the precedent does. On
    a refusal the precedent IS the finding -- there is no recommendation to
    explain. On a success the rationale is the answer to "why this?" and the
    source is the evidence for it, which is the order they get read in.
    """
    if not precedent:
        return justification
    return f"{justification}\n\n{precedent}" if justification else precedent


def success_from_docs(
    recommendation: str,
    justification: str,
    top_score: float,
    chunks: Sequence,
    closest_matching_solution: str = "",
) -> PipelineResult:
    """Tier 2: audited recommendation plus the standards sections it came from.

    The citations are taken from the retrieved chunks, not from the model's
    prose. The Judge checks that what the model wrote matches these, but the
    column itself is never the model's to author -- the same reason confidence
    and SPS IDs are supplied from measurement in Tier 1.
    """
    return PipelineResult(
        ai_recommendation=recommendation,
        justification=_with_precedent(justification, closest_matching_solution),
        confidence=f"{score_to_percent(top_score)}%",
        referenced_sources=_dedupe([c.citation for c in chunks]),
        resolution_source=SOURCE_DOCUMENTATION,
        closest_matching_solution=closest_matching_solution,
    )


# =============================================================================
# Writing workbooks
# =============================================================================
#
# Excel handoff for the UiPath Performer.
#
# The resolver hands back two workbooks: a status sheet written on every run, and
# a result sheet written whenever the run reached a conclusion -- including the
# conclusion that neither tier had an answer.
#
# openpyxl is imported inside the writer, so a command that writes no workbook
# never loads it.

# openpyxl refuses to write a cell longer than this, and Excel itself will not
# display more. Truncating with a visible marker beats failing the handoff.
EXCEL_MAX_CELL = 32767
_TRUNCATION_NOTE = " […truncated for Excel]"

# Characters an .xlsx cannot hold at all: XML 1.0 forbids them, and openpyxl
# raised on them, failing the write -- which the resolver then reported as an
# infrastructure fault for the robot to retry, forever, on the same text. They
# arrive in archive text pasted from Word, where a vertical tab is a manual line
# break and a form feed a page break; those two become line breaks, and the
# rest, which carry nothing, are dropped. Same reasoning as the truncation.
_EXCEL_LINE_BREAKS = str.maketrans({"\x0b": "\n", "\x0c": "\n"})
_EXCEL_UNSTORABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _fit_cell(value: str) -> str:
    unstorable = len(_EXCEL_UNSTORABLE.findall(value))
    if unstorable:
        logger.warning("Replacing %d character(s) an Excel cell cannot hold", unstorable)
        value = _EXCEL_UNSTORABLE.sub("", value.translate(_EXCEL_LINE_BREAKS))
    if len(value) <= EXCEL_MAX_CELL:
        return value
    logger.warning("Truncating a %d-character cell to Excel's 32767 limit", len(value))
    return value[: EXCEL_MAX_CELL - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


# --------------------------------------------------------------------------
# Resolver workbooks: a status sheet written on every run, and a result sheet
# written whenever a verdict was reached, "Solution not found." included.
# --------------------------------------------------------------------------

# Embedding_Model is appended LAST on purpose: a caller reading the first four
# columns positionally is unaffected by its arrival.
STATUS_COLUMNS = (
    "Execution_Timestamp",
    "Status",
    "Status_Code",
    "Reason",
    "Embedding_Model",
)
# Referenced_Sources replaces the former Referenced_SPS_IDs: with two tiers the
# column holds SPS IDs or document citations, and Resolution_Source says which.
# Both are a BREAKING change for a caller that reads this sheet by column name.
# Closest_Matching_Solution is appended LAST, like Embedding_Model above and for
# the same reason. It is raw archive text that has passed none of the
# supplier-facing checks -- it carries its own banner saying so -- and it exists
# because a reviewer looking at a refusal asked to see what we found rather than
# only that we found nothing.
RESULT_COLUMNS = (
    "Part_Number",
    "AI_Recommendation",
    "Justification",
    "Confidence_Score",
    "Referenced_Sources",
    "Resolution_Source",
    "Closest_Matching_Solution",
    # Blank unless the recommendation is archive text passed through unchanged
    # and that text carries something the reviewer should see first.
    "Cascade_Warnings",
    # 1 to MAX_SOLUTIONS, one row each, best first; blank on the single row
    # written when there is no recommendation. Appended last, like the columns
    # above, so a caller reading the first ones positionally is unaffected --
    # but a caller that read only the first ROW now sees only the best answer.
    "Solution_Rank",
)


def write_rows(path: Path | str, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write a sheet atomically, every cell as text.

    Text throughout, so a Confidence_Score of "84%" or a part number of
    "0012-43951" never reaches Excel as a number or a date. Marked as text cell
    by cell, too: left to itself openpyxl stores text beginning with "=" as a
    formula and "#N/A" as an error value, so an archive solution reading
    "=> regrind the seam" would open as a broken formula.

    One sheet named Sheet1, header bold with thin borders, centred: the layout
    pandas gave these workbooks when it wrote them, kept so that nothing reading
    them notices the change.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side

    path = Path(path)
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"

    edge = Side(style="thin")
    for column, name in enumerate(columns, start=1):
        cell = _text_cell(sheet, 1, column, _fit_cell(str(name)))
        cell.font = Font(bold=True)
        cell.border = Border(left=edge, right=edge, top=edge, bottom=edge)
        cell.alignment = Alignment(horizontal="center", vertical="top")
    for row_number, row in enumerate(rows, start=2):
        for column, name in enumerate(columns, start=1):
            _text_cell(sheet, row_number, column, _fit_cell(str(row.get(name, ""))))

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".xlsx")
    os.close(handle)
    try:
        book.save(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _text_cell(sheet, row: int, column: int, text: str):
    cell = sheet.cell(row=row, column=column)
    cell.value = text
    cell.data_type = "s"  # never a formula or an error value; see write_rows
    return cell


# =============================================================================
# Command: resolve
# =============================================================================
#
# Resolve one SPS ticket against a historical file. No vector database.
#
#     python sps.py resolve --ticket-file ticket.xlsx --history-file history.csv --output-dir .\out
#
# Writes two workbooks into --output-dir:
#
#   status.xlsx   ALWAYS, including an early abort or an unhandled exception.
#                 Execution_Timestamp, Status (PASS/FAIL), Status_Code, Reason,
#                 Embedding_Model. The last names the embedding deployment that
#                 ran; it is blank when the run aborted before anything was
#                 encoded.
#   output.xlsx   Whenever the run REACHED A CONCLUSION, which includes
#                 concluding that neither tier had an answer. Part_Number,
#                 AI_Recommendation, Justification, Confidence_Score,
#                 Referenced_Sources, Resolution_Source. On a failure the
#                 recommendation is "Solution not found." and Resolution_Source
#                 is NONE -- a caller merging this into its own records needs one
#                 row per ticket, and a missing row cannot be told apart from a
#                 ticket that was never processed.
#
#                 NOT written for an infrastructure fault (exit 1 or 2): the run
#                 formed no verdict, and the robot is expected to retry it.
#
# Both are written atomically, and both are cleared before any work starts, so a
# process killed outright leaves no stale result to be mistaken for this run's.
#
# Exit codes are retained alongside the status sheet, since a caller can branch on
# them without opening a workbook: 0 the run completed (PASS or a legitimate
# FAIL), 1 an infrastructure fault, 2 the inputs could not be read.
#
# Two tiers. Tier 1 answers from the part's own historical SPS records. When
# that produces nothing usable -- no matching part, nothing similar enough, or no
# record the intent scorer judges to ask the same question -- Tier 2 answers
# from the 0250 engineering standards instead, and says so in
# Resolution_Source.
#
# Order of work is cheapest-first, so nothing expensive runs for a ticket that
# cannot succeed: validate, then sync the history store, fetch the family (or the
# part), embed what is new, rank, gate, then the LLM. Tier 2 only runs after all
# of that has failed, which is why it can afford to parse and embed a document
# corpus.



EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_BAD_INPUT = 2

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

# The two tiers are delineated in the status code, so a caller can tell a
# precedent-backed answer from a standards-derived one without opening
# output.xlsx. Status itself stays PASS/FAIL for both, so a workflow branching
# on Status is unaffected.
CODE_SUCCESS_HISTORICAL = "SUCCESS_HISTORICAL"
CODE_SUCCESS_DOC = "SUCCESS_0250_DOC"
# When neither tier resolves the ticket, the code says WHY the pipeline ran out
# of options, because the robot routes on it: an unknown part goes to Master
# Data, a novel defect goes to a Reliability Engineer, and a draft the Judge
# refused goes to a human reviewer. One collapsed code would send all three to
# the same queue.
CODE_NO_MATCHES = "NO_MATCHES"
CODE_BELOW_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"
CODE_AUDIT_REJECTED = "LLM_AUDIT_REJECTED"
CODE_INVALID_INPUT = "INVALID_INPUT"
CODE_INFRASTRUCTURE = "INFRASTRUCTURE_ERROR"

# `Status == PASS` has one definition.
SUCCESS_CODES = frozenset({CODE_SUCCESS_HISTORICAL, CODE_SUCCESS_DOC})

# How far a tier got before it stopped. Ordered, because the reported code is
# the FURTHEST stage either tier reached: a run where history reached the Judge
# and was refused, while the standards had nothing to say, is an audit
# rejection -- routing it to Master Data as NO_MATCHES would be wrong, since
# the part was perfectly well known.
STAGE_NOTHING = 0    # nothing retrieved at all
STAGE_GATED = 1      # retrieved, nothing cleared the threshold
STAGE_REJECTED = 2   # cleared the threshold, then refused: intent score, Actor or Judge

STAGE_CODES = {
    STAGE_NOTHING: CODE_NO_MATCHES,
    STAGE_GATED: CODE_BELOW_THRESHOLD,
    STAGE_REJECTED: CODE_AUDIT_REJECTED,
}

STATUS_FILE = "status.xlsx"
OUTPUT_FILE = "output.xlsx"


def resolve_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python sps.py resolve",
        description="Resolve one SPS ticket against a historical file.",
        epilog=(
            "Azure credentials are read from the environment only "
            "(AZURE_OPENAI_ENDPOINT / _API_KEY / _DEPLOYMENT)."
        ),
    )
    parser.add_argument("--ticket-file", required=True, help="Single-ticket .xlsx or .csv")
    parser.add_argument("--history-file", required=True, help="Historical records .csv or .xlsx")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Where status.xlsx and output.xlsx are written (default: current directory)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Tier-1 confidence threshold (default {AZURE_EMBEDDING_THRESHOLD} for "
        "Azure embeddings, or SPS_CONFIDENCE_THRESHOLD)",
    )
    parser.add_argument(
        "--tier2-threshold",
        type=float,
        default=None,
        help=f"Tier-2 gate (default {TIER2_AZURE_THRESHOLD}, or SPS_TIER2_THRESHOLD). "
        "Separate from "
        "--threshold because the two tiers score in different ranges.",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="0250 standards folder for the Tier-2 fallback "
        f"(default {DEFAULT_DOCS_DIR}, or SPS_0250_DOCS_DIR). "
        "Tier 2 is skipped when the folder holds no .docx.",
    )
    parser.add_argument(
        "--no-tier2",
        action="store_true",
        help="Answer from history only, as before the 0250 fallback existed.",
    )
    parser.add_argument(
        "--commodity-file",
        default=None,
        help=f"Commodity groups .csv/.xlsx (Commodity_Code, Group), or {COMMODITY_GROUPS_ENV}. "
        "Without one, a ticket is matched on its part number alone.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def read_ticket(path: Path) -> dict[str, Any]:
    """Read the first data row of a single-ticket file.

    Format-agnostic: .csv and .xlsx both arrive here as a header row plus data
    rows, and the keys are returned verbatim. IncomingTicket.from_dict resolves
    spelling, matching keys case-insensitively.
    """

    headers, rows = read_header_and_rows(path, "Ticket file")
    for row in rows:
        if is_blank(row):
            continue
        return {
            str(h).strip(): (row[i] if i < len(row) else "")
            for i, h in enumerate(headers)
            if str(h or "").strip()
        }
    raise FileReadError(f"Ticket file {path.name!r} has a header but no data row.")


@dataclass(frozen=True, slots=True)
class ResolveOutcome:
    """What one run produced.

    A dataclass rather than a tuple because the batch evaluator needs the raw
    similarity alongside the outcome: calibrating a threshold means seeing the
    score distribution, including for the cases the threshold rejected.
    """

    code: str
    reason: str
    exit_code: int
    embedding_model: str = ""
    # The best cosine found, whether or not it cleared the gate. Zero means
    # nothing was scored -- no history for the part, or an abort before
    # embedding.
    top_score: float = 0.0
    threshold_used: float = 0.0
    candidates_considered: int = 0
    # Tier 2, reported separately rather than folded into the fields above: the
    # two tiers score in different ranges, so a single "top score" column that
    # sometimes meant one and sometimes the other could not be calibrated
    # against anything.
    tier2_top_score: float = 0.0
    tier2_threshold: float = 0.0
    tier2_chunks: int = 0
    tier2_cache_state: str = ""
    resolution_source: str = ""
    # Everything main() needs to write output.xlsx, so that both workbooks are
    # produced in one place. Blank part number means the run stopped before the
    # ticket could be read.
    part_number: str = ""
    # One PipelineResult per solution, best first; empty when there is no
    # recommendation, in which case main() builds the "Solution not found."
    # row. Tier 1 can send up to MAX_SOLUTIONS; Tier 2 writes one.
    results: tuple = ()
    # The shortlisted records' solutions (Tier 1) or the sections the Tier-2
    # Actor was given. Operational only -- it never reaches output.xlsx, which
    # is supplier-facing -- but it is what lets a reviewer see WHY a
    # recommendation says what it does, in the same glance that shows what it
    # said. Two fields rather than one because the tiers are populated from
    # different call frames and a single key would collide in the shared
    # **measured expansion.
    evidence: tuple[str, ...] = ()
    tier2_evidence: tuple[str, ...] = ()
    # The single best precedent, banner and all, ready to write. Unlike
    # `evidence` above this DOES reach output.xlsx -- a reviewer looking at a
    # refusal asked to see what we found, not merely that we found nothing.
    closest_match: str = ""
    # The scored source list, used on a success where closest_match is not.
    source_list: str = ""
    # The intent scorer's verdict, kept apart from `top_score` because they
    # measure different things: top_score is the cosine that shortlisted a
    # record, intent_score is the judgement that it asks the same question.
    # One number standing for both could not be calibrated against either.
    intent_score: float = 0.0
    # The scorer's one-line reading of what the ticket asks for. Recorded so a
    # wrong match can be traced to a wrong reading rather than being an
    # unexplained number.
    ticket_intent: str = ""
    # Which check ran out of road, as a value rather than as prose, so a batch
    # can be tallied by cause instead of read one row at a time.
    stop_reason: str = ""
    # The intent gate the run applied, beside the score it was applied to.
    intent_threshold: float = 0.0

    @property
    def result(self):
        """The best solution, or None."""
        return self.results[0] if self.results else None


def _reason_with_marker(reason: str, embedding_model: str) -> str:
    """Append [Azure] so the encoder is visible in Reason itself.

    [Injected] marks an encoder handed in through the test seam, so a stand-in
    can never pass for the deployment in a status sheet.
    """
    flat = " ".join(str(reason).split())
    if not embedding_model:
        return flat
    marker = "[Azure]" if embedding_model.startswith("azure") else "[Injected]"
    return f"{flat} {marker}"


def _describe_backend(stats) -> str:
    """One short token naming the encoder, for the status sheet.

    Names the deployment rather than just "azure", so a run against the wrong
    embedding deployment is visible in the sheet rather than only in its scores.
    """
    if not stats.backend:
        return ""
    detail = stats.backend_detail or stats.backend
    return f"{stats.backend}:{detail}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(output_dir: Path, code: str, reason: str, embedding_model: str = "") -> None:

    status = STATUS_PASS if code in SUCCESS_CODES else STATUS_FAIL
    write_rows(
        output_dir / STATUS_FILE,
        STATUS_COLUMNS,
        [
            {
                "Execution_Timestamp": _now(),
                "Status": status,
                "Status_Code": code,
                # The marker is appended to Reason as well as carried in its
                # own column: a support engineer skimming the sheet, or a
                # caller reading only the first four columns, still sees which
                # encoder ran without having to know the column exists.
                "Reason": _reason_with_marker(reason, embedding_model),
                # Blank when the run aborted before embedding, which is itself
                # information: nothing was encoded.
                "Embedding_Model": embedding_model,
            }
        ],
    )
    logger.info("status: %s / %s [%s] -- %s", status, code, embedding_model or "none", reason)


# Long enough to tell boilerplate from a real disposition, short enough that
# fifteen candidates stay readable in one cell. The workbook writer truncates at
# Excel's hard limit anyway; this keeps it from getting that far.
EVIDENCE_CHARS = 500


def _evidence(label: str, text: str) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) > EVIDENCE_CHARS:
        flat = flat[: EVIDENCE_CHARS - 1] + "\u2026"
    return f"{label}: {flat}"


# DEA copies the recommendation into the SPS portal by hand. A field holding
# raw archive text therefore has to announce itself in the first few words, or
# it gets pasted too -- which is precisely how "ESW#20033465 is submitted for
# these issues" would reach a supplier. The banner is a control, not a caption.
RAW_BANNER = "[RAW HISTORY - not checked for supplier use]"
WEAK_BANNER = "[RAW HISTORY - WEAK MATCH, below the confidence gate]"


def _closest_match(records: Sequence[tuple[str, str, float]], gated: bool) -> str:
    """The closest precedents found, banner-prefixed, or '' if there were none.

    `records` is (label, text, score), best first. Several rather than one: a
    reviewer with no recommendation is reading these as options to answer the
    ticket from, and the top-scoring record is not always the one that happens
    to describe the fix.
    """
    if not records:
        return ""

    banner = WEAK_BANNER if gated else RAW_BANNER
    header = f"{banner} {len(records)} closest record(s):"
    blocks = [
        _evidence(f"{label} ({score_to_percent(score)}%)", text)
        for label, text, score in records
    ]
    return header + "\n\n" + "\n\n".join(blocks)


def _source_list(records: Sequence[tuple[str, float]], label: str = "Synthesized from") -> str:
    """Where a recommendation came from, without reproducing it.

    On a success the answer is already written; repeating the archive text
    underneath it adds length without adding information. In Tier 2 the Actor
    may combine several sections, so showing one section's text would read as
    THE source and invite a reviewer to check a step against one it did not
    come from. In Tier 1 one record's solution is sent, and the list is every
    record the intent scorer read, in the order its scores put them.

    What is missing from `Referenced_Sources` is the per-record score -- it
    lists the ids and cannot say which matched at 94% and which at 71%, which
    is the one thing that says where to look first.
    """
    if not records:
        return ""

    listed = ", ".join(f"{name} ({score_to_percent(score)}%)" for name, score in records)
    return f"{label} {len(records)} record(s): {listed}"


# Tier 1 sends records as they are rather than combining them, so its list
# says what was scored, not what was synthesized.
TIER1_SOURCE_LABEL = "Intent scores across"


# Resolver-level stops, alongside the loop's own STOP_* values. These happen
# before any draft is written, so the loop cannot report them.
STOP_NO_HISTORY = "NO_HISTORY_FOR_PART"
STOP_BELOW_THRESHOLD = "BELOW_THRESHOLD"
# Records were found and read, and none of them asks the same question. A
# different finding from BELOW_THRESHOLD, which means nothing was similar
# enough to be worth reading -- and it routes differently, because this one
# says the archive has no answer rather than that retrieval missed.
STOP_INTENT_BELOW = "INTENT_BELOW_THRESHOLD"
STOP_NO_PART_NUMBER = "NO_PART_NUMBER"


def _business_justification(outcome: "ResolveOutcome") -> str:
    """Why there is no recommendation, for the person who reads the portal.

    The diagnostic prose -- scores, thresholds, row counts -- stays in
    status.xlsx's Reason, where the robot and support already read it. It was
    being shown to DEA as well, which is what "please say it in softer words"
    was about. The closest precedent leads, because a reviewer would rather see
    the near-miss first and the explanation second.
    """

    WHY = {
        STOP_BELOW_THRESHOLD: (
            "This is the closest past problem we found, but it is not similar "
            "enough to rely on. Not much historical data to infer the solution "
            "or recommendation."
        ),
        STOP_NO_PART_NUMBER: (
            "No part number was supplied, so the part's own history could not "
            "be searched."
        ),
        STOP_ACTOR_ABSTAINED: (
            "This is the closest past solution we found. We are not "
            "recommending it because it does not answer what this ticket asks."
        ),
        STOP_GATE_UNTRANSFERABLE: (
            "This is the closest past solution we found. We are not "
            "recommending it as it stands because it refers to an attachment, "
            "a prior conversation or a work request belonging to a different "
            "ticket, none of which the supplier can act on."
        ),
        STOP_GATE_MISATTRIBUTED: (
            "This is the closest past solution we found. We are not "
            "recommending it as it stands because it asks the supplier to "
            "perform an action only AMAT can perform."
        ),
        STOP_JUDGE_REFUSED: (
            "This is the closest past solution we found. We are not "
            "recommending it as it stands because it did not pass review."
        ),
    }

    if not outcome.closest_match:
        # No precedent to show, but there may still be a specific reason worth
        # giving. "Not much historical data" is true of an unknown part and
        # misleading about a ticket that arrived with no part number at all --
        # one is a gap in the archive, the other is a gap in the ticket, and
        # only the second is something the sender can fix.
        return WHY.get(
            outcome.stop_reason,
            "Not much historical data to infer the solution or recommendation.",
        )

    # The intent case leads with the number, because the number is the finding
    # -- "we read these records and none of them asks your question, here is
    # the nearest" is a different statement from "we found nothing". It is the
    # one entry that needs a value, so it is built rather than looked up.
    if outcome.stop_reason == STOP_INTENT_BELOW:
        lead = (
            f"Confidence is {score_to_percent(outcome.intent_score)}% hence we are not "
            "recommending the solution, but this is the closest match from "
            "historical data."
        )
        return lead + "\n\n" + outcome.closest_match

    why = WHY.get(
        outcome.stop_reason,
        "This is the closest past solution we found. We are not recommending "
        "it as it stands.",
    )
    # Precedent first, explanation second -- the order the reviewer asked for.
    return outcome.closest_match + "\n\n" + why


def unresolved_result(outcome: "ResolveOutcome"):
    """The result row for a run that concluded without a recommendation.

    A caller merging output.xlsx into its own records needs one row per ticket.
    Without this, "no historical data for this part" produced no row at all, and
    a ticket that found nothing was indistinguishable from a ticket that never
    ran -- which is exactly the case a reviewer most needs to see.
    """

    # Blank, not 0%, when nothing was ever scored. An unknown part and a near
    # miss at 42% are different findings, and "0%" reads as the second one.
    # Reason carries both tiers' scores in full either way.
    #
    # Tier 2's score is used when Tier 1 scored nothing: reporting a flat 0%
    # for a run where the standards matched at 0.41 described the wrong tier.
    # The intent score when there is one: on this path the number a reviewer
    # reads is the one that decided the outcome, and the cosine only
    # shortlisted. Falls back to the retrieval scores where no intent judgement
    # was ever made -- an unknown part, or a Tier-2-only run.
    score = outcome.intent_score or outcome.top_score or outcome.tier2_top_score
    confidence = f"{score_to_percent(score)}%" if score > 0 else ""
    return PipelineResult(
        ai_recommendation=NO_RECOMMENDATION,
        justification=_business_justification(outcome),
        confidence=confidence,
        referenced_sources=[],
        resolution_source=SOURCE_NONE,
        closest_matching_solution=outcome.closest_match,
    )


def solution_rank(result, position: int) -> str:
    """1, 2, 3 for a recommendation; blank for the no-recommendation row."""
    return "" if result.resolution_source == SOURCE_NONE else str(position)


def write_output(output_dir: Path, part_number: str, results: Sequence) -> None:
    """One row per solution, best first -- or the one no-recommendation row."""
    write_rows(
        output_dir / OUTPUT_FILE,
        RESULT_COLUMNS,
        [
            {
                "Part_Number": part_number,
                "AI_Recommendation": result.ai_recommendation,
                "Justification": result.justification,
                "Confidence_Score": result.confidence,
                # SPS IDs or document citations depending on the tier;
                # Resolution_Source alongside says which kind these are.
                "Referenced_Sources": ", ".join(result.referenced_sources),
                "Resolution_Source": result.resolution_source,
                "Closest_Matching_Solution": result.closest_matching_solution,
                "Cascade_Warnings": result.cascade_warnings,
                "Solution_Rank": solution_rank(result, position),
            }
            for position, result in enumerate(results, 1)
        ],
    )


def commodity_file(args: argparse.Namespace) -> str | None:
    """The groups file: --commodity-file, else SPS_COMMODITY_GROUPS, else none."""
    chosen = getattr(args, "commodity_file", None) or os.environ.get(COMMODITY_GROUPS_ENV, "")
    return chosen.strip() or None


def _matched_reason(winners, assessment, threshold: float, scope_note: str) -> str:
    """The Reason line for a Tier-1 answer of one or more solutions."""
    read = len(assessment.scored)
    if len(winners) == 1:
        best = winners[0]
        found = (
            f"Intent matched {best.candidate.sps_id} at {best.percent}% across {read} "
            f"record(s); its solution was sent unchanged."
        )
    else:
        listed = ", ".join(f"{s.candidate.sps_id} ({s.percent}%)" for s in winners)
        found = (
            f"Intent matched {len(winners)} record(s) at or above {int(threshold * 100)}% "
            f"across {read}: {listed}; their solutions were sent unchanged, best first."
        )
    return f"{found} {scope_note} Ticket intent: {assessment.ticket_intent}"


def resolve(args: argparse.Namespace, output_dir: Path) -> ResolveOutcome:
    """Run the pipeline once and report what happened."""
    import asyncio
    import os



    ticket_path = Path(args.ticket_file)
    history_path = Path(args.history_file)

    # Extension check first, on both files, before anything is opened, any model
    # is loaded or any history is scanned. The wrong attachment is a business
    # problem for whoever assembled the ticket, not an I/O fault: it reports
    # INVALID_INPUT and exits 0, so the caller faults the item without retrying.
    for label, path in (("Ticket file", ticket_path), ("History file", history_path)):
        try:
            validate_file_type(path, label)
        except UnsupportedFileType as exc:
            return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_OK)

    # A file that is the right type but absent or unreadable is genuine I/O
    # trouble, and keeps exit 2 so a human is alerted rather than the item
    # being quietly faulted.
    for label, path in (("Ticket file", ticket_path), ("History file", history_path)):
        if not path.exists():
            return ResolveOutcome(
                CODE_INVALID_INPUT, f"{label} not found: {path}", EXIT_BAD_INPUT
            )

    try:
        raw = read_ticket(ticket_path)
    except (FileReadError, ValueError, OSError) as exc:
        return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT)

    ticket = IncomingTicket.from_dict(raw)
    part_number = normalize_part_number(ticket.part_number)

    # Gate before anything expensive: no file scan, no model load, no LLM.
    #
    # The two halves of the old validate_ticket() are now separated, because
    # they stop different amounts of the pipeline.
    #
    # A missing problem description is terminal. Both tiers match on that text
    # -- Tier 2 builds its query from it too -- so there is nothing to search
    # with on either side, and falling through would be searching for nothing.
    invalid = validate_problem_description(ticket.problem_description)
    if invalid:
        return ResolveOutcome(
            invalid.code, invalid.reason, EXIT_OK, part_number=part_number
        )

    # A missing part number stops Tier 1 only. Tier 1 filters history by exact
    # part, so there is nothing for it to retrieve; Tier 2 searches standards
    # by defect text and never looks at the part number.
    bad_part = validate_part_number(ticket.part_number)

    # An explicit --threshold (or SPS_CONFIDENCE_THRESHOLD) overrides the
    # Tier-1 gate; otherwise it is AZURE_EMBEDDING_THRESHOLD, from the
    # environment when set there.
    threshold_override = args.threshold
    if threshold_override is None:
        threshold_override = _env_float("SPS_CONFIDENCE_THRESHOLD", None)
    _threshold = _env_float

    # Constructed before the part-number branch because Tier 2 needs the loop
    # on both paths. Constructing either opens no connection.
    #
    # The client is built separately rather than read back off the loop: the
    # intent scorer and the Tier-2 loop are two callers of one deployment, and
    # having one reach through the other for its transport made the dependency
    # invisible at the call site.
    chat_client = AzureOpenAIChatClient(LLMSettings.from_env())
    loop = ActorCriticLoop(chat_client, LLMSettings.from_env())

    if bad_part:
        logger.info("%s Tier 1 skipped; trying the standards.", bad_part.reason)
        return _resolve_from_docs(
            args=args,
            ticket=ticket,
            part_number=part_number,
            output_dir=output_dir,
            loop=loop,
            tier1_detail=bad_part.reason,
            tier1_stage=STAGE_NOTHING,
            tier1_stop=STOP_NO_PART_NUMBER,
            measured=dict(part_number=part_number),
        )

    intent_settings = IntentSettings.from_env()
    retriever = HistoryRetriever(
        history_path=history_path,
        confidence_threshold=threshold_override,
        azure_threshold=_threshold("AZURE_EMBEDDING_THRESHOLD", AZURE_EMBEDDING_THRESHOLD),
        max_context_records=intent_settings.top_k,
        commodity_file=commodity_file(args),
    )

    try:
        candidates = retriever.retrieve(ticket)
    except HistoryError as exc:
        return ResolveOutcome(CODE_INVALID_INPUT, str(exc), EXIT_BAD_INPUT)
    except HistoryStoreConfigError as exc:
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_BAD_INPUT)
    except AzureEmbeddingNotConfigured as exc:
        # Exit 2, not 1. No number of retries produces an API key, and a robot
        # cycling through fifty of them only delays the human who has to go and
        # set one.
        logger.error("embedding is not configured: %s", exc)
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_BAD_INPUT)
    except AzureEmbeddingError as exc:
        # Transient: network, timeout, throttling, a bad response. Worth a
        # retry, and there is no longer a local encoder to absorb it.
        logger.error("embedding failed: %s", exc)
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_INFRASTRUCTURE)
    except (HistoryStoreError, sqlite3.Error) as exc:
        # Another robot holding the store past the wait, a full disk. Worth a
        # retry; the history file itself is fine.
        logger.error("history store failed: %s", exc)
        return ResolveOutcome(
            CODE_INFRASTRUCTURE, f"History store failed: {exc}", EXIT_INFRASTRUCTURE
        )

    stats = retriever.stats
    logger.info("retrieval stats: %s", stats.as_dict())
    model = _describe_backend(stats)
    scope_note = stats.scope_note

    measured = dict(
        part_number=part_number,
        # The shortlisted records' solutions. Present even on the gated path,
        # where seeing the near-miss text is the whole point.
        evidence=tuple(_evidence(c.sps_id, c.actual_solution) for c in candidates),
        # The closest rows, gate or no gate. `candidates` is empty on a gated
        # run, which is exactly when a reviewer most wants to see what was
        # almost good enough -- so this reads stats, not candidates.
        #
        # Only ever shown when there is NO recommendation. On a success the
        # source list below replaces it: see `_source_list`.
        closest_match=_closest_match(
            [(c.sps_id, c.actual_solution, c.composite_score) for c in stats.top_candidates],
            gated=not candidates,
        ),
        # Every shortlisted record with its cosine. Replaced by the intent-
        # ordered list once the scorer has run; this is what a run that stops
        # before it reports.
        source_list=_source_list(
            [(c.sps_id, c.composite_score) for c in candidates], TIER1_SOURCE_LABEL
        ),
        embedding_model=model,
        top_score=stats.top_score,
        threshold_used=stats.threshold_used,
        candidates_considered=stats.considered,
        intent_threshold=intent_settings.threshold,
    )

    # -- Tier 1: the family's history, or the part's own -------------------

    where = f"commodity family {stats.family}" if stats.family else f"part {part_number}"
    if stats.usable == 0:
        tier1_stage = STAGE_NOTHING
        tier1_stop = STOP_NO_HISTORY
        tier1_detail = (
            f"No usable history for {where}: {stats.part_matches} row(s) "
            f"in scope out of {stats.rows_scanned} stored. {scope_note}"
        )
    elif not candidates:
        tier1_stage = STAGE_GATED
        tier1_stop = STOP_BELOW_THRESHOLD
        tier1_detail = (
            f"Best historical match {stats.top_score:.4f} is below the "
            f"{stats.threshold_used:.2f} threshold across {stats.considered} candidate(s). "
            f"{scope_note}"
        )
    else:
        assessment = asyncio.run(score_intent(chat_client, ticket, candidates))

        if assessment.failed:
            # Fails closed, and as an outage rather than a refusal. "Nothing
            # matched" is a business outcome the robot files; "we could not
            # tell" is something it should retry. Tier 2 shares the same
            # deployment, so there is nothing to fall back to.
            logger.error("dependency failure: %s", assessment.failure_reason)
            return ResolveOutcome(
                CODE_INFRASTRUCTURE, assessment.failure_reason, EXIT_INFRASTRUCTURE, **measured
            )

        best = assessment.best
        measured["ticket_intent"] = assessment.ticket_intent
        measured["intent_score"] = best.intent_match if best else 0.0
        # Re-derived from the intent scores. Retrieval ordered these by cosine,
        # which is precisely the ordering this design stopped trusting, so what
        # the reviewer is shown has to be the order that decided the outcome.
        measured["source_list"] = _source_list(
            [(s.candidate.sps_id, s.intent_match) for s in assessment.scored],
            TIER1_SOURCE_LABEL,
        )
        measured["closest_match"] = _closest_match(
            [
                (s.candidate.sps_id, s.candidate.actual_solution, s.intent_match)
                for s in assessment.scored[:CASCADE_LIMIT]
            ],
            gated=True,
        )

        # Every record that clears the gate, best first, up to MAX_SOLUTIONS.
        winners = [
            s for s in assessment.scored if s.intent_match >= intent_settings.threshold
        ][:MAX_SOLUTIONS]
        if winners:
            # Each sent exactly as recorded. No strip, no renumbering, no
            # normalisation -- a reviewer can diff each cell against its source
            # record and expect a character-for-character match, and anything
            # tidier would break that.
            results = tuple(
                cascade(
                    solution=s.candidate.actual_solution,
                    candidate=s.candidate,
                    intent_match=s.intent_match,
                    reason=s.reason,
                    source_list=measured["source_list"],
                )
                for s in winners
            )
            for s, result in zip(winners, results):
                if result.cascade_warnings:
                    logger.warning(
                        "Ticket %r cascading %s unchanged, and it %s",
                        ticket.sps_id, s.candidate.sps_id, result.cascade_warnings,
                    )
            return ResolveOutcome(
                CODE_SUCCESS_HISTORICAL,
                _matched_reason(winners, assessment, intent_settings.threshold, scope_note),
                EXIT_OK,
                resolution_source=results[0].resolution_source,
                results=results,
                **measured,
            )

        # Records were read and none asks the same question. STAGE_REJECTED,
        # not STAGE_GATED: retrieval did its job and a judgement was made on
        # the content, which is the same shape of outcome the Judge used to
        # produce and routes to the same place.
        tier1_stage = STAGE_REJECTED
        tier1_stop = STOP_INTENT_BELOW
        best_pct = best.percent if best else 0
        tier1_detail = (
            f"Best intent match {best_pct}% is below the "
            f"{int(intent_settings.threshold * 100)}% threshold across "
            f"{len(assessment.scored)} record(s). {scope_note} "
            f"Ticket intent: {assessment.ticket_intent}"
        )

    logger.info("Tier 1 produced no resolution: %s", tier1_detail)

    # -- Tier 2: the 0250 engineering standards ----------------------------

    return _resolve_from_docs(
        args=args,
        ticket=ticket,
        part_number=part_number,
        output_dir=output_dir,
        loop=loop,
        tier1_detail=tier1_detail,
        tier1_stage=tier1_stage,
        tier1_stop=tier1_stop,
        measured=measured,
    )


def _resolve_from_docs(
    args: argparse.Namespace,
    ticket,
    part_number: str,
    output_dir: Path,
    loop,
    tier1_detail: str,
    tier1_stage: int,
    tier1_stop: str,
    measured: dict,
) -> ResolveOutcome:
    """Tier 2. Reached only once Tier 1 has produced nothing usable."""
    import asyncio
    import os


    def _no_resolution(extra: str, tier2_stage: int, **fields) -> ResolveOutcome:
        """Both tiers tried, neither resolved it.

        The code is the furthest stage either tier reached, so the robot can
        route on how the pipeline ran out of options. Reason carries the detail
        from both tiers, including each one's best score, because that is what
        tells a human whether this was a near miss worth a threshold change or
        a genuine absence of evidence.
        """
        # Merged rather than double-splatted: Tier 2 may supply a
        # closest_match when Tier 1 had none, and two **expansions carrying the
        # same key is a TypeError.
        merged = {**measured, **fields}
        # The furthest stage wins for the code, so the stop reason has to agree
        # with it -- reporting Tier 1's cause beside Tier 2's code would send a
        # reader looking in the wrong tier.
        merged.setdefault(
            "stop_reason", tier1_stop if tier1_stage >= tier2_stage else merged.get("stop_reason", "")
        )
        return ResolveOutcome(
            STAGE_CODES[max(tier1_stage, tier2_stage)],
            f"{tier1_detail} {extra}".strip(),
            EXIT_OK,
            **merged,
        )

    if args.no_tier2:
        # Nothing was retrieved because nothing was looked for, so Tier 1's
        # stage stands on its own.
        return _no_resolution("Tier 2 disabled by --no-tier2.", STAGE_NOTHING)

    # The 0250 standards are scoped to particular reason codes, so most
    # tickets never reach them.
    #
    # The two ways of not being allowed are reported differently on purpose.
    # An empty list disables Tier 2 for every ticket in the deployment, and if
    # that happened because a .env line was lost, the only place it would ever
    # show is this sentence. A code simply not being listed is routine.

    intent_settings = IntentSettings.from_env()
    reason_code = (ticket.problem_reason_code or "").strip()
    if not intent_settings.tier2_reason_codes:
        return _no_resolution(
            "Tier 2 skipped: no reason codes are configured "
            "(SPS_TIER2_REASON_CODES is empty).",
            STAGE_NOTHING,
        )
    if not intent_settings.tier2_allowed(reason_code):
        return _no_resolution(
            f"Tier 2 skipped: reason code {reason_code or '(blank)'!r} is not "
            f"one of the {len(intent_settings.tier2_reason_codes)} configured.",
            STAGE_NOTHING,
        )

    docs_dir = (
        args.docs_dir
        or os.environ.get("SPS_0250_DOCS_DIR", "").strip()
        or DEFAULT_DOCS_DIR
    )

    _threshold = _env_float

    # Deliberately NOT --threshold. That number is calibrated against Tier 1's
    # ticket-to-ticket distribution; against Tier 2's ticket-to-document
    # distribution, where the right section scores far lower, the same number
    # could reject every chunk. One override spanning both would be a single number standing for
    # two different embedding spaces, which is the mistake this codebase keeps
    # not making.
    tier2_override = args.tier2_threshold
    if tier2_override is None:
        tier2_override = _env_float("SPS_TIER2_THRESHOLD", None)

    docs = DocRetriever(
        docs_dir=docs_dir,
        confidence_threshold=tier2_override,
        azure_threshold=_threshold("TIER2_AZURE_THRESHOLD", TIER2_AZURE_THRESHOLD),
    )

    if not docs.available:
        # The normal state of a deployment whose standards have not been loaded
        # yet. Not an error: the ticket reports exactly the Tier-1 outcome it
        # would have reported before Tier 2 existed.
        return _no_resolution(
            f"No 0250 documents found in {docs_dir}.", STAGE_NOTHING
        )

    try:
        chunks = docs.retrieve(ticket)
    except AzureEmbeddingNotConfigured as exc:
        logger.error("Tier 2 embedding is not configured: %s", exc)
        return ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_BAD_INPUT, **measured)
    except AzureEmbeddingError as exc:
        # Tier 1 embedded successfully or the run would have ended already, so
        # Azure going down between the tiers is a genuine outage -- not a
        # corpus the robot should stop retrying.
        logger.error("Tier 2 embedding failed: %s", exc)
        return ResolveOutcome(
            CODE_INFRASTRUCTURE, str(exc), EXIT_INFRASTRUCTURE, **measured
        )
    except Exception as exc:
        # A corpus that cannot be PARSED is different: Tier 2 is a fallback, and
        # an unreadable document must not turn a legitimate Tier-1 "no
        # resolution" into a fault the robot retries forever.
        logger.error("Tier 2 failed; reporting the Tier-1 outcome: %s", exc)
        return _no_resolution(f"Tier 2 unavailable: {exc}", STAGE_NOTHING)

    tier2_stats = docs.stats
    logger.info("tier 2 stats: %s", tier2_stats.as_dict())

    # Tier 1 can stop before encoding anything -- an unknown part scans the
    # history, matches no row and never reaches the model. If Tier 2 then
    # encoded, the status sheet must name the encoder that actually ran, or a
    # run where only Tier 2 embedded would report no encoder at all.
    #
    # Rebinding rather than mutating, and _no_resolution reads `measured` from
    # this scope when it is called, so every later return picks this up.
    if not measured.get("embedding_model") and tier2_stats.backend:
        measured = dict(
            measured,
            embedding_model=f"{tier2_stats.backend}:{tier2_stats.backend_detail}",
        )

    tier2_measured = dict(
        tier2_top_score=tier2_stats.top_score,
        tier2_threshold=tier2_stats.threshold_used,
        tier2_chunks=tier2_stats.chunks,
        tier2_cache_state=tier2_stats.cache_state,
    )

    # Only when Tier 1 found nothing at all. A part's own history outranks a
    # general standard as the thing to show a reviewer, so a Tier-1 near-miss
    # is never displaced by a Tier-2 one.
    if tier2_stats.top_chunks and not measured.get("closest_match"):
        tier2_measured["closest_match"] = _closest_match(
            [(c.citation, c.chunk.text, c.score) for c in tier2_stats.top_chunks],
            gated=not chunks,
        )

    if not chunks:
        if not tier2_stats.chunks:
            # Documents were present but yielded no usable text at all -- every
            # one unparseable, or every section too short to embed. Nothing was
            # scored, so nothing was gated.
            return _no_resolution(
                f"No usable text in {tier2_stats.documents} 0250 document(s).",
                STAGE_NOTHING,
                **tier2_measured,
            )
        return _no_resolution(
            f"Best 0250 match {tier2_stats.top_score:.4f} is below the "
            f"{tier2_stats.threshold_used:.2f} threshold across "
            f"{tier2_stats.chunks} chunk(s).",
            STAGE_GATED,
            stop_reason=STOP_BELOW_THRESHOLD,
            **tier2_measured,
        )

    tier2_measured["tier2_evidence"] = tuple(
        _evidence(c.citation, c.chunk.text) for c in chunks
    )

    outcome = asyncio.run(loop.run_grounded(ticket, documentation_grounding(chunks)))

    if not outcome.succeeded or outcome.draft is None:
        if outcome.infrastructure_failure:
            logger.error("dependency failure in Tier 2: %s", outcome.failure_reason)
            return ResolveOutcome(
                CODE_INFRASTRUCTURE,
                outcome.failure_reason,
                EXIT_INFRASTRUCTURE,
                # Merged, not double-splatted: both can carry closest_match
                # when Tier 1 found no history, and two ** expansions with the
                # same key are a TypeError rather than an override.
                **{**measured, **tier2_measured},
            )
        return _no_resolution(
            f"{outcome.failure_reason} Best 0250 match "
            f"{tier2_stats.top_score:.4f}.",
            STAGE_REJECTED,
            stop_reason=outcome.stop_reason,
            **tier2_measured,
        )


    result = success_from_docs(
        recommendation=outcome.draft.recommendation,
        justification=outcome.draft.justification
        or f"Derived from {len(chunks)} 0250 standard section(s).",
        top_score=tier2_stats.top_score,
        chunks=chunks,
        closest_matching_solution=_source_list([(c.citation, c.score) for c in chunks]),
    )
    return ResolveOutcome(
        CODE_SUCCESS_DOC,
        f"{tier1_detail} Resolved instead from {len(chunks)} 0250 section(s) at "
        f"{result.confidence} confidence: {', '.join(result.referenced_sources)}.",
        EXIT_OK,
        resolution_source=result.resolution_source,
        results=(result,),
        # Merged for the same reason as the infrastructure path above. A part
        # with no history whose standards lookup SUCCEEDED crashed here with
        # "got multiple values for keyword argument 'closest_match'" -- and
        # reported it as an infrastructure fault, so the robot retried a
        # ticket that would fail identically every time.
        **{**measured, **tier2_measured},
    )


def resolve_main(argv: list[str] | None = None) -> int:
    args = resolve_parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    load_env_file()

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Clear both before any work: a process killed mid-run must leave no
        # stale workbook that reads as this run's answer.
        for name in (STATUS_FILE, OUTPUT_FILE):
            (output_dir / name).unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Cannot prepare output directory %s: %s", output_dir, exc)
        return EXIT_INFRASTRUCTURE

    try:
        outcome = resolve(args, output_dir)
    except SettingError as exc:
        logger.error("%s", exc)
        outcome = ResolveOutcome(CODE_INFRASTRUCTURE, str(exc), EXIT_BAD_INPUT)
    except Exception:
        # status.xlsx is written even here: an unhandled fault must still leave
        # the caller a row explaining why, not an empty directory.
        logger.error("Unhandled error:\n%s", traceback.format_exc())
        outcome = ResolveOutcome(
            CODE_INFRASTRUCTURE,
            "Unhandled error; see stderr for the traceback.",
            EXIT_INFRASTRUCTURE,
        )

    # A run that reached a conclusion always leaves a result row, even when the
    # conclusion is "no solution": the caller merges output.xlsx into its own
    # records, and a missing row cannot be told apart from a ticket that was
    # never processed.
    #
    # An infrastructure fault writes none. It reached no conclusion, the robot
    # is expected to retry it, and recording "Solution not found." for an Azure
    # outage would enter a verdict the pipeline never actually formed.
    if outcome.exit_code == EXIT_OK:
        try:
            write_output(
                output_dir,
                outcome.part_number,
                outcome.results or (unresolved_result(outcome),),
            )
        except Exception:
            logger.error("Could not write %s:\n%s", OUTPUT_FILE, traceback.format_exc())
            # Leave no half-written workbook behind to be merged as a result.
            (output_dir / OUTPUT_FILE).unlink(missing_ok=True)
            outcome = replace(
                outcome,
                code=CODE_INFRASTRUCTURE,
                reason=f"Could not write {OUTPUT_FILE}; see stderr for the traceback.",
                exit_code=EXIT_INFRASTRUCTURE,
            )

    try:
        write_status(output_dir, outcome.code, outcome.reason, outcome.embedding_model)
    except Exception:
        logger.error("Could not write %s:\n%s", STATUS_FILE, traceback.format_exc())
        return EXIT_INFRASTRUCTURE
    return outcome.exit_code


# =============================================================================
# Command: bulk
# =============================================================================
#
# Run a workbook of tickets through the real pipeline, one row at a time.
#
#     python sps.py bulk --tickets tickets.xlsx --history history.csv
#
# Reads a sheet where each row is a ticket, resolves every one, and writes a copy
# carrying **every original column verbatim plus the result columns**. The input
# is never modified: it stays a clean, re-runnable fixture, and two runs can be
# diffed against each other.
#
# This is the review tool. `sps.py eval` is the calibration tool -- it takes
# a directory of one-ticket-per-file cases and reports scores, not prose. Use this
# one to read what the system actually recommended; use that one to decide where a
# threshold belongs.
#
# Nothing here reimplements the pipeline. Each row goes through the same
# `resolve()` the UiPath wrapper calls, so a result in this sheet is the result
# production would have produced for that ticket.
#
# Each row goes through `resolve()` exactly as production does, history store
# included: the first row syncs the store if the export changed (or `ingest`
# already has), and every later row reads only its family from it.

# The same cell renderer the ticket contracts use. Shared rather than copied:
# it is what stops an Excel-sourced part number of 1243951.0 reaching the
# pipeline as "1243951.0", an id that matches nothing, and a second copy of
# that rule is a second place for the fix to be lost.
clean_cell = _clean



# Appended to the right of whatever the input already had. Order is
# reading order: the verdict, then what a reviewer judges, then the numbers
# that say how close a rejected row came.
BULK_RESULT_COLUMNS = (
    "Status",
    "Status_Code",
    "Reason",
    # A ticket gets one row per solution, up to MAX_SOLUTIONS, each repeating
    # the ticket's own columns so the sheet still filters and sorts as one
    # table. 1 is the best; blank on a ticket with no recommendation.
    "Solution_Rank",
    "AI_Recommendation",
    "Justification",
    "Confidence_Score",
    "Referenced_Sources",
    "Resolution_Source",
    "Tier1_Score",
    "Tier2_Score",
    # Why this row produced nothing, as one of a fixed set of values rather
    # than as prose. Reason says it in English already; this is the column you
    # can group by, which is what turns "your checks are too strict" from an
    # argument into a count.
    "No_Recommendation_Reason",
    # The single best precedent, gate or no gate, exactly as output.xlsx shows
    # it to the reviewer.
    "Closest_Matching_Solution",
    # What the reviewer should read before pasting, when the recommendation is
    # archive text sent unchanged.
    "Cascade_Warnings",
    # The shortlisted historical solutions, or the 0250 sections the Tier-2
    # Actor was shown.
    # Put beside its recommendation because the commonest question about a bad
    # answer is "what was it looking at?", and answering it from the SPS IDs
    # alone means a lookup per row.
    "Matched_Solutions",
    "Embedding_Model",
    "Duration_Seconds",
)

# A row that was never attempted, so the output still lines up with the input.
NOT_RUN = "NOT RUN"

DEFAULT_STOP_AFTER = 5


def read_tickets(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (column names, rows) with every column preserved verbatim."""

    header, raw_rows = read_header_and_rows(path, "Ticket workbook")
    columns = [str(h).strip() for h in header if str(h or "").strip()]
    if not columns:
        raise UnsupportedFileType(f"{path.name!r} has no column headings.")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if is_blank(raw):
            continue
        rows.append(
            {
                name: clean_cell(raw[i]) if i < len(raw) else ""
                for i, name in enumerate(columns)
            }
        )
    return columns, rows


def resolve_columns(original: Sequence[str]) -> dict[str, str]:
    """Map each result column to the name it will be written under.

    A ticket sheet that already has a `Status` column would otherwise produce
    two columns of that name, and whichever pandas kept would be anyone's
    guess. Collisions get an `_AI` suffix instead.
    """
    taken = {c.casefold() for c in original}
    mapping: dict[str, str] = {}
    for name in BULK_RESULT_COLUMNS:
        final = name if name.casefold() not in taken else f"{name}_AI"
        mapping[name] = final
        taken.add(final.casefold())
    return mapping


def write_ticket_file(row: dict[str, Any], columns: Sequence[str], path: Path) -> Path:
    """One row as a single-ticket CSV, which is what resolve() consumes.

    Every column goes in, not just the four the pipeline reads: whatever
    spelling of Problem_Description or Issue_Type this sheet uses, the ticket
    contract resolves it case-insensitively at the other end.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow([row.get(c, "") for c in columns])
    return path


def result_for(outcome, duration: float) -> list[dict[str, str]]:
    """Flatten one ResolveOutcome into the appended columns: a row per solution."""

    results: tuple = outcome.results
    if not results and outcome.exit_code == EXIT_OK:
        # The run concluded without a recommendation. This is the same row
        # main() writes into output.xlsx for that case, reused rather than
        # rebuilt so the bulk sheet and the per-ticket workbook cannot
        # disagree about what a refusal looks like.
        #
        # Deliberately not done for exit 1 or 2: no verdict was formed there,
        # and "Solution not found." would record one that never existed.
        results = (unresolved_result(outcome),)

    passed = outcome.code in SUCCESS_CODES
    shared = {
        "Status": "PASS" if passed else "FAIL",
        "Status_Code": outcome.code,
        "No_Recommendation_Reason": "" if passed else (outcome.stop_reason or outcome.code),
        "Closest_Matching_Solution": outcome.closest_match,
        "Reason": " ".join(str(outcome.reason).split()),
        # Blank rather than 0 when nothing was scored: an unknown part never
        # reaches the encoder, and 0 reads as a match that scored badly.
        "Tier1_Score": round(outcome.top_score, 4) if outcome.top_score else "",
        "Tier2_Score": round(outcome.tier2_top_score, 4) if outcome.tier2_top_score else "",
        "Matched_Solutions": "\n\n".join(outcome.evidence + outcome.tier2_evidence),
        "Embedding_Model": outcome.embedding_model,
        "Duration_Seconds": round(duration, 2),
    }
    if not results:
        return [{**shared, "Solution_Rank": "", "Cascade_Warnings": "", "AI_Recommendation": "",
                 "Justification": "", "Confidence_Score": "", "Referenced_Sources": "",
                 "Resolution_Source": ""}]
    return [
        {
            **shared,
            "Solution_Rank": solution_rank(result, position),
            "Cascade_Warnings": result.cascade_warnings,
            "AI_Recommendation": result.ai_recommendation,
            "Justification": result.justification,
            "Confidence_Score": result.confidence,
            "Referenced_Sources": ", ".join(result.referenced_sources),
            "Resolution_Source": result.resolution_source,
        }
        for position, result in enumerate(results, 1)
    ]


def blank_result(reason: str) -> dict[str, str]:
    """The appended columns for a row that was never attempted."""
    values = {name: "" for name in BULK_RESULT_COLUMNS}
    values["Status"] = NOT_RUN
    values["Reason"] = reason
    return values


def run_row(row: dict[str, Any], columns: Sequence[str], work: Path, args) -> list[dict[str, str]]:
    """Resolve one ticket through the real pipeline: its rows, one per solution."""

    started = time.time()
    ticket_file = write_ticket_file(row, columns, work / "ticket.csv")

    argv = [
        "--ticket-file", str(ticket_file),
        "--history-file", str(args.history),
        "--output-dir", str(work / "run"),
    ]
    if args.threshold is not None:
        argv += ["--threshold", str(args.threshold)]
    if args.tier2_threshold is not None:
        argv += ["--tier2-threshold", str(args.tier2_threshold)]
    if args.docs_dir:
        argv += ["--docs-dir", str(args.docs_dir)]
    if args.no_tier2:
        argv.append("--no-tier2")
    if args.commodity_file:
        argv += ["--commodity-file", str(args.commodity_file)]

    try:
        outcome = resolve(resolve_parse_args(argv), work / "run")
    except Exception as exc:
        # One bad row must not end a run that may have taken an hour.
        logger.error("row raised:\n%s", traceback.format_exc())
        values = blank_result(f"{type(exc).__name__}: {exc}")
        values["Status"] = "FAIL"
        values["Status_Code"] = "INFRASTRUCTURE_ERROR"
        values["Duration_Seconds"] = round(time.time() - started, 2)
        return [values]

    return result_for(outcome, time.time() - started)


def summarise_bulk(results: Sequence[dict[str, str]]) -> str:
    """Counts per TICKET: pass each ticket's first row, not every solution row."""
    from collections import Counter

    lines = [f"\n{len(results)} row(s)"]
    counts = Counter(r.get("Status_Code") or r.get("Status") or "?" for r in results)
    lines.append("  by status:")
    for key, count in counts.most_common():
        lines.append(f"    {count:5d}  {key}")

    recommended = sum(1 for r in results if r.get("Status") == "PASS")
    lines.append(f"  {recommended} of {len(results)} produced a recommendation")

    # The breakdown that answers "why not more?". Status_Code above says how far
    # the pipeline got; this says what stopped it, which is a different question
    # and the one worth arguing from. GUARD/JUDGE rows mean our checks are the
    # binding constraint; the rest mean the archive is.
    reasons = Counter(
        r["No_Recommendation_Reason"]
        for r in results
        if str(r.get("No_Recommendation_Reason", "")) != ""
    )
    if reasons:
        blocked = sum(v for k, v in reasons.items() if k.startswith(("GATE_", "JUDGE_")))
        lines.append("  of those without one, why:")
        for key, count in reasons.most_common():
            lines.append(f"    {count:5d}  {key}")
        total = sum(reasons.values())
        lines.append(
            f"    -> {blocked} of {total} were stopped by a supplier-safety check; "
            f"{total - blocked} by the data."
        )

    scored = sorted(
        float(r["Tier1_Score"]) for r in results if str(r.get("Tier1_Score", "")) != ""
    )
    if scored:
        def pct(fraction: float) -> float:
            return scored[min(int(fraction * (len(scored) - 1)), len(scored) - 1)]

        lines.append(f"  Tier-1 score over {len(scored)} scored row(s):")
        lines.append(f"    min {scored[0]:.4f}   median {pct(0.5):.4f}   max {scored[-1]:.4f}")
        lines.append(
            "  Sort the sheet by Tier1_Score and read where the good answers stop: "
            "that is where the threshold belongs."
        )
    return "\n".join(lines)


def bulk_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python sps.py bulk",
        description="Resolve a workbook of tickets and write a copy with the results appended.",
    )
    parser.add_argument("--tickets", required=True, help="Sheet of tickets, one per row")
    parser.add_argument("--history", required=True, help="Historical records (.csv preferred)")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the results copy (default: <tickets>_results.xlsx beside the input). "
        "The input is never modified.",
    )
    parser.add_argument("--docs-dir", default=None, help="0250 standards folder for Tier 2")
    parser.add_argument("--no-tier2", action="store_true", help="History only")
    parser.add_argument("--threshold", type=float, default=None, help="Tier-1 gate override")
    parser.add_argument("--tier2-threshold", type=float, default=None, help="Tier-2 gate override")
    parser.add_argument(
        "--commodity-file", default=None,
        help=f"Commodity groups .csv/.xlsx, or {COMMODITY_GROUPS_ENV}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Resolve only the first N rows. The rest still appear, marked NOT RUN, "
        "so the sheet stays aligned with the input.",
    )
    parser.add_argument(
        "--stop-after-errors", type=int, default=DEFAULT_STOP_AFTER,
        help=f"Give up after this many CONSECUTIVE infrastructure errors "
        f"(default {DEFAULT_STOP_AFTER}; 0 disables). Wrong credentials would "
        "otherwise burn one failing call per row for the whole sheet.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def default_output(tickets: Path) -> Path:
    return tickets.with_name(f"{tickets.stem}_results.xlsx")


def bulk_main(argv: list[str] | None = None) -> int:
    args = bulk_parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    load_env_file()

    tickets_path = Path(args.tickets)
    history_path = Path(args.history)
    try:
        validate_file_type(tickets_path, "Ticket workbook")
        validate_file_type(history_path, "History file")
        for label, path in (("Ticket workbook", tickets_path), ("History file", history_path)):
            if not path.exists():
                print(f"{label} not found: {path}", file=sys.stderr)
                return EXIT_BAD_INPUT
        columns, rows = read_tickets(tickets_path)
    except (UnsupportedFileType, OSError, ValueError) as exc:
        print(f"Could not read the inputs: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if not rows:
        print(f"No ticket rows in {tickets_path}", file=sys.stderr)
        return EXIT_BAD_INPUT

    naming = resolve_columns(columns)
    clashed = [n for n, final in naming.items() if n != final]
    if clashed:
        print(
            f"{len(clashed)} result column(s) clash with the input and were suffixed "
            f"'_AI': {', '.join(clashed)}",
            file=sys.stderr,
        )

    output_path = Path(args.output) if args.output else default_output(tickets_path)
    started = time.time()
    # Per ticket: its rows, one per solution.
    results: list[list[dict[str, str]]] = []
    consecutive = 0
    stopped = ""

    with tempfile.TemporaryDirectory(prefix="sps_bulk_") as tmp:
        work = Path(tmp)
        for position, row in enumerate(rows, 1):
            if stopped:
                results.append([blank_result(stopped)])
                continue
            if args.limit is not None and position > args.limit:
                results.append([blank_result(f"Beyond --limit {args.limit}.")])
                continue

            print(f"[{position}/{len(rows)}]", file=sys.stderr, flush=True)
            ticket_rows = run_row(row, columns, work, args)
            results.append(ticket_rows)

            if ticket_rows[0].get("Status_Code") == "INFRASTRUCTURE_ERROR":
                consecutive += 1
                if args.stop_after_errors and consecutive >= args.stop_after_errors:
                    stopped = (
                        f"Stopped: {consecutive} consecutive infrastructure errors. "
                        "Check the Reason on the rows above."
                    )
                    print(f"\n{stopped}", file=sys.stderr)
            else:
                consecutive = 0

    # Original columns first, verbatim, then the appended ones -- so the sheet
    # reads as the input with answers alongside it.
    out_columns = list(columns) + [naming[n] for n in BULK_RESULT_COLUMNS]
    merged = [
        {**row, **{naming[n]: values.get(n, "") for n in BULK_RESULT_COLUMNS}}
        for row, ticket_rows in zip(rows, results)
        for values in ticket_rows
    ]
    firsts = [ticket_rows[0] for ticket_rows in results]


    try:
        write_rows(output_path, out_columns, merged)
    except Exception:
        logger.error("Could not write %s:\n%s", output_path, traceback.format_exc())
        return EXIT_INFRASTRUCTURE

    print(summarise_bulk(firsts), file=sys.stderr)
    print(
        f"\n{len(rows)} row(s) in {time.time() - started:.1f}s -> {output_path}",
        file=sys.stderr,
    )

    broken = sum(1 for r in firsts if r.get("Status_Code") == "INFRASTRUCTURE_ERROR")
    if broken:
        print(f"{broken} row(s) hit an infrastructure error.", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    return EXIT_OK


# =============================================================================
# Command: eval
# =============================================================================
#
# Run many tickets in one process and write a single results workbook.
#
# Built for threshold calibration, which needs the score distribution rather than
# pass/fail: **every row carries the raw top cosine, including the cases the gate
# rejected.** A run that only reported successes would show you the scores above
# the threshold and hide exactly the ones you need to decide whether the threshold
# is right.
#
#     python sps.py eval --test-dir test_cases
#     python sps.py eval --test-dir test_cases --history-file master.csv
#     python sps.py eval --run-list runs.csv --output-dir results
#
# Case discovery, in order of precedence:
#
#   --run-list      a .csv/.xlsx with columns Test_ID, Ticket_File, History_File
#                   (History_File optional if --history-file is given). Paths are
#                   resolved relative to the list's own directory.
#   --test-dir      files named <id>_ticket.<ext> paired with <id>_history.<ext>.
#                   A case with no history of its own uses --history-file, which
#                   is the usual shape: fifty tickets against one extract.
#
# One process for the whole batch, so a local-model cold start is paid once rather
# than fifty times. One failing case never stops the run: it becomes a row saying
# what went wrong.

RESULTS_FILE = "eval_results.xlsx"

EVAL_RESULT_COLUMNS = (
    "Test_ID",
    "Status",
    "Status_Code",
    "Reason",
    "Embedding_Model",
    "Confidence_Score",
    # Beyond the five requested, and all of them earn their place when the point
    # is calibration rather than a pass/fail report.
    "Threshold_Applied",
    "Cleared_Threshold",
    "Candidates_Considered",
    # The score that actually decides a Tier-1 answer, and its gate. The cosine
    # above only shortlists; calibrating SPS_INTENT_THRESHOLD needs these, read
    # against your own judgement of which cases should have matched.
    "Intent_Score",
    "Intent_Threshold",
    "Solutions_Sent",
    # Tier 2 is calibrated the same way and needs the same columns. They are
    # kept separate from the Tier-1 ones rather than reusing them: the two
    # tiers score in different ranges, so one pooled column could not be
    # calibrated against anything.
    "Resolution_Source",
    "Tier2_Score",
    "Tier2_Threshold",
    "Tier2_Chunks",
    "Tier2_Cache",
    "Duration_Seconds",
    "Ticket_File",
    "History_File",
)

TICKET_MARKER = "_ticket"
HISTORY_MARKER = "_history"


@dataclass(frozen=True, slots=True)
class Case:
    test_id: str
    ticket: Path
    # None, not Path(""), for "no history was found". Path("") is Path("."),
    # which is truthy and would sail past the guard in run_case to fail later
    # with a confusing message about "." having no extension.
    history: Path | None


@dataclass
class Row:
    """One line of eval_results.xlsx."""

    test_id: str
    status: str = "FAIL"
    status_code: str = ""
    reason: str = ""
    embedding_model: str = ""
    confidence_score: float | str = ""
    threshold_applied: float | str = ""
    cleared: str = ""
    candidates: int | str = ""
    intent_score: float | str = ""
    intent_threshold: float | str = ""
    solutions: int | str = ""
    resolution_source: str = ""
    tier2_score: float | str = ""
    tier2_threshold: float | str = ""
    tier2_chunks: int | str = ""
    tier2_cache: str = ""
    duration: float = 0.0
    ticket: str = ""
    history: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "Test_ID": self.test_id,
            "Status": self.status,
            "Status_Code": self.status_code,
            "Reason": self.reason,
            "Embedding_Model": self.embedding_model,
            "Confidence_Score": self.confidence_score,
            "Threshold_Applied": self.threshold_applied,
            "Cleared_Threshold": self.cleared,
            "Candidates_Considered": self.candidates,
            "Intent_Score": self.intent_score,
            "Intent_Threshold": self.intent_threshold,
            "Solutions_Sent": self.solutions,
            "Resolution_Source": self.resolution_source,
            "Tier2_Score": self.tier2_score,
            "Tier2_Threshold": self.tier2_threshold,
            "Tier2_Chunks": self.tier2_chunks,
            "Tier2_Cache": self.tier2_cache,
            "Duration_Seconds": round(self.duration, 2),
            "Ticket_File": self.ticket,
            "History_File": self.history,
        }


# --------------------------------------------------------------- discovery


def discover_from_dir(test_dir: Path, shared_history: Path | None) -> list[Case]:
    """Pair <id>_ticket.<ext> with <id>_history.<ext>.

    A ticket whose history is missing and with no --history-file to fall back on
    is left out here and reported as an error row, rather than silently skipped:
    a case that never ran is not the same as a case that failed.
    """
    tickets: dict[str, Path] = {}
    histories: dict[str, Path] = {}

    for path in sorted(test_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stem = path.stem
        lowered = stem.lower()
        if lowered.endswith(TICKET_MARKER):
            tickets[stem[: -len(TICKET_MARKER)]] = path
        elif lowered.endswith(HISTORY_MARKER):
            histories[stem[: -len(HISTORY_MARKER)]] = path

    cases = []
    for test_id, ticket in sorted(tickets.items()):
        history = histories.get(test_id) or shared_history
        cases.append(Case(test_id, ticket, Path(history) if history else None))

    orphan_histories = sorted(set(histories) - set(tickets))
    if orphan_histories:
        logger.warning(
            "%d history file(s) have no matching ticket and were ignored: %s",
            len(orphan_histories),
            ", ".join(orphan_histories),
        )
    return cases


def discover_from_list(run_list: Path, shared_history: Path | None) -> list[Case]:
    """Read Test_ID / Ticket_File / History_File from a run list.

    Paths are resolved relative to the list itself, so the whole test set stays
    portable between machines -- which matters when it is assembled on one
    laptop and run on another.
    """

    header, rows = read_header_and_rows(run_list, "Run list")
    index = {str(h).strip().casefold().replace(" ", "_"): i for i, h in enumerate(header) if h}
    for required in ("test_id", "ticket_file"):
        if required not in index:
            raise UnsupportedFileType(
                f"Run list must have a {required!r} column. Found: "
                + ", ".join(repr(str(h)) for h in header if str(h).strip())
            )

    base = run_list.parent
    cases = []
    for row in rows:
        if is_blank(row):
            continue

        def cell(name: str) -> str:
            position = index.get(name)
            if position is None or position >= len(row):
                return ""
            return str(row[position] or "").strip()

        history_cell = cell("history_file")
        history = (base / history_cell) if history_cell else shared_history
        cases.append(
            Case(
                test_id=cell("test_id") or f"row{len(cases) + 1}",
                ticket=base / cell("ticket_file"),
                history=Path(history) if history else None,
            )
        )
    return cases


# ------------------------------------------------------------------ running


def run_case(case: Case, work_dir: Path, threshold: float | None,
             commodity_file: str | None = None) -> Row:
    """Run one ticket through the real resolver path."""

    row = Row(
        test_id=case.test_id,
        ticket=case.ticket.name,
        history=case.history.name if case.history else "",
    )
    started = time.time()

    if case.history is None:
        row.status_code = "INVALID_INPUT"
        row.reason = (
            f"No history file for case {case.test_id!r}: expected "
            f"{case.test_id}{HISTORY_MARKER}.csv/.xlsx beside the ticket, "
            f"or pass --history-file."
        )
        row.duration = time.time() - started
        return row

    argv = [
        "--ticket-file", str(case.ticket),
        "--history-file", str(case.history),
        "--output-dir", str(work_dir / case.test_id),
    ]
    if threshold is not None:
        argv += ["--threshold", str(threshold)]
    if commodity_file:
        argv += ["--commodity-file", str(commodity_file)]

    try:
        outcome = resolve(resolve_parse_args(argv), work_dir / case.test_id)
    except Exception as exc:
        # One bad case must not end the batch.
        logger.error("case %s raised:\n%s", case.test_id, traceback.format_exc())
        row.status_code = "INFRASTRUCTURE_ERROR"
        row.reason = f"{type(exc).__name__}: {exc}"
        row.duration = time.time() - started
        return row

    row.status = "PASS" if outcome.code in SUCCESS_CODES else "FAIL"
    row.status_code = outcome.code
    row.reason = " ".join(outcome.reason.split())
    row.embedding_model = outcome.embedding_model
    row.duration = time.time() - started

    # Populated whenever anything was scored, which is the point: the rejected
    # cases are the ones that tell you whether the threshold is set correctly.
    if outcome.embedding_model:
        row.confidence_score = round(outcome.top_score, 4)
        row.threshold_applied = round(outcome.threshold_used, 4)
        row.cleared = "YES" if outcome.top_score >= outcome.threshold_used else "NO"
        row.candidates = outcome.candidates_considered
    # Blank unless the scorer ran: a case gated on its cosine was never judged.
    if outcome.ticket_intent:
        row.intent_score = round(outcome.intent_score, 4)
        row.intent_threshold = round(outcome.intent_threshold, 4)
    if outcome.code == CODE_SUCCESS_HISTORICAL:
        row.solutions = len(outcome.results)

    row.resolution_source = outcome.resolution_source
    row.tier2_cache = outcome.tier2_cache_state
    # Populated only when Tier 2 actually ran, so a blank means "Tier 1
    # answered" rather than "Tier 2 scored zero".
    if outcome.tier2_cache_state and outcome.tier2_cache_state != "absent":
        row.tier2_score = round(outcome.tier2_top_score, 4)
        row.tier2_threshold = round(outcome.tier2_threshold, 4)
        row.tier2_chunks = outcome.tier2_chunks
    return row


def summarise_eval(rows: list[Row]) -> str:
    """A console digest, so the distribution is visible without opening Excel."""
    from collections import Counter

    lines = [f"\n{len(rows)} case(s)"]
    for label, counts in (
        ("status", Counter(r.status_code or "ERROR" for r in rows)),
        ("encoder", Counter(r.embedding_model or "(none)" for r in rows)),
    ):
        lines.append(f"  by {label}:")
        for key, count in counts.most_common():
            lines.append(f"    {count:4d}  {key}")

    scored = sorted(float(r.confidence_score) for r in rows if r.confidence_score != "")
    if scored:
        def pct(fraction: float) -> float:
            return scored[min(int(fraction * (len(scored) - 1)), len(scored) - 1)]

        lines.append(f"  score distribution over {len(scored)} scored case(s):")
        lines.append(f"    min {scored[0]:.4f}   p25 {pct(0.25):.4f}   median {pct(0.5):.4f}")
        lines.append(f"    p75 {pct(0.75):.4f}   p90 {pct(0.90):.4f}   max {scored[-1]:.4f}")
        lines.append(
            "  Use these to set the threshold: it should sit above the scores of "
            "cases you judged wrong and below the ones you judged right."
        )

    # A threshold belongs to one embedding space. If cases were scored by
    # different deployments, the percentiles above pool two distributions and
    # describe neither, and nothing else in the run would flag it.
    encoders = {r.embedding_model for r in rows if r.embedding_model}
    if len(encoders) > 1:
        lines.append(
            f"  WARNING: {len(encoders)} encoders ran in this batch ("
            + ", ".join(sorted(encoders))
            + "). The scores above are pooled across different embedding spaces, "
            "so the distribution describes neither. Get the encoder consistent "
            "and re-run before setting a threshold from it."
        )
    return "\n".join(lines)


def eval_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python sps.py eval",
        description="Run a directory of test tickets and write eval_results.xlsx.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-dir", help="Directory of <id>_ticket.* / <id>_history.* pairs")
    source.add_argument("--run-list", help="A .csv/.xlsx of Test_ID, Ticket_File, History_File")
    parser.add_argument(
        "--history-file",
        default=None,
        help="History used by any case without one of its own -- the usual shape, "
        "where many tickets share a single extract.",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Where eval_results.xlsx is written (default: cwd)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the gate for every case. Leave unset to use the per-encoder default; "
        "the raw score is reported either way, so the threshold does not affect calibration.",
    )
    parser.add_argument(
        "--commodity-file", default=None,
        help=f"Commodity groups .csv/.xlsx, or {COMMODITY_GROUPS_ENV}",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def eval_main(argv: list[str] | None = None) -> int:
    args = eval_parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # The batch is the run that calibrates the Azure threshold, so it has to
    # see the same credentials the resolver would. Without this every case
    # fails to embed and the batch measures nothing.
    load_env_file()

    output_dir = Path(args.output_dir)
    shared_history = Path(args.history_file) if args.history_file else None

    try:
        if shared_history is not None:
            validate_file_type(shared_history, "Shared history file")
        if args.run_list:
            cases = discover_from_list(validate_file_type(Path(args.run_list), "Run list"),
                                       shared_history)
        else:
            test_dir = Path(args.test_dir)
            if not test_dir.is_dir():
                print(f"Not a directory: {test_dir}", file=sys.stderr)
                return EXIT_BAD_INPUT
            cases = discover_from_dir(test_dir, shared_history)
    except (UnsupportedFileType, OSError, ValueError) as exc:
        print(f"Could not build the case list: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if not cases:
        print(
            "No cases found. Expected files named <id>_ticket.csv/.xlsx "
            "(optionally paired with <id>_history.csv/.xlsx).",
            file=sys.stderr,
        )
        return EXIT_BAD_INPUT

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_runs"

    rows: list[Row] = []
    started = time.time()
    for position, case in enumerate(cases, 1):
        print(f"[{position}/{len(cases)}] {case.test_id}", file=sys.stderr, flush=True)
        rows.append(run_case(case, work_dir, args.threshold, args.commodity_file))


    results_path = output_dir / RESULTS_FILE
    write_rows(results_path, EVAL_RESULT_COLUMNS, [r.as_dict() for r in rows])

    print(summarise_eval(rows), file=sys.stderr)
    print(f"\n{len(rows)} case(s) in {time.time() - started:.1f}s -> {results_path}",
          file=sys.stderr)

    # Non-zero only if something was genuinely broken. A case that legitimately
    # failed its gate is a result, not an error.
    broken = sum(1 for r in rows if r.status_code == "INFRASTRUCTURE_ERROR")
    if broken:
        print(f"{broken} case(s) hit an infrastructure error.", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    return EXIT_OK


# =============================================================================
# Command: ingest
# =============================================================================
#
# Bring the history store up to date ahead of the robots:
#
#     python sps.py ingest --history history.csv
#
# Reads the export once if it changed, then embeds every usable record the store
# has not seen, in requests of up to 2,048, committing as it goes -- so an
# interrupted ingest resumes where it stopped. Run it after each export.
#
# A ticket syncs the store and embeds its own family on demand anyway, so a
# forgotten ingest is slow rather than broken. But at family scale the first
# ticket after an export can then spend minutes embedding, past what a robot
# is willing to wait.
#
# Exit 0 done, 1 Azure or the store failed (retry), 2 the file or the
# configuration is wrong (a person has to fix it).


def ingest_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python sps.py ingest",
        description="Bring the history store up to date with an export, ahead of the robots.",
    )
    parser.add_argument("--history", required=True, help="Historical records (.csv preferred)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def ingest_main(argv: list[str] | None = None) -> int:
    args = ingest_parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    load_env_file()

    path = Path(args.history)
    try:
        validate_file_type(path, "History file")
    except UnsupportedFileType as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT
    if not path.is_file():
        print(f"History file not found: {path}", file=sys.stderr)
        return EXIT_BAD_INPUT
    settings = AzureEmbeddingSettings.from_env()
    if not settings.configured:
        print("Azure embeddings are not configured: " + ", ".join(settings.missing()),
              file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        store = open_history_store(path)
    except HistoryStoreConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT
    if not isinstance(store, SqliteHistoryStore):
        print(f"Nothing to ingest into: the history store is {store.location}. "
              f"Set {HISTORY_STORE_ENV}=sqlite and {HISTORY_STORE_DIR_ENV} to a writable "
              "local folder.", file=sys.stderr)
        return EXIT_BAD_INPUT

    encoder = azure_encoder(settings)
    started = time.time()
    try:
        synced = store.sync(path)
        dropped = store.drop_other_models(encoder.identity)
        missing, distinct = store.missing_texts(encoder)
        keys = list(missing)
        for start in range(0, len(keys), AZURE_EMBEDDING_BATCH_LIMIT):
            chunk = keys[start:start + AZURE_EMBEDDING_BATCH_LIMIT]
            store.remember(encoder, chunk, encoder.passages([missing[k] for k in chunk]))
            print(f"  embedded {start + len(chunk)} of {len(keys)}", file=sys.stderr, flush=True)
    except HistoryError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT
    except AzureEmbeddingNotConfigured as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT
    except (AzureEmbeddingError, HistoryStoreError, sqlite3.Error) as exc:
        print(f"Ingest stopped, and will resume from here when re-run: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    finally:
        store.close()

    print(f"History store: {store.location}")
    print(f"  rows:     {synced.rows}" + (" (re-read)" if synced.reread else " (file unchanged)"))
    print(f"  embedded: {len(keys)} new problem text(s); {distinct - len(keys)} already stored")
    if dropped:
        print(f"  removed:  {dropped} vector(s) from another embedding deployment")
    print(f"  in {time.time() - started:.1f}s via {encoder.identity}")
    return EXIT_OK


# =============================================================================
# Command: check
# =============================================================================
#
# Acceptance checks for the embedder the pipeline actually uses.
#
# Everything else in the suite runs against a deterministic stand-in. This script
# exercises a real encoder, to confirm the properties the ranking maths depends
# on and to print real cosine numbers against the live gate -- which is the
# fastest way to find out whether a threshold is anywhere near right on a new
# deployment.
#
#     python sps.py check                # the Azure deployment
#     python sps.py check --quiet
#     python sps.py check --probe        # the calibration sweep
#
# AZURE (the default, and the only encoder in the pipeline) checks that the
# deployment is reachable, that its vectors are unit length so the NumPy matmul
# really is a cosine, that a batch comes back in the order it was sent, and then
# scores real SPS text against the 0.50 Tier-1 gate and a standards passage
# against the 0.35 Tier-2 gate. Both of those numbers are unmeasured guesses, so
# this is the first honest look at them.
#
# PROBE is for setting those numbers rather than sanity-checking them. It scores
# labelled pairs -- your judgement of which belong together -- and reports the
# gap between the weakest pair you called a match and the strongest you did not.
# That gap is the only thing a threshold can live in. If it is negative, no
# threshold works and the probe says so instead of suggesting one.

TOLERANCE = 1e-5

# Realistic SPS text: two paraphrases of one defect, plus an unrelated problem.
PROBLEM = "Bracket weld seam cracking observed during incoming inspection"
PARAPHRASE = "Cracks found in the weld seam of the mounting bracket at goods-in"
NEAR = "Weld porosity detected on the bracket joint during inspection"
UNRELATED = "Outer carton label misprinted on the shipment packaging"

# A Tier-2 comparison needs a standards passage, not another defect sentence:
# the whole reason Tier 2 has its own threshold is that a section answers a
# ticket without resembling it.
STANDARD_RELEVANT = (
    "[0250-Weld-Standards.docx \u00a7 4.2 Weld Seam Cracking] \n"
    "Cracking in a fillet or butt weld seam is cause for rejection of the "
    "affected part. On discovery at incoming inspection, segregate the affected "
    "lot and quarantine it pending disposition. The cracked seam shall be ground "
    "out to sound metal for the full length of the crack plus 25 mm beyond each "
    "visible end, then re-welded to the original joint profile and re-inspected "
    "by dye penetrant."
)
STANDARD_UNRELATED = (
    "[0250-Packaging-Standards.docx \u00a7 2.1 Carton Labelling] \n"
    "Each outer carton shall carry a label showing the part number, the "
    "revision, the quantity contained and the date of packing. Labels shall be "
    "printed at 300 dpi or better and shall remain legible after transit."
)
TIER2_QUERY = f"Issue Type: Quality | Defect: {PROBLEM}"


def l2(vector) -> float:
    return math.sqrt(sum(v * v for v in vector))


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b) -> float:
    na, nb = l2(a), l2(b)
    return dot(a, b) / (na * nb) if na and nb else 0.0


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
        if not passed:
            self.failures.append(name)
        return passed

    def note(self, name: str, detail: str = "") -> None:
        """A reading worth seeing that passes or fails nothing."""
        print(f"  [INFO] {name}" + (f"  --  {detail}" if detail else ""))


def verify_azure(checks: "Checks", quiet: bool) -> None:
    """Check the live Azure deployment, and read the gates against it."""

    settings = AzureEmbeddingSettings.from_env()
    if not checks.check(
        "Azure embedding deployment is configured",
        settings.configured,
        "missing " + ", ".join(settings.missing()) if not settings.configured else "",
    ):
        print("\n  Set them in .env, then re-run. Nothing below can be checked.")
        return

    print(f"  deployment: {settings.deployment}  endpoint: {settings.endpoint}")

    # ------------------------------------------------------------- 1. reachable
    print("\n1. Reachability and shape")
    corpus = {
        "exact": PROBLEM,
        "paraphrase": PARAPHRASE,
        "near": NEAR,
        "unrelated": UNRELATED,
    }
    started = time.time()
    try:
        vectors = AzureEmbedder(settings).embed_batch([PROBLEM] + list(corpus.values()))
    except AzureEmbeddingError as exc:
        checks.check("embedding request succeeds", False, str(exc))
        return
    elapsed = time.time() - started
    checks.check("embedding request succeeds", True, f"{len(vectors)} vectors in {elapsed:.2f}s")

    dimension = len(vectors[0])
    checks.check(
        "every vector has the same dimension",
        all(len(v) == dimension for v in vectors),
        f"{dimension} dimensions",
    )
    # Not asserted against a fixed number: the right value depends on the
    # deployment (3072 for text-embedding-3-large, 1536 for -small and ada-002).
    # What matters is that it is consistent and that the caller knows which.
    print(f"       {dimension} dimensions -- expected 3072 for text-embedding-3-large")

    # --------------------------------------------------------- 2. normalisation
    print("\n2. Normalisation")
    norms = [l2(v) for v in vectors]
    checks.check(
        "vectors are unit length",
        all(abs(n - 1.0) < TOLERANCE for n in norms),
        f"min {min(norms):.8f} max {max(norms):.8f}",
    )
    d, c = dot(vectors[1], vectors[2]), cosine(vectors[1], vectors[2])
    checks.check(
        "dot product equals cosine similarity",
        abs(d - c) < TOLERANCE,
        f"dot={d:.8f} cosine={c:.8f}",
    )

    # ------------------------------------------------------------- 3. ordering
    print("\n3. Batch ordering")
    # The query was sent first and repeated as "exact", so those two vectors
    # must be identical. If the response were reordered, every candidate would
    # be paired with the wrong score and nothing downstream would notice.
    query_vec, exact_vec = vectors[0], vectors[1]
    checks.check(
        "a repeated text comes back in the position it was sent",
        cosine(query_vec, exact_vec) > 1 - TOLERANCE,
        f"cosine(sent first, sent second) = {cosine(query_vec, exact_vec):.8f}",
    )

    # ------------------------------------------------- 4. Tier-1 calibration
    print(f"\n4. Tier-1 cosine behaviour (gate is {AZURE_EMBEDDING_THRESHOLD})")
    scored = {k: cosine(query_vec, v) for k, v in zip(corpus, vectors[1:])}
    if not quiet:
        print(f"\n    {'candidate':12s} {'cosine':>10s}   {'vs gate':>8s}")
        for key, value in sorted(scored.items(), key=lambda kv: -kv[1]):
            print(f"    {key:12s} {value:10.4f}   "
                  f"{'PASS' if value >= AZURE_EMBEDDING_THRESHOLD else 'blocked':>8s}")
        print()

    checks.check(
        "exact match outranks the unrelated record",
        scored["exact"] > scored["unrelated"],
        f"{scored['exact']:.4f} > {scored['unrelated']:.4f}",
    )
    checks.check(
        "paraphrase outranks the unrelated record",
        scored["paraphrase"] > scored["unrelated"],
        f"{scored['paraphrase']:.4f} > {scored['unrelated']:.4f}",
    )
    checks.check(
        f"exact match clears the {AZURE_EMBEDDING_THRESHOLD} gate",
        scored["exact"] >= AZURE_EMBEDDING_THRESHOLD,
        f"{scored['exact']:.4f}",
    )
    # Recall. An exact match clearing the gate proves very little -- the same
    # text scores 1.0 on any encoder. A paraphrase is what history actually
    # looks like: nobody writes the defect up twice the same way. If this
    # fails, real precedent exists and is being gated out, and nothing
    # downstream can tell that apart from there being no precedent at all.
    checks.check(
        f"paraphrase clears the {AZURE_EMBEDDING_THRESHOLD} gate",
        scored["paraphrase"] >= AZURE_EMBEDDING_THRESHOLD,
        f"{scored['paraphrase']:.4f} -- if this FAILS the gate is too high and real matches are lost",
    )
    # The one that actually matters. If an unrelated defect also clears the
    # gate, the gate is not a gate -- which is exactly the ada-002 failure mode
    # the threshold comment warns about.
    checks.check(
        f"unrelated record is blocked by the {AZURE_EMBEDDING_THRESHOLD} gate",
        scored["unrelated"] < AZURE_EMBEDDING_THRESHOLD,
        f"{scored['unrelated']:.4f} -- if this FAILS the gate admits everything, raise it",
    )
    # Reported, not checked. `near` is weld porosity against weld cracking:
    # same part, same inspection, same vocabulary, different fix -- the shape
    # of the near-miss that produced two wrong recommendations on real
    # tickets. This used to be a gate failure. The cosine is now a recall
    # filter and the intent scorer is what tells these two apart, so a score
    # above the gate here is expected; raising the gate to block it would
    # discard real precedent worded differently, which is the failure the
    # recall filter exists to prevent.
    checks.note(
        "a different defect on the same part",
        f"{scored['near']:.4f} -- porosity vs cracking. Above {AZURE_EMBEDDING_THRESHOLD} "
        f"is expected: telling these apart is the intent scorer's job, not the cosine's.",
    )

    # ------------------------------------------------- 5. Tier-2 calibration
    print(f"\n5. Tier-2 cosine behaviour (gate is {TIER2_AZURE_THRESHOLD})")
    try:
        t2 = AzureEmbedder(settings).embed_batch(
            [TIER2_QUERY, STANDARD_RELEVANT, STANDARD_UNRELATED]
        )
    except AzureEmbeddingError as exc:
        checks.check("Tier-2 embedding request succeeds", False, str(exc))
        return
    relevant = cosine(t2[0], t2[1])
    unrelated = cosine(t2[0], t2[2])
    if not quiet:
        print(f"\n    {'passage':24s} {'cosine':>10s}")
        print(f"    {'relevant standard':24s} {relevant:10.4f}")
        print(f"    {'unrelated standard':24s} {unrelated:10.4f}\n")
    checks.check(
        "the relevant standard outranks the unrelated one",
        relevant > unrelated,
        f"{relevant:.4f} > {unrelated:.4f}",
    )
    checks.check(
        f"the relevant standard clears the {TIER2_AZURE_THRESHOLD} gate",
        relevant >= TIER2_AZURE_THRESHOLD,
        f"{relevant:.4f}",
    )
    checks.check(
        f"the unrelated standard is blocked by the {TIER2_AZURE_THRESHOLD} gate",
        unrelated < TIER2_AZURE_THRESHOLD,
        f"{unrelated:.4f} -- set SPS_TIER2_THRESHOLD between the two numbers above",
    )

    # A recall filter: it must admit everything that could be the answer --
    # the paraphrase above all -- and need only keep out what plainly is not.
    # `near` is not in the must-block set: the intent scorer rejects it.
    must_admit = min(scored["exact"], scored["paraphrase"])
    must_block = scored["unrelated"]
    if must_block < must_admit:
        print(
            f"\n  Suggested gates from this run: Tier 1 between "
            f"{must_block:.4f} and {must_admit:.4f} (nearer the low end: it is a recall "
            f"filter), "
            f"Tier 2 between {unrelated:.4f} and {relevant:.4f}."
        )
    else:
        print(
            f"\n  No Tier-1 gate works on this sample: the weakest text that must be "
            f"admitted scores {must_admit:.4f}, below the {must_block:.4f} of one that "
            f"must be blocked. A threshold cannot separate them."
        )
    print("  One sample each -- widen it with --probe, then python sps.py eval.")


# ===========================================================================
# The similarity probe
# ===========================================================================
#
# The checks above score four fixed texts. That is enough to catch a broken
# deployment and not nearly enough to set a threshold, because a threshold is
# only meaningful in the gap between what you judged similar and what you
# judged different -- and one sample of each gives no gap, just two points.
#
# The probe takes labelled pairs instead. Each pair isolates one linguistic
# property (word order, vocabulary, length, negation, a changed request) so
# that when the numbers come out wrong it is clear WHICH property the encoder
# could not see.

PAIR_HEADERS = ("class", "text_a", "text_b", "expect")
MATCH = "match"
NO_MATCH = "no-match"
VALID_EXPECT = (MATCH, NO_MATCH)

DEFAULT_PAIRS = PROJECT_DIR / "samples" / "similarity_pairs.csv"


@dataclass(frozen=True)
class Pair:
    label: str
    text_a: str
    text_b: str
    expect: str


@dataclass(frozen=True)
class Scored:
    pair: Pair
    cosine: float


class PairFileError(ValueError):
    """The pairs file is unusable, and the message says which row."""


def read_pairs(path: Path) -> list[Pair]:
    """Read labelled pairs from a .csv or .xlsx.

    Reuses the pipeline's own reader so a pairs file can be an .xlsx pasted
    together in Excel -- which is how real ticket text is going to arrive.
    """

    header, rows = read_header_and_rows(path, label="Pairs file")
    index = {str(name).strip().casefold(): i for i, name in enumerate(header or [])}

    missing = [h for h in PAIR_HEADERS if h not in index]
    if missing:
        raise PairFileError(
            f"{path.name} is missing column(s): {', '.join(missing)}. "
            f"Expected {', '.join(PAIR_HEADERS)}; found "
            f"{', '.join(str(h) for h in (header or [])) or 'nothing'}."
        )

    def cell(row, name: str) -> str:
        i = index[name]
        return str(row[i]).strip() if i < len(row) and row[i] is not None else ""

    pairs: list[Pair] = []
    for number, row in enumerate(rows, start=2):  # row 1 is the header
        if not any(str(c).strip() for c in row if c is not None):
            continue
        expect = cell(row, "expect").casefold()
        if expect not in VALID_EXPECT:
            raise PairFileError(
                f"{path.name} row {number}: Expect is {cell(row, 'expect')!r}, "
                f"must be one of {', '.join(VALID_EXPECT)}."
            )
        text_a, text_b = cell(row, "text_a"), cell(row, "text_b")
        if not text_a or not text_b:
            raise PairFileError(f"{path.name} row {number}: Text_A and Text_B cannot be blank.")
        pairs.append(Pair(cell(row, "class") or f"row {number}", text_a, text_b, expect))

    if not pairs:
        raise PairFileError(f"{path.name} has a header but no rows.")
    return pairs


def score_pairs(pairs: list[Pair], encode) -> list[Scored]:
    """Cosine for every pair, encoding each distinct text exactly once.

    `encode` takes a list of strings and returns their vectors, so the maths
    here is testable without a deployment. De-duplicating matters: a pair set
    built around one ticket repeats that ticket's text on many rows, and
    paying for it once per row would be the bulk of the bill.
    """
    texts = list(dict.fromkeys(t for p in pairs for t in (p.text_a, p.text_b)))
    vectors = dict(zip(texts, encode(texts)))
    return [Scored(p, cosine(vectors[p.text_a], vectors[p.text_b])) for p in pairs]


@dataclass(frozen=True)
class Separation:
    """Where the two populations sit, and whether anything can divide them."""

    weakest_match: Scored | None
    strongest_no_match: Scored | None

    @property
    def measurable(self) -> bool:
        return self.weakest_match is not None and self.strongest_no_match is not None

    @property
    def margin(self) -> float:
        return self.weakest_match.cosine - self.strongest_no_match.cosine

    @property
    def separable(self) -> bool:
        return self.measurable and self.margin > 0

    @property
    def suggested(self) -> float:
        """Just below the weakest pair called a match: the recall gate.

        The Tier-1 cosine no longer decides; it chooses what the intent scorer
        reads, and the scorer rejects. So the errors are no longer the ones
        this used to weigh. A false match here costs a slightly longer prompt
        the scorer turns down. A missed match is a record the scorer never
        sees -- refused without anyone judging it, and indistinguishable
        downstream from there being no precedent at all. The gate therefore
        sits just under the weakest match, whatever that admits.
        """
        return round(max(self.weakest_match.cosine - 0.01, 0.0), 4)


def separation(scored: list[Scored]) -> Separation:
    matches = [s for s in scored if s.pair.expect == MATCH]
    no_matches = [s for s in scored if s.pair.expect == NO_MATCH]
    return Separation(
        weakest_match=min(matches, key=lambda s: s.cosine) if matches else None,
        strongest_no_match=max(no_matches, key=lambda s: s.cosine) if no_matches else None,
    )


def report_probe(scored: list[Scored], gate: float, source: Path) -> None:
    """Print the table, then the only number that decides anything."""
    print(f"\nSimilarity probe -- {len(scored)} pairs from {source}")
    print(f"Gate under test: {gate}\n")

    width = max(len(s.pair.label) for s in scored)
    print(f"  {'class':{width}s} {'cosine':>8s}  {'expect':<9s} {'gate says':<10s}")
    for s in sorted(scored, key=lambda s: -s.cosine):
        admits = s.cosine >= gate
        wanted = s.pair.expect == MATCH
        verdict = "admits" if admits else "blocks"
        # Only a blocked match is an error now: an admitted no-match goes on
        # to the intent scorer, which is what turns it down.
        flag = "   <-- MISSED" if wanted and not admits else ("   (scorer)" if admits and not wanted else "")
        print(f"  {s.pair.label:{width}s} {s.cosine:8.4f}  {s.pair.expect:<9s} {verdict:<10s}{flag}")

    sep = separation(scored)
    if not sep.measurable:
        print(
            f"\n  No margin: the set needs at least one {MATCH} pair and one "
            f"{NO_MATCH} pair. A threshold is the boundary between two "
            f"populations, and only one is present."
        )
        return

    print(
        f"\n  weakest  match     {sep.weakest_match.cosine:7.4f}   ({sep.weakest_match.pair.label})"
        f"\n  strongest no-match {sep.strongest_no_match.cosine:7.4f}   ({sep.strongest_no_match.pair.label})"
        f"\n  separation margin  {sep.margin:+7.4f}   {'' if sep.separable else '<-- NEGATIVE'}"
    )

    admitted = [s for s in scored if s.pair.expect == NO_MATCH and s.cosine >= sep.suggested]
    print(f"\n  Suggested recall gate: {sep.suggested} (just below the weakest match).")
    if admitted:
        print(f"  It admits {len(admitted)} pair(s) you called {NO_MATCH}; the intent scorer is")
        print("  what has to turn these down, so they belong in its evaluation set:")
        for s in sorted(admitted, key=lambda s: -s.cosine):
            print(f"    {s.pair.label:20s} {s.cosine:.4f}")
    else:
        print(f"  It admits nothing you called {NO_MATCH}.")

    if sep.separable:
        print("  The cosine alone separates this set -- the scorer has an easy job here.")
        return
    overlap = [
        s for s in scored
        if s.pair.expect == NO_MATCH and s.cosine >= sep.weakest_match.cosine
    ]
    print(
        f"\n  No cosine threshold can separate these pairs: {len(overlap)} {NO_MATCH} pair(s)\n"
        f"  score at or above the weakest {MATCH}. Expected for a changed request or a\n"
        f"  negation -- wording alike, intent not -- and why the intent scorer decides.\n"
        f"  If the overlap is in classes that differ in substance, check those first."
    )


PROBE_COLUMNS = (
    "Class",
    "Expect",
    "Cosine",
    "Gate",
    "Gate_Admits",
    "Gate_Agrees",
    "Text_A",
    "Text_B",
)


def write_probe(scored: list[Scored], gate: float, path: Path) -> None:
    """One row per pair, sorted by score, so the crossover is visible by eye.

    Goes through the resolver's own writer: atomic, and it handles Excel's
    32767-character cell limit, which a pasted-in ticket narrative can reach.
    That writer renders every cell as text. For a cosine that is harmless --
    these are all fixed-width decimals in [0, 1], so Excel's text sort puts
    them in numeric order anyway, and the console report above has already
    done the arithmetic.
    """

    write_rows(
        path,
        PROBE_COLUMNS,
        [
            {
                "Class": s.pair.label,
                "Expect": s.pair.expect,
                "Cosine": f"{s.cosine:.6f}",
                "Gate": f"{gate}",
                "Gate_Admits": "yes" if s.cosine >= gate else "no",
                # A recall gate is wrong only when it blocks a match.
                "Gate_Agrees": "NO" if s.pair.expect == MATCH and s.cosine < gate else "yes",
                "Text_A": s.pair.text_a,
                "Text_B": s.pair.text_b,
            }
            for s in sorted(scored, key=lambda s: -s.cosine)
        ],
    )
    print(f"\n  Wrote {path}")


def run_probe(pairs_path: Path, out_path: Path | None, checks: "Checks") -> None:

    # Pairs first. It is local and free, and a mistyped filename should not
    # need credentials to discover.
    #
    # FileReadError and UnsupportedFileType are the reader's own -- a missing
    # file and a .pdf respectively. Neither is an OSError, so leaving them out
    # turns a typo'd filename into a traceback.

    try:
        pairs = read_pairs(pairs_path)
    except (PairFileError, FileReadError, UnsupportedFileType, OSError) as exc:
        checks.check("pairs file is readable", False, str(exc))
        return
    print(f"  {len(pairs)} pairs read from {pairs_path}")

    settings = AzureEmbeddingSettings.from_env()
    if not checks.check(
        "Azure embedding deployment is configured",
        settings.configured,
        "missing " + ", ".join(settings.missing()) if not settings.configured else "",
    ):
        print("\n  Set them in .env, then re-run.")
        return

    embedder = AzureEmbedder(settings)
    try:
        scored = score_pairs(pairs, embedder.embed_batch)
    except AzureEmbeddingError as exc:
        checks.check("embedding request succeeds", False, str(exc))
        return

    report_probe(scored, AZURE_EMBEDDING_THRESHOLD, pairs_path)

    # The probe reports; it does not pass or fail. Whether a wrong verdict on
    # an invented pair matters is a judgement about the pair, not about the
    # deployment, and a red FAIL would imply the script had made that call.
    sep = separation(scored)
    if sep.measurable and not sep.separable:
        print("\n  Probe finished. The margin is negative -- see above.")
    else:
        print("\n  Probe finished.")

    if out_path:
        write_probe(scored, AZURE_EMBEDDING_THRESHOLD, out_path)


def check_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python sps.py check", description="Verify the embedder the pipeline uses"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the similarity table")
    parser.add_argument(
        "--probe",
        nargs="?",
        const=str(DEFAULT_PAIRS),
        metavar="PAIRS",
        help="score labelled text pairs and report the separation margin, instead of "
        f"running the acceptance checks. Defaults to {DEFAULT_PAIRS.name}; "
        "point it at your own .csv or .xlsx of real ticket pairs",
    )
    parser.add_argument(
        "--out",
        metavar="XLSX",
        help="with --probe, also write the scored pairs to this workbook",
    )
    args = parser.parse_args(argv)

    # Same .env the resolver reads. Without this the script sees only real
    # environment variables and reports every credential missing on a machine
    # whose .env is perfectly well filled in.

    env_file = load_env_file()

    checks = Checks()
    if args.probe:
        print("Probing the Azure embedding deployment.")
        print(f"  .env: {env_file if env_file else 'not found -- using the environment only'}\n")
        run_probe(Path(args.probe), Path(args.out) if args.out else None, checks)
        return _report(checks)

    print("Checking the Azure embedding deployment.")
    print(f"  .env: {env_file if env_file else 'not found -- using the environment only'}\n")
    verify_azure(checks, args.quiet)
    return _report(checks)


def _report(checks: "Checks") -> int:
    print()
    if checks.failures:
        print(f"FAILED: {len(checks.failures)} check(s): {', '.join(checks.failures)}")
        return 1
    print("All embedder checks passed.")
    return 0


# =============================================================================
# Entry point
# =============================================================================

COMMANDS = {
    "resolve": resolve_main,
    "bulk": bulk_main,
    "eval": eval_main,
    "ingest": ingest_main,
    "check": check_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python sps.py {" + " | ".join(COMMANDS) + "} [options]", file=sys.stderr)
        print("       python sps.py <command> --help", file=sys.stderr)
        return EXIT_BAD_INPUT
    try:
        return COMMANDS[argv[0]](argv[1:])
    except SettingError as exc:
        # resolve reports this in status.xlsx itself; every other command
        # stops here, naming the setting, rather than with a traceback.
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
