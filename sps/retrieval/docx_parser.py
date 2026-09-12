"""Parse 0250 engineering standards into citable, embeddable chunks.

Two things make a chunk useful to a grounded LLM, and both are done here:

* It is **section-bounded**. A chunk never spans a heading, so the citation
  attached to it is actually true. A chunk that straddled 4.2 and 4.3 could be
  cited as either and would be wrong half the time.
* It **carries its own provenance**. The document name and section trail are
  prepended to the text itself, so the citation travels with the content into
  the model's context rather than living in a side-channel the model cannot see.

`python-docx` is imported lazily: the rest of the package, and its tests, still
import on a machine without it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

DOCX_SUFFIX = ".docx"
LEGACY_DOC_SUFFIX = ".doc"

# 250-350 words per the standard's own guidance. At ~1.1 tokens per word for
# technical prose that is 270-390 tokens, comfortably inside bge-small's 512
# limit even once the citation prefix is added -- measured at 382 tokens for
# 350 words, leaving ~130 tokens of headroom.
TARGET_CHUNK_WORDS = 300
MIN_CHUNK_WORDS = 250
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
_WHITESPACE = re.compile(r"\s+")


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


def _heading_level(paragraph) -> int | None:
    """Word heading level, or None for body text."""
    style = getattr(paragraph, "style", None)
    if style is None:
        return None
    for value, pattern in ((getattr(style, "name", ""), _HEADING_NAME),
                           (getattr(style, "style_id", ""), _HEADING_ID)):
        match = pattern.match(str(value or "").strip())
        if match:
            return int(match.group(1))
    # "Title" heads the document but is not numbered.
    if str(getattr(style, "style_id", "") or "").strip().casefold() == "title":
        return 0
    return None


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def _iter_blocks(document) -> Iterator[tuple[str, object]]:
    """Yield ("paragraph", Paragraph) / ("table", Table) in document order.

    `document.paragraphs` silently omits every table, and in a standards
    document the tables are where the numbers live -- porosity limits, torque
    values, acceptance criteria. Walking the body element is the only way to
    get both, in the order a reader would meet them.
    """
    from docx.document import Document as _Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if not isinstance(document, _Document):
        raise DocParseError("Expected a python-docx Document")

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, document)


def _table_text(table) -> str:
    """Flatten a table to one line per row.

    Row-wise with a separator keeps a limits table readable as
    "Porosity | 2 percent max", which is the shape the answer is actually in.
    """
    lines = []
    for row in table.rows:
        cells = [_clean(cell.text) for cell in row.cells]
        # Merged cells repeat their text across the span; collapse the repeats.
        deduped: list[str] = []
        for cell in cells:
            if cell and (not deduped or deduped[-1] != cell):
                deduped.append(cell)
        if deduped:
            lines.append(" | ".join(deduped))
    return "\n".join(lines)


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
        text = _clean(text)
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
        from docx import Document
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DocParseError(
            "python-docx is required to read 0250 documents. "
            "Install it with: pip install python-docx"
        ) from exc

    try:
        document = Document(str(path))
    except Exception as exc:
        # python-docx raises a variety of types for a corrupt or password
        # protected file; the caller only needs "this one is unusable".
        raise DocParseError(f"Could not read {path.name!r}: {exc}") from exc

    accumulator = _Accumulator(path.name)
    trail: list[str] = []

    for kind, block in _iter_blocks(document):
        if kind == "table":
            accumulator.add(_table_text(block))
            continue

        level = _heading_level(block)
        text = _clean(block.text)
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

    A .doc is not a zip container and `python-docx` cannot open it. The only
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
