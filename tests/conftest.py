"""Test doubles for the embedder and the chat client, and a .docx writer.

Azure is not needed to exercise the resolver, and Word is not needed to build
the standards documents it reads.
"""

from __future__ import annotations

import math
import re
import zipfile
import zlib
from typing import Sequence
from xml.sax.saxutils import escape

import pytest


DIM = 128
_TOKEN = re.compile(r"[a-z0-9]+")


class TokenOverlapEmbedder:
    """Deterministic stand-in for an embedding model.

    Hashes tokens into a fixed-width space and L2-normalizes, so cosine
    similarity tracks token overlap: identical text scores 1.0, disjoint text
    scores 0.0. That makes retrieval assertions exact without a real model.
    """

    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall((text or "").lower()):
            vector[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passage_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)


class ScriptedChatClient:
    """Chat client that replays a fixed list of JSON responses.

    The loop alternates Actor / Judge calls, so a script reads as
    [actor_1, judge_1, actor_2, judge_2, ...].
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: Sequence[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedChatClient ran out of scripted responses")
        return self.responses.pop(0)

    async def complete_model(self, messages: Sequence[dict[str, str]], model):
        """Scripts stay plain JSON strings and are validated through the same
        pydantic models the real client is constrained by, so a script that the
        deployment could not have produced fails here too."""
        from sps import validate_json

        return validate_json(await self.complete(messages), model)

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def embedder() -> TokenOverlapEmbedder:
    return TokenOverlapEmbedder()


@pytest.fixture(autouse=True)
def isolated_0250_corpus(tmp_path_factory, monkeypatch):
    """Point Tier 2 at an empty folder for every test that does not set its own.

    Without this the default corpus path is resolved against the working
    directory, so a deployment that had actually loaded its 0250 documents
    would change the outcome of tests written about Tier 1 -- passing or
    failing depending on what was in a folder the test never mentions.
    """
    monkeypatch.setenv(
        "SPS_0250_DOCS_DIR", str(tmp_path_factory.mktemp("no_0250_docs"))
    )


class FakeAzureEmbedder:
    """Offline stand-in for the Azure embedding deployment.

    Same surface and the same semantics that matter: one `embed_batch` call for
    query and passages together, no instruction prefix, unit-length vectors
    out. Scores therefore track token overlap, so retrieval assertions stay
    exact without a network call.
    """

    def __init__(self, settings, client=None) -> None:
        self.settings = settings
        self._inner = TokenOverlapEmbedder()
        self.batches: list[list[str]] = []

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        return [self._inner._vector(t) for t in texts]


@pytest.fixture
def azure_embeddings(monkeypatch):
    """Configure and stub the Azure deployment for a whole run.

    With the local encoder out of the pipeline nothing embeds offline, so any
    test that drives the resolver end to end needs this. Before, the same tests
    ran on the bge-small fallback without noticing they had no credentials.
    """
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    # Patched on the module, not the call site: both retrievers import
    # AzureEmbedder inside the function that uses it, so the lookup is fresh.
    monkeypatch.setattr("sps.AzureEmbedder", FakeAzureEmbedder)


# ------------------------------------------------------------------ .docx files
#
# Written with the standard library, like the reader: python-docx left the
# install along with the code that needed it. Styles carry Word's own ids and
# names -- "Heading1" called "heading 1", "Title" -- so the reader is shown what
# Word itself writes.

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_WORDML = "application/vnd.openxmlformats-officedocument.wordprocessingml"
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    f'<Override PartName="/word/document.xml" ContentType="{_WORDML}.document.main+xml"/>'
    "{styles}</Types>"
)
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
_PACKAGE_RELS = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{_RELS_NS}">'
    f'<Relationship Id="rId1" Type="{_REL_TYPE}officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)
_DOCUMENT_RELS = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{_RELS_NS}">'
    "{styles}</Relationships>"
)


def docx_style(style_id: str, name: str, default: bool = False) -> str:
    """One paragraph style, as styles.xml declares it."""
    flag = ' w:default="1"' if default else ""
    return (f'<w:style w:type="paragraph"{flag} w:styleId="{style_id}">'
            f'<w:name w:val="{name}"/></w:style>')


WORD_STYLES = (
    docx_style("Normal", "Normal", default=True)
    + docx_style("Title", "Title")
    + "".join(docx_style(f"Heading{n}", f"heading {n}") for n in range(1, 10))
)


def _docx_paragraph(text: str, style: str | None = None) -> str:
    properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return (f'<w:p>{properties}<w:r><w:t xml:space="preserve">{escape(text)}</w:t>'
            "</w:r></w:p>")


def _docx_table(rows: Sequence[Sequence[str]]) -> str:
    return "<w:tbl>" + "".join(
        "<w:tr>" + "".join(f"<w:tc>{_docx_paragraph(text)}</w:tc>" for text in row) + "</w:tr>"
        for row in rows
    ) + "</w:tbl>"


def write_docx(path, blocks, styles: str | None = WORD_STYLES):
    """Write a .docx whose body is `blocks`, in order. Each block is one of

        ("title", text)   ("heading", level, text)   ("paragraph", text)
        ("table", [[cell, ...], ...])   ("xml", raw body XML, w: prefix bound)

    `styles` is the <w:style> elements of styles.xml; None leaves the styles
    part out altogether.
    """
    body = []
    for kind, *args in blocks:
        if kind == "title":
            body.append(_docx_paragraph(args[0], "Title"))
        elif kind == "heading":
            body.append(_docx_paragraph(args[1], f"Heading{args[0]}"))
        elif kind == "paragraph":
            body.append(_docx_paragraph(args[0]))
        elif kind == "table":
            body.append(_docx_table(args[0]))
        elif kind == "xml":
            body.append(args[0])
        else:
            raise ValueError(f"unknown block kind {kind!r}")
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{_W_NS}">'
        f'<w:body>{"".join(body)}</w:body></w:document>'
    )
    styled = styles is not None
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", _CONTENT_TYPES.format(styles=(
            f'<Override PartName="/word/styles.xml" ContentType="{_WORDML}.styles+xml"/>'
            if styled else "")))
        package.writestr("_rels/.rels", _PACKAGE_RELS)
        package.writestr("word/document.xml", document)
        package.writestr("word/_rels/document.xml.rels", _DOCUMENT_RELS.format(styles=(
            f'<Relationship Id="rId1" Type="{_REL_TYPE}styles" Target="styles.xml"/>'
            if styled else "")))
        if styled:
            package.writestr(
                "word/styles.xml",
                f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<w:styles xmlns:w="{_W_NS}">{styles}</w:styles>',
            )
    return path
