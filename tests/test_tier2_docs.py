"""Tier 2: parsing 0250 standards, caching their vectors, and falling back.

The load-bearing tests here are the ones about *not* answering. Tier 2 reaches
for a document when precedent has already run out, which is exactly the moment a
model is most inclined to supply a plausible engineering fix from its own
knowledge -- so the refusal paths, the citation integrity and the
one-embedding-space rule get more attention than the happy path.
"""

from __future__ import annotations

import json

import pytest

import sps as resolver
from sps import IncomingTicket
from sps import (
    CACHE_FILENAMES,
    TIER2_AZURE_THRESHOLD,
    DocRetriever,
    ScoredChunk,
    build_query,
    corpus_hash,
    load_cache,
    save_cache,
)
from sps import (
    MIN_USABLE_WORDS,
    DocChunk,
    iter_doc_files,
    parse_docx,
)
from tests.conftest import TokenOverlapEmbedder, docx_style, write_docx
from tests.test_resolver import (
    NEAR_PROBLEM, PART, PROBLEM, read_sheet, row, write_history, write_ticket,
)


@pytest.fixture(autouse=True)
def _stubbed_azure(azure_embeddings):
    """Azure is the only encoder now, so every test in this module embeds
    through the stub. A test that wants the unconfigured path deletes the
    variables itself."""


CRACKING = (
    "Cracking in a fillet or butt weld seam is cause for rejection of the affected "
    "part. On discovery at incoming inspection, segregate the affected lot and "
    "quarantine it pending disposition. The cracked seam shall be ground out to "
    "sound metal for the full length of the crack plus 25 mm beyond each visible "
    "end, then re-welded to the original joint profile."
)
PACKAGING = (
    "Each outer carton shall carry a label showing the part number, the revision, "
    "the quantity contained and the date of packing. Labels shall be printed at "
    "300 dpi or better and shall remain legible after transit. Where a label is "
    "unreadable on arrival the carton shall be held and the supplier notified."
)


def write_doc(path, sections, title="0250 Standard"):
    """Build a .docx with a title and (heading, [paragraphs]) sections."""
    blocks = [("heading", 1, title)]
    for heading, paragraphs in sections:
        blocks.append(("heading", 2, heading))
        blocks.extend(("paragraph", text) for text in paragraphs)
    return write_docx(path, blocks)


def weld_corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    write_doc(
        directory / "0250-Weld-Standards.docx",
        [("4.2 Weld Seam Cracking", [CRACKING])],
        title="0250 Weld Standards",
    )
    write_doc(
        directory / "0250-Packaging-Standards.docx",
        [("2.1 Carton Labelling", [PACKAGING])],
        title="0250 Packaging Standards",
    )
    return directory


def ticket(problem=PROBLEM, issue="Quality", part=PART):
    return IncomingTicket(
        problem_description=problem, part_number=part, issue_type=issue, sps_id="T-1"
    )


# ------------------------------------------------------------------ parsing


def test_a_section_becomes_a_citable_chunk(tmp_path):
    path = write_doc(tmp_path / "0250-Weld.docx", [("4.2 Weld Seam Cracking", [CRACKING])])
    (chunk,) = parse_docx(path)

    assert chunk.document == "0250-Weld.docx"
    assert chunk.section == "4.2 Weld Seam Cracking"
    assert chunk.citation == "0250-Weld.docx § 4.2 Weld Seam Cracking"


def test_a_chunk_never_spans_two_sections(tmp_path):
    """A chunk that straddled 4.2 and 4.3 could be cited as either, and would
    be wrong half the time."""
    path = write_doc(
        tmp_path / "d.docx",
        [("4.2 Cracking", [CRACKING]), ("4.3 Porosity", [PACKAGING])],
    )
    chunks = parse_docx(path)

    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"4.2 Cracking", "4.3 Porosity"}
    assert "Cracking in a fillet" in chunks[0].text
    assert "Cracking in a fillet" not in chunks[1].text


def test_the_citation_travels_inside_the_embedded_text(tmp_path):
    """The string that was ranked is the string the model is asked to cite, so
    nothing can drift between retrieval and attribution."""
    path = write_doc(tmp_path / "d.docx", [("4.2 Cracking", [CRACKING])])
    (chunk,) = parse_docx(path)

    assert chunk.embed_text.startswith(f"[{chunk.citation}] ")
    assert chunk.text in chunk.embed_text


def test_table_text_is_captured(tmp_path):
    """`document.paragraphs` omits every table, and a limits table is where a
    standard keeps its numbers."""
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "4.3 Porosity Limits"),
        ("table", [["Scattered porosity", "2 percent by area max over any 25 mm of seam"],
                   ["Clustered porosity and wormholes", "Rejectable at any level whatsoever"]]),
    ])

    (chunk,) = parse_docx(path)
    assert "2 percent by area max" in chunk.text
    assert "Scattered porosity |" in chunk.text


# ---------------------------------------------------------------- the reader
#
# A .docx is read with the standard library. It began as python-docx 1.1.2's
# rules, proven against python-docx before it left the install; it now also
# reads what python-docx skipped and Word shows -- content controls, smart tags,
# tracked insertions, nested tables -- and keeps a table's rows apart.

FILLER = "Applies to every welded bracket supplied under this standard, whatever the alloy."


def _cell(text, properties=""):
    tc_pr = f"<w:tcPr>{properties}</w:tcPr>" if properties else ""
    return f"<w:tc>{tc_pr}<w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>"


def test_run_content_is_read_as_word_shows_it(tmp_path):
    """Tabs, line breaks and non-breaking hyphens are text; a hyperlink's
    runs are part of its paragraph."""
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "4.2 Cracking"),
        ("xml", '<w:p><w:r><w:t>Grind</w:t><w:tab/><w:t>out</w:t><w:br/><w:t>the seam</w:t>'
                '<w:noBreakHyphen/><w:t>end</w:t></w:r><w:hyperlink><w:r>'
                '<w:t xml:space="preserve"> per drawing 7.</w:t></w:r></w:hyperlink></w:p>'),
        ("paragraph", FILLER),
    ])
    (chunk,) = parse_docx(path)
    assert chunk.text == f"Grind out the seam-end per drawing 7.\n{FILLER}"


def test_tracked_changes_read_as_word_shows_them(tmp_path, caplog):
    """Insertions in, deletions out -- Word's default view -- and the document
    named, because its text is not settled. python-docx dropped insertions and
    deletions alike, so it read neither the old text nor the new."""
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "6.1 Torque"),
        ("xml", '<w:p><w:r><w:t xml:space="preserve">Torque the bracket fasteners to 12 Nm</w:t></w:r>'
                '<w:ins w:id="1" w:author="a"><w:r><w:t xml:space="preserve"> and then to 14 Nm</w:t>'
                '</w:r></w:ins><w:del w:id="2" w:author="a"><w:r>'
                '<w:delText xml:space="preserve"> or by hand</w:delText></w:r></w:del>'
                "<w:r><w:t>.</w:t></w:r></w:p>"),
        ("paragraph", FILLER),
    ])
    with caplog.at_level("WARNING"):
        (chunk,) = parse_docx(path)
    assert chunk.text == f"Torque the bracket fasteners to 12 Nm and then to 14 Nm.\n{FILLER}"
    assert "d.docx has tracked changes" in caplog.text


def test_a_heading_from_a_localized_word_is_still_a_heading(tmp_path):
    """The style name is localized; the style id stays English."""
    styles = docx_style("Normal", "Normal", default=True) + docx_style("Heading2", "Überschrift 2")
    path = write_docx(tmp_path / "d.docx", [("heading", 2, "4.2 Risse"), ("paragraph", FILLER)],
                      styles=styles)
    (chunk,) = parse_docx(path)
    assert chunk.section == "4.2 Risse"


def test_without_a_styles_part_nothing_is_a_heading(tmp_path):
    """python-docx stood in a template defining only Normal, so a heading
    style in a file with no styles part read as body text."""
    path = write_docx(tmp_path / "d.docx", [("heading", 2, "4.2 Cracking"), ("paragraph", FILLER)],
                      styles=None)
    (chunk,) = parse_docx(path)
    assert (chunk.section, chunk.text) == ("", f"4.2 Cracking\n{FILLER}")


def test_merged_cells_are_read_once(tmp_path):
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "4.3 Porosity Limits"),
        ("xml", "<w:tbl>"
                "<w:tr>" + _cell("Porosity", '<w:vMerge w:val="restart"/>') + _cell("Scattered")
                + _cell("2 percent by area max") + "</w:tr>"
                "<w:tr>" + _cell("", "<w:vMerge/>") + _cell("Clustered")
                + _cell("Rejectable at any level") + "</w:tr>"
                "<w:tr>" + _cell("Applies to every seam over 25 mm long", '<w:gridSpan w:val="3"/>')
                + "</w:tr></w:tbl>"),
    ])
    (chunk,) = parse_docx(path)
    # One row per line. Run together, "2 percent by area max" and the next
    # row's "Porosity" read as a single claim -- which is how they were cached
    # until the rows were kept apart.
    assert chunk.text == ("Porosity | Scattered | 2 percent by area max\n"
                          "Porosity | Clustered | Rejectable at any level\n"
                          "Applies to every seam over 25 mm long")


def test_a_merge_down_a_long_table_is_read_in_full(tmp_path):
    """python-docx followed a vertical merge upward one recursive call per row,
    and a merge past about a thousand rows raised RecursionError."""
    rows = ["<w:tr>" + _cell("Porosity", '<w:vMerge w:val="restart"/>') + _cell("row 0") + "</w:tr>"]
    rows += ["<w:tr>" + _cell("", "<w:vMerge/>") + _cell(f"row {i}") + "</w:tr>" for i in range(1, 1500)]
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "4.3 Porosity Limits"), ("xml", "<w:tbl>" + "".join(rows) + "</w:tbl>"),
    ])

    chunks = parse_docx(path)
    lines = [line for chunk in chunks for line in chunk.text.split("\n")]
    # Every row once, in order, each whole. A chunk that continues the table
    # opens with its first row, so the first row recurs and nothing else does.
    assert len(chunks) > 1
    assert [l for l in lines if l != "Porosity | row 0"] == \
        [f"Porosity | row {i}" for i in range(1, 1500)]
    assert all(chunk.text.startswith("Porosity | row 0") for chunk in chunks)


def test_text_in_content_controls_and_smart_tags_is_read(tmp_path):
    """Word shows it; python-docx read none of it. A whole section wrapped in a
    content control -- common in template-built standards -- was invisible."""
    path = write_docx(tmp_path / "d.docx", [
        ("xml", "<w:sdt><w:sdtPr><w:alias w:val='Section'/></w:sdtPr><w:sdtContent>"
                '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>4.2 Cracking</w:t></w:r></w:p>'
                "<w:p><w:r><w:t xml:space='preserve'>Grind out to </w:t></w:r>"
                "<w:smartTag><w:r><w:t>sound metal</w:t></w:r></w:smartTag>"
                "<w:sdt><w:sdtContent><w:r><w:t xml:space='preserve'> and re-weld.</w:t></w:r>"
                "</w:sdtContent></w:sdt></w:p>"
                "</w:sdtContent></w:sdt>"),
        ("paragraph", FILLER),
    ])
    (chunk,) = parse_docx(path)
    assert chunk.section == "4.2 Cracking"
    assert chunk.text.startswith("Grind out to sound metal and re-weld.")


def test_a_table_nested_in_a_cell_is_read(tmp_path):
    nested = "<w:tbl><w:tr>" + _cell("Ra") + _cell("1.6 max") + "</w:tr></w:tbl>"
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "3.2 Surface Finish"),
        ("xml", "<w:tbl><w:tr>" + _cell("Machined faces")
                + "<w:tc>" + nested + "<w:p/></w:tc></w:tr></w:tbl>"),
        ("paragraph", FILLER),
    ])
    (chunk,) = parse_docx(path)
    assert chunk.text.startswith("Machined faces | Ra | 1.6 max")


def test_changing_the_reading_rules_rebuilds_the_cache(tmp_path):
    from sps import DOC_READER_VERSION

    chunks = [DocChunk(document="d.docx", section="1", text="x")]
    path = tmp_path / CACHE_FILENAMES["injected"]
    save_cache(path, chunks, [[1.0]],
               {"format": 2, "reader": DOC_READER_VERSION - 1, "corpus_hash": "h", "model": "m"})
    assert load_cache(path, "h", "m") is None


def test_a_merge_with_nothing_above_it_costs_one_cell_not_the_document(tmp_path):
    """A deliberate difference: python-docx raised here mid-document, past the
    guard that skips one unreadable file, and Tier 2 lost every document."""
    path = write_docx(tmp_path / "d.docx", [
        ("heading", 2, "4.3 Porosity Limits"),
        ("xml", "<w:tbl><w:tr>" + _cell("orphan", "<w:vMerge/>")
                + _cell("Scattered porosity is limited to 2 percent by area over any 25 mm of seam")
                + "</w:tr></w:tbl>"),
    ])
    (chunk,) = parse_docx(path)
    assert chunk.text == "Scattered porosity is limited to 2 percent by area over any 25 mm of seam"


def test_the_section_trail_does_not_restate_the_document_title(tmp_path):
    """The H1 is nearly always the document's own name, which the citation
    already carries two words earlier."""
    path = write_doc(
        tmp_path / "0250-Weld-Standards.docx",
        [("4.2 Cracking", [CRACKING])],
        title="0250 Weld Standards",
    )
    (chunk,) = parse_docx(path)
    assert chunk.section == "4.2 Cracking"


def test_a_heading_with_nothing_under_it_is_dropped(tmp_path):
    """Its vector would match everything weakly and nothing well."""
    path = write_doc(tmp_path / "d.docx", [("4.1 Scope", ["See below."]),
                                           ("4.2 Cracking", [CRACKING])])
    chunks = parse_docx(path)

    assert [c.section for c in chunks] == ["4.2 Cracking"]


def test_an_over_long_passage_is_split_on_sentences(tmp_path):
    """Splitting mid-sentence would hand the model a fragment ending in
    'shall not exceed'."""
    long_text = " ".join(
        f"Requirement number {i} states that the seam shall be inspected "
        f"thoroughly before release." for i in range(60)
    )
    path = write_doc(tmp_path / "d.docx", [("4.2 Cracking", [long_text])])
    chunks = parse_docx(path)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.word_count >= MIN_USABLE_WORDS
        assert chunk.text.rstrip().endswith(".")


def test_word_count_stays_within_the_model_window(tmp_path):
    body = " ".join(["The weld seam shall be ground out and re-inspected."] * 120)
    path = write_doc(tmp_path / "d.docx", [("4.2 Cracking", [body])])

    for chunk in parse_docx(path):
        # 350 words measured at 382 tokens against bge-small's 512 limit.
        assert chunk.word_count <= 350


# ------------------------------------------------------------- file listing


def test_legacy_doc_files_are_refused_with_a_warning(tmp_path, caplog):
    """A .doc is not a zip container. The only way to read one on Windows is
    Word via COM, which blocks a headless robot on a modal dialog."""
    weld_corpus(tmp_path)
    (tmp_path / "0250-Old-Standard.doc").write_bytes(b"\xd0\xcf\x11\xe0 legacy OLE")

    with caplog.at_level("WARNING"):
        files = iter_doc_files(tmp_path)

    assert [f.suffix for f in files] == [".docx", ".docx"]
    assert "0250-Old-Standard.doc" in caplog.text
    assert ".docx" in caplog.text


def test_word_lock_files_are_skipped(tmp_path):
    """~$name.docx means someone has the document open, not that there are two."""
    weld_corpus(tmp_path)
    (tmp_path / "~$0250-Weld-Standards.docx").write_bytes(b"lock")

    assert not any(f.name.startswith("~$") for f in iter_doc_files(tmp_path))


def test_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert iter_doc_files(tmp_path / "absent") == []


def test_an_unreadable_document_does_not_lose_the_corpus(tmp_path, caplog):
    weld_corpus(tmp_path)
    (tmp_path / "0250-Corrupt.docx").write_bytes(b"not a zip at all")

    retriever = DocRetriever(docs_dir=tmp_path, embedder=TokenOverlapEmbedder())
    with caplog.at_level("WARNING"):
        hits = retriever.retrieve(ticket())

    assert "0250-Corrupt.docx" in caplog.text
    assert retriever.stats.chunks == 2
    assert hits or retriever.stats.top_score > 0


def test_malformed_markup_mid_document_loses_only_that_document(tmp_path, caplog):
    """A valid package whose table markup does not add up fails while it is
    being walked, not while it is opened. That has to be caught per document
    too, or one bad table would make Tier 2 unavailable for every ticket."""
    weld_corpus(tmp_path)
    write_docx(tmp_path / "0250-Bad-Table.docx", [
        ("heading", 2, "4.3 Porosity Limits"),
        ("xml", "<w:tbl><w:tr>" + _cell("Porosity", '<w:gridSpan w:val="two"/>') + "</w:tr></w:tbl>"),
    ])

    retriever = DocRetriever(docs_dir=tmp_path, embedder=TokenOverlapEmbedder())
    with caplog.at_level("WARNING"):
        retriever.retrieve(ticket())

    assert "0250-Bad-Table.docx" in caplog.text
    assert retriever.stats.chunks == 2


# ------------------------------------------------------------------ hashing


def test_the_hash_changes_when_a_document_changes(tmp_path):
    weld_corpus(tmp_path)
    before = corpus_hash(iter_doc_files(tmp_path))

    write_doc(tmp_path / "0250-Weld-Standards.docx", [("4.2 Cracking", [PACKAGING])])
    assert corpus_hash(iter_doc_files(tmp_path)) != before


def test_the_hash_changes_when_a_document_is_renamed(tmp_path):
    """A rename changes every citation the document produces, even though not
    one byte of its text moved."""
    weld_corpus(tmp_path)
    before = corpus_hash(iter_doc_files(tmp_path))

    (tmp_path / "0250-Weld-Standards.docx").rename(tmp_path / "0250-Welding-Rev-B.docx")
    assert corpus_hash(iter_doc_files(tmp_path)) != before


def test_the_hash_is_stable_across_calls(tmp_path):
    weld_corpus(tmp_path)
    files = iter_doc_files(tmp_path)
    assert corpus_hash(files) == corpus_hash(list(reversed(files)))


# ----------------------------------------------------------------- the cache


def test_a_second_run_loads_from_cache_without_reembedding(tmp_path):
    weld_corpus(tmp_path)
    embedder = TokenOverlapEmbedder()

    first = DocRetriever(docs_dir=tmp_path, embedder=embedder)
    first.retrieve(ticket())
    passages_after_first = len(embedder.passage_calls)

    second = DocRetriever(docs_dir=tmp_path, embedder=embedder)
    second.retrieve(ticket())

    assert first.stats.cache_state == "rebuilt"
    assert second.stats.cache_state == "hit"
    # The query is still encoded; the corpus is not.
    assert len(embedder.passage_calls) == passages_after_first


def test_editing_a_document_invalidates_the_cache(tmp_path):
    weld_corpus(tmp_path)
    embedder = TokenOverlapEmbedder()
    DocRetriever(docs_dir=tmp_path, embedder=embedder).retrieve(ticket())

    write_doc(
        tmp_path / "0250-Weld-Standards.docx",
        [("4.2 Cracking", [CRACKING]), ("4.4 New Section", [CRACKING])],
    )
    rebuilt = DocRetriever(docs_dir=tmp_path, embedder=embedder)
    rebuilt.retrieve(ticket())

    assert rebuilt.stats.cache_state == "rebuilt"
    assert rebuilt.stats.chunks == 3


def test_each_encoder_gets_its_own_cache_file(tmp_path):
    """A vector belongs to one embedding space; one file could not hold two."""
    assert CACHE_FILENAMES["azure"] != CACHE_FILENAMES["injected"]
    weld_corpus(tmp_path)
    DocRetriever(docs_dir=tmp_path, embedder=TokenOverlapEmbedder()).retrieve(ticket())

    assert (tmp_path / CACHE_FILENAMES["injected"]).exists()
    assert not (tmp_path / CACHE_FILENAMES["azure"]).exists()


def test_a_changed_model_invalidates_the_cache(tmp_path):
    """Two Azure deployments share a filename but not an embedding space, so
    the model that wrote the vectors is recorded and checked."""
    chunks = [DocChunk("d.docx", "4.2", CRACKING)]
    path = tmp_path / CACHE_FILENAMES["injected"]
    save_cache(path, chunks, [[0.1, 0.2]], {
        "format": 2, "reader": resolver.DOC_READER_VERSION, "corpus_hash": "abc", "model": "azure:text-embedding-3-large",
    })

    assert load_cache(path, "abc", "azure:text-embedding-3-large") is not None
    assert load_cache(path, "abc", "azure:text-embedding-3-small") is None


def test_a_stale_hash_invalidates_the_cache(tmp_path):
    path = tmp_path / CACHE_FILENAMES["injected"]
    save_cache(path, [DocChunk("d.docx", "4.2", CRACKING)], [[0.1, 0.2]],
               {"format": 2, "reader": resolver.DOC_READER_VERSION, "corpus_hash": "abc", "model": "m"})

    assert load_cache(path, "different", "m") is None


def test_a_corrupt_cache_is_a_rebuild_not_a_crash(tmp_path, caplog):
    """The documents on disk are always the source of truth."""
    path = tmp_path / CACHE_FILENAMES["injected"]
    path.write_bytes(b"this is not an npz archive")

    with caplog.at_level("WARNING"):
        assert load_cache(path, "abc", "m") is None
    assert "unreadable cache" in caplog.text


def test_the_cache_round_trips_chunks_exactly(tmp_path):
    chunks = [
        DocChunk("0250-Weld-Standards.docx", "4.2 Weld Seam Cracking", CRACKING),
        DocChunk("0250-Packaging-Standards.docx", "2.1 Carton Labelling", PACKAGING),
    ]
    path = tmp_path / CACHE_FILENAMES["injected"]
    save_cache(path, chunks, [[1.0, 0.0], [0.0, 1.0]],
               {"format": 2, "reader": resolver.DOC_READER_VERSION, "corpus_hash": "h", "model": "m"})

    loaded, vectors = load_cache(path, "h", "m")
    assert loaded == chunks
    assert vectors.shape == (2, 2)


def test_the_cache_is_never_unpickled(tmp_path):
    """A cache file is data. A pickle loader would execute whatever it held."""
    import numpy as np

    path = tmp_path / CACHE_FILENAMES["injected"]
    save_cache(path, [DocChunk("d.docx", "s", CRACKING)], [[1.0]],
               {"format": 2, "reader": resolver.DOC_READER_VERSION, "corpus_hash": "h", "model": "m"})

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) >= {"vectors", "documents", "sections", "texts", "meta"}
        assert json.loads(str(archive["meta"]))["corpus_hash"] == "h"


def test_an_unwritable_cache_directory_still_answers(tmp_path, monkeypatch):
    """A read-only deployment works; it just re-embeds every run."""
    weld_corpus(tmp_path)

    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("sps.save_cache", refuse)
    retriever = DocRetriever(docs_dir=tmp_path, embedder=TokenOverlapEmbedder())

    assert retriever.retrieve(ticket()) is not None
    assert retriever.stats.chunks == 2


# ------------------------------------------------------------------- search


def test_the_query_is_enriched_with_the_issue_type():
    assert build_query(ticket(issue="Quality")) == (
        f"Issue Type: Quality | Defect: {PROBLEM}"
    )


def test_a_ticket_without_an_issue_type_still_queries():
    assert build_query(ticket(issue="")) == f"Defect: {PROBLEM}"


def test_tier2_has_its_own_threshold():
    """A 300-word standards chunk that genuinely answers the ticket scores far
    lower than two matching defect sentences do, so Tier 2 cannot share Tier 1's
    number. One per tier -- and none per encoder, since production has one."""
    assert TIER2_AZURE_THRESHOLD == 0.35

    fields = DocRetriever.__dataclass_fields__
    assert "local_threshold" not in fields
    assert fields["azure_threshold"].default == TIER2_AZURE_THRESHOLD
    assert fields["confidence_threshold"].default is None


def test_chunks_below_the_gate_are_withheld(tmp_path):
    weld_corpus(tmp_path)
    retriever = DocRetriever(
        docs_dir=tmp_path, embedder=TokenOverlapEmbedder(), confidence_threshold=0.99
    )

    assert retriever.retrieve(ticket()) == []
    # The score is still measured, which is what calibration needs.
    assert retriever.stats.top_score > 0
    assert retriever.stats.threshold_used == 0.99


def test_results_come_back_best_first(tmp_path):
    weld_corpus(tmp_path)
    retriever = DocRetriever(
        docs_dir=tmp_path, embedder=TokenOverlapEmbedder(), confidence_threshold=0.0
    )
    hits = retriever.retrieve(ticket())

    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert "Weld" in hits[0].chunk.document


def test_top_k_caps_what_reaches_the_model(tmp_path):
    directory = tmp_path / "many"
    directory.mkdir()
    write_doc(
        directory / "0250-Many.docx",
        [(f"4.{i} Section", [CRACKING]) for i in range(9)],
    )
    retriever = DocRetriever(
        docs_dir=directory, embedder=TokenOverlapEmbedder(),
        confidence_threshold=0.0, top_k=3,
    )

    assert len(retriever.retrieve(ticket())) == 3


def test_an_empty_corpus_is_not_available(tmp_path):
    assert DocRetriever(docs_dir=tmp_path).available is False
    assert DocRetriever(docs_dir=tmp_path).retrieve(ticket()) == []


def test_a_populated_corpus_is_available(tmp_path):
    weld_corpus(tmp_path)
    assert DocRetriever(docs_dir=tmp_path).available is True


def test_a_failed_corpus_batch_aborts_rather_than_ranking_half_a_corpus(tmp_path):
    """The query encodes, then the corpus batch fails. Nothing is ranked.

    There is no second encoder to reach for now, so the only wrong answer
    available is ranking the query against a partial corpus -- which would look
    like a successful search that simply found nothing relevant.
    """
    from sps import AzureEmbeddingError

    weld_corpus(tmp_path)
    retriever = DocRetriever(docs_dir=tmp_path)
    calls = []

    def azure(texts):
        calls.append(len(texts))
        if len(texts) == 1:                      # the query encodes
            class S:
                deployment = "text-embedding-3-large"
            return S(), [[1.0, 0.0]]
        raise AzureEmbeddingError("corpus batch failed")   # the corpus does not

    retriever._embed_azure = azure
    with pytest.raises(AzureEmbeddingError):
        retriever.retrieve(ticket())

    # Both calls were attempted, and the failure was not swallowed into an
    # empty result the caller would read as "no standard covers this".
    assert calls == [1, 2]


def test_missing_credentials_are_their_own_error(tmp_path, monkeypatch):
    """Distinguished from a transient failure so the caller can exit 2 instead
    of asking a robot to retry its way to an API key."""
    from sps import AzureEmbeddingError, AzureEmbeddingNotConfigured

    for name in ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY",
                 "AZURE_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    weld_corpus(tmp_path)

    with pytest.raises(AzureEmbeddingNotConfigured) as info:
        DocRetriever(docs_dir=tmp_path).retrieve(ticket())

    assert issubclass(AzureEmbeddingNotConfigured, AzureEmbeddingError)
    assert "AZURE_EMBEDDING_API_KEY" in str(info.value)


# ------------------------------------------------------------------ prompts


def test_the_tier2_actor_must_acknowledge_the_history_gap():
    from sps_prompts import TIER2_ACTOR_SYSTEM_PROMPT

    assert "Historical records yielded no resolution." in TIER2_ACTOR_SYSTEM_PROMPT


def test_the_tier2_actor_is_told_a_limit_is_not_a_disposition():
    """The likeliest wrong answer here is the engineering-plausible one: a
    standard that states a tolerance does not state what to do about a part
    that misses it."""
    from sps_prompts import TIER2_ACTOR_SYSTEM_PROMPT

    assert "A STANDARD IS NOT AUTOMATICALLY A SOLUTION" in TIER2_ACTOR_SYSTEM_PROMPT


def test_the_tier2_judge_checks_citations_and_dispositions():
    from sps_prompts import TIER2_JUDGE_SYSTEM_PROMPT

    assert "CITATION INTEGRITY" in TIER2_JUDGE_SYSTEM_PROMPT
    assert "UNGROUNDED DISPOSITION" in TIER2_JUDGE_SYSTEM_PROMPT


def test_the_rendered_context_shows_each_citation():
    from sps_prompts import build_tier2_actor_messages

    chunk = ScoredChunk(DocChunk("0250-Weld.docx", "4.2 Cracking", CRACKING), 0.81)
    messages = build_tier2_actor_messages(ticket(), [chunk])
    user = messages[1]["content"]

    assert "[0250-Weld.docx § 4.2 Cracking]" in user
    assert "ISSUE TYPE" in user and "Quality" in user
    assert "usable resolution" in user and "HISTORICAL SPS RECORDS" in user


def test_the_tier2_grounding_carries_its_own_abstention_reason():
    from sps import documentation_grounding

    assert "0250" in documentation_grounding([]).abstention_reason


# -------------------------------------------------------- through the CLI


class _Draft:
    def __init__(self, recommendation, justification):
        self.recommendation = recommendation
        self.justification = justification


class _Outcome:
    def __init__(
        self,
        draft=None,
        failure_reason="",
        infrastructure_failure=False,
        stop_reason="JUDGE_REFUSED",
    ):
        self.draft = draft
        self.attempts = 1
        self.critiques = []
        self.failure_reason = failure_reason
        self.infrastructure_failure = infrastructure_failure
        self.stop_reason = stop_reason
        self.tier = "0250"

    @property
    def succeeded(self):
        return self.draft is not None


@pytest.fixture
def tiered_llm(monkeypatch):
    """Script Tier 1 and Tier 2 independently."""

    def install(tier1: _Outcome, tier2: _Outcome):
        seen = {}

        class Loop:
            def __init__(self, *a, **k):
                pass

            async def run(self, ticket, candidates):
                seen["tier1"] = list(candidates)
                return tier1

            async def run_grounded(self, ticket, grounding):
                seen["tier2"] = list(grounding.items)
                return tier2

        # Tier 1 no longer runs the Actor/Judge loop -- it scores intent and
        # cascades. The tier1 _Outcome is still how these tests say "Tier 1
        # succeeds" or "Tier 1 fails", so it is translated into the intent
        # verdict that produces the same outcome.
        async def fake_score_intent(client, ticket, candidates):
            from sps import IntentOutcome, ScoredCandidate

            seen["tier1"] = list(candidates)
            if tier1.infrastructure_failure:
                return IntentOutcome(failure_reason=tier1.failure_reason)
            # Comfortably either side of any threshold these tests set, so a
            # change to the default gate does not silently flip them.
            top = 0.95 if tier1.draft is not None else 0.10
            return IntentOutcome(
                ticket_intent="stubbed ticket intent",
                scored=tuple(
                    ScoredCandidate(
                        candidate=c,
                        intent_match=top if i == 0 else 0.05,
                        reason="stubbed",
                    )
                    for i, c in enumerate(candidates)
                ),
            )

        monkeypatch.setattr("sps.ActorCriticLoop", Loop)
        monkeypatch.setattr("sps.AzureOpenAIChatClient", lambda *a, **k: object())
        monkeypatch.setattr("sps.score_intent", fake_score_intent)
        return seen

    return install


# Tier 2 is scoped to configured reason codes. These tests are about what
# Tier 2 does once it runs, so they configure the gate open and carry a
# matching code; the gate itself is tested in test_reason_code_gate.py.
TIER2_CODE = "RC-EQUIV"


@pytest.fixture(autouse=True)
def _tier2_reason_code_allowed(monkeypatch):
    monkeypatch.setenv("SPS_TIER2_REASON_CODES", TIER2_CODE)


def run_cli(tmp_path, docs_dir=None, threshold=0.99, part=PART,
            tier2_threshold=None, extra=(), reason_code=TIER2_CODE):
    """Threshold 0.99 by default so Tier 1 gates and Tier 2 is reached."""
    out = tmp_path / "out"
    # A near-duplicate, not a copy: identical text scores exactly 1.0 through
    # the stubbed Azure encoder and would clear the 0.99 gate these tests use
    # to force Tier 1 to fail.
    write_history(tmp_path / "h.xlsx", [row("SPS-1001", problem=NEAR_PROBLEM)])
    write_ticket(tmp_path / "t.xlsx", part=part, reason_code=reason_code)
    argv = [
        "--ticket-file", str(tmp_path / "t.xlsx"),
        "--history-file", str(tmp_path / "h.xlsx"),
        "--output-dir", str(out),
        "--threshold", str(threshold),
        *extra,
    ]
    if docs_dir is not None:
        argv += ["--docs-dir", str(docs_dir)]
    if tier2_threshold is not None:
        argv += ["--tier2-threshold", str(tier2_threshold)]
    return resolver.resolve_main(argv), out


def test_tier2_answers_when_tier1_finds_nothing(tmp_path, tiered_llm):
    tiered_llm(
        tier1=_Outcome(failure_reason="Historical records did not address it."),
        tier2=_Outcome(_Draft("1. Grind out the cracked seam.", "Historical records "
                              "yielded no resolution. Per 0250-Weld-Standards.docx.")),
    )
    docs = weld_corpus(tmp_path / "0250")

    code, out = run_cli(tmp_path, docs_dir=docs)

    assert code == resolver.EXIT_OK
    status = read_sheet(out / "status.xlsx")[0]
    assert status["Status"] == "PASS"
    assert status["Status_Code"] == "SUCCESS_0250_DOC"

    result = read_sheet(out / "output.xlsx")[0]
    assert result["Resolution_Source"] == "0250_DOCUMENTATION"
    assert "0250-Weld-Standards.docx" in result["Referenced_Sources"]
    assert "§" in result["Referenced_Sources"]
    # No historical record backs a standards answer; its sections do.
    assert all(result[c] == "" for c in resolver.RECORD_COLUMNS)
    assert result["Standard_Document"].startswith("[1] 0250-Weld-Standards.docx")
    assert result["Standard_Section"].startswith("[1] 4.2 Weld Seam Cracking")
    assert result["Standard_Section_Text"].startswith("[1] Cracking in a fillet")
    # Best first, so [1] is the section the confidence was measured on.
    assert result["Standard_Match_Percent"].split("\n")[0] == f"[1] {result['Confidence_Score']}"
    # One entry per section, in the same order, in every column.
    assert len({len(result[c].split("\n"))
                for c in ("Standard_Document", "Standard_Section", "Standard_Match_Percent")}) == 1


def test_the_standard_columns_list_every_section_by_number():
    """Every section the answer drew on, not the best one: shown alone, one
    would read as THE source when the Actor may have combined several."""
    result = resolver.PipelineResult(
        ai_recommendation="1. Grind out the seam.", justification="", confidence="42%",
        resolution_source=resolver.SOURCE_DOCUMENTATION,
        sections=(
            ScoredChunk(DocChunk("0250-Weld.docx", "4.2 Cracking", "Grind it out."), 0.42),
            ScoredChunk(DocChunk("0250-Weld.docx", "", "This standard covers welds."), 0.31),
        ),
    )

    assert resolver.standard_columns(result) == {
        "Standard_Document": "[1] 0250-Weld.docx\n[2] 0250-Weld.docx",
        "Standard_Section": "[1] 4.2 Cracking\n[2] (before the first heading)",
        "Standard_Section_Text": "[1] Grind it out.\n\n[2] This standard covers welds.",
        "Standard_Match_Percent": "[1] 42%\n[2] 31%",
    }


def test_a_part_with_no_history_can_still_be_answered_from_the_standards(tmp_path, tiered_llm):
    """The case the test above misses, because its part HAS history.

    With no history at all, Tier 1 leaves closest_match empty but present, and
    Tier 2 then supplies its own. Both reached ResolveOutcome as separate **
    expansions, which is a TypeError -- reported as an infrastructure fault, so
    the robot retried a ticket that would crash identically every time. Found
    by the golden master, not by any test that existed.
    """
    tiered_llm(
        tier1=_Outcome(failure_reason="unused: no history to score"),
        tier2=_Outcome(_Draft("1. Grind out the cracked seam.", "Historical records "
                              "yielded no resolution. Per 0250-Weld-Standards.docx.")),
    )
    docs = weld_corpus(tmp_path / "0250")

    code, out = run_cli(tmp_path, docs_dir=docs, part="0099-99999", tier2_threshold=0.05)

    assert code == resolver.EXIT_OK
    assert read_sheet(out / "status.xlsx")[0]["Status_Code"] == "SUCCESS_0250_DOC"
    assert read_sheet(out / "output.xlsx")[0]["Resolution_Source"] == "0250_DOCUMENTATION"


def test_the_citations_come_from_retrieval_not_from_the_model(tmp_path, tiered_llm):
    """The same reason confidence and SPS IDs are supplied from measurement in
    Tier 1: a model asked to author its own citation can invent one."""
    tiered_llm(
        tier1=_Outcome(failure_reason="none"),
        tier2=_Outcome(_Draft("1. Do the thing.",
                              "Per 0250-Invented-Document.docx section 9.9.")),
    )
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs)

    sources = read_sheet(out / "output.xlsx")[0]["Referenced_Sources"]
    assert "0250-Invented-Document.docx" not in sources
    assert "0250-Weld-Standards.docx" in sources


def test_the_reason_logs_the_best_score_from_both_tiers(tmp_path, tiered_llm):
    tiered_llm(tier1=_Outcome(failure_reason="none"), tier2=_Outcome())
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs, threshold=0.999)

    status = read_sheet(out / "status.xlsx")[0]
    # Tier 1 gated, but Tier 2 cleared its own gate and the Actor then declined:
    # the furthest stage reached was the audit, so that is what is reported.
    assert status["Status_Code"] == "LLM_AUDIT_REJECTED"
    assert "Best historical match" in status["Reason"]
    assert "Best 0250 match" in status["Reason"]


def test_no_corpus_reports_the_tier1_outcome(tmp_path, tiered_llm):
    """The normal state before the standards are loaded. Not an error."""
    tiered_llm(tier1=_Outcome(failure_reason="none"), tier2=_Outcome())
    empty = tmp_path / "empty"
    empty.mkdir()

    code, out = run_cli(tmp_path, docs_dir=empty)

    assert code == resolver.EXIT_OK
    status = read_sheet(out / "status.xlsx")[0]
    # Nothing retrieved from Tier 2, so Tier 1's own stage stands.
    assert status["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"
    assert "No 0250 documents found" in status["Reason"]


def test_no_tier2_restores_the_single_tier_behaviour(tmp_path, tiered_llm):
    tiered_llm(tier1=_Outcome(failure_reason="none"), tier2=_Outcome(_Draft("x", "y")))
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs, extra=["--no-tier2"])

    status = read_sheet(out / "status.xlsx")[0]
    assert status["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"
    assert "disabled by --no-tier2" in status["Reason"]
    # A concluded run, so it still leaves a row -- one that says no solution.
    assert read_sheet(out / "output.xlsx")[0]["Resolution_Source"] == "NONE"


def test_a_tier1_outage_does_not_fall_through_to_tier2(tmp_path, tiered_llm):
    """Tier 2 needs the same Azure deployment. Trying it would fail again,
    slower, and an outage must stay retryable rather than becoming a refusal."""
    seen = tiered_llm(
        tier1=_Outcome(failure_reason="Azure unreachable", infrastructure_failure=True),
        tier2=_Outcome(_Draft("x", "y")),
    )
    docs = weld_corpus(tmp_path / "0250")

    code, out = run_cli(tmp_path, docs_dir=docs, threshold=0.1)

    assert code == resolver.EXIT_INFRASTRUCTURE
    assert read_sheet(out / "status.xlsx")[0]["Status_Code"] == "INFRASTRUCTURE_ERROR"
    assert "tier2" not in seen


def test_tier1_success_never_reaches_tier2(tmp_path, tiered_llm):
    seen = tiered_llm(
        tier1=_Outcome(_Draft("1. Rework the seam.", "From SPS-1001.")),
        tier2=_Outcome(_Draft("should not be used", "")),
    )
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs, threshold=0.1)

    status = read_sheet(out / "status.xlsx")[0]
    assert status["Status_Code"] == "SUCCESS_HISTORICAL"
    assert read_sheet(out / "output.xlsx")[0]["Resolution_Source"] == "HISTORICAL_DATA"
    assert "tier2" not in seen


def test_a_broken_corpus_does_not_become_an_infrastructure_fault(tmp_path, tiered_llm):
    """Tier 2 is a fallback. A corpus it cannot read must not turn a legitimate
    'no resolution' into something the robot retries forever."""
    tiered_llm(tier1=_Outcome(failure_reason="none"), tier2=_Outcome())
    docs = weld_corpus(tmp_path / "0250")

    import sps as doc_cache

    original = doc_cache.DocRetriever.retrieve

    def explode(self, ticket):
        raise RuntimeError("corpus on fire")

    doc_cache.DocRetriever.retrieve = explode
    try:
        code, out = run_cli(tmp_path, docs_dir=docs)
    finally:
        doc_cache.DocRetriever.retrieve = original

    assert code == resolver.EXIT_OK
    status = read_sheet(out / "status.xlsx")[0]
    assert status["Status_Code"] == "BELOW_CONFIDENCE_THRESHOLD"
    assert "Tier 2 unavailable" in status["Reason"]


def test_status_stays_pass_or_fail_for_both_tiers():
    """A workflow branching on Status is unaffected by the new codes."""
    assert resolver.SUCCESS_CODES == {"SUCCESS_HISTORICAL", "SUCCESS_0250_DOC"}


# ------------------------------------------------- the routing state machine


UNKNOWN_PART = "0099-99999"


def _empty_dir(tmp_path):
    directory = tmp_path / "no_docs"
    directory.mkdir(exist_ok=True)
    return directory


@pytest.mark.parametrize(
    "tier1,tier2,expected",
    [
        # Neither tier had anything to offer. The part is unknown to history
        # and no standard covers it -- Master Data's problem.
        ("nothing", "nothing", "NO_MATCHES"),
        # Something was retrieved somewhere, but nothing cleared its gate.
        ("nothing", "gated", "BELOW_CONFIDENCE_THRESHOLD"),
        ("gated", "nothing", "BELOW_CONFIDENCE_THRESHOLD"),
        ("gated", "gated", "BELOW_CONFIDENCE_THRESHOLD"),
        # The maths was satisfied somewhere and the Actor or Judge still
        # refused. A human reviewer's problem, not a data one.
        ("nothing", "rejected", "LLM_AUDIT_REJECTED"),
        ("gated", "rejected", "LLM_AUDIT_REJECTED"),
        ("rejected", "nothing", "LLM_AUDIT_REJECTED"),
        ("rejected", "gated", "LLM_AUDIT_REJECTED"),
        ("rejected", "rejected", "LLM_AUDIT_REJECTED"),
    ],
)
def test_the_reported_code_is_the_furthest_stage_either_tier_reached(
    tmp_path, tiered_llm, tier1, tier2, expected
):
    """The robot routes on this column, so the combination rule is a contract.

    A run where history reached the Judge and was refused, while the standards
    had nothing to say, is an audit rejection -- reporting NO_MATCHES would
    send a perfectly well-known part to Master Data.
    """
    tiered_llm(tier1=_Outcome(failure_reason="tier 1 declined."),
               tier2=_Outcome(failure_reason="tier 2 declined."))

    # Tier 1: unknown part retrieves nothing; 0.99 gates what it does retrieve;
    # 0.1 lets it through to the Actor, which the fixture makes decline.
    part = UNKNOWN_PART if tier1 == "nothing" else PART
    threshold = {"nothing": 0.5, "gated": 0.99, "rejected": 0.1}[tier1]

    # Tier 2: an empty folder retrieves nothing; 0.99 gates the demo corpus;
    # 0.1 lets it through.
    docs = _empty_dir(tmp_path) if tier2 == "nothing" else weld_corpus(tmp_path / "0250")
    tier2_threshold = {"nothing": None, "gated": 0.99, "rejected": 0.1}[tier2]

    code, out = run_cli(
        tmp_path, docs_dir=docs, threshold=threshold, part=part,
        tier2_threshold=tier2_threshold,
    )

    assert code == resolver.EXIT_OK
    assert read_sheet(out / "status.xlsx")[0]["Status_Code"] == expected


def test_the_reason_still_carries_both_tiers_whatever_the_code(tmp_path, tiered_llm):
    """The code is for the robot's switch; the Reason is for the human who has
    to decide what to do about it."""
    tiered_llm(tier1=_Outcome(failure_reason="tier 1 declined."), tier2=_Outcome())
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs, threshold=0.99, tier2_threshold=0.99)

    reason = read_sheet(out / "status.xlsx")[0]["Reason"]
    assert "Best historical match 0.9354" in reason
    assert "Best 0250 match" in reason


def test_a_stage_maps_to_exactly_one_code():
    """Ordered, and the order is the routing precedence."""
    assert resolver.STAGE_NOTHING < resolver.STAGE_GATED < resolver.STAGE_REJECTED
    assert resolver.STAGE_CODES == {
        resolver.STAGE_NOTHING: "NO_MATCHES",
        resolver.STAGE_GATED: "BELOW_CONFIDENCE_THRESHOLD",
        resolver.STAGE_REJECTED: "LLM_AUDIT_REJECTED",
    }


def test_every_status_code_the_robot_can_see_is_named():
    """The full switch a UiPath workflow has to handle: two success codes,
    three exhaustion codes, and the two that were never tier-specific."""
    codes = {
        resolver.CODE_SUCCESS_HISTORICAL,
        resolver.CODE_SUCCESS_DOC,
        resolver.CODE_NO_MATCHES,
        resolver.CODE_BELOW_THRESHOLD,
        resolver.CODE_AUDIT_REJECTED,
        resolver.CODE_INVALID_INPUT,
        resolver.CODE_INFRASTRUCTURE,
    }
    assert codes == {
        "SUCCESS_HISTORICAL", "SUCCESS_0250_DOC", "NO_MATCHES",
        "BELOW_CONFIDENCE_THRESHOLD", "LLM_AUDIT_REJECTED",
        "INVALID_INPUT", "INFRASTRUCTURE_ERROR",
    }
    assert set(resolver.STAGE_CODES.values()) <= codes


def test_a_tier2_only_encode_still_names_its_encoder(tmp_path, tiered_llm):
    """An unknown part stops Tier 1 before it embeds anything. Support counts
    Azure fallbacks from Embedding_Model, so a run where only Tier 2 encoded
    must not report an empty one."""
    tiered_llm(tier1=_Outcome(failure_reason="none"), tier2=_Outcome())
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs, part=UNKNOWN_PART, tier2_threshold=0.99)

    status = read_sheet(out / "status.xlsx")[0]
    assert status["Embedding_Model"].startswith("azure:")
    assert status["Reason"].endswith("[Azure]")


def test_tier1_keeps_naming_the_encoder_when_it_did_embed(tmp_path, tiered_llm):
    """Tier 2 must not overwrite an encoder Tier 1 legitimately reported."""
    tiered_llm(tier1=_Outcome(failure_reason="none"), tier2=_Outcome())
    docs = weld_corpus(tmp_path / "0250")

    _, out = run_cli(tmp_path, docs_dir=docs, threshold=0.99, tier2_threshold=0.99)

    assert read_sheet(out / "status.xlsx")[0]["Embedding_Model"].startswith("azure:")
