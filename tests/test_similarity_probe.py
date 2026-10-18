"""The similarity probe: pair parsing, the margin maths, and what it reports.

None of this touches Azure. The encoder is a dict of hand-written vectors, so
every cosine in here is a number chosen on purpose -- which is the only way to
test a conclusion like "no threshold can separate these" without depending on
what some deployment happens to return today.
"""

from __future__ import annotations

import math

import pytest

from sps import (
    MATCH,
    NO_MATCH,
    Pair,
    PairFileError,
    Scored,
    read_pairs,
    report_probe,
    score_pairs,
    separation,
)

HEADER = "Class,Text_A,Text_B,Expect\n"


def write_pairs(tmp_path, body: str, name: str = "pairs.csv"):
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def unit(x: float, y: float) -> list[float]:
    """A 2-D unit vector at the given point, so cosines are exact."""
    norm = math.hypot(x, y)
    return [x / norm, y / norm]


def encoder(table: dict[str, list[float]]):
    """A stand-in encoder that also records what it was asked to encode."""
    calls: list[list[str]] = []

    def encode(texts):
        calls.append(list(texts))
        return [table[t] for t in texts]

    encode.calls = calls
    return encode


# ---------------------------------------------------------------- parsing


def test_reads_a_well_formed_file(tmp_path):
    path = write_pairs(tmp_path, "reorder,alpha,beta,match\nnegation,gamma,delta,no-match\n")
    pairs = read_pairs(path)
    assert [p.label for p in pairs] == ["reorder", "negation"]
    assert pairs[0] == Pair("reorder", "alpha", "beta", MATCH)
    assert pairs[1].expect == NO_MATCH


def test_expect_is_case_insensitive(tmp_path):
    path = write_pairs(tmp_path, "a,one,two,MATCH\nb,three,four,No-Match\n")
    assert [p.expect for p in read_pairs(path)] == [MATCH, NO_MATCH]


def test_headers_are_case_and_order_insensitive(tmp_path):
    """The file is going to be pasted together in Excel, where nobody preserves
    a column order they cannot see the point of."""
    path = tmp_path / "shuffled.csv"
    path.write_text("EXPECT,text_b,Text_A,class\nmatch,beta,alpha,reorder\n", encoding="utf-8")
    assert read_pairs(path) == [Pair("reorder", "alpha", "beta", MATCH)]


def test_blank_rows_are_skipped(tmp_path):
    path = write_pairs(tmp_path, "a,one,two,match\n,,,\nb,three,four,no-match\n")
    assert len(read_pairs(path)) == 2


def test_missing_column_names_it(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Class,Text_A,Expect\nreorder,alpha,match\n", encoding="utf-8")
    with pytest.raises(PairFileError, match="text_b"):
        read_pairs(path)


def test_bad_expect_names_the_row(tmp_path):
    path = write_pairs(tmp_path, "a,one,two,match\nb,three,four,probably\n")
    with pytest.raises(PairFileError) as exc:
        read_pairs(path)
    # Row 3, not row 2: the header is row 1, and an off-by-one here sends
    # someone to the wrong line of a 200-row sheet.
    assert "row 3" in str(exc.value)
    assert "probably" in str(exc.value)


def test_blank_text_names_the_row(tmp_path):
    path = write_pairs(tmp_path, "a,one,,match\n")
    with pytest.raises(PairFileError, match="row 2"):
        read_pairs(path)


def test_header_only_file_is_rejected(tmp_path):
    with pytest.raises(PairFileError, match="no rows"):
        read_pairs(write_pairs(tmp_path, ""))


def test_reads_xlsx_too(tmp_path):
    """Reuses the pipeline's reader, so .xlsx has to work with no extra code."""
    import openpyxl

    path = tmp_path / "pairs.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Class", "Text_A", "Text_B", "Expect"])
    sheet.append(["reorder", "alpha", "beta", "match"])
    book.save(path)
    assert read_pairs(path) == [Pair("reorder", "alpha", "beta", MATCH)]


# ---------------------------------------------------------------- scoring


def test_each_distinct_text_is_encoded_exactly_once():
    """A pair set built around one ticket repeats its text on every row. Paying
    per row rather than per text would be most of the bill."""
    table = {"q": unit(1, 0), "a": unit(1, 1), "b": unit(0, 1)}
    encode = encoder(table)
    pairs = [
        Pair("one", "q", "a", MATCH),
        Pair("two", "q", "b", NO_MATCH),
        Pair("three", "q", "a", MATCH),  # same texts as "one"
    ]
    score_pairs(pairs, encode)
    assert encode.calls == [["q", "a", "b"]]


def test_cosine_is_the_reported_number():
    table = {"q": unit(1, 0), "a": unit(1, 1)}
    scored = score_pairs([Pair("p", "q", "a", MATCH)], encoder(table))
    assert scored[0].cosine == pytest.approx(math.sqrt(0.5))


def test_pair_order_is_preserved():
    table = {"q": unit(1, 0), "a": unit(1, 1), "b": unit(0, 1)}
    pairs = [Pair("one", "q", "a", MATCH), Pair("two", "q", "b", NO_MATCH)]
    assert [s.pair.label for s in score_pairs(pairs, encoder(table))] == ["one", "two"]


# ---------------------------------------------------------------- the margin


def scored(*values: tuple[str, float, str]) -> list[Scored]:
    return [Scored(Pair(label, "a", "b", expect), value) for label, value, expect in values]


def test_positive_margin_is_separable():
    sep = separation(scored(
        ("lexical", 0.88, MATCH),
        ("verbose", 0.74, MATCH),
        ("negation", 0.61, NO_MATCH),
    ))
    assert sep.separable
    assert sep.margin == pytest.approx(0.13)
    assert sep.weakest_match.pair.label == "verbose"
    assert sep.strongest_no_match.pair.label == "negation"


def test_the_suggested_gate_sits_just_below_the_weakest_match():
    """The Tier-1 cosine is a recall filter: what it admits, the intent scorer
    judges, and what it blocks nobody ever judges. So it sits under the weakest
    match -- not above the strongest no-match, as it did when it decided."""
    sep = separation(scored(("m", 0.90, MATCH), ("w", 0.74, MATCH), ("n", 0.61, NO_MATCH)))
    assert sep.suggested == 0.73
    # A negative margin no longer leaves it with nothing to suggest.
    overlapping = separation(scored(("w", 0.74, MATCH), ("n", 0.94, NO_MATCH)))
    assert overlapping.suggested == 0.73


def test_negative_margin_is_not_separable():
    sep = separation(scored(
        ("verbose", 0.74, MATCH),
        ("negation", 0.94, NO_MATCH),
    ))
    assert not sep.separable
    assert sep.margin == pytest.approx(-0.20)


def test_a_margin_of_exactly_zero_is_not_separable():
    """Equal scores mean the two are the same point. No cut lies between them."""
    sep = separation(scored(("m", 0.80, MATCH), ("n", 0.80, NO_MATCH)))
    assert sep.margin == 0
    assert not sep.separable


def test_margin_needs_both_populations():
    assert not separation(scored(("m", 0.9, MATCH))).measurable
    assert not separation(scored(("n", 0.3, NO_MATCH))).measurable
    assert separation(scored(("m", 0.9, MATCH), ("n", 0.3, NO_MATCH))).measurable


# ---------------------------------------------------------------- reporting


def test_report_names_every_overlapping_pair(capsys, tmp_path):
    report_probe(
        scored(
            ("lexical", 0.88, MATCH),
            ("verbose", 0.74, MATCH),
            ("negation", 0.94, NO_MATCH),
            ("changed-ask", 0.81, NO_MATCH),
            ("unrelated", 0.30, NO_MATCH),
        ),
        gate=0.50,
        source=tmp_path / "pairs.csv",
    )
    out = capsys.readouterr().out
    assert "NEGATIVE" in out
    # Both no-match pairs above the weakest match, and not the one below it.
    assert "negation" in out and "changed-ask" in out
    assert "2 no-match pair(s)" in out
    # Reported as the scorer's work, not as a reason to rebuild retrieval.
    assert "why the intent scorer decides" in out
    assert "Suggested recall gate: 0.73" in out


def test_report_names_the_no_match_pairs_the_scorer_must_turn_down(capsys, tmp_path):
    """The suggested gate admits them by design; naming them says what the
    intent scorer's evaluation set has to include."""
    report_probe(
        scored(
            ("lexical", 0.88, MATCH),
            ("verbose", 0.60, MATCH),
            ("negation", 0.70, NO_MATCH),
            ("unrelated", 0.20, NO_MATCH),
        ),
        gate=0.50,
        source=tmp_path / "pairs.csv",
    )
    out = capsys.readouterr().out
    assert "Suggested recall gate: 0.59" in out
    assert "admits 1 pair(s) you called no-match" in out
    assert "    negation" in out and "    unrelated" not in out


def test_report_marks_the_no_match_pairs_the_gate_admits(capsys, tmp_path):
    report_probe(
        scored(("negation", 0.94, NO_MATCH), ("lexical", 0.88, MATCH)),
        gate=0.50,
        source=tmp_path / "pairs.csv",
    )
    lines = capsys.readouterr().out.splitlines()
    negation = next(l for l in lines if l.strip().startswith("negation"))
    lexical = next(l for l in lines if l.strip().startswith("lexical"))
    assert "(scorer)" in negation   # admitted: the intent scorer's to turn down
    assert "MISSED" not in lexical


def test_only_a_blocked_match_is_flagged_as_missed(capsys, tmp_path):
    report_probe(scored(("paraphrase", 0.40, MATCH), ("n", 0.10, NO_MATCH)),
                 gate=0.50, source=tmp_path / "pairs.csv")
    paraphrase = next(l for l in capsys.readouterr().out.splitlines()
                      if l.strip().startswith("paraphrase"))
    assert "MISSED" in paraphrase


def test_report_survives_one_sided_input(capsys, tmp_path):
    report_probe(scored(("m", 0.9, MATCH)), gate=0.50, source=tmp_path / "pairs.csv")
    out = capsys.readouterr().out
    assert "No margin" in out
    assert "Traceback" not in out


# ---------------------------------------------------------- the file gate
#
# run_probe reads the pairs before it looks at credentials, so these reach the
# reader without a deployment. The bug being guarded: the reader raises its own
# FileReadError and UnsupportedFileType, neither of which is an OSError, so a
# mistyped filename came back as a traceback instead of a message.


@pytest.mark.parametrize(
    "name, expected",
    [
        ("nope.csv", "not found"),
        ("README.md", "unsupported type"),
    ],
)
def test_an_unusable_pairs_path_reports_instead_of_raising(name, expected, capsys):
    from sps import Checks, run_probe
    from pathlib import Path

    checks = Checks()
    run_probe(Path(name), None, checks)  # must not raise

    out = capsys.readouterr().out
    assert expected in out
    assert "Traceback" not in out
    assert checks.failures == ["pairs file is readable"]


def test_a_bad_pairs_file_is_rejected_before_credentials_are_needed(tmp_path, capsys, monkeypatch):
    """Ordering, not just handling. Discovering a typo should not require a
    configured deployment -- reading the file is local and free."""
    from sps import Checks, run_probe

    for var in ("AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY", "AZURE_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(var, raising=False)

    checks = Checks()
    run_probe(tmp_path / "missing.csv", None, checks)

    # The file complaint, not the credential complaint.
    assert checks.failures == ["pairs file is readable"]
    assert "is configured" not in capsys.readouterr().out


# ------------------------------------------------------- the shipped fixture


def test_shipped_pair_set_is_valid_and_two_sided():
    """The file the probe defaults to has to parse, and has to contain both
    populations -- a set with only matches in it cannot calibrate anything."""
    from sps import DEFAULT_PAIRS

    pairs = read_pairs(DEFAULT_PAIRS)
    expectations = {p.expect for p in pairs}
    assert expectations == {MATCH, NO_MATCH}
    assert len(pairs) >= 10
    # Every label distinct, so a row named in the report identifies one pair.
    labels = [p.label for p in pairs]
    assert len(labels) == len(set(labels))


def test_shipped_pair_set_covers_the_two_failures_that_motivated_it():
    from sps import DEFAULT_PAIRS

    by_label = {p.label: p for p in read_pairs(DEFAULT_PAIRS)}
    # Negation: near-identical form, opposite meaning.
    assert by_label["negation"].expect == NO_MATCH
    # The NTK case: same background paragraph, different request.
    changed = by_label["changed-ask"]
    assert changed.expect == NO_MATCH
    shared = "engraved with revision C"
    assert shared in changed.text_a and shared in changed.text_b
    assert changed.text_a != changed.text_b


# ------------------------------------------------------------ check's gates


def _fake_deployment(monkeypatch, near=0.70, unrelated=0.10):
    """An AzureEmbedder whose cosines against the check's query are set exactly."""
    import math

    import sps

    def at(cosine_to_query, axis=0):
        other = math.sqrt(max(1 - cosine_to_query ** 2, 0.0))
        return [cosine_to_query, other, 0.0] if axis == 0 else [0.0, other, cosine_to_query]

    table = {
        sps.PROBLEM: at(1.0), sps.PARAPHRASE: at(0.80), sps.NEAR: at(near),
        sps.UNRELATED: at(unrelated),
        sps.TIER2_QUERY: [0.0, 0.0, 1.0],
        sps.STANDARD_RELEVANT: at(0.60, axis=2), sps.STANDARD_UNRELATED: at(0.20, axis=2),
    }

    class Fake:
        def __init__(self, settings, client=None):
            pass

        def embed_batch(self, texts):
            return [table[t] for t in texts]

    monkeypatch.setattr(sps, "AzureEmbedder", Fake)
    for name, value in (("AZURE_EMBEDDING_ENDPOINT", "https://x.openai.azure.com/"),
                        ("AZURE_EMBEDDING_API_KEY", "k"),
                        ("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")):
        monkeypatch.setenv(name, value)


def test_a_different_defect_above_the_recall_gate_is_reported_not_failed(monkeypatch, capsys):
    """Porosity against cracking scoring above 0.30 is what a recall filter
    should do; the intent scorer separates them. It used to fail the check and
    tell the operator to tune the gate -- which would discard real precedent."""
    import sps

    _fake_deployment(monkeypatch, near=0.70)
    checks = sps.Checks()
    sps.verify_azure(checks, quiet=True)

    out = capsys.readouterr().out
    assert checks.failures == []
    assert "[INFO] a different defect on the same part" in out
    assert "Tier 1 between 0.1000 and 0.8000" in out


def test_an_unrelated_record_over_the_gate_still_fails(monkeypatch, capsys):
    """The one precision check a recall filter keeps: a gate that admits a
    packaging defect for a weld ticket admits everything."""
    import sps

    _fake_deployment(monkeypatch, unrelated=0.50)
    checks = sps.Checks()
    sps.verify_azure(checks, quiet=True)
    assert any("unrelated record is blocked" in f for f in checks.failures)
