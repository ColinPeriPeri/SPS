"""Acceptance checks for the embedder the pipeline actually uses.

Everything else in the suite runs against a deterministic stand-in. This script
exercises a real encoder, to confirm the properties the ranking maths depends
on and to print real cosine numbers against the live gate -- which is the
fastest way to find out whether a threshold is anywhere near right on a new
deployment.

    python -m scripts.verify_embedder                # the Azure deployment
    python -m scripts.verify_embedder --quiet
    python -m scripts.verify_embedder --local        # the dormant bge-small path
    python -m scripts.verify_embedder --probe        # the calibration sweep

AZURE (the default, and the only encoder in the pipeline) checks that the
deployment is reachable, that its vectors are unit length so the NumPy matmul
really is a cosine, that a batch comes back in the order it was sent, and then
scores real SPS text against the 0.50 Tier-1 gate and a standards passage
against the 0.35 Tier-2 gate. Both of those numbers are unmeasured guesses, so
this is the first honest look at them.

PROBE is for setting those numbers rather than sanity-checking them. It scores
labelled pairs -- your judgement of which belong together -- and reports the
gap between the weakest pair you called a match and the strongest you did not.
That gap is the only thing a threshold can live in. If it is negative, no
threshold works and the probe says so instead of suggesting one.

LOCAL is the bge-small path, kept working but out of the pipeline. It needs
torch and sentence-transformers, which the pruned requirements.txt no longer
installs, so it reports what to install rather than dying on an ImportError.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from sps.config import EmbeddingSettings, BGE_QUERY_INSTRUCTION

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


def verify_azure(checks: "Checks", quiet: bool) -> None:
    """Check the live Azure deployment, and read the gates against it."""
    from sps.config import AzureEmbeddingSettings
    from sps.embedding import AzureEmbedder, AzureEmbeddingError
    from sps.retrieval.doc_cache import TIER2_AZURE_THRESHOLD
    from sps.retrieval.in_memory import AZURE_EMBEDDING_THRESHOLD

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
    # Precision, and the harder half of it. `unrelated` is a packaging defect
    # against a weld defect -- any encoder separates those. `near` is weld
    # porosity against weld cracking: same part, same inspection, same
    # vocabulary, different fix. That is the shape of the near-miss that
    # produced two wrong recommendations on real tickets, and it is the pair
    # a gate has to get right to be worth having.
    checks.check(
        f"a different defect on the same part is blocked by the {AZURE_EMBEDDING_THRESHOLD} gate",
        scored["near"] < AZURE_EMBEDDING_THRESHOLD,
        f"{scored['near']:.4f} -- porosity vs cracking; if this FAILS the gate cannot "
        f"tell two defects apart. Run --probe before changing the number.",
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

    # The floor is the HIGHEST thing that must be blocked, not the lowest.
    # `unrelated` alone would flatter the gate: a packaging defect scores far
    # below a weld defect, so a gate that clears it may still admit `near`.
    must_admit = min(scored["exact"], scored["paraphrase"])
    must_block = max(scored["unrelated"], scored["near"])
    if must_block < must_admit:
        print(
            f"\n  Suggested gates from this run: Tier 1 between "
            f"{must_block:.4f} and {must_admit:.4f}, "
            f"Tier 2 between {unrelated:.4f} and {relevant:.4f}."
        )
    else:
        print(
            f"\n  No Tier-1 gate works on this sample: the weakest text that must be "
            f"admitted scores {must_admit:.4f}, below the {must_block:.4f} of one that "
            f"must be blocked. A threshold cannot separate them."
        )
    print("  One sample each -- widen it with --probe, then scripts/run_eval_batch.py.")


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

DEFAULT_PAIRS = Path(__file__).resolve().parent.parent / "samples" / "similarity_pairs.csv"


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
    from sps.file_reader import read_header_and_rows

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
        """Just above the strongest thing that must be blocked.

        Deliberately not the midpoint. The midpoint treats a missed match and
        a false match as equally bad; here they are not. A missed match costs
        a refusal, which a reviewer sees and can act on. A false match costs a
        wrong instruction sent to a supplier, which looks exactly like a right
        one. Sitting just above the no-match ceiling buys precision with
        whatever recall the margin can spare.
        """
        return round(self.strongest_no_match.cosine + 0.01, 4)


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
        flag = "" if admits == wanted else "   <-- WRONG"
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

    if sep.separable:
        blocked = [s for s in scored if s.pair.expect == MATCH and s.cosine < sep.suggested]
        print(f"\n  A gate exists. Suggested: {sep.suggested} (just above the no-match ceiling).")
        if blocked:
            print(f"  It would also block {len(blocked)} pair(s) you called a match:")
            for s in blocked:
                print(f"    {s.pair.label:20s} {s.cosine:.4f}")
            print("  That is the recall this precision costs. Judge it before committing.")
        else:
            print("  It blocks nothing you called a match.")
        return

    overlap = [
        s for s in scored
        if s.pair.expect == NO_MATCH and s.cosine >= sep.weakest_match.cosine
    ]
    print(
        f"\n  No threshold can separate these pairs. {len(overlap)} pair(s) you called "
        f"{NO_MATCH} score at or above the weakest {MATCH}:"
    )
    for s in sorted(overlap, key=lambda s: -s.cosine):
        print(f"    {s.pair.label:20s} {s.cosine:.4f}  >=  {sep.weakest_match.cosine:.4f}")
    print(
        "\n  This is the evidence that retrieval itself has to change -- moving the\n"
        "  number can only trade one error for the other. Check WHICH classes\n"
        "  overlap before deciding what to change."
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
    from service.excel_output import write_rows

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
                "Gate_Agrees": "yes" if (s.cosine >= gate) == (s.pair.expect == MATCH) else "NO",
                "Text_A": s.pair.text_a,
                "Text_B": s.pair.text_b,
            }
            for s in sorted(scored, key=lambda s: -s.cosine)
        ],
    )
    print(f"\n  Wrote {path}")


def run_probe(pairs_path: Path, out_path: Path | None, checks: "Checks") -> None:
    from sps.config import AzureEmbeddingSettings
    from sps.embedding import AzureEmbedder, AzureEmbeddingError
    from sps.retrieval.in_memory import AZURE_EMBEDDING_THRESHOLD

    # Pairs first. It is local and free, and a mistyped filename should not
    # need credentials to discover.
    #
    # FileReadError and UnsupportedFileType are the reader's own -- a missing
    # file and a .pdf respectively. Neither is an OSError, so leaving them out
    # turns a typo'd filename into a traceback.
    from sps.file_reader import FileReadError, UnsupportedFileType

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the embedder the pipeline uses")
    parser.add_argument("--quiet", action="store_true", help="suppress the similarity table")
    parser.add_argument(
        "--local",
        action="store_true",
        help="check the dormant bge-small path instead of Azure "
        "(needs torch + sentence-transformers, which requirements.txt no longer installs)",
    )
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
    from sps.config import load_env_file

    env_file = load_env_file()

    checks = Checks()
    if args.probe:
        if args.local:
            parser.error("--probe runs against Azure; it does not support --local")
        print("Probing the Azure embedding deployment.")
        print(f"  .env: {env_file if env_file else 'not found -- using the environment only'}\n")
        run_probe(Path(args.probe), Path(args.out) if args.out else None, checks)
        return _report(checks)

    if not args.local:
        print("Checking the Azure embedding deployment.")
        print(f"  .env: {env_file if env_file else 'not found -- using the environment only'}\n")
        verify_azure(checks, args.quiet)
        return _report(checks)

    try:
        from sps.embedding import BGEEmbedder
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        print(f"The local path needs torch and sentence-transformers: {exc}")
        print("  pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install sentence-transformers==3.0.1")
        return 1

    return _verify_local(checks, args, BGEEmbedder)


def _report(checks: "Checks") -> int:
    print()
    if checks.failures:
        print(f"FAILED: {len(checks.failures)} check(s): {', '.join(checks.failures)}")
        return 1
    print("All embedder checks passed.")
    return 0


def _verify_local(checks: "Checks", args, BGEEmbedder) -> int:
    settings = EmbeddingSettings()

    print(f"Loading {settings.model_name} on {settings.device} ...")
    started = time.time()
    embedder = BGEEmbedder(settings)
    _ = embedder.model  # force the download / load
    print(f"Loaded in {time.time() - started:.1f}s\n")

    # ---------------------------------------------------------------- 1. dims
    print("1. Dimensions")
    reported = embedder.model.get_sentence_embedding_dimension()
    checks.check("model reports 384 dimensions", reported == 384, f"got {reported}")

    passages = embedder.embed_passages([PROBLEM, PARAPHRASE, UNRELATED])
    checks.check(
        "embed_passages returns 384-dim vectors",
        all(len(v) == 384 for v in passages),
        f"lengths {[len(v) for v in passages]}",
    )
    query_vec = embedder.embed_query(PROBLEM)
    checks.check("embed_query returns a 384-dim vector", len(query_vec) == 384,
                 f"got {len(query_vec)}")
    checks.check("configured dimension matches the model", embedder.dimension == reported)

    # --------------------------------------------------------- 2. normalization
    print("\n2. Normalization")
    norms = [l2(v) for v in passages]
    checks.check(
        "passage vectors are unit length",
        all(abs(n - 1.0) < TOLERANCE for n in norms),
        f"norms {[round(n, 8) for n in norms]}",
    )
    checks.check(
        "query vector is unit length",
        abs(l2(query_vec) - 1.0) < TOLERANCE,
        f"norm {l2(query_vec):.8f}",
    )
    # If vectors are normalized, dot product == cosine -- the assumption the
    # NumPy matmul ranking and the 0.89 gate both rest on.
    d, c = dot(passages[0], passages[1]), cosine(passages[0], passages[1])
    checks.check(
        "dot product equals cosine similarity",
        abs(d - c) < TOLERANCE,
        f"dot={d:.8f} cosine={c:.8f}",
    )

    # ------------------------------------------------------ 3. query instruction
    print("\n3. BGE query-instruction prefix")
    plain = replace(settings, use_query_instruction=False)
    plain_embedder = BGEEmbedder(plain)
    plain_embedder._model = embedder.model  # reuse the loaded weights

    prefixed_q = embedder.embed_query(PROBLEM)
    plain_q = plain_embedder.embed_query(PROBLEM)
    checks.check(
        "prefix changes the query vector (it is actually applied)",
        cosine(prefixed_q, plain_q) < 0.9999,
        f"cosine(prefixed, plain) = {cosine(prefixed_q, plain_q):.6f}",
    )

    # The passage side must be identical to an unprefixed query: proof that
    # embed_passages never prepends the instruction.
    passage_only = embedder.embed_passages([PROBLEM])[0]
    checks.check(
        "embed_passages applies NO prefix",
        cosine(passage_only, plain_q) > 1 - TOLERANCE,
        f"cosine(passage, unprefixed query) = {cosine(passage_only, plain_q):.8f}",
    )
    checks.check(
        "prefix constant is the documented BGE string",
        BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: ",
    )

    # ------------------------------------------------- 4. retrieval sanity/calibration
    print("\n4. Cosine behaviour on real SPS text")
    corpus = {"exact": PROBLEM, "paraphrase": PARAPHRASE, "near": NEAR, "unrelated": UNRELATED}
    vectors = dict(zip(corpus, embedder.embed_passages(list(corpus.values()))))

    scored_prefixed = {k: cosine(prefixed_q, v) for k, v in vectors.items()}
    scored_plain = {k: cosine(plain_q, v) for k, v in vectors.items()}

    if not args.quiet:
        print(f"\n    {'candidate':12s} {'with prefix':>12s} {'no prefix':>12s}")
        for key in corpus:
            print(f"    {key:12s} {scored_prefixed[key]:12.4f} {scored_plain[key]:12.4f}")
        print()

    checks.check(
        "exact match outranks the unrelated record",
        scored_prefixed["exact"] > scored_prefixed["unrelated"],
        f"{scored_prefixed['exact']:.4f} > {scored_prefixed['unrelated']:.4f}",
    )
    checks.check(
        "paraphrase outranks the unrelated record",
        scored_prefixed["paraphrase"] > scored_prefixed["unrelated"],
        f"{scored_prefixed['paraphrase']:.4f} > {scored_prefixed['unrelated']:.4f}",
    )
    checks.check(
        "exact match clears the 0.89 gate",
        scored_prefixed["exact"] >= 0.89,
        f"{scored_prefixed['exact']:.4f}",
    )
    checks.check(
        "unrelated record is blocked by the 0.89 gate",
        scored_prefixed["unrelated"] < 0.89,
        f"{scored_prefixed['unrelated']:.4f}",
    )
    checks.check(
        "ordering is identical with and without the prefix",
        sorted(scored_prefixed, key=scored_prefixed.get, reverse=True)
        == sorted(scored_plain, key=scored_plain.get, reverse=True),
    )

    # --------------------------------------------------------------- 5. determinism
    print("\n5. Determinism")
    again = embedder.embed_query(PROBLEM)
    checks.check(
        "repeated encoding is bit-stable",
        all(abs(a - b) < 1e-9 for a, b in zip(prefixed_q, again)),
    )

    return _report(checks)


if __name__ == "__main__":
    sys.exit(main())
