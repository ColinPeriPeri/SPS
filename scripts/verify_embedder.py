"""Acceptance checks for the real BAAI/bge-small-en-v1.5 embedder.

Everything else in the suite runs against a deterministic stand-in. This script
exercises the actual model to confirm the three properties the pipeline's maths
depends on:

  1. 384 dimensions exactly.
  2. L2-normalized output -- ranking is a dot product of normalised vectors,
     which is the cosine the 0.89 gate is calibrated on.
  3. The BGE query-instruction prefix is applied to queries and NOT to passages.

It also reports real cosine numbers for related vs unrelated SPS text, so the
0.89 threshold can be sanity-checked against the actual embedding space.

    python -m scripts.verify_embedder            # downloads ~130 MB on first run
    python -m scripts.verify_embedder --quiet
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace

from sps.config import EmbeddingSettings, BGE_QUERY_INSTRUCTION
from sps.embedding import BGEEmbedder

TOLERANCE = 1e-5

# Realistic SPS text: two paraphrases of one defect, plus an unrelated problem.
PROBLEM = "Bracket weld seam cracking observed during incoming inspection"
PARAPHRASE = "Cracks found in the weld seam of the mounting bracket at goods-in"
NEAR = "Weld porosity detected on the bracket joint during inspection"
UNRELATED = "Outer carton label misprinted on the shipment packaging"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the real BGE embedder")
    parser.add_argument("--quiet", action="store_true", help="suppress the similarity table")
    args = parser.parse_args(argv)

    checks = Checks()
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

    print()
    if checks.failures:
        print(f"FAILED: {len(checks.failures)} check(s): {', '.join(checks.failures)}")
        return 1
    print("All embedder checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
