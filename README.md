# Supplier Problem Sheet (SPS) — Constrained RAG Pipeline

A constrained Retrieval-Augmented Generation pipeline that answers
supplier-submitted problem sheets **strictly from historical precedent**. It never
introduces external domain knowledge, never exposes internal tooling to suppliers,
and always emits a status workbook for human admin review.

Runs on a **CPU-only host**: `BAAI/bge-small-en-v1.5` locally via
`sentence-transformers`, generation on an Azure OpenAI deployment at
`temperature=0.0`, `top_p=0.1`. **No vector database** — history is filtered by
part number and embedded per ticket.

---

## Quick start

Install PyTorch **first**, from PyTorch's own index. The CPU wheel is only
published there: on Windows, plain PyPI `torch==2.3.1` resolves to the
CUDA-bundled build, which is a far larger download and roughly 2.4 GB unpacked
for no benefit on a CPU-only host.

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

Then the rest. torch is already satisfied, so it is not refetched:

```bash
python -m pip install -r requirements.txt
```

Verify the install before running anything. torch, numpy, scipy, scikit-learn
and httpx are pinned as one set and have to land together; this is the command
that proves they did:

```bash
python -m pip check
python -c "import torch; print(torch.__version__)"   # expect 2.3.1+cpu
```

`pip check` must print `No broken requirements found.` A `+cpu` suffix on the
torch version confirms the right wheel; a bare `2.3.1` means the CUDA build was
installed instead.

Run the test suite:

```bash
python -m pytest -q
```

72 tests, none of which needs a server, an Azure key or a model download. The
real-model checks are opt-in (~130 MB of weights):

```bash
SPS_MODEL_TESTS=1 python -m pytest -q
```

---

## Layout

| Path | Role |
| --- | --- |
| `scripts/run_resolver.py` | **Entry point** — ticket in, two workbooks out |
| `scripts/run_resolver.cmd` | UiPath wrapper; propagates the exit code |
| `sps/validators.py` | Part-number canonicalisation + ticket gatekeeping |
| `sps/retrieval/in_memory.py` | Filter history by part, embed, rank, gate |
| `sps/embedding.py` | BGE wrapper (lazy load, L2-normalized, CPU) |
| `sps/generation/` | Actor / Judge loop, schema-constrained |
| `sps/schemas.py` | Pydantic response schemas for the LLM |
| `sps/contracts.py` | Ticket and result shapes |
| `sps/output.py` | Builds the resolved recommendation |
| `service/excel_output.py` | Atomic workbook writer |
| `scripts/verify_embedder.py` | Acceptance checks against the real model |


**Gatekeeping depends on nothing but the standard library.** `sps/validators.py`
imports without torch, openai, pandas or pydantic, so the rules that decide
whether a ticket is even processable are testable in milliseconds. Everything
heavier sits behind lazy imports and the `Embedder` / `ChatClient` protocols.

---

## The resolver (primary path)

`scripts/run_resolver.py` resolves one ticket against a history file with **no
vector database**. Because the part-number filter means semantic search only ever
runs against one part's history, that history is small enough to embed on demand
-- so the index, the indexer schedule, the embedded-storage lock and payload
drift all stop existing.

```bash
python -m scripts.run_resolver --ticket-file ticket.xlsx \
    --history-file history.csv --output-dir .\out
```

| Output | When | Columns |
| --- | --- | --- |
| `status.xlsx` | **Always**, including early aborts and unhandled exceptions | `Execution_Timestamp`, `Status` (PASS/FAIL), `Status_Code`, `Reason` |
| `output.xlsx` | Only when `Status` is PASS | `Part_Number`, `AI_Recommendation`, `Justification`, `Confidence_Score`, `Referenced_SPS_IDs` |

Status codes: `SUCCESS`, `INVALID_INPUT`, `NO_MATCHES`,
`BELOW_CONFIDENCE_THRESHOLD`, `LLM_AUDIT_REJECTED`, `INFRASTRUCTURE_ERROR`.

Exit codes are kept alongside the sheet so a caller can branch without opening a
workbook: **0** the run completed (PASS, or a legitimate FAIL such as a gated
ticket), **1** an infrastructure fault worth retrying, **2** the inputs could not
be read. Both workbooks are written atomically and cleared before any work
starts, so a process killed outright leaves no stale result.

Work is ordered cheapest-first, so nothing expensive runs for a ticket that
cannot succeed:

```
validate -> filter history by part -> cap to newest 300 -> embed -> rank -> gate -> LLM
```

A malformed part number is rejected before the model is even loaded.

### Measured latency, 300k-row history

| step | cost |
| --- | --- |
| history scan, CSV | 1.45 s |
| embed 242 candidates + query | 1.39 s |
| **per ticket, warm process** | **2.84 s** |
| model load, once per process | ~8 s |
| **per ticket, cold CLI** | **~11 s** |

Two things follow, and both matter operationally:

1. **Use CSV for a large history.** The same 300k rows take **~40 s** as `.xlsx`
   against ~1.5 s as `.csv`, because openpyxl inflates and parses XML per row
   while a CSV is a linear read. The engine reads both; the format is the
   difference between meeting the 3 s budget and missing it by 13x.
2. **The 3 s budget only holds for a warm process.** Model load dominates a cold
   CLI invocation. One ticket per process costs ~11 s regardless of how fast the
   retrieval is; keep a resident process, or batch tickets, to amortise it.

### Part-number gatekeeping

`sps/validators.py` canonicalises with `.strip().upper()` plus removal of
invisible characters -- zero-width space, zero-width joiner, BOM, word joiner,
soft hyphen -- and interior whitespace. Non-breaking spaces from web forms and
zero-width characters from copy-paste are invisible to whoever pasted them but
turn an exact-match filter into a total miss.

Structural delimiters are preserved: `0012-43951`, `0012/43951`, `0012_43951`
and `0012.43951` remain four distinct identifiers.

A ticket whose part number is missing, blank, or nothing but delimiters is
rejected as `INVALID_INPUT` **before** any embedding or LLM call.

### Threshold: recalibrate for bge-small

The resolver uses `bge-small-en-v1.5` (384d), which is about ten times faster
than `bge-large` on CPU -- 1.0 s versus 11 s to encode 300 candidates -- and is
what makes per-ticket embedding viable at all.

**It also scores systematically higher, so the 0.82 threshold does not carry
over.** Same probe texts, same query:

| candidate | bge-small | bge-large |
| --- | --- | --- |
| identical text | 0.9641 | 0.9389 |
| paraphrase | 0.8736 | 0.7926 |
| **different defect** (weld porosity vs seam cracking) | **0.8442** | 0.7831 |
| unrelated | 0.5111 | 0.4395 |

At 0.82, bge-large admits only the identical text; bge-small admits the
paraphrase **and a materially different defect**. Carrying 0.82 across the model
change would silently loosen the gate, so **the resolver default is 0.89**
(`DEFAULT_CONFIDENCE_THRESHOLD` in `sps/retrieval/in_memory.py`, which the CLI
imports rather than repeating).

That figure comes from four probe sentences, not a measurement -- the eval set
should confirm or move it.

> **The threshold is coupled to the model.** 0.89 belongs to bge-small's scoring
> distribution. Swapping the embedding model invalidates it, and the number lives
> in `sps/retrieval/in_memory.py` beside the engine that applies it for exactly
> that reason. Override per run with `--threshold`, or with
> `SPS_CONFIDENCE_THRESHOLD`.

Metadata boosting is gone from this path entirely: part number is an exact
filter, and the other boosts existed to discriminate within a mixed-part result
set that no longer occurs. `Confidence_Score` is the cosine alone.

---

## Configuration

Copy `.env.example` to `.env`; the resolver loads it if present, and real
environment variables always win. A UiPath robot does not necessarily inherit an
interactive shell's environment, which is why the file is read at all.

Azure credentials are read **only** from the environment. There is no CLI flag
for them: command lines are visible in the Windows process list and in UiPath
job logs.

Precedence for the threshold is `--threshold` — `SPS_CONFIDENCE_THRESHOLD` —
the built-in 0.89.

---

## Assumptions and known boundaries

Stated explicitly rather than buried:

1. **BGE query instruction.** Queries are prefixed with
   `"Represent this sentence for searching relevant passages: "`, the convention
   `bge-*-v1.5` was trained with; history passages are not. Verified against the
   real model: a passage vector is bit-identical to an unprefixed query vector
   (cosine 1.00000000), and ranking is unchanged with the prefix on or off, though
   absolute scores shift. Disable with `SPS_USE_BGE_QUERY_INSTRUCTION=false`.

2. **Part numbers are alphanumeric and free of stray spaces.** Canonicalisation
   still strips invisible characters and upper-cases defensively, but on a house
   format like `0012-43951` both steps are no-ops. A purely numeric part number
   would be a problem Excel creates before this code sees the file: it stores
   numbers as doubles, so leading zeros are lost in the sheet itself. Format such
   columns as Text, or supply CSV.

3. **A blank part number is rejected, not broadened.** A ticket without one is
   `INVALID_INPUT`; it does not fall back to searching the whole history. As a
   consequence, history rows with a blank `Part_Number` are unreachable to any
   ticket that supplies one — worth measuring the blank rate in the extract
   before an evaluation run.

4. **The 300-candidate cap drops by recency.** For a part with more history than
   that, the oldest records are not considered. An old fix is likelier to have
   been superseded, and encoding cost is linear in what survives.

5. **Prompt effectiveness is not unit-testable.** The tests pin the loop's
   *control flow* (refinement, the circuit breaker, fail-closed behaviour) and
   that the mandated constraints appear in the prompts. Whether the Judge
   actually catches a given hallucination is a model-behaviour question needing
   an evaluation set of labelled drafts against the real deployment.

---

## Test coverage

```
tests/test_resolver.py                 32   validation, part filtering, capping, dual workbooks, threshold
tests/test_component_c_actor_critic.py 19   refinement, circuit breaker, fail-closed, prompt isolation
tests/test_structured_outputs.py       12   strict response_format, fallback, schema boundaries
tests/test_config.py                    9   env loading, model/threshold single-sourcing
tests/test_real_embedder.py            14   the real bge-small model (opt-in)
```

The real-model tests load ~130 MB of weights and are therefore opt-in:

```bash
SPS_MODEL_TESTS=1 python -m pytest -q
```

They pin the properties the ranking maths assumes: 384 dimensions, unit-length
vectors (so the NumPy matmul *is* the cosine the 0.89 gate is calibrated on), and
the query instruction applied to queries but never to history passages.

`python -m scripts.verify_embedder` runs the same checks as a standalone report
and prints real cosine numbers for related versus unrelated SPS text, which is
the fastest way to sanity-check the threshold on a new machine.
