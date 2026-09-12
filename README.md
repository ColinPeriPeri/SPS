# Supplier Problem Sheet (SPS) — Constrained RAG Pipeline

A constrained Retrieval-Augmented Generation pipeline that answers
supplier-submitted problem sheets **strictly from historical precedent**. It never
introduces external domain knowledge, never exposes internal tooling to suppliers,
and always emits a status workbook for human admin review.

Two tiers of evidence. **Tier 1** answers from the part's own historical SPS
records. When those produce nothing usable, **Tier 2** answers from the company's
0250 engineering standards instead and cites the document and section it used.
Neither tier is allowed to answer from the model's own knowledge.

Runs on a **CPU-only host**: `BAAI/bge-small-en-v1.5` locally via
`sentence-transformers`, generation on an Azure OpenAI deployment at
`temperature=0.0`, `top_p=0.1`. **No vector database** — history is filtered by
part number and embedded per ticket; the standards corpus is embedded once and
cached against a hash of itself.

---

## Quick start

Setting this up on a fresh Windows machine? **[`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)**
is the copy-paste version of this section, with the checks, the `.env` values and
a smoke test that proves the install before any real data touches it.

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

202 tests, none of which needs a server, an Azure key or a model download. The
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
| `scripts/run_eval_batch.py` | Batch evaluator — many tickets, one results workbook |
| `scripts/run_eval.cmd` | Wrapper for the above |
| `sps/validators.py` | Part-number canonicalisation + ticket gatekeeping |
| `sps/retrieval/in_memory.py` | **Tier 1** — filter by part, embed (Azure or local), rank, gate |
| `sps/retrieval/docx_parser.py` | **Tier 2** — parse 0250 .docx into citable, section-bounded chunks |
| `sps/retrieval/doc_cache.py` | **Tier 2** — hash-keyed vector cache, enriched query, search |
| `sps/embedding.py` | Azure + local BGE encoders |
| `sps/generation/` | Actor / Judge loop, schema-constrained |
| `sps/schemas.py` | Pydantic response schemas for the LLM |
| `sps/contracts.py` | Ticket and result shapes |
| `sps/output.py` | Builds the resolved recommendation |
| `sps/file_reader.py` | .csv / .xlsx dispatch + strict type gate |
| `service/excel_output.py` | Atomic workbook writer |
| `scripts/verify_embedder.py` | Acceptance checks against the real model |
| `scripts/make_sample_docs.py` | Regenerates the demo 0250 documents |
| `samples/` | A ticket + history for smoke tests, six eval cases, three demo standards |
| `data/0250_docs/` | Where the real 0250 standards go (ships empty) |


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
| `status.xlsx` | **Always**, including early aborts and unhandled exceptions | `Execution_Timestamp`, `Status` (PASS/FAIL), `Status_Code`, `Reason`, `Embedding_Model` |
| `output.xlsx` | Only when `Status` is PASS | `Part_Number`, `AI_Recommendation`, `Justification`, `Confidence_Score`, `Referenced_Sources`, `Resolution_Source` |

Status codes: `SUCCESS_HISTORICAL`, `SUCCESS_0250_DOC`, `NO_RESOLUTION_FOUND`,
`INVALID_INPUT`, `INFRASTRUCTURE_ERROR`. `Status` itself is still PASS or FAIL
for both success codes, so a caller branching on `Status` is unaffected by the
second tier.

`Resolution_Source` is `HISTORICAL_DATA` or `0250_DOCUMENTATION`, and
`Referenced_Sources` holds SPS IDs or document citations to match. Both replace
the former `Referenced_SPS_IDs` column — **a breaking change** for anything
reading that sheet by column name.

Exit codes are kept alongside the sheet so a caller can branch without opening a
workbook: **0** the run completed (PASS, or a legitimate FAIL such as a gated
ticket), **1** an infrastructure fault worth retrying, **2** the inputs could not
be read. Both workbooks are written atomically and cleared before any work
starts, so a process killed outright leaves no stale result.

Work is ordered cheapest-first, so nothing expensive runs for a ticket that
cannot succeed:

```
validate -> filter history by part -> cap to newest 300 -> embed -> rank -> gate -> LLM
                                                                                    |
                                                        (nothing usable) -----------+
                                                                 |
                                        Tier 2:  load cache -> rank chunks -> gate -> LLM
```

A malformed part number is rejected before the model is even loaded. Tier 2 runs
only once all of Tier 1 has failed, which is what lets it afford to parse and
embed a document corpus.

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

### File formats

Both `--ticket-file` and `--history-file` accept `.csv` and `.xlsx`, dispatched
on the extension. `.xlsm` is accepted too: it is the same format as far as
openpyxl is concerned, a workbook that happens to carry macros, and business
users hand those over routinely.

`sps/file_reader.py` is the only place that knows the difference. Everything
past it works on a header row plus data rows, so column mapping, part
canonicalisation and filtering are format-blind. The same history in both
formats produces byte-identical results, down to the similarity score:

```
xlsx  exit=0  BELOW_CONFIDENCE_THRESHOLD  Best match 0.5658 ... [Local]
csv   exit=0  BELOW_CONFIDENCE_THRESHOLD  Best match 0.5658 ... [Local]
```

**Anything else hard-fails immediately**, on the extension alone, before a file
is opened, a model is loaded or a history is scanned:

```
exit=0  FAIL / INVALID_INPUT
reason: History file 'history.pdf' has an unsupported type (.pdf).
        Expected one of: .csv, .xlsm, .xlsx.
```

**Exit 0, not 2**, and that distinction is deliberate: the wrong attachment is a
business problem for whoever assembled the ticket, so the item is faulted and
not retried. A file of the *right* type that is missing or unreadable keeps
exit 2 — that is genuine I/O trouble and a human should look at it.

Prefer CSV for a large history: at 300k rows the same data takes about 1.5 s as
`.csv` against about 40 s as `.xlsx`, because openpyxl parses XML per row.

### Embedding: Azure primary, local fallback

Azure embeddings are the primary encoder; local `bge-small-en-v1.5` is the
fallback. The choice is **all-or-nothing per run**: the query and every
candidate are always encoded by the same model, because a cosine between vectors
from two different embedding spaces is not a similarity, it is noise that
happens to land between -1 and 1.

```
try  Azure: one request, [query] + candidates      -> AZURE_EMBEDDING_THRESHOLD
except network / auth / timeout / rate limit / not configured
     log AZURE_EMBEDDING_FAILED_FALLING_BACK
     re-encode the WHOLE batch locally             -> LOCAL_EMBEDDING_THRESHOLD
```

Anything Azure managed to return before failing is discarded rather than topped
up locally. A short or reordered response is treated as a failure for the same
reason: the response carries a per-item index, and the batch is re-sorted on it
rather than trusting arrival order, because a silently reordered batch would
pair every candidate with another candidate's score.

**The local model is not loaded when Azure succeeds.** Constructing `BGEEmbedder`
is free (0.000 s, no torch import); the ~15 s of torch import and weight loading
lands on first *encode*. So the retriever takes a factory, not an instance, and
calls it only inside the `except` branch. A test asserts the factory is never
invoked on a successful primary run.

The Azure path deliberately sends **no BGE instruction prefix**.
`"Represent this sentence for searching relevant passages: "` is a convention
`bge-*-v1.5` was trained with; Azure's models were not, so prepending it would
inject a constant meaningless string into every query. That asymmetry is also
why the local path needs two encode calls (queries prefixed, passages not) where
Azure needs one.

Vectors are L2-normalised on both paths. OpenAI returns unit-length embeddings
today, but the ranking is a bare dot product that stops being a cosine if that
ever changes, so the guarantee is made here rather than assumed.

### Two thresholds, one per embedding space

A threshold is a property of one model's scoring distribution and does not
survive a change of encoder. The gate applies whichever belongs to the model
that actually answered, and `status.xlsx` records which one that was.

| | value | basis |
| --- | --- | --- |
| `LOCAL_EMBEDDING_THRESHOLD` | **0.89** | Measured. bge-small scores systematically higher than bge-large — a materially different defect reaches 0.8442 against 0.7831 — so 0.82 would have loosened the gate. |
| `AZURE_EMBEDDING_THRESHOLD` | **0.50** | **Provisional, not measured.** |

> **0.50 is a placeholder, and how wrong it is depends on which model backs the
> deployment.** `text-embedding-3-small` / `-3-large` put unrelated text around
> 0.1—0.3, so 0.50 is a plausible starting point. `text-embedding-ada-002`
> is notorious for keeping even unrelated pairs above 0.7 — against that model
> 0.50 admits essentially everything and the gate stops existing. Measure the
> distribution on your own deployment before the evaluation run; the local 0.89
> was derived from exactly four probe sentences and still wants the eval set to
> confirm it.

`SPS_CONFIDENCE_THRESHOLD` and `--threshold` override **both**, and are honoured
whichever encoder answers — an explicit operator instruction is not
second-guessed by the backend that happened to respond.

### Tracking how often the fallback fires

`status.xlsx` carries `Embedding_Model`, appended as the **last** column so a
caller reading the first four positionally is unaffected:

```
azure:text-embedding-3-small     primary path
local:BAAI/bge-small-en-v1.5     fallback fired
(blank)                          aborted before anything was encoded
```

`Reason` also ends with `[Azure]` or `[Local]`, so a support engineer skimming
the sheet — or a caller reading only the first four columns — sees which
encoder ran without needing to know the column exists.

Every fallback also logs `AZURE_EMBEDDING_FAILED_FALLING_BACK` on stderr with
the cause, at WARNING, with a fixed marker so it can be counted from the job
logs. A deployment where Azure is quietly misconfigured still works — it just
pays the local cold start on every ticket, which is exactly the situation this
column exists to make visible.

Metadata boosting is gone from this path entirely: part number is an exact
filter, and the other boosts existed to discriminate within a mixed-part result
set that no longer occurs. `Confidence_Score` is the cosine alone.

---

## Tier 2: the 0250 standards fallback

When Tier 1 produces no recommendation — no matching part, nothing similar
enough, or an Actor that declines the precedent it was shown — the resolver
searches the 0250 engineering standards in `data/0250_docs/` instead.

```bash
python -m scripts.run_resolver --ticket-file t.csv --history-file h.csv \
    --output-dir .\out --docs-dir data\0250_docs
```

**An empty corpus is a valid state.** With no `.docx` in the folder, Tier 2 does
not run and every ticket reports exactly the Tier-1 outcome it would have
reported before Tier 2 existed. `--no-tier2` forces that behaviour explicitly.

### Parsing

Chunks are cut at Word heading styles and never span a section, because the
citation attached to a chunk has to be true — one straddling 4.2 and 4.3 could
be cited as either and would be wrong half the time. Each chunk carries its own
provenance *inside the embedded text*:

```
[0250-Weld-Standards.docx § 4.2 Weld Seam Cracking]
Cracking in a fillet or butt weld seam is cause for rejection ...
```

so the string that was ranked is the string the model is asked to cite, and
nothing can drift between retrieval and attribution.

Tables are included, flattened one row per line. `document.paragraphs` omits
them entirely, and a limits table is where a standard keeps its numbers —
dropping tables would lose "2 percent by area maximum" from a porosity section
while keeping the prose around it.

Legacy `.doc` files are named in a warning and skipped. A `.doc` is not a zip
container, so the only way to read one on Windows is to drive Word through COM
automation, which on a headless robot blocks on a modal dialog and hangs the run
rather than failing it.

### The cache

The corpus is the same on every ticket, so it is embedded once. Two cache files,
`0250_cache_azure.npz` and `0250_cache_local.npz`, because a vector belongs to
one embedding space and one file could not hold two.

Validity is keyed on a SHA-256 of every document's **name and bytes** — a rename
changes every citation the document produces, so it invalidates too — plus the
identity of the model that wrote the vectors, since two Azure deployments share a
filename but not an embedding space. Edit a document and the next run rebuilds;
touch nothing and it loads in milliseconds. A corrupt or hand-edited cache is a
rebuild, not a failure: the documents on disk are always the source of truth.

### The enriched query

```
Issue Type: {Issue_Type} | Defect: {Problem_Description}
```

A standards corpus is organised by issue class — welding, packaging, plating —
while a defect sentence often names only the symptom. The enrichment costs a
little raw similarity (0.8191 → 0.7800 on the measured example, since the header
tokens are not defect language) and buys discrimination between two standards
describing the same symptom under different issue classes.

### A third threshold

Tier 2 scores in a different range from Tier 1, and this is not a detail. Tier 1
compares one short defect sentence against another; Tier 2 compares a defect
sentence against 300 words of standards prose that answers it **without
resembling it**. Measured with bge-small against the demo corpus:

| query | top-ranked section | score |
| --- | --- | --- |
| weld seam cracking | `§ 4.2 Weld Seam Cracking` | 0.7462 |
| weld porosity | `§ 4.3 Weld Porosity Limits` | 0.8095 |
| carton label misprint | `§ 2.1 Carton Labelling` | 0.7507 |
| hydraulic pump pressure — *not covered by any document* | `§ 4.2 Weld Seam Cracking` | **0.5944** |

Tier 1's gate is 0.89. Reusing it would reject the correct chunk on every ticket
and Tier 2 would never fire once. `TIER2_LOCAL_THRESHOLD` is **0.62** — above the
uncovered-defect peak, below every genuine match.

`--threshold` is deliberately Tier 1's alone; Tier 2 has `--tier2-threshold` and
`SPS_TIER2_THRESHOLD`. One override spanning both would be a single number
standing for two different embedding spaces.

**Re-measure 0.62 once the real standards are loaded.** A larger corpus gives an
irrelevant section more chances to score highly, so the uncovered-defect peak
rises with corpus size, and 0.5944 leaves only 0.026 of headroom.

### Grounding

The Tier-2 Actor is under one constraint the Tier-1 Actor is not: **a standard is
not automatically a solution.** A section stating a limit, a tolerance or an
acceptance criterion without saying what to do about a part that violates it has
not supplied a resolution, and deriving the disposition is fabrication — the more
dangerous kind, because the invented step is usually the engineering-plausible
one. The Judge audits this as a fourth check alongside hallucination, tool
leakage, and citation integrity.

The Actor is also required to open its justification with *"Historical records
yielded no resolution."*, so a human reviewer can see at a glance that this
answer came from a standard rather than from precedent.

Citations in `Referenced_Sources` are taken from the retrieved chunks, never from
the model's prose — the same reason confidence and SPS IDs are supplied from
measurement in Tier 1. A model asked to author its own citation can invent one.

---

## The batch evaluator

`scripts/run_eval_batch.py` runs a directory of test tickets and writes one
`eval_results.xlsx`. It exists to calibrate a threshold, which shapes both of
its design decisions.

```bash
python -m scripts.run_eval_batch --test-dir cases --history-file master.csv \
    --output-dir eval_out
scripts\run_eval.cmd cases eval_out master.csv
```

**Every row carries the raw top cosine, including the cases the gate rejected.**
A run that reported only its successes would show you the scores above the
threshold and hide exactly the ones you need in order to judge whether the
threshold belongs where it is.

**The whole batch runs in one process**, so the local model's ~7 s cold start is
paid once rather than once per case. Fifty cases that would take some minutes as
fifty subprocesses take roughly as long as fifty embeddings. One failing case
never ends the run: it becomes a row saying what went wrong, and the batch
carries on.

| Column | |
| --- | --- |
| `Test_ID`, `Status`, `Status_Code`, `Reason` | as `status.xlsx` |
| `Embedding_Model` | `azure:<deployment>` or `local:<model>` |
| `Confidence_Score` | the top cosine, **populated even when the gate rejected it** |
| `Threshold_Applied`, `Cleared_Threshold` | what it was measured against, and the verdict |
| `Candidates_Considered`, `Duration_Seconds` | |
| `Ticket_File`, `History_File` | which inputs produced this row |

Blank score cells mean nothing was ever encoded — an unknown part, a bad file
type — which is different from a score of zero, and kept distinct so those cases
do not drag the distribution down.

Cases are discovered as `<id>_ticket.csv|xlsx` optionally paired with
`<id>_history.csv|xlsx`; a case with no history of its own uses `--history-file`.
`--run-list` takes a `Test_ID` / `Ticket_File` / `History_File` sheet instead,
with paths resolved relative to the list, so a test set stays portable between
machines.

The console digest prints the score distribution — min, p25, median, p75, p90,
max — so the shape is visible without opening Excel. It also **warns if more
than one encoder ran in the batch**: a threshold belongs to one embedding space,
so a distribution pooled across Azure and the local fallback describes neither.
Nothing else would report that, because falling back is normal behaviour rather
than an error.

Exit code 0 means the batch ran, whatever the individual cases did; 1 means at
least one case hit an infrastructure error; 2 means the case list could not be
built.

`samples/` holds a worked example: `sample_ticket.csv` with `sample_history.csv`
for the smoke test in [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md), and
`samples/eval_cases/` with six cases covering a clean match, a near-duplicate, a
case with its own history, a materially different defect, an unknown part and a
blank part number.

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
tests/test_tier2_docs.py               48   docx parsing, cache invalidation, Tier-2 gating and fallback
tests/test_resolver.py                 32   validation, part filtering, capping, dual workbooks, threshold
tests/test_eval_batch.py               32   case discovery, per-case isolation, the score columns
tests/test_file_reader.py              31   format dispatch, strict type gate, format agnosticism
tests/test_embedding_fallback.py       19   Azure primary, all-or-nothing fallback, dual threshold
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
