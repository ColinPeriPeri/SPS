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

122 tests, none of which needs a server, an Azure key or a model download. The
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
| `sps/retrieval/in_memory.py` | Filter by part, embed (Azure or local), rank, gate |
| `sps/embedding.py` | Azure + local BGE encoders |
| `sps/generation/` | Actor / Judge loop, schema-constrained |
| `sps/schemas.py` | Pydantic response schemas for the LLM |
| `sps/contracts.py` | Ticket and result shapes |
| `sps/output.py` | Builds the resolved recommendation |
| `sps/file_reader.py` | .csv / .xlsx dispatch + strict type gate |
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
| `status.xlsx` | **Always**, including early aborts and unhandled exceptions | `Execution_Timestamp`, `Status` (PASS/FAIL), `Status_Code`, `Reason`, `Embedding_Model` |
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
tests/test_embedding_fallback.py       19   Azure primary, all-or-nothing fallback, dual threshold
tests/test_file_reader.py              31   format dispatch, strict type gate, format agnosticism
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
