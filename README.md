# Supplier Problem Sheet (SPS) — Constrained RAG Pipeline

A constrained Retrieval-Augmented Generation pipeline that answers
supplier-submitted problem sheets **strictly from historical precedent**. It never
introduces external domain knowledge, never exposes internal tooling to suppliers,
and always emits a status workbook for human admin review.

Two tiers of evidence. **Tier 1** answers from the part's own historical SPS
records. When those produce nothing usable, **Tier 2** answers from the company's
0250 engineering standards instead and cites the document and section it used.
Neither tier is allowed to answer from the model's own knowledge.

Embeddings and generation both run on **Azure OpenAI**, at `temperature=0.0`,
`top_p=0.1`. **No vector database** — history is filtered by part number and
embedded per ticket; the standards corpus is embedded once and cached against a
hash of itself.

> **The local `bge-small` encoder is currently disabled.** Azure is the only
> encoder in the pipeline. The model wrapper, its acceptance checks and its
> thresholds are all still in the tree and untouched — see
> [The local encoder, disabled](#the-local-encoder-disabled) — but nothing
> reaches them, and `requirements.txt` no longer installs torch. The practical
> consequence: **an Azure embedding failure now ends the run** instead of being
> absorbed, and there is no offline mode.

---

## Quick start

Setting this up on a fresh Windows machine? **[`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)**
is the copy-paste version of this section, with the checks, the `.env` values and
a smoke test that proves the install before any real data touches it.

```bash
python -m pip install -r requirements.txt
```

No torch, no CUDA, no model download — 33 packages. Verify before running
anything:

```bash
python -m pip check
python -c "from openai import AzureOpenAI; from lxml import etree; import docx"
```

`pip check` must print `No broken requirements found.`, and the second line must
print nothing at all. Two pins exist purely to make that second line work on a
managed Windows machine: **jiter below 0.17** and **lxml below 6**, whose native
DLLs are blocked by Windows Application Control. Both fail with *"An Application
Control policy has blocked this file"*, which reads like a corrupt install
rather than a policy decision — and the jiter one takes every Azure call with
it, since `openai` imports it.

Run the test suite:

```bash
python -m pytest -q
```

297 tests in about six seconds, none of which needs a server, an Azure key or
a model download. The real-model checks are opt-in, and now also need the
disabled dependencies reinstalled (~130 MB of weights plus torch):

```bash
SPS_MODEL_TESTS=1 python -m pytest -q
```

---

## Layout

| Path | Role |
| --- | --- |
| `scripts/run_resolver.py` | **Entry point** — ticket in, two workbooks out |
| `scripts/run_resolver.cmd` | UiPath wrapper; propagates the exit code |
| `scripts/run_bulk_test.py` | **Bulk test** — a sheet of tickets in, the same sheet plus answers out |
| `scripts/run_bulk_test.cmd` | Wrapper for the above |
| `scripts/run_eval_batch.py` | Batch evaluator — threshold calibration from a directory of cases |
| `scripts/run_eval.cmd` | Wrapper for the above |
| `sps/validators.py` | Part-number canonicalisation + ticket gatekeeping |
| `sps/retrieval/in_memory.py` | **Tier 1** — filter by part, embed via Azure, rank, gate |
| `sps/retrieval/docx_parser.py` | **Tier 2** — parse 0250 .docx into citable, section-bounded chunks |
| `sps/retrieval/doc_cache.py` | **Tier 2** — hash-keyed vector cache, enriched query, search |
| `sps/embedding.py` | Azure encoder; the BGE wrapper, disabled but intact |
| `sps/generation/` | Actor / Judge loop, schema-constrained |
| `sps/generation/transferable.py` | The deterministic gate on record-specific references |
| `sps/schemas.py` | Pydantic response schemas for the LLM |
| `sps/contracts.py` | Ticket and result shapes |
| `sps/output.py` | Builds the resolved recommendation |
| `sps/file_reader.py` | .csv / .xlsx dispatch + strict type gate |
| `service/excel_output.py` | Atomic workbook writer |
| `scripts/verify_embedder.py` | Acceptance checks + threshold reading against the live encoder |
| `scripts/make_sample_docs.py` | Regenerates the demo 0250 documents |
| `samples/` | A ticket + history for smoke tests, a bulk sheet, six eval cases, three demo standards |
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
| `output.xlsx` | Whenever the run **reached a conclusion** (exit 0), including "no solution". Never on exit 1 or 2 | `Part_Number`, `AI_Recommendation`, `Justification`, `Confidence_Score`, `Referenced_Sources`, `Resolution_Source` |

### Status codes

The robot routes exceptions on this column, so it says *how* the pipeline ran
out of options, not merely that it did.

| Code | Meaning | Typical routing |
| --- | --- | --- |
| `SUCCESS_HISTORICAL` | Answered from the part's own precedent | Admin review |
| `SUCCESS_0250_DOC` | Answered from a 0250 standard | Admin review |
| `NO_MATCHES` | No history for the part **and** no standard covered it | Master Data |
| `BELOW_CONFIDENCE_THRESHOLD` | Candidates found in either tier, none cleared its gate | Reliability Engineer — a genuinely novel defect |
| `LLM_AUDIT_REJECTED` | Candidates cleared the maths, the Actor or Judge refused | Human reviewer |
| `INVALID_INPUT` | Bad part number, missing file, unsupported format | Fault the item |
| `INFRASTRUCTURE_ERROR` | Azure outage, unhandled crash | Retry |

With two tiers the reported code is the **furthest stage either tier reached** —
nothing retrieved < gated < audited and refused. A run where history reached the
Judge and was refused while the standards had nothing to say is an audit
rejection: reporting `NO_MATCHES` would send a perfectly well-known part to
Master Data.

A corollary worth knowing: with no 0250 corpus loaded, Tier 2 retrieves nothing
and every code is Tier 1's own, identical to the behaviour before Tier 2 existed.

`Reason` carries the detail underneath — including the best score from *each*
tier — because the code is for the robot's switch and the Reason is for the
human who has to act on it.

`Status` itself is still PASS or FAIL for both success codes, so a caller
branching on `Status` is unaffected by the second tier.

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

### The local encoder, disabled

Azure OpenAI is the only encoder the pipeline reaches. The `bge-small` path is
disabled, not deleted:

| Still in the tree | State |
| --- | --- |
| `sps/embedding.py` → `BGEEmbedder` | Untouched |
| `scripts/verify_embedder.py --local` | Works, once torch is reinstalled |
| `tests/test_real_embedder.py` | Skips cleanly without `sentence_transformers` |
| `LOCAL_EMBEDDING_THRESHOLD` (0.89), `TIER2_LOCAL_THRESHOLD` (0.62) | Dormant; kept so re-enabling is a wiring change, not a re-measurement |
| The `embedder=` injection seam on both retrievers | Live — tests use it, and it is where a local model drops back in |

Re-wiring it means restoring one branch at each of the three sites marked
`LOCAL MODEL DISABLED`, in `sps/retrieval/in_memory.py`,
`sps/retrieval/doc_cache.py` and `scripts/run_resolver.py`, and uncommenting
four pins in `requirements.txt`. The steps are written out at the bottom of that
file.

**What changed behaviourally.** A failure used to be absorbed: the run fell back
to `bge-small`, scored in a different embedding space, applied a different
threshold, and still reported success — with only the `Embedding_Model` column
to say so. Now it ends the run, and the exit code says which kind of failure it
was:

| Situation | `Status_Code` | Exit | Why |
| --- | --- | --- | --- |
| Network, timeout, throttling, bad response | `INFRASTRUCTURE_ERROR` | **1** | Transient; retry |
| `AZURE_EMBEDDING_*` missing | `INFRASTRUCTURE_ERROR` | **2** | No number of retries produces an API key |

That split is the whole reason `AzureEmbeddingNotConfigured` subclasses
`AzureEmbeddingError`: existing `except` clauses still catch both, while the
resolver checks the specific type first.

The batch evaluator gains something from this too — with one encoder there is no
longer any way for a run to pool scores from two embedding spaces, which was the
failure its mixed-encoder warning existed to catch.

### Thresholds, one per embedding space

A threshold is a property of one model's scoring distribution and does not
survive a change of encoder. Four exist; **two are live and two are dormant**,
and `status.xlsx` records which encoder actually answered.

| | value | state | basis |
| --- | --- | --- | --- |
| `AZURE_EMBEDDING_THRESHOLD` | **0.50** | **live** | **Provisional, never measured.** |
| `TIER2_AZURE_THRESHOLD` | **0.35** | **live** | **Provisional twice over** — derived from the 0.50 above, itself a guess. |
| `LOCAL_EMBEDDING_THRESHOLD` | 0.89 | dormant | Measured, on bge-small. |
| `TIER2_LOCAL_THRESHOLD` | 0.62 | dormant | Measured, on bge-small against the demo corpus. |

> **Every gating decision the pipeline now makes rests on an unmeasured
> number.** Disabling the local encoder retired both measured thresholds and
> promoted both guesses. How wrong 0.50 is depends on which model backs the
> deployment: `text-embedding-3-small` / `-3-large` put unrelated text around
> 0.1—0.3, so it is a plausible starting point; `text-embedding-ada-002` is
> notorious for keeping even unrelated pairs above 0.7, where 0.50 admits
> essentially everything and the gate stops existing.
>
> `python -m scripts.verify_embedder` reads both live gates against the real
> deployment and suggests a range for each. Run it before trusting either, and
> widen it with `scripts/run_eval_batch.py`.

`SPS_CONFIDENCE_THRESHOLD` and `--threshold` override the Tier-1 pair;
`SPS_TIER2_THRESHOLD` and `--tier2-threshold` override the Tier-2 pair. An
explicit operator instruction is not second-guessed by the backend that
happened to respond.

### Which encoder answered

`status.xlsx` carries `Embedding_Model`, appended as the **last** column so a
caller reading the first four positionally is unaffected:

```
azure:text-embedding-3-large     the encoder
(blank)                          aborted before anything was encoded
local:<model>                    only reachable through an injected encoder
```

`Reason` also ends with `[Azure]` or `[Local]`, so a support engineer skimming
the sheet — or a caller reading only the first four columns — sees which encoder
ran without needing to know the column exists.

The column kept its value after the fallback was removed: a ticket whose Tier 1
never embedded — an unknown part stops before the model is reached — still
reports Tier 2's encoder if Tier 2 ran, so a run that embedded is never recorded
as one that did not.

Metadata boosting is gone from this path entirely: part number is an exact
filter, and the other boosts existed to discriminate within a mixed-part result
set that no longer occurs. `Confidence_Score` is the cosine alone.

---

## Transferability: the third gate

Grounding answers *"did this text come from the source?"*. It does not answer
*"is it still true of the ticket in front of us?"*, and those have different
answers. Two live recommendations made the gap concrete:

```
1. See the feedback in the attachment.
2. Per discussed, rework as attachment shown.
...
4. ESW#20033465 is submitted for these issues.
```

Neither is a hallucination. Both restate historical `Solution_Text` accurately,
so `CHECK 1` passed them. Neither tells the supplier to use an internal system,
so `CHECK 2` passed them. Both were wrong anyway:

- `ESW#20033465` is an internal work request raised for a **different** issue.
- There is no attachment — `output.xlsx` is text.
- No discussion has taken place with this supplier.

A whole class of facts is true of the *record* and false about the *ticket*:
tracking IDs, attachments, prior conversations, dates, lot and PO numbers, names.

### Two layers

**`CHECK 3 — CONTEXT TRANSFER`** in the Judge, whose load-bearing sentence is
*"appearing in the historical solutions is not a defence for this check"*.
`CHECK 1` trains the model toward "in the source ⇒ fine", which is exactly why
the ESW number survived an audit it should have failed. Tier 2 gets the
equivalent as `CHECK 5 — ONWARD CROSS-REFERENCES`: citing the extract you used
is required, sending the supplier to a section they were not given is not.

**`sps/generation/transferable.py`**, a deterministic scan, because an LLM asked
to catch its own class of error is what just failed. It runs **before** the
Judge — it is local and free where the Judge is a paid call — and a hit feeds
back through the same critique path a Judge rejection uses, so the Actor
rewrites and the existing circuit breaker still bounds the retries. Three
failures trip it and no supplier-facing text is produced.

### Not flagging good text is the harder half

A scanner that rejects sound drafts costs three Actor round trips and then
refuses a ticket that deserved an answer. Deliberately untouched:

| Left alone | Why |
| --- | --- |
| `0012-43951` | Part numbers are digit-led; the ID rule requires leading letters |
| `25 mm`, `Ra 1.6`, `12.5 Nm` | Measurements are not identifiers |
| `within 30 days` | A duration is not a date |
| "Attach photos to your response" | `attach` the verb is an action the supplier can take; `attachment` the noun is a document they do not have |
| `Segregate lot 5` | A count, not a lot number — identifiers need three digits or more |

Named individuals are left to the Judge: no pattern separates a person's name
from a material or a process, and a false rejection costs more than a retry.

`KNOWN_TRACKING_PREFIXES` is a module constant — extend it with your own
systems rather than editing a regex.

### The consequence to expect

The PASS rate falls. Where the history is process boilerplate rather than
technical content, constraint 7 now tells the Actor to answer
`Solution not found.` instead of restating it, and Tier 2 gets its turn. That is
the honest number: the previous rate counted recommendations a supplier could
not act on. `Matched_Solutions` in the bulk sheet shows how much of your history
is like this.

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

## Bulk test runs

`scripts/run_bulk_test.py` resolves a **sheet of tickets, one per row**, and
writes a copy carrying every original column plus the answers.

```bash
python -m scripts.run_bulk_test --tickets tickets.xlsx --history history.csv
scripts\run_bulk_test.cmd tickets.xlsx history.csv
```

The input is never modified. It stays a clean, re-runnable fixture, a crashed
run cannot destroy the test set, and two runs can be diffed against each other.
Output defaults to `<tickets>_results.xlsx` beside the input.

| | |
| --- | --- |
| **Original columns** | Preserved verbatim, in order, including ones the pipeline never reads |
| **Appended** | `Status`, `Status_Code`, `Reason`, `AI_Recommendation`, `Justification`, `Confidence_Score`, `Referenced_Sources`, `Resolution_Source`, `Tier1_Score`, `Tier2_Score`, `Embedding_Model`, `Duration_Seconds` |
| **Rows** | Exactly one per input row, in the same order — a ticket that resolved nothing still has a row saying why |

Nothing here reimplements the pipeline. Each row goes through the same
`resolve()` the UiPath wrapper calls, so a result in this sheet is the result
production would have produced for that ticket. The "Solution not found." row is
built by the same function that writes `output.xlsx`, so the two cannot disagree
about what a refusal looks like.

A column that clashes with an existing heading is suffixed `_AI` rather than
duplicated — a sheet that already has `Status` would otherwise end up with two,
and which one survived would be down to pandas.

### Two bulk tools, different jobs

| | `run_bulk_test.py` | `run_eval_batch.py` |
| --- | --- | --- |
| Input | One sheet, a ticket per row | A directory of `<id>_ticket.*` files |
| Output | Your sheet plus the answers | A fixed results workbook |
| Carries | The recommendation prose | Scores and gate decisions |
| For | Reading what the system recommended | Deciding where a threshold belongs |

### Cost and time

Sequential by design, one ticket at a time. Two things dominate a large run:

- **The history is re-scanned per ticket**, because that is what `resolve()`
  does per invocation and this deliberately does not work around it. Prefer a
  CSV history: the same 300k rows take ~1.5 s as CSV against ~40 s as `.xlsx`,
  *per ticket*.
- **Each resolved ticket costs an embedding call and up to six LLM calls**
  (three Actor/Judge rounds, twice if Tier 2 runs).

`--limit N` resolves only the first N rows so you can price a trial before
committing to 500; the rest still appear, marked `NOT RUN`, so the sheet stays
aligned with the input. `--stop-after-errors N` (default 5) abandons the run
after that many **consecutive** infrastructure errors — wrong credentials would
otherwise burn one failing call per row for the whole sheet, and every row would
carry the same useless message.

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
so a pooled distribution describes neither. With the local encoder out of the
pipeline nothing can currently trigger it — which is the point of keeping it.
It was written for a fallback that fired silently, and it is what would catch
the same mistake if one is wired back in.

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
tests/test_tier2_docs.py               62   docx parsing, cache invalidation, Tier-2 gating, the routing matrix
tests/test_resolver.py                 32   validation, part filtering, capping, dual workbooks, threshold
tests/test_eval_batch.py               32   case discovery, per-case isolation, the score columns
tests/test_file_reader.py              31   format dispatch, strict type gate, format agnosticism
tests/test_azure_embeddings.py         22   the only encoder: hard failure, config vs transient, threshold
tests/test_bulk_test.py                24   column preservation, row alignment, the not-attempted markers
tests/test_transferable.py             36   the transferability gate, and what it must NOT flag
tests/test_component_c_actor_critic.py 26   refinement, circuit breaker, fail-closed, the transferability gate
tests/test_structured_outputs.py       12   strict response_format, fallback, schema boundaries
tests/test_config.py                    9   env loading, model/threshold single-sourcing
tests/test_real_embedder.py            14   the real bge-small model (opt-in, needs the disabled deps)
```

The real-model tests exercise the disabled local encoder. They need torch and
sentence-transformers reinstalled, and load ~130 MB of weights, so they are
opt-in and skip cleanly when those are absent:

```bash
SPS_MODEL_TESTS=1 python -m pytest -q
```

They pin the properties the ranking maths assumes: 384 dimensions, unit-length
vectors (so the NumPy matmul *is* the cosine the 0.89 gate is calibrated on), and
the query instruction applied to queries but never to history passages.

`python -m scripts.verify_embedder` checks the **live Azure deployment**:
reachability, vector dimension, unit length, batch ordering, and then real
cosine numbers for related versus unrelated SPS text scored against both gates.
It ends by suggesting a range for each. That is the fastest way to find out
whether 0.50 and 0.35 are anywhere near right on a new deployment — both are
guesses until it has been run. `--local` runs the original bge-small checks
instead, and reports what to install if torch is absent.
