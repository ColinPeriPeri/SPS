# Supplier Problem Sheet (SPS) — Constrained RAG Pipeline

An asynchronous, constrained Retrieval-Augmented Generation pipeline that answers
supplier-submitted problem sheets **strictly from historical precedent**. It never
introduces external domain knowledge, never exposes internal tooling to suppliers,
and always emits a fixed JSON contract for human admin review.

Runs on a **CPU-only host with 10–12 GB RAM**: `BAAI/bge-large-en-v1.5` locally via
`sentence-transformers`, generation on an Azure OpenAI deployment at
`temperature=0.0`, `top_p=0.1`.

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

See the whole pipeline run end-to-end — no Qdrant, no Azure, no model download:

```bash
python -m scripts.demo
```

Run the test suite:

```bash
python -m pytest -q
```

349 tests with the full stack installed; 149 still run with no third-party
packages at all. No test needs a running server or an Azure key. The 15
real-model tests are opt-in (they load 1.3 GB of weights) and bring the total
to 364:

```bash
SPS_MODEL_TESTS=1 python -m pytest -q
```

---

## Layout

| Path | Role |
| --- | --- |
| `sps/contracts.py` | Shared dataclasses + the Section 4 output schema |
| `sps/config.py` | Every tunable from the spec, in one auditable place |
| `sps/sanitize.py` | **A.2** — null/short filtering, dedup by SHA-256 content hash |
| `sps/embedding.py` | BGE wrapper (lazy load, L2-normalized, CPU) |
| `sps/vectorstore/` | `VectorStore` protocol + Qdrant and in-memory adapters |
| `sps/indexing/` | **A** — watermark, delta reader, micro-batching indexer |
| `sps/indexing/flat_file.py` | **A** — .csv / .xlsx source, streaming, SQL override |
| `sps/validators.py` | Part-number canonicalisation + ticket gatekeeping |
| `sps/retrieval/in_memory.py` | **Primary retrieval** — filter, embed, rank per ticket |
| `sps/retrieval/scoring.py` | **B.3** — metadata boosting (legacy Qdrant path) |
| `sps/retrieval/retriever.py` | **B** — validation, retrieval, the 75% gate |
| `sps/schemas.py` | Pydantic: LLM response schemas + the 4-key contract |
| `sps/generation/prompts.py` | **C** — Actor and Judge system prompts |
| `sps/generation/actor_critic.py` | **C** — the loop and the circuit breaker |
| `sps/pipeline.py` | Orchestration; the only place the components meet |
| `service/run_inference.py` | UiPath-facing CLI + callable (no web framework) |
| `service/excel_output.py` | Contract — DataFrame — atomic .xlsx |
| `service/status_file.py` | STATUS / EXIT_CODE / REASON side-channel |
| `scripts/run_inference.cmd` | Reference Windows batch wrapper for the Performer |
| `scripts/run_resolver.py` | **Primary entry point** — ticket in, two workbooks out |
| `scripts/` | `run_indexer` (legacy), `audit_part_numbers`, `demo`, `verify_embedder` |

**Components A and B depend on nothing but the standard library.** Sanitization,
dedup, the boosting arithmetic and the confidence gate import cleanly without
torch, qdrant-client, openai, pandas or pydantic — so the rules that decide
whether a supplier gets an answer at all are testable in under a second on any
machine. Component C needs pydantic (its LLM responses are schema-validated)
and the Excel handoff needs pandas/openpyxl; both sit behind lazy imports, and
their tests skip cleanly where those packages are absent. The heavy pieces sit
behind `Embedder`, `VectorStore` and `ChatClient` protocols.

---

## Component A — offline incremental indexing

```bash
python -m scripts.run_indexer                          # nightly / weekly SQL delta
python -m scripts.run_indexer --source-file dump.csv   # local file instead of SQL
python -m scripts.run_indexer --source-file dump.xlsx
python -m scripts.run_indexer --source-file dump.csv --dry-run
python -m scripts.run_indexer --reset-watermark        # deliberate full rebuild
```

### Building from a flat file

`--source-file` bypasses the SQL connection entirely — `SPS_SOURCE_DSN` is not
needed and not read. Everything downstream is unchanged: the same sanitization,
the same `content_hash` deduplication, the same micro-batching, the same payload.

| | |
| --- | --- |
| Required columns | `SPS_ID`, `Problem_Description`, and the solution column |
| Solution column | `Solution_Text` **or** `Actual_Solution` — both spellings are accepted |
| Optional columns | `Part_Number`, `Issue_Type`, `Problem_Reason_Code`, `Part_Description`, `Item_Status`, `Last_Modified_Date` |
| Formats | `.csv` / `.tsv` (pandas, chunked) and `.xlsx` / `.xlsm` (openpyxl, read-only) |

Headers are matched case- and separator-insensitively, so `problem description`
and `Problem_Description` both work. A missing **required** column fails before a
single row is read, naming what it expected and what it actually found — far
better than discovering 300k empty problem descriptions after the embedding run.

**Both formats stream.** CSV goes through `read_csv(chunksize=…)`;
XLSX uses `openpyxl` in `read_only` mode, which iterates rows rather than
materialising the sheet. `pandas.read_excel` has no chunked mode and would pull
all 300k rows into memory at once — the very spike that makes CSV preferable —
so the workbook path uses openpyxl directly. Either format is safe on the
10–12 GB host.

**A flat load is a full load, not a delta.** Every row is read regardless of the
high-water mark, which is safe to repeat because point IDs derive from the
`SPS_ID` — re-running the same file updates in place rather than duplicating.

**Without `Last_Modified_Date`** two things change, and the indexer warns about
each: duplicate pairs are resolved by `SPS_ID` order instead of recency, and the
watermark does not advance (so a later SQL run still starts from scratch rather
than silently skipping rows the file never contained). Include the column if the
extract has it and both behaviours return to normal.

**Numeric-looking identifiers need care in Excel.** Excel holds every number as
a double, so a part number or SPS_ID that looks numeric can arrive as
`1243951.0`; stored verbatim that is an identifier which matches nothing and
cites nothing. Whole floats are rendered without the fractional part at ingest
(genuine decimals are left alone), and CSV is read with `dtype=str`.

What cannot be recovered is **leading zeros**: if a sheet stored `0012-43951` as
a number rather than text, the zeros were gone before this code saw the file.
Format part-number columns as Text, or supply CSV. A hyphenated value like
`0012-43951` is text to Excel already and is safe either way.

**Without `Problem_Reason_Code`** the `+0.02` boost cannot fire, so an exact
text match tops out around 96% rather than 98%. (An exact match scores 0.9389,
not 1.0, because the query carries the BGE instruction prefix and the passage
does not.) Worth knowing before tuning `SPS_CONFIDENCE_THRESHOLD` against an
index built from a partial extract.

**Delta tracking.** The high-water mark is a composite `(Last_Modified_Date, SPS_ID)`,
not a bare timestamp — a bare timestamp silently skips records that share the
boundary second. It is committed **after every batch** and written atomically
(temp file + `os.replace`), so a crash mid-run resumes at the last completed batch
instead of restarting the delta or, worse, reading a truncated mark.

**Sanitization.** Rows are dropped when `Problem_Description` or `Actual_Solution`
is null, empty, or under 15 characters. Whitespace is collapsed *before* the length
check, so a field of newlines cannot pass the gate.

**Deduplication.** Identical problem–solution pairs collapse to the latest SPS_ID,
ordered by `(Last_Modified_Date, SPS_ID)` so the winner is stable when duplicates
share a timestamp. Dedup operates at two levels:

- *Within a run* — the `DedupeLedger` carries SHA-256 content hashes across the
  whole run, not just a batch, because duplicates routinely straddle a batch
  boundary. The source stream is ordered ascending, so the later SPS_ID wins.
- *Across runs* — the hash is persisted as `content_hash` in the vector payload,
  and each batch does **one** filtered lookup against the index, not one per
  record. An identical pair submitted weeks later under a new SPS_ID replaces the
  incumbent instead of joining it, which is what keeps retrieval from degrading
  into semantically identical neighbours.

`content_hash` is SHA-256 over the *sanitized* pair, joined on an ASCII unit
separator so `("ab", "c")` and `("a", "bc")` cannot collide, and case- and
whitespace-folded so cosmetic reformatting does not read as a new record.

**Micro-batching.** Records stream through batches of 250–500 (default 400); the
band is enforced at construction, and a value outside it is a hard configuration
error rather than a silent OOM at 3 a.m. Each batch is embedded, upserted, then
`del`-ed with an explicit `gc.collect()` before the next chunk is pulled. The
source itself uses keyset pagination, so nothing loads the full table into RAM.

**Embedding target.** Only the cleansed `Problem_Description` is embedded — no
metadata, no solution text (`test_only_the_problem_description_is_embedded` pins
this).

**Upsert payload.**

```json
{
  "sps_id": "SPS-1099",
  "content_hash": "a8f2699b…  (SHA-256 of sanitized problem + solution)",
  "actual_solution": "Rework the weld seam per the original joint profile…",
  "part_number": "PN-1000",
  "part_description": "Mounting bracket",
  "item_status": "Active",
  "problem_reason_code": "RC-WELD",
  "issue_type": "Quality"
}
```

`content_hash` is ETL bookkeeping and never reaches the model: `Candidate` has no
field for it and the prompt builders read named attributes only, so an added
payload key cannot leak into the Actor's or Judge's context
(`test_new_payload_fields_cannot_reach_the_actor_or_judge` pins this).

**Eviction.** Incremental indexing is kept equivalent to a full rebuild: records
that become unusable (a solution blanked in the source) or are superseded by a
duplicate are **deleted** from the index, not merely skipped. Point IDs are
`uuid5(namespace, sps_id)`, so re-indexing a modified record updates it in place.

---

## Component B — retrieval and the confidence gate

1. **Validation** — `Problem_Description` must be ≥ 10 characters. Invalid tickets
   abort before the embedder is touched, so a malformed submission costs nothing.
2. **Retrieval** — Top 15 by cosine similarity, **hard-filtered to the
   incoming `Part_Number`**. Semantic similarity is only ever computed against
   history for the same part; a ticket with no part number searches the whole
   index rather than being pinned to records whose part number is also blank.
3. **Boosting** — composite score = cosine (clamped to `[0,1]`) plus
   `+0.03` issue type and `+0.02` reason code; re-sorted by the composite.
   Part number is **not** scored: it is a hard filter on the search, so every
   candidate reaching this point already matches it.
4. **Gate** — a top composite below `SPS_CONFIDENCE_THRESHOLD` aborts **before
   any LLM call**. The spec default is 0.75; see *Threshold calibration* below for
   why the shipped `.env.example` sets 0.82 pending eval-set tuning. The
   `Justification` text follows whatever is configured.

Two deliberate decisions inside the arithmetic:

- **Blank ≠ match.** Two records both missing an `Issue_Type` do not earn that
  boost. Only a shared, populated value does. Otherwise sparse metadata would
  inflate every score uniformly and push weak matches through the gate.
- **Clamping.** Negative cosine is floored at 0 (unrelated is not *worse* than
  unrelated, and boosts must not rescue an irrelevant record), and the composite is
  capped at 1.0 so confidence never reports above 100%.

Ties break on raw cosine then SPS_ID, so a given query always produces the same
ordering — required for an auditable recommendation.

### The part-number filter

Semantic similarity is only ever computed against history for the same part.
Measured on a three-record index (`SPS-1099`/`PN-1000`, `SPS-1002`/`PN-1000`,
`SPS-1003`/`PN-2000`), same query text throughout:

| incoming part | outcome | confidence |
| --- | --- | --- |
| `PN-1000` | ok | 93% (raw cosine 0.9389) |
| `pn-1000` | ok | 93% — canonicalised, same result |
| `PN-2000` | below threshold | 46% |
| `PN-NEW` | **no matches** | 0% |
| *(none)* | ok | searches the whole index |

**Part numbers are canonicalised** — `.strip().upper()` — at ingest, on the
incoming ticket, and on the filter value itself. Without that, `pn-1000` would
report `NO_MATCHES` for a part that is plainly indexed. Matching is otherwise
exact: `PN-100` does not match `PN-1000`.

> **Check for drift before relying on the filter.** An index built before this
> change holds part numbers exactly as the source wrote them, so any lower- or
> mixed-case value is unreachable. Audit it first — the fix is a payload
> rewrite, not a re-embed:
>
> ```bash
> python -m scripts.audit_part_numbers          # read-only report
> python -m scripts.audit_part_numbers --fix    # rewrite, then re-verify
> ```
>
> `part_number` is payload only — the vector encodes the cleansed
> `Problem_Description` and nothing else — so `set_payload` corrects it in
> seconds rather than hours of re-encoding. Re-running a `--source-file` load
> also fixes drift, since a flat load rewrites every payload.
>
> The same pass reports two data-quality figures for free: the **blank**
> part-number rate (blank history is unreachable to any ticket that supplies a
> part number) and the count of **purely numeric** part numbers (whose leading
> zeros Excel destroys in the sheet itself).
>
> Where the house format is alphanumeric and free of stray spaces — e.g.
> `0012-43951` — both canonicalisation steps are no-ops and every figure
> should read zero. That makes this an **assumption check**: a non-zero result
> means something about the source changed and is worth looking at before an
> evaluation run.

**A blank part number applies no filter.** A ticket that arrives without one
searches the whole index rather than being pinned to records whose part number
is also blank.

**Two consequences worth carrying into the evaluation:**

1. **`NO_MATCHES` is now routine.** It used to mean an empty index; it now also
   means "no history for this part", which any new or low-volume part hits
   immediately. The contract still returns correctly at `Confidence: 0%`.

2. **History with a blank `Part_Number` is unreachable** to any ticket that
   supplies one — accepted as intended, but worth measuring the blank rate in
   the extract before the full run, since that history is otherwise dark.

The spec's `+0.05` part-number boost was **removed** along with this change.
With the filter in place every candidate matched the part by construction, so
the boost fired on all of them, stopped discriminating, and became a constant
that lifted every score — softening a configured 0.82 gate to an effective
0.77. Confidence is now the cosine on the problem text plus only the boosts that
still discriminate. The same query that read 98% before reads 93% now; that is
the floor being removed, not a regression.

### Threshold calibration — measured, and worth a second look

`python -m scripts.verify_embedder` reports real cosine numbers from
`bge-large-en-v1.5`. Query: *"Bracket weld seam cracking observed during incoming
inspection"*.

| candidate | with query prefix | without prefix |
| --- | --- | --- |
| identical text | 0.9389 | 1.0000 |
| paraphrase (*"Cracks found in the weld seam of the mounting bracket at goods-in"*) | 0.7926 | 0.8572 |
| **different defect, same part** (*"Weld porosity detected on the bracket joint"*) | **0.7831** | 0.8515 |
| unrelated (*"Outer carton label misprinted"*) | 0.4395 | 0.4848 |

Two things follow, neither of which changes the spec but both of which matter
operationally:

1. **The query prefix costs ~0.06–0.07 of raw score.** The encoding is
   deliberately asymmetric — queries are prefixed, passages are not — so even an
   *identical* problem statement scores 0.9389 rather than 1.0. Ranking is
   unaffected (order is identical with and without), but absolute scores sit
   lower than an unprefixed intuition would suggest, and the reported
   `Confidence` inherits that.

2. **0.75 is a loose gate in this embedding space.** Clearly unrelated text is
   rejected decisively (0.44). But *weld porosity* — a genuinely different defect
   that would need a different fix — scores 0.7831 against *weld seam cracking*
   and clears the gate. The part filter does not help here: both are the same
   part, which is exactly when the confusion arises. With a matching issue type
   and reason code it collects the remaining `+0.05`, reaching ~0.83.

So the gate reliably separates "unrelated" from "same subject area", not
"same problem" from "different problem". The Judge is the real defence against a
wrong-precedent recommendation, which is an argument for keeping it strict rather
than relaxing it later for cost. The eval set should measure this distribution
over labelled relevant/irrelevant pairs; if it holds up, the threshold likely
wants to move up (0.82–0.85) rather than stay at 0.75. `SPS_CONFIDENCE_THRESHOLD`
is already an env override, so that is a config change, not a code change.

---

## Component C — the actor–critic loop

**Actor** — a closed-book synthesizer. Its context is the incoming problem and the
verbatim `Actual_Solution` text of the qualifying records, *and nothing else*. This
is structural, not just instructional: the Actor is never shown internal metadata,
so there is none in its context to leak.

**Judge** — a strict compliance auditor running exactly the two mandated checks
(domain hallucination, internal tool leakage) and returning
`{"status":"PASS"}` or `{"status":"FAIL","critique":"..."}`.

**Both are schema-constrained, not schema-requested.** Each call passes a
pydantic model to Azure OpenAI as a strict `response_format`, so the deployment
cannot return a shape the pipeline then has to defend against. `JudgeVerdict`
types `status` as `Literal["PASS", "FAIL"]`, which is why an unrecognised
verdict raises rather than being read as a pass. Deployments or API versions
without `json_schema` support fall back to JSON mode plus client-side pydantic
validation, reaching the same guarantee one round-trip later; the fallback is
remembered so it is probed once, not per call.

**What the LLM is *not* allowed to emit.** `ActorDraft` carries exactly two
fields, `recommendation` and `justification`. `Confidence` is the retrieval
composite score and `SPS_IDs_Referred` is the set of records the retriever
actually passed to the Actor — both are measured, not generated. Putting them
in the LLM's schema would let it state a confidence it never computed and cite
records it was never shown, which is exactly the fabrication the rest of the
architecture exists to prevent. The pipeline supplies those two fields; the
assembled result is then validated against `SPSContract` before handoff.

**Refinement** — a `FAIL` routes the critique *and the rejected draft* back to the
Actor, so the rewrite is targeted rather than a blind resample (`temperature=0.0`
would otherwise likely reproduce the same draft).

**Circuit breaker** — 3 attempts total (1 draft + 2 retries). Still failing on the
third → `"Solution not found."`

Everything **fails closed**:

| Situation | Behaviour |
| --- | --- |
| Judge call errors out | No recommendation. An unaudited draft is never shipped. |
| Judge returns an unrecognised status | Treated as a rejection, not a pass. |
| Actor abstains (`"Solution not found."`) | Returned immediately — retrying would only pressure it into inventing something. |
| Any unexpected exception | Caught in `pipeline.process()`; the contract is still emitted, and the fault is logged for ops. |

---

## Output contract

Every exit path — success, invalid input, gate failure, circuit breaker,
unexpected fault — emits exactly these four keys:

```json
{
  "AI_Recommendation": "1. Segregate the affected lot.\n2. Rework the cracked weld seam...",
  "Justification": "SPS-1099 records the same weld seam cracking on this bracket...",
  "Confidence": "84%",
  "SPS_IDs_Referred": ["SPS-1099"]
}
```

`Confidence` **truncates** rather than rounds. A system that gates at 75% must
never report `"75%"` for a candidate it rejected at 0.7499, and understating
confidence is the safe direction for a compliance-reviewed recommendation.

Note: on a circuit-breaker failure the reported `Confidence` is the *retrieval*
confidence, which was genuinely high — the failure was in generation, not
matching. `SPS_IDs_Referred` is empty because no vetted recommendation was
produced from those records.

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

> **The threshold is coupled to the model.** 0.89 belongs to bge-small. The
> legacy `service/run_inference.py` path runs bge-large, where 0.89 would reject
> even a close paraphrase (0.7926); its default stays at the spec's 0.75.
> `SPS_CONFIDENCE_THRESHOLD` is read by both entry points, so set it per
> invocation, or pass `--threshold`, if you run the legacy path.

Metadata boosting is gone from this path entirely: part number is an exact
filter, and the other boosts existed to discriminate within a mixed-part result
set that no longer occurs. `Confidence_Score` is the cosine alone.

---

## UiPath integration

Deployed under a UiPath Dispatcher-Performer on Windows Server. **No web
framework, no server process, no database access from Python.** UiPath owns all
SQL reads and writes; this codebase is a callable that takes a ticket and prints
a JSON contract.

### Invocation

Integration is pure file-in / file-out plus an exit code. Nothing is passed on
the command line except paths.

```bash
python -m service.run_inference --payload-file in.txt     --output-file out.xlsx --status-file status.txt

python -m service.run_inference --batch-file queue.jsonl     --output-file results.xlsx --status-file status.txt

python -m service.run_inference --payload-file in.txt --output-file out.txt   # JSON instead
```

A reference wrapper ships at `scripts/run_inference.cmd`:

```bat
scripts\run_inference.cmd <payload-file> <output-file.xlsx> <status-file>
```

It resolves the venv, runs from the project directory, swallows stdout, leaves
stderr attached for the job log, and propagates the exit code verbatim.

Or import it, skipping the process boundary entirely:

```python
from service.run_inference import run_inference, run_batch
contract = run_inference({"problem_description": "...", "part_number": "PN-1000"})
```

### The integration contract

| Channel | Guarantee |
| --- | --- |
| `--output-file` **`.xlsx`** | A workbook: header row plus one row per ticket, columns `AI_Recommendation`, `Justification`, `Confidence`, `SPS_IDs_Referred`. Written with `to_excel(index=False)`, so there is no leading index column to shift every reference in the workflow. |
| `--output-file` (any other extension) | The same contract as JSON text, UTF-8, no BOM. |
| `--status-file` | Three lines — `STATUS`, `EXIT_CODE`, `REASON`. Written on every exit path, and written **last**, only once the data file is complete. |
| stdout | The contract as JSON, always, whatever the file format. The process stays debuggable by hand. |
| stderr | Logs, diagnostics, fault traces. Never mixed into stdout. |
| exit 0 | The pipeline ran. **Includes `"Solution not found."`** — a refusal is an expected business outcome. |
| exit 1 | Infrastructure fault: Azure unreachable, storage locked, bad config, or the output file could not be written. |
| exit 2 | The payload could not be parsed as JSON. Fault the item; do not retry. |

Both file formats are written **atomically** (temp file in the destination
directory + `os.replace`). For the workbook this matters more than for JSON: a
partially written .xlsx is not merely truncated, it is a corrupt zip container
that Excel and UiPath both refuse to open.

Three further properties the file channel depends on, each pinned by a test:

- **The file is written on every exit path** — success, gated, circuit-breaker,
  outage, unparseable payload. A file-based caller is never left with no answer.
- **A stale file is deleted before any work begins.** If the process is killed
  outright, the caller finds *nothing* rather than the previous run's result,
  which it would otherwise read as this transaction's answer. Missing file and
  non-zero exit both mean "do not trust this transaction".
- **An unwritable output path is exit 1**, even when the pipeline itself
  succeeded: the answer exists but could not be handed over, which is a system
  fault, not a business outcome.
- **Input files are read as `utf-8-sig`.** .NET writes UTF-8 *with* a BOM by
  default, so a payload produced by a Write Text File activity normally starts
  with one. Read as strict `utf-8` that fails with `Unexpected UTF-8 BOM` and
  every ticket would be rejected as an invalid payload. Output stays strict
  `utf-8` with no BOM, which is what RFC 8259 requires of JSON.

**The 0-versus-1 split is the part that matters most.** Both a gated ticket and
an Azure outage produce `"Solution not found."`, and without the distinction a
Performer would mark every item Successful during an outage and quietly burn the
whole queue. Retryable faults ride on `PipelineResult.infrastructure_failure`,
which is deliberately *not* part of `to_contract()` — the JSON stays exactly four
keys.

### The workbook

Every cell is written as text. Without that, pandas would infer dtypes and a
`Confidence` of `"84%"` could reach Excel as a number or, worse, a date.

`SPS_IDs_Referred` is a list, which has no faithful single-cell representation.
It is joined on `", "` — `SPS-100, SPS-101` — which reads well for the support
team and splits trivially in UiPath. Left as a list, a DataFrame would write the
Python repr `['SPS-100', 'SPS-101']` into the cell. An **empty** list is an empty
cell, which openpyxl reports as `null` and pandas as `NaN`; treat both as "no
records cited".

Cells longer than Excel's 32,767-character limit are truncated with a visible
`[…truncated for Excel]` marker rather than failing the handoff. The full text is
always available on stdout.

### The status file

```
STATUS: SUCCESS
EXIT_CODE: 0
REASON: Processed successfully.
```

Three lines, so exception handling never has to parse a workbook to find out
what happened, and a support engineer opening the file sees the cause in plain
text.

**STATUS is `SUCCESS` if and only if `EXIT_CODE` is 0.** They cannot disagree by
construction. That is deliberate: if they could, the workflow's behaviour would
depend on which one it happened to read, which is the exact class of bug this
file exists to remove.

One consequence worth knowing: a gated ticket reports **`STATUS: SUCCESS` with
`EXIT_CODE: 0`**, because the pipeline ran correctly and produced a legitimate
answer — it just was not a recommendation. `REASON` says
`Confidence below 82% threshold.` so the outcome is unmistakable. Marking that
`FAILURE` would contradict the exit code and fault a queue item that succeeded.

`EXIT_CODE` carries the real code, including **2** for an unparseable payload.
That is a business exception (fault the item, do not retry), distinct from the
system exception that 1 warrants.

`REASON` is always one line — recommendation and justification text is
routinely multi-line, so it is collapsed and capped rather than trusted — and
it carries the *specific* diagnostic on a fault, naming the missing credential
variables where the supplier-facing `Justification` stays generic. Values are
never included, only names.

**Ordering matters and is guaranteed:** the status file is written **last**, only
after the data file is complete. So `SUCCESS` always means the workbook beside it
is present and readable. Both files are cleared before any work begins, status
first, so there is never an instant where a stale `SUCCESS` points at a deleted
data file. If the process is killed outright, neither file exists.

### Ticket payload keys

Both spellings are accepted, so a hand-written ticket works either way:

```json
{ "SPS_ID": "SPS-999", "Problem_Description": "...", "Part_Number": "PN-1000" }
{ "sps_id": "SPS-999", "problem_description": "...", "part_number": "PN-1000" }
```

Matching ignores case, underscores and spaces. This matters more than it looks:
every other interface in the system names fields the SQL way, so a payload
written as `Problem_Description` is the natural thing to produce. Reading only
the lower-case spelling yielded an empty ticket — the description judged
invalid, the part number dropped so the search filter failed open — and the
process exited **0**, indistinguishable from a legitimate refusal.

### Credentials

Azure credentials are read **only** from the environment:

```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
```

There is no CLI flag for any of them, and `--api-key`-style arguments are
rejected by the parser. Command lines are visible in the Windows process list
and in UiPath job logs, so a credential passed as an argument is a credential
disclosed. A `.env` beside the project is loaded if present (real environment
variables always win), which covers robots that do not inherit an interactive
shell's environment.

Missing credentials are reported on stderr **by variable name, never by value**,
and outage detail never reaches `Justification` — an admin reviewing a ticket
should not be shown internal endpoints, and the field may eventually surface to
a supplier.

### Embedded Qdrant

```ini
SPS_QDRANT_PATH=./qdrant_data
```

Setting a path switches Qdrant to embedded mode: in-process, persistent local
directory, no Docker and no Windows service. `SPS_QDRANT_URL` is then ignored.

Embedded Qdrant is **single-writer** — it holds an exclusive lock on the
directory. The UiPath schedule guarantees exclusivity by pausing the Performer
queue during an indexing run, and the code fails loudly rather than silently if
that is ever violated: a second opener gets `VectorStoreBusyError` naming the
directory. Both CLIs call `store.close()` in a `finally`, so a crashed run does
not leave the storage locked for the next scheduled process.

Two consequences of embedded mode, measured rather than assumed (50k points,
1024-dim, this hardware):

| operation | at 50k | extrapolated to 300k |
| --- | --- | --- |
| vector search, top-15 | 0.17 s | ~1.0 s |
| `find_by_content_hash`, 400 hashes | 0.95 s | ~5.7 s |

1. **Search is exact, not approximate.** Local mode builds no HNSW index, so
   query cost is linear: about a second per ticket at 300k. Fine for an async
   queue; it would not be for an interactive UI.
2. **Payload indexes are a no-op locally** (qdrant-client says so on startup).
   The per-batch `content_hash` filter is therefore a full scan. Incremental runs
   touch one or two batches and cost seconds. A 300k *backfill* would be 750
   batches — roughly 36 minutes of pure scanning — so the indexer now **skips the
   cross-run lookup entirely when the run starts from the initial watermark**. A
   rebuild reads the whole table, so the in-run ledger already sees every record
   and the store lookup is redundant. Dedup is unaffected
   (`test_rebuild_still_deduplicates_within_the_run` pins that).

Switching to a Qdrant server later is a config change — clear `SPS_QDRANT_PATH`,
set `SPS_QDRANT_URL` — and brings back HNSW and real payload indexes.

### Sequencing

```
UiPath Scheduler
  |
  +-- (nightly)  pause Performer queue
  |              python -m scripts.run_indexer          -> exit 0/1/2
  |              resume Performer queue
  |
  +-- (ongoing)  Dispatcher : SQL -> queue items
                 Performer  : write ticket JSON  -> in.txt
                              run_inference.cmd in.txt out.xlsx status.txt
                              read status.txt
                                SUCCESS  -> read out.xlsx, write rows to SQL
                                FAILURE + EXIT_CODE 1 -> system exception, retry
                                FAILURE + EXIT_CODE 2 -> business exception, fault
                                file missing          -> process was killed; retry
```

The indexer and the Performer must never overlap: embedded Qdrant is
single-writer, which is why the schedule pauses the queue for the nightly run.

---

## Configuration

Copy `.env.example` to `.env`. Spec-fixed values (`TOP_K=15`, threshold `0.75`,
min query length `10`, `MAX_ATTEMPTS=3`, batch band `250–500`) are module constants
in `sps/config.py`; the env vars exist for operational override, and the batch-size
band is validated rather than trusted.

---

## Assumptions and known boundaries

Stated explicitly rather than buried:

1. **BGE query instruction.** Queries are prefixed with
   `"Represent this sentence for searching relevant passages: "`, the convention
   `bge-*-v1.5` was trained with; indexed passages are not. This is a model-level
   encoding instruction, not record metadata, so Component A.4's "no appended
   metadata" rule is unaffected — the embedded content is still only the
   `Problem_Description`. Disable with `SPS_USE_BGE_QUERY_INSTRUCTION=false`.
   Verified against the real model: a passage vector is bit-identical to an
   unprefixed query vector (cosine 1.00000000), and ranking is unchanged with the
   prefix on or off. See *Threshold calibration* for its effect on absolute scores.

2. **Backfilling `content_hash` on an existing collection.** Cross-run dedup
   matches on the persisted `content_hash`, so points written *before* that field
   existed are invisible to it until they are re-indexed. On an index built by an
   earlier version, run `--reset-watermark` once to backfill. `ensure_collection`
   creates the `content_hash` payload index on every run, not just at creation, so
   a pre-existing collection picks it up without manual intervention.

   The cross-run winner is decided by an ordering invariant rather than a stored
   timestamp: the delta query only returns rows past the watermark, and everything
   already indexed was read at or below it, so an incoming record is always the
   newer one. A `--reset-watermark` rebuild breaks that invariant transiently but
   re-reads the table in ascending order, so the run still converges on the latest
   SPS_ID. Keeping the decision on that invariant is what let the payload stay at
   the eight approved fields instead of also persisting `Last_Modified_Date`.

3. **NumPy must stay below 2.x.** `torch==2.3.1` wheels are built against the
   NumPy 1.x C ABI. With NumPy 2.x installed, torch imports but its array bridge
   is dead, and any `tensor.numpy()` raises `RuntimeError: Numpy is not available`
   — which is where `sentence-transformers` lands by default. `requirements.txt`
   pins `numpy>=1.26,<2`. Independently, `BGEEmbedder` now encodes with
   `convert_to_tensor=True` and converts via `Tensor.tolist()`, so the hot path
   never crosses the numpy bridge at all and survives a mismatched environment.
   Both fixes are wanted: the pin makes fresh installs correct, the code change
   makes the embedder robust when someone's environment drifts.

4. **Actor context size.** All candidates ≥ 75% are passed to the Actor
   (`SPS_MAX_CONTEXT_RECORDS`, default 15 = the retrieval top-k, i.e. no
   truncation). Lower it only to trim LLM cost.

5. **`SqlRecordSource` dialect.** The delta query uses `SELECT TOP (:chunk)`
   (SQL Server / `pyodbc`, matching the shipped driver pin). For PostgreSQL or
   MySQL, change it to `LIMIT :chunk` — one line in `sps/indexing/source.py`. The
   table name is validated against a strict identifier allowlist before
   interpolation, since identifiers cannot be bound as parameters.

6. **Prompt effectiveness is not unit-testable.** The tests pin the loop's
   *control flow* (refinement, the breaker, fail-closed behaviour) and the
   presence of the mandated constraints in the prompts. Whether the Judge actually
   catches a given hallucination is a model-behaviour question and needs an
   evaluation set of labelled drafts against the real deployment before go-live.

---

## Test coverage

```
tests/test_excel_output.py             39   schema validation, DataFrame shape, atomic workbook write
tests/test_component_a_indexing.py     29   sanitization, dedup (in-run + cross-run), watermark, batching, eviction
tests/test_status_file.py              26   STATUS/EXIT_CODE agreement, reasons, write ordering
tests/test_cli_output_file.py          24   output file: atomicity, BOM both ways, stale safety, credentials
tests/test_component_b_retrieval.py    40   validation, boosting, ranking, the gate, part filter, payload keys
tests/test_cli_inference.py            21   stdout purity, exit codes, input modes, lock release
tests/test_component_c_actor_critic.py 19   refinement, circuit breaker, fail-closed, prompt isolation
tests/test_flat_file_source.py         31   .csv/.xlsx parity, header mapping, numeric identifiers, ingest logic
tests/test_qdrant_adapter.py           27   the adapter against a real Qdrant engine, server and embedded
tests/test_audit_part_numbers.py       21   drift detection, payload-only repair, assumption checks
tests/test_resolver.py                 33   validation, part filtering, capping, dual workbooks, threshold
tests/test_structured_outputs.py       17   strict response_format, fallback, schema boundaries
tests/test_pipeline_contract.py        13   every exit path emits a valid contract
tests/test_config.py                    9   env loading, spec constants, batch-band validation
tests/test_real_embedder.py            15   the real BGE model (opt-in, see below)
```

`test_real_embedder.py` loads the actual 1.3 GB `bge-large-en-v1.5` weights and is
therefore opt-in:

```bash
SPS_MODEL_TESTS=1 python -m pytest tests/test_real_embedder.py -q
```

It pins the three properties the rest of the pipeline's arithmetic assumes: 1024
dimensions, unit-length vectors (so the vector DB's cosine distance really is a
plain dot product and the 0.75 gate is calibrated on cosine), and the query
instruction applied to queries but never to indexed passages. There is also a
standalone report, `python -m scripts.verify_embedder`, which prints the real
cosine numbers for related vs unrelated SPS text.

`test_qdrant_adapter.py` runs against `QdrantClient(":memory:")` — qdrant-client's
local mode, which executes the real filtering and similarity code in-process. So
cosine ordering, the `content_hash` `MatchAny` filter behind cross-run dedup,
payload round-tripping and `uuid5` point identity are verified against the actual
engine rather than a mock. It skips cleanly where qdrant-client is absent.

