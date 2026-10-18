# Supplier Problem Sheet (SPS) — Constrained RAG Pipeline

A constrained Retrieval-Augmented Generation pipeline that answers
supplier-submitted problem sheets **strictly from historical precedent**. It never
introduces external domain knowledge, never exposes internal tooling to suppliers,
and always emits a status workbook for human admin review.

Two tiers of evidence, and they work differently.

**Tier 1** answers from the part's own historical SPS records. It does not
write anything: an LLM reads the shortlisted records, decides which is making
the same *request* as the ticket, and above a confidence gate the pipeline
sends that record's solution **exactly as recorded** — no rewrite, no audit.

**Tier 2** answers from the company's 0250 engineering standards, for a
configured set of reason codes only, and it keeps the full Actor/Judge
apparatus: a model drafts, a compliance auditor and two deterministic gates
review, and an unverified draft is never sent.

Neither tier is allowed to answer from the model's own knowledge.

Embeddings and generation both run on **Azure OpenAI**, at `temperature=0.0`,
`top_p=0.1`. **No vector database** — history is filtered by part number and
embedded per ticket; the standards corpus is embedded once and cached against a
hash of itself.

> **Azure is the only encoder.** The local `bge-small` encoder was disabled and
> has since been removed — see [The local encoder, removed](#the-local-encoder-removed).
> The practical consequence: **an Azure embedding failure ends the run**
> instead of being absorbed, and there is no offline mode.

All of it is two files: **`sps.py`**, everything that runs, and
**`sps_prompts.py`**, everything the model reads. The code imports four
packages — `openai`, `numpy`, `pydantic`, `openpyxl` — and otherwise only the
standard library, which is also what reads a Word document and loads `.env`.

---

## Quick start

Setting this up on a fresh Windows machine? **[`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)**
is the copy-paste version of this section, with the checks, the `.env` values and
a smoke test that proves the install before any real data touches it.

```bash
python -m pip install -r requirements.txt
```

No torch, no CUDA, no model download — 24 packages, test runner included.
Verify before running anything:

```bash
python -m pip check
python -c "from openai import AzureOpenAI; import numpy, openpyxl, pydantic"
```

`pip check` must print `No broken requirements found.`, and the second line must
print nothing at all. One pin exists purely to make that second line work on a
managed Windows machine: **jiter below 0.17**, whose native DLL is blocked by
Windows Application Control. It fails with *"An Application Control policy has
blocked this file"*, which reads like a corrupt install rather than a policy
decision — and takes every Azure call with it, since `openai` imports it. lxml 6
failed the same way; it is no longer installed at all.

Run the test suite:

```bash
python -m pytest -q
```

459 tests in about six seconds, none of which needs a server or an Azure key.

---

## Layout

| Path | Role |
| --- | --- |
| `sps.py` | **Everything that runs.** `python sps.py resolve \| bulk \| eval \| check` |
| `sps_prompts.py` | **Everything the model reads** — the intent scorer, the Tier-2 Actor and Judge, and their message builders |
| `scripts/run_resolver.cmd` | UiPath wrapper for `resolve`; propagates the exit code |
| `scripts/run_bulk_test.cmd` | Wrapper for `bulk` |
| `scripts/run_eval.cmd` | Wrapper for `eval` |
| `tests/` | The suite, run with `python -m pytest` |
| `samples/` | A ticket + history for smoke tests, a bulk sheet, six eval cases, twelve similarity pairs, three demo standards |
| `data/0250_docs/` | Where the real 0250 standards go (ships empty) |
| `docs/` | Architecture diagrams |

`sps.py` reads top to bottom in the order a ticket passes through it:

| Section | |
| --- | --- |
| Validation and part-number canonicalisation | the rules that decide whether a ticket is processable at all |
| Data contracts | ticket and result shapes |
| Configuration | environment settings, and the `.env` reader |
| Reading .csv and .xlsx | format dispatch and the strict type gate |
| LLM response schemas | the pydantic models every model response is validated against |
| Embedding | the Azure encoder |
| Tier 2: reading the 0250 .docx standards | citable, section-bounded chunks |
| Tier 2: the standards vector cache | hash-keyed, enriched query, search |
| Tier 1: history retrieval | filter by part, embed, rank, gate |
| The Azure OpenAI chat client | structured outputs, with a JSON-mode fallback |
| The deterministic gates | record-specific references, and misattributed actions |
| Tier 1: intent scoring | which past record asks the same question |
| The Actor/Judge loop | schema-constrained drafting and audit — **Tier 2 only** |
| Building results, Writing workbooks | the rows, and the atomic writer |
| Command: resolve, bulk, eval, check | the four commands, then the similarity probe and the entry point |

Prompt wording lives in its own file so a prompt change is a diff of prompt
text only. Azure is reached only through the `Embedder` / `ChatClient` seams,
which is what lets the suite run with neither a key nor a network.

---

## The resolver (primary path)

`python sps.py resolve` resolves one ticket against a history file with **no
vector database**. Because the part-number filter means semantic search only ever
runs against one part's history, that history is small enough to embed on demand
-- so the index, the indexer schedule, the embedded-storage lock and payload
drift all stop existing.

```bash
python sps.py resolve --ticket-file ticket.xlsx \
    --history-file history.csv --output-dir .\out
```

| Output | When | Columns |
| --- | --- | --- |
| `status.xlsx` | **Always**, including early aborts and unhandled exceptions | `Execution_Timestamp`, `Status` (PASS/FAIL), `Status_Code`, `Reason`, `Embedding_Model` |
| `output.xlsx` | Whenever the run **reached a conclusion** (exit 0), including "no recommendation". Never on exit 1 or 2 | `Part_Number`, `AI_Recommendation`, `Justification`, `Confidence_Score`, `Referenced_Sources`, `Resolution_Source`, `Closest_Matching_Solution`, `Cascade_Warnings` |

Both are one sheet named `Sheet1`, every cell stored as text — so `0012-43951`
never becomes a date, and archive text beginning with `=` never becomes a
formula. A character Excel cannot store (Word's manual line break is one) is
replaced on the way in rather than failing the write, and a cell over Excel's
32,767-character limit is cut with a visible marker.

### Status codes

The robot routes exceptions on this column, so it says *how* the pipeline ran
out of options, not merely that it did.

| Code | Meaning | Typical routing |
| --- | --- | --- |
| `SUCCESS_HISTORICAL` | Answered from the part's own precedent | Admin review |
| `SUCCESS_0250_DOC` | Answered from a 0250 standard | Admin review |
| `NO_MATCHES` | No history for the part **and** no standard covered it | Master Data |
| `BELOW_CONFIDENCE_THRESHOLD` | Candidates found in either tier, none cleared its gate | Reliability Engineer — a genuinely novel defect |
| `LLM_AUDIT_REJECTED` | Candidates cleared the maths, then were refused: no record asked the same question (Tier 1's intent scorer), or the Tier-2 Actor or Judge declined | Human reviewer |
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

### When Tier 1 finds a match

The matched record's `Solution_Text` becomes `AI_Recommendation` **character for
character**. Nothing is reworded, renumbered or reformatted; the one thing that
changes is that the history reader trims each cell at load, as it does for every
field. A reviewer can diff the cell against the source record and expect it to
match.

`Referenced_Sources` names the single record the solution came from — not the
whole shortlist, because one record's solution was sent.
`Closest_Matching_Solution` carries the scored shortlist so the rejected
candidates stay visible.

**`Cascade_Warnings` is the only thing standing between the archive and the
supplier.** It runs the same two scanners that *block* a draft in Tier 2, and
here they only annotate:

```
Contains: internal tracking identifier: 'ESW#20033465'; reference to an
attachment: 'attachment'; reference to a prior conversation: 'Per discussed'
```

The asymmetry is deliberate. Tier-2 text is written by a model, so rejecting it
costs a retry. Tier-1 text is the archive's own and is sent as recorded —
rejecting it would mean refusing a precedent the business asked to be sent. So
the risk is named and the decision sits with the reviewer.

### When there is no recommendation

DEA copies `AI_Recommendation` into the SPS portal by hand, and it reaches the
supplier from there essentially unedited. Everything below follows from that one
fact.

`AI_Recommendation` reads **"No recommendation available."** and `Justification`
explains it in the reviewer's language. The diagnostics — cosines, thresholds,
row counts — stay in `status.xlsx`'s `Reason`, where the robot and support
already read them. They used to be shown to DEA as well.

What `Closest_Matching_Solution` holds depends on whether there is an answer,
because the two cases need different things:

| Outcome | The column holds | Why |
| --- | --- | --- |
| No recommendation | the **3 closest records**, in full, banner-prefixed | There is no answer, so the raw records *are* the deliverable &mdash; a reviewer now writing the reply themselves is reading these as options |
| Resolved | a **scored source list**, no archive text | The recommendation is already the distilled answer |

Both also appear at the end of `Justification`, which is the field a reviewer is
already reading. On a refusal the records lead and the explanation follows; on a
success the rationale leads and the sources follow it.

### Why a resolved row does not quote its source

On a success the answer is already in `AI_Recommendation`, so repeating archive
text beneath it would add length without adding information. A resolved row
carries a **scored source list** instead:

```
Synthesized from 2 record(s): SPS-1001 (100%), SPS-1003 (31%)
```

In Tier 1 that is every record the intent scorer read, ordered by intent score,
with the record whose solution was sent first. In Tier 2 it is every standards
section the Actor was given, and there a recommendation may combine several —
quoting one section would read as *the* source and invite a reviewer to check a
step against a section it did not come from. The label reads "Synthesized from"
in both tiers, which is only accurate for Tier 2.

`Referenced_Sources` already lists the ids. What it cannot say is which scored
100% and which 31%, and that is the one thing telling a reviewer where to look
first.

On a refusal the limit is **3**, not all of them: enough that a near-miss is not
hidden behind a marginally better one, few enough that the cell stays readable.

`Closest_Matching_Solution` carries the best precedent found **whether or not it
was recommended, and whether or not it cleared the gate**:

```
[RAW HISTORY - WEAK MATCH, below the confidence gate] SPS-5002 (47%): 1. For
the feedback please see the attachment. 2. Per discussed, rework as attachment
shown. 4. ESW#20033465 is submitted for these issues.
```

The banner is a control, not a caption. That text names an attachment that does
not exist, a conversation the supplier was not party to, and another supplier's
work request — it is genuinely useful to a reviewer and genuinely unsafe to
paste, and the only thing standing between those two facts is the label.

This is the answer to *"if we can provide the solution as is from matching SPS
we are almost there"*. The precedent is shown in full; what it is not allowed to
do is bypass the checks on its way to a supplier.

> Retaining the below-gate row needed a change in the retriever: `retrieve()`
> returns only what qualified, so on a gated run the near-miss text existed
> nowhere at all. `RetrievalStats.top_candidates` — the closest three — is
> captured beside `top_score`, before the filter, and deliberately kept out of
> `as_dict()`: that feeds the log line, which does not want 500 characters of
> solution text.

### Why there was no recommendation

`Status_Code` says how far the pipeline got. `No_Recommendation_Reason`, in the
bulk sheet, says what stopped it — a different question, and the one worth
arguing from:

| Value | Constraint it points at |
| --- | --- |
| `NO_HISTORY_FOR_PART` | The archive — this part is new |
| `BELOW_THRESHOLD` | The gate, or the encoder |
| `INTENT_BELOW_THRESHOLD` | The archive — records found and read, none asks the same question |
| `ACTOR_ABSTAINED` | Tier 2 — the standards found do not answer this defect |
| `GATE_UNTRANSFERABLE` | **Our checks** — attachment, ticket number, prior conversation |
| `GATE_MISATTRIBUTED` | **Our checks** — the supplier was told to do the customer's job |
| `JUDGE_REFUSED` | **Our checks** — failed the compliance audit |

The last three stop a Tier-2 draft. A Tier-1 answer is never stopped by them:
it is the archive's own text, sent as recorded, and the same scanners annotate
it in `Cascade_Warnings` instead.

`python sps.py bulk` tallies these at the end of a run and prints the split:

```
  of those without one, why:
      14  ACTOR_ABSTAINED
       6  BELOW_THRESHOLD
       3  GATE_UNTRANSFERABLE
    -> 3 of 23 were stopped by a supplier-safety check; 20 by the data.
```

That last line exists because "your checks are too strict" and "the archive has
no answer" lead to opposite fixes, and the difference between them is countable
rather than arguable. Run it before changing a guard.

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
validate -> filter history by part -> cap to newest 300 -> embed -> rank -> gate -> intent scorer
                                                                                         |
                                                             (nothing usable) -----------+
                                                                      |
                                       Tier 2:  load cache -> rank chunks -> gate -> Actor/Judge
```

A malformed part number is rejected before anything is embedded. Tier 2 runs
only once all of Tier 1 has failed, which is what lets it afford to parse and
embed a document corpus.

### Measured latency, 300k-row history

| step | cost |
| --- | --- |
| process start and imports | 0.3 s |
| history scan, CSV | 1.45 s |
| embedding and the intent-scoring call | Azure round trips; not yet measured against the deployment |

The process-start figure is the whole cold-start cost now: there is no model to
load, so a ticket per process costs no more than a resident process would. (The
figures this table once carried for embedding and model load were measured on
the since-removed local encoder.)

**Use CSV for a large history.** The same 300k rows take **~40 s** as `.xlsx`
against ~1.5 s as `.csv`, because openpyxl inflates and parses XML per row while
a CSV is a linear read. The engine reads both; the format is the difference
between meeting the 3 s budget and missing it by 13x.

### Part-number gatekeeping

The first section of `sps.py` canonicalises with `.strip().upper()` plus removal of
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

The reader in `sps.py` ("Reading .csv and .xlsx") is the only place that knows
the difference. Everything past it works on a header row plus data rows, so
column mapping, part canonicalisation and filtering are format-blind. The same
history in both formats produces identical results, down to the similarity
score, and the suite holds it to that.

**Anything else hard-fails immediately**, on the extension alone, before a file
is opened, anything is embedded or a history is scanned:

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

### The local encoder, removed

Azure OpenAI is the only encoder. The `bge-small` path was first disabled and
then removed outright — the wrapper, its thresholds (0.89 for Tier 1, 0.62 for
Tier 2), `check --local` and the opt-in real-model tests — because it needed
torch to run at all and nothing reached it. It is recoverable from git history.
The `embedder=` seam on both retrievers stays: the tests inject their stand-in
through it, and anything injected is labelled `injected` in the status sheet,
never mistaken for the deployment.

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

### Thresholds

A threshold is a property of one model's scoring distribution and does not
survive a change of encoder. Three gate the pipeline, one per comparison:

| | default | what it decides | basis |
| --- | --- | --- | --- |
| `AZURE_EMBEDDING_THRESHOLD` | **0.30** | Tier 1's cosine — a **recall filter**: which records the intent scorer gets to read | **Provisional, never measured.** Lowered from 0.50 when intent scoring took over the decision, so wording similarity cannot silently discard a record whose request matches |
| `SPS_INTENT_THRESHOLD` | **0.75** | Tier 1's **decision**: the intent score a record must reach for its solution to be sent | An LLM's score, not a calibrated probability |
| `TIER2_AZURE_THRESHOLD` | **0.35** | Tier 2's cosine gate | **Provisional twice over** — the old 0.50 scaled by bge-small's Tier-1:Tier-2 ratio |

> **Every gating decision the pipeline makes rests on an unmeasured number.**
> Removing the local encoder retired the two measured thresholds. How far off
> the Azure numbers are depends on which model backs the deployment:
> `text-embedding-3-small` / `-3-large` put unrelated text around 0.1—0.3;
> `text-embedding-ada-002` keeps even unrelated pairs above 0.7, where these
> gates would admit essentially everything.
>
> `python sps.py check` reads both cosine gates against the real deployment and
> suggests a range for each. Run it before trusting either, and widen it with
> `python sps.py eval`.

Tier 1's cosine gate is overridden by `--threshold`, then
`SPS_CONFIDENCE_THRESHOLD`, then `AZURE_EMBEDDING_THRESHOLD`, before the
built-in 0.30. Tier 2's by `--tier2-threshold`, then `SPS_TIER2_THRESHOLD`, then
`TIER2_AZURE_THRESHOLD`. The intent gate by `SPS_INTENT_THRESHOLD`. One override
spanning both tiers would be a single number standing for two comparisons.

### Which encoder answered

`status.xlsx` carries `Embedding_Model`, appended as the **last** column so a
caller reading the first four positionally is unaffected:

```
azure:text-embedding-3-large     the encoder
(blank)                          aborted before anything was encoded
injected:<name>                  a stand-in handed in through the test seam
```

`Reason` also ends with `[Azure]` (or `[Injected]`), so a support engineer
skimming the sheet — or a caller reading only the first four columns — sees which
encoder ran without needing to know the column exists.

The column kept its value after the fallback was removed: a ticket whose Tier 1
never embedded — an unknown part stops before anything is encoded — still
reports Tier 2's encoder if Tier 2 ran, so a run that embedded is never recorded
as one that did not.

Metadata boosting is gone from this path entirely: part number is an exact
filter, and the other boosts existed to discriminate within a mixed-part result
set that no longer occurs. On a Tier-1 answer `Confidence_Score` is the intent
score of the record sent; on a Tier-2 answer, the best cosine among the sections
used.

---

## What must never reach a supplier

Two kinds of text reach `AI_Recommendation`, and they are guarded differently.
**Tier-2 text is written by a model**, so it is drafted, scanned by two
deterministic gates and audited by a Judge, and a draft that fails is rewritten
or refused. **Tier-1 text is the archive's own**, sent as recorded: the same two
scanners run over it, and their findings are written to `Cascade_Warnings`
rather than blocking it. Most of this chapter is the history of why each guard
exists; where a guard once ran in Tier 1, it now lives in Tier 2 or in that
column.

### Both fields, not just the recommendation

Every gate and the Judge read `recommendation`. For a long time nothing read
`justification` -- which is written to `output.xlsx` and read by the reviewer.
So this passed every check:

```
recommendation : 1. Rework the weld seam. 2. Re-inspect under 10x magnification.
justification  : Based on SPS-100, for which ESW#20033465 was submitted; see the attachment.
```

The gate rejects that text the moment it is handed it; it was simply never
handed it. `_sanitised()`, in the Actor/Judge loop, now scans the justification
on the way out of a PASS and **blanks it on a finding rather than rejecting the
draft**. The result builder already substitutes generated text for an empty
justification, so there is no new path, no second model call and no retry.

That asymmetry is deliberate. The recommendation is already clean by then;
discarding it over its rationale would spend a retry -- often the whole ticket
-- to fix prose that is context for the reviewer rather than the text DEA sends
on. A careless explanation is not worth a lost answer.

### The gap the gates exist to close

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
so the Judge's grounding check passed them. Neither tells the supplier to use an
internal system, so its tool-leakage check passed them. Both were wrong anyway:

- `ESW#20033465` is an internal work request raised for a **different** issue.
- There is no attachment — `output.xlsx` is text.
- No discussion has taken place with this supplier.

A whole class of facts is true of the *record* and false about the *ticket*:
tracking IDs, attachments, prior conversations, dates, lot and PO numbers, names.

### Two layers

**A Judge check whose load-bearing sentence is that appearing in the source is
not a defence.** A grounding check trains the model toward "in the source ⇒
fine", which is exactly why the ESW number survived an audit it should have
failed. The Tier-1 Judge carried this as `CHECK 3 — CONTEXT TRANSFER` until
Tier 1 stopped drafting; the Tier-2 Judge carries it as
`CHECK 5 — ONWARD CROSS-REFERENCES`: citing the extract you used is required,
sending the supplier to a section they were not given is not.

**The deterministic gates** ("The deterministic gates" in `sps.py`), because an
LLM asked to catch its own class of error is what just failed. In Tier 2 they
run **before** the Judge — they are local and free where the Judge is a paid
call — and a hit feeds back through the same critique path a Judge rejection
uses, so the Actor rewrites and the existing circuit breaker still bounds the
retries. Three failures trip it and no supplier-facing text is produced. In
Tier 1 the same scans fill `Cascade_Warnings`.

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

### The second gate: who is being told to act

A later pair of tickets showed the same shape of failure with a different cause.
Both asked AMAT to issue an ESW; both answers came back at 99% and 96%:

```
1. Issue an ESW.
2. Do not ship the parts until the ESW is fully approved.
```

Step 2 is right — the supplier controls shipment. **Step 1 tells the supplier to
perform the customer's action.** ESW issuance is internal; the supplier may
request one, and in both tickets that is exactly what they did.

The Actor's and the Judge's prompts already forbade this in prose. Neither
fired, because **nothing in the system knew who issues an ESW.** The historical
`Solution_Text` is an internal engineer's own to-do note, carrying no actor, and
the Judge has no domain knowledge to supply one. `INTERNAL_INSTRUMENTS` and
`INTERNAL_ACTION_VERBS` supply it as data.

The rule is a **base-form verb directly governing an internal instrument**, and
the base form is the whole trick — it is what separates an imperative from an
outcome:

| Rejected | Kept |
| --- | --- |
| `Issue an ESW.` | `Do not ship until the ESW is fully approved.` |
| `Please issue an ESW for this unit.` | `An ESW will be issued once review completes.` |
| `Rework the part and issue an ESW.` | `Request an ESW before shipment.` |
| `1. Waive the requirement.` | `we will waive it if the result is ok` |

`request` is deliberately absent from the verb list: the tickets themselves show
suppliers requesting ESWs. Only granting is internal.

That last row cost a redesign. A bare `waive` initially flagged *"we will waive
it if the result is ok"* — real history, and legitimate supplier-facing text
where the customer states an outcome. Bare verbs now fire only in imperative
position, at the start of a step, because position is the only thing separating
the two.

The critique names the required rewrite rather than only the rejection:
*"An ESW has been requested; do not ship until it is approved."* Both gates share
one critique, so a draft with both problems is rewritten once rather than
spending two of its three attempts.

### Responsiveness

The same pair exposed something the gates cannot catch. The two tickets asked
for **different dispositions** — one "we will re-engrave, please issue an ESW",
the other "we have no experience with that rework, please approve shipping
as-is" — and got identical answers.

A Judge check could only turn that confident non-answer into an honest refusal.
The fix had to be in matching, and that is what Tier 1's intent scorer is: it
reads the ticket's request and each record's problem, and scores whether they
ask for the same thing, so wording that matches while the request differs no
longer counts as precedent. In Tier 2, where text is still drafted, the Judge's
`CHECK 6 — RESPONSIVENESS` fails a draft that does not address what the ticket
asks.

### The consequence to expect

Archive text is sent as recorded, so whatever process boilerplate the history
holds reaches the reviewer too — marked in `Cascade_Warnings` when it names an
attachment, a ticket number or a prior conversation. `Matched_Solutions` in the
bulk sheet shows how much of your history is like this.

---

## Tier 2: the 0250 standards fallback

When Tier 1 produces no recommendation — no matching part, nothing similar
enough, or no record asking the same question — and the ticket's reason code is
one listed in `SPS_TIER2_REASON_CODES`, the resolver searches the 0250
engineering standards in `data/0250_docs/` instead. With that list empty, Tier 2
never runs.

```bash
python sps.py resolve --ticket-file t.csv --history-file h.csv \
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

Tables are included, row by row with cells joined by ` | `, so a limits table
reads as `Porosity | 2 percent by area max`. A reader that took only the
paragraphs would lose exactly those numbers while keeping the prose around them.
(The rows of one table currently run together in the chunk text rather than
staying a line each.)

A `.docx` is a zip of XML, and it is read with the standard library — to the
same rules python-docx applied, proven document by document against it before
it was dropped, so every chunk is what it was. Two consequences of those rules
are worth knowing, because text Word shows does not reach Tier 2:

- Text inside a **content control** is skipped — including a whole section
  wrapped in one, which documents built from a template often are — as are
  **tracked insertions not yet accepted**, fields and smart tags.
- A table **nested inside a cell** contributes nothing.

Where python-docx gave up on a document, the reader does not: a malformed merge,
or one running down more than a thousand rows, is read; anything that still
fails costs only that one document, with a warning naming it.

Legacy `.doc` files are named in a warning and skipped. A `.doc` is not a zip
container, so the only way to read one on Windows is to drive Word through COM
automation, which on a headless robot blocks on a modal dialog and hangs the run
rather than failing it.

### The cache

The corpus is the same on every ticket, so it is embedded once, into
`0250_cache_azure.npz` beside the documents. (An encoder injected by the tests
writes `0250_cache_injected.npz` instead: a vector belongs to one embedding
space, and one file could not hold two.)

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
resembling it**. Measured with the since-removed bge-small against the demo
corpus:

| query | top-ranked section | score |
| --- | --- | --- |
| weld seam cracking | `§ 4.2 Weld Seam Cracking` | 0.7462 |
| weld porosity | `§ 4.3 Weld Porosity Limits` | 0.8095 |
| carton label misprint | `§ 2.1 Carton Labelling` | 0.7507 |
| hydraulic pump pressure — *not covered by any document* | `§ 4.2 Weld Seam Cracking` | **0.5944** |

Tier 1's gate on that encoder was 0.89. Reusing it would have rejected the
correct chunk on every ticket and Tier 2 would never have fired once. Tier 2's
gate there was **0.62** — above the uncovered-defect peak, below every genuine
match. The Azure gate, `TIER2_AZURE_THRESHOLD` = **0.35**, is that ratio applied
to Tier 1's former Azure gate, and has not been measured.

`--threshold` is deliberately Tier 1's alone; Tier 2 has `--tier2-threshold` and
`SPS_TIER2_THRESHOLD`. One override spanning both would be a single number
standing for two different comparisons.

**Measure 0.35 once the real standards are loaded.** A larger corpus gives an
irrelevant section more chances to score highly, so the uncovered-defect peak
rises with corpus size; on bge-small it left only 0.026 of headroom.

### Grounding

The Tier-2 Actor works under a constraint precedent never needed: **a standard is
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

`python sps.py bulk` resolves a **sheet of tickets, one per row**, and writes a
copy carrying every original column plus the answers.

```bash
python sps.py bulk --tickets tickets.xlsx --history history.csv
scripts\run_bulk_test.cmd tickets.xlsx history.csv
```

The input is never modified. It stays a clean, re-runnable fixture, a crashed
run cannot destroy the test set, and two runs can be diffed against each other.
Output defaults to `<tickets>_results.xlsx` beside the input.

| | |
| --- | --- |
| **Original columns** | Preserved verbatim, in order, including ones the pipeline never reads |
| **Appended** | `Status`, `Status_Code`, `Reason`, `AI_Recommendation`, `Justification`, `Confidence_Score`, `Referenced_Sources`, `Resolution_Source`, `Tier1_Score`, `Tier2_Score`, `No_Recommendation_Reason`, `Closest_Matching_Solution`, `Cascade_Warnings`, `Matched_Solutions`, `Embedding_Model`, `Duration_Seconds` |
| **Rows** | Exactly one per input row, in the same order — a ticket that resolved nothing still has a row saying why |

Nothing here reimplements the pipeline. Each row goes through the same
`resolve()` the UiPath wrapper calls, so a result in this sheet is the result
production would have produced for that ticket. The "Solution not found." row is
built by the same function that writes `output.xlsx`, so the two cannot disagree
about what a refusal looks like.

A column that clashes with an existing heading is suffixed `_AI` rather than
duplicated — a sheet that already has `Status` would otherwise end up with two,
and which one a reader picked up would be anyone's guess.

### Two bulk tools, different jobs

| | `sps.py bulk` | `sps.py eval` |
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
- **A ticket costs one embedding call in Tier 1, and one intent-scoring call if
  anything clears the cosine gate.** When Tier 2 runs it adds a query embedding
  and up to six more LLM calls (three Actor/Judge rounds).

`--limit N` resolves only the first N rows so you can price a trial before
committing to 500; the rest still appear, marked `NOT RUN`, so the sheet stays
aligned with the input. `--stop-after-errors N` (default 5) abandons the run
after that many **consecutive** infrastructure errors — wrong credentials would
otherwise burn one failing call per row for the whole sheet, and every row would
carry the same useless message.

---

## The batch evaluator

`python sps.py eval` runs a directory of test tickets and writes one
`eval_results.xlsx`. It exists to calibrate a threshold, which shapes both of
its design decisions.

```bash
python sps.py eval --test-dir cases --history-file master.csv \
    --output-dir eval_out
scripts\run_eval.cmd cases eval_out master.csv
```

**Every row carries the raw top cosine, including the cases the gate rejected.**
A run that reported only its successes would show you the scores above the
threshold and hide exactly the ones you need in order to judge whether the
threshold belongs where it is.

**The whole batch runs in one process**, and one failing case never ends the
run: it becomes a row saying what went wrong, and the batch carries on.

| Column | |
| --- | --- |
| `Test_ID`, `Status`, `Status_Code`, `Reason` | as `status.xlsx` |
| `Embedding_Model` | `azure:<deployment>` |
| `Confidence_Score` | the top cosine, **populated even when the gate rejected it** |
| `Threshold_Applied`, `Cleared_Threshold` | what it was measured against, and the verdict |
| `Candidates_Considered` | |
| `Resolution_Source`, `Tier2_Score`, `Tier2_Threshold`, `Tier2_Chunks`, `Tier2_Cache` | the same for Tier 2, kept apart because the tiers score in different ranges |
| `Duration_Seconds` | |
| `Ticket_File`, `History_File` | which inputs produced this row |

It calibrates the cosine gates. The intent score that decides a Tier-1 answer
is not among its columns — only its verdict, in `Status`.

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
so a pooled distribution describes neither. With one encoder nothing can
currently trigger it. It was written for a fallback that fired silently, and it
is kept so the same mistake cannot come back unannounced.

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

Copy `.env.example` to `.env`. Every command loads it if present — from the
working directory, else from beside `sps.py` — and real environment variables
always win. A UiPath robot does not necessarily inherit an interactive shell's
environment, which is why the file is read at all.

The file follows the usual `.env` rules: `KEY=value`, an optional `export`,
`#` comments (one after a bare value needs a space before it), single or double
quotes, and `${VAR}` or `${VAR:-default}` expansion. Save it as UTF-8. A
byte-order mark is fine, and so is the UTF-16 that `echo KEY=value > .env`
writes in Windows PowerShell; a file in any other encoding is skipped with an
error naming it, and the run then reports the missing settings. A line that
cannot be parsed is skipped with a warning giving its line number.

Azure credentials are read **only** from the environment. There is no CLI flag
for them: command lines are visible in the Windows process list and in UiPath
job logs.

Threshold precedence is under [Thresholds](#thresholds).

---

## Assumptions and known boundaries

Stated explicitly rather than buried:

1. **No query instruction.** Queries and history are embedded the same way.
   Some open models are trained to expect an instruction prefixed to queries;
   Azure's embedding models are not, and a constant prefix would only shift
   every query vector.

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
   actually catches a given hallucination, or the intent scorer a different
   request in the same words, is a model-behaviour question needing an
   evaluation set of labelled cases against the real deployment.

---

## Test coverage

```
tests/test_tier2_docs.py               71   docx reading, cache invalidation, Tier-2 gating, the routing matrix
tests/test_transferable.py             62   the two local gates, and what they must NOT flag
tests/test_resolver.py                 53   validation, part filtering, capping, dual workbooks, the writer, threshold
tests/test_component_c_actor_critic.py 40   refinement, breaker, fail-closed, both gates, stop_reason, justification
tests/test_eval_batch.py               32   case discovery, per-case isolation, the score columns
tests/test_file_reader.py              31   format dispatch, strict type gate, format agnosticism
tests/test_config.py                   29   the .env reader, the command line, model/threshold single-sourcing
tests/test_similarity_probe.py         26   pair parsing, the margin maths, both margin verdicts, the file gate
tests/test_intent_matching.py          25   intent scoring, failing closed, the reason-code gate
tests/test_cascade_and_wording.py      24   verbatim cascade, the warning column, the refusal wording
tests/test_bulk_test.py                24   column preservation, row alignment, the not-attempted markers
tests/test_azure_embeddings.py         22   the only encoder: hard failure, config vs transient, threshold
tests/test_structured_outputs.py       12   strict response_format, fallback, schema boundaries
tests/test_golden.py                    6   the whole pipeline's output, pinned
tests/test_dependencies.py              2   nothing beyond the standard library and four packages
```

`test_golden.py` runs the real pipeline over `samples/` with Azure stubbed and
compares every workbook cell and every prompt the model was sent against
`tests/golden/golden.json`. A change that alters what the pipeline produces —
intended or not — fails it; `SPS_GOLDEN_CAPTURE=1` re-records it once the
change is meant.

`python sps.py check` checks the **live Azure deployment**: reachability,
vector dimension, unit length, batch ordering, and then real cosine numbers for
related versus unrelated SPS text scored against both cosine gates. It ends by
suggesting a range for each. That is the fastest way to find out whether 0.30
and 0.35 are anywhere near right on a new deployment — both are guesses until
it has been run.

Four fixtures are scored, and all four are now asserted on. Two of those
assertions were missing for a migration:

| Check | Catches |
| --- | --- |
| `paraphrase clears the gate` | The gate is too **high**: real precedent exists and is silently gated out, indistinguishable downstream from there being none |
| `a different defect on the same part is blocked` | The gate is too **low**: weld porosity answers a weld cracking ticket |

The second is the harder one. `unrelated` is a packaging defect against a weld
defect and any encoder separates those, so a gate that clears it can still be
useless. Porosity against cracking is the real test — same part, same
inspection, same vocabulary, different fix — and it is the shape that produced
two wrong recommendations on live tickets.

### Setting the gate, rather than sanity-checking it

Four fixtures cannot calibrate a threshold. A threshold lives in the gap
between what you judged similar and what you did not, and one sample of each
gives no gap — just two points.

```bat
python sps.py check --probe
python sps.py check --probe my_pairs.xlsx --out probe.xlsx
```

`--probe` scores labelled pairs and reports that gap. Each pair in
[`samples/similarity_pairs.csv`](samples/similarity_pairs.csv) changes exactly
one property — word order, vocabulary, abbreviation, length, negation, the
request — so a bad number identifies *which* property the encoder missed rather
than just that something is wrong.

The suggested gate sits **just above the strongest pair you called a no-match**,
not at the midpoint. The two errors are not symmetric: a missed match is a
refusal a reviewer sees and can act on, while a false match is a wrong
instruction that looks exactly like a right one. The report then says how much
recall that precision costs, by name.

If the gap is **negative** the probe suggests nothing, names every overlapping
pair, and says so — that is the evidence that retrieval itself has to change,
because moving the number can then only trade one error for the other.
