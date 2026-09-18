# Sample data

Small, synthetic, and safe to commit. Two jobs.

## Smoke test

`sample_ticket.csv` + `sample_history.csv` — one ticket, five history rows
across two part numbers. Used by the install check in
[`../MIGRATION_GUIDE.md`](../MIGRATION_GUIDE.md):

```bat
python -m scripts.run_resolver --ticket-file samples\sample_ticket.csv --history-file samples\sample_history.csv --output-dir smoke --threshold 0.99
```

The forced 0.99 threshold gates the run before the LLM, so it costs one
embedding call and no generation while still proving the reader, the part
filter, the Azure encoder, cosine ranking and the Excel writer.

It does need a working `AZURE_EMBEDDING_*` configuration: the local encoder is
out of the pipeline, so there is no offline path. The top score depends on your
deployment.

## Bulk test sheet

`bulk_tickets.csv` is six tickets in one sheet, for `scripts/run_bulk_test.py`.
It carries a `Supplier` column the pipeline never reads, which is the point:
whatever columns you put in come back untouched alongside the answers.

```bat
scripts\run_bulk_test.cmd samples\bulk_tickets.csv samples\sample_history.csv
```

Against `sample_history.csv` the six rows cover a clean match, a paraphrase, a
different part, a defect nothing in history covers, an unknown part number and a
blank one -- so one run exercises every `Status_Code` the sheet can report.

## Boilerplate history

`boilerplate_history.csv` + `boilerplate_tickets.csv` reproduce the shape real
history takes, as opposed to the idealised fixtures everywhere else here. Two of
its three records defer to an attachment, a prior conversation and an internal
work request rather than describing a fix:

```
1. See the feedback in the attachment. 2. Per discussed, rework as attachment
shown. 3. After rework, provide photos and related data. 4. ESW#20033465 is
submitted for these issues.
```

```bat
scripts\run_bulk_test.cmd samples\boilerplate_tickets.csv samples\boilerplate_history.csv
```

The weld ticket should come back **without** a recommendation: the
transferability gate rejects the draft, the circuit breaker trips, and the
`Reason` names what it found. The packaging ticket, whose record contains an
actual disposition, resolves normally. `Matched_Solutions` shows the difference
between the two at a glance.

These fixtures exist because the rest of the samples could never have surfaced
that bug -- every other `Solution_Text` here is clean, actionable prose.

## Eval cases

`eval_cases/` is a worked example for `scripts/run_eval_batch.py`. Six cases
against `sample_history.csv`, chosen to put one of each outcome in the results
workbook:

| Case | Shape | Expected |
| --- | --- | --- |
| `case01` | Verbatim match of a history row | High score, `SUCCESS` |
| `case02` | Same defect, different wording | High score — the one that shows paraphrase survives encoding |
| `case03` | Different part, **ships its own history** (`case03_history.csv`) | High score against 2 candidates, not 3 |
| `case04` | Hydraulic pump problem filed against a bracket part | ~0.57 — a materially different defect, and the case the threshold has to reject |
| `case05` | Part number absent from the history | `NO_MATCHES`, nothing encoded |
| `case06` | Blank part number | `INVALID_INPUT`, rejected before the model loads |

```bat
scripts\run_eval.cmd samples\eval_cases eval_out samples\sample_history.csv
```

`case04` is the load-bearing one. A threshold is only meaningful in the gap
between the scores of cases you judged wrong and the scores of cases you judged
right, so a set with no `case04` in it cannot calibrate anything.

## Demo 0250 standards

`0250_docs/` holds three invented engineering standards, so Tier 2 can be
exercised before a real document is loaded. Regenerate them with
`python -m scripts.make_sample_docs`.

| Document | Contains | Why it is here |
| --- | --- | --- |
| `0250-Weld-Standards.docx` | 4.1 Scope, 4.2 Weld Seam Cracking, 4.3 Porosity Limits (a **table**), 4.4 Undercut, 7.1 Re-inspection | The section a weld ticket should land on, plus a limits table that `document.paragraphs` would drop |
| `0250-Packaging-Standards.docx` | 2.1 Carton Labelling, 2.4 Barcode Symbology | A different issue class, so the enriched query has something to discriminate against |
| `0250-Surface-Finish.docx` | 3.2 Surface Finish | **States a limit and no disposition.** It retrieves at 0.75 and the Actor is then required to decline it — the fabrication Tier 2's grounding check exists to catch |

```bat
python -m scripts.run_resolver --ticket-file samples\sample_ticket.csv --history-file samples\sample_history.csv --output-dir smoke --docs-dir samples\0250_docs --threshold 0.99 --tier2-threshold 0.99
```

Both gates forced to 0.99 stop before the LLM; the Reason then names a score
from each tier. The numbers below were measured with **bge-small**, the encoder
that is currently disabled, and are what `TIER2_LOCAL_THRESHOLD = 0.62` was
derived from. Your Azure deployment will produce a different distribution --
`python -m scripts.verify_embedder` reads it:

| query | top section | score |
| --- | --- | --- |
| weld seam cracking | `§ 4.2 Weld Seam Cracking` | 0.7462 |
| weld porosity | `§ 4.3 Weld Porosity Limits` | 0.8095 |
| carton label misprint | `§ 2.1 Carton Labelling` | 0.7507 |
| hydraulic pump pressure (`case04`) | — nothing covers it — | 0.5944 |

That last row is why `TIER2_LOCAL_THRESHOLD` is 0.62.

## Adding your own

Name files `<id>_ticket.csv|xlsx`, optionally paired with
`<id>_history.csv|xlsx`. A case with no history of its own uses the one passed on
the command line. `.csv` and `.xlsx` mix freely within a directory.

Include cases you expect to **fail**, and record which ones those are — that
judgement is the only thing that turns a score distribution into a threshold.
