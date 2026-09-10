# Sample data

Small, synthetic, and safe to commit. Two jobs.

## Smoke test

`sample_ticket.csv` + `sample_history.csv` — one ticket, five history rows
across two part numbers. Used by the install check in
[`../MIGRATION_GUIDE.md`](../MIGRATION_GUIDE.md):

```bat
python -m scripts.run_resolver --ticket-file samples\sample_ticket.csv --history-file samples\sample_history.csv --output-dir smoke --threshold 0.99
```

The forced 0.99 threshold gates the run before the LLM, so it proves the reader,
the part filter, the torch/numpy stack, the embedding model, cosine ranking and
the Excel writer without needing an Azure key. The real top score is `0.9641`.

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

## Adding your own

Name files `<id>_ticket.csv|xlsx`, optionally paired with
`<id>_history.csv|xlsx`. A case with no history of its own uses the one passed on
the command line. `.csv` and `.xlsx` mix freely within a directory.

Include cases you expect to **fail**, and record which ones those are — that
judgement is the only thing that turns a score distribution into a threshold.
