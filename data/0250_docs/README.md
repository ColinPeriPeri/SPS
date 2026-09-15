# 0250 engineering standards — put the real documents here

This folder is the Tier-2 corpus. The resolver reads it when the historical SPS
records produce no usable resolution, and answers from these standards instead.

**It ships empty.** An empty folder is a valid state: Tier 2 simply does not run
and every ticket reports the Tier-1 outcome it would have reported before Tier 2
existed. Nothing fails.

## Loading it

Copy the `.docx` standards in. That is the whole procedure — there is no index
to build and no command to run. The first ticket after a change parses the
documents, embeds them and writes a cache; every ticket after that loads the
cache in milliseconds.

```
data/0250_docs/
├── 0250-Weld-Standards.docx
├── 0250-Packaging-Standards.docx
├── 0250_cache_local.npz      ← written automatically, safe to delete
└── 0250_cache_azure.npz      ← written automatically, safe to delete
```

## Rules

- **`.docx` only.** A legacy `.doc` is named in a warning and skipped. It is not
  a zip container, so the only way to read one on Windows is to drive Word
  through COM automation — which on a headless robot blocks on a modal dialog
  and hangs the run rather than failing it. Open it in Word and *Save As* .docx.
- **Use Word's heading styles.** Chunks are cut at headings, and the heading
  becomes the citation the supplier is shown. A document whose section titles
  are bold body text rather than `Heading 2` parses as one undifferentiated
  block and cites only the filename.
- **Tables are read.** Limits tables are included in the text, flattened one row
  per line, so an acceptance criterion in a table is retrievable.
- **The cache invalidates itself.** It is keyed on a SHA-256 of every document's
  name and bytes, so editing, adding, removing or renaming one rebuilds it on
  the next run. You never need to clear it by hand; deleting the `.npz` files is
  harmless if you want to force a rebuild anyway.

## Before you trust the scores

`TIER2_AZURE_THRESHOLD` is **0.35, and has never been measured** — it was
derived from the Tier-1 Azure figure, itself a guess, scaled by a ratio measured
on the now-disabled local encoder. It is the number that actually gates your
runs.

`TIER2_LOCAL_THRESHOLD` (0.62) *was* measured, against the three-document demo
corpus in `samples/0250_docs/`, but it belongs to the local encoder and is
dormant.

**Measure before trusting either.** A larger corpus has more chances for an
irrelevant section to score highly, so the top score for an uncovered defect
rises with corpus size — on the demo set it peaked at 0.5944 against a 0.62
gate, leaving only 0.026 of headroom. `python -m scripts.verify_embedder` gives
a first reading against your own deployment in seconds.

`scripts/run_eval_batch.py` reports `Tier2_Score` on every row for exactly this.
