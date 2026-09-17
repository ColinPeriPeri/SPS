# Migration guide — standing this up on a fresh Windows laptop

Copy-paste, top to bottom, in a **Command Prompt** (`cmd.exe`) opened in the
folder you want the project in. About five minutes, most of it `pip install`.

You need working Azure credentials to get past step 7. There is no offline
mode: the local encoder is out of the pipeline, so nothing resolves without a
reachable embedding deployment.

Every step ends with something to check. If a check fails, jump to
[Troubleshooting](#troubleshooting) rather than continuing — each step depends
on the one above it, and a broken install fails much later with a confusing
error.

> **PowerShell instead of cmd?** Everything works except `copy` (use
> `Copy-Item`), `set VAR=value` (use `$env:VAR="value"`), and activation, which
> is `venv\Scripts\Activate.ps1`. If PowerShell refuses to run it, either use
> `cmd.exe` or run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

---

## In a hurry

The whole thing, assuming nothing goes wrong. Each line is a step below, and
every one of them has a check worth reading if it does go wrong.

```bat
git clone https://github.com/ColinPeriPeri/SPS.git
cd SPS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip check
copy .env.example .env
notepad .env
python -m pytest -q
python -m scripts.verify_embedder
scripts
un_resolver.cmd samples\sample_ticket.csv samples\sample_history.csv smoke
```

The two that most often surprise people: `pip check` must also survive
`python -c "from openai import AzureOpenAI"` on a managed laptop
([step 4](#4-verify-the-install)), and `verify_embedder` is the only thing that
will tell you whether the shipped thresholds suit your deployment
([step 7](#7-check-the-azure-deployment)).

---

## 0. Prerequisites

| Need | Check | Notes |
|---|---|---|
| Python 3.11 or 3.12 | `python --version` | Built and tested on 3.12.0. 3.13 is untested. |
| Git | `git --version` | Or download the repo as a zip. |
| Network to `pypi.org` | | That is the whole list. No model download, no PyTorch index. |
| Your Azure OpenAI endpoint and keys | | **Required.** Two deployments: one chat, one embedding. The local encoder is disabled, so nothing resolves without these. See [step 5](#5-configure-env). |
| Your 0250 standards as `.docx` | | Optional. Tier 2 is skipped entirely if the folder is empty — see [step 10](#10-load-the-0250-standards-tier-2). |

No admin rights, no database, no service to install. The whole thing is a
Python process the UiPath Performer invokes per ticket.

---

## 1. Get the code

```bat
git clone https://github.com/ColinPeriPeri/SPS.git
cd SPS
```

**Check:** `dir` lists `requirements.txt`, `sps`, `scripts`, `samples`.

---

## 2. Create and activate the virtual environment

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

**Check:** your prompt now starts with `(venv)`. Everything below assumes it is
still active — if you open a new terminal, run `venv\Scripts\activate` again.

The `.cmd` wrappers look for `venv\` first, then `env\`, so either name works.
To point them at an interpreter somewhere else entirely, set `SPS_PYTHON` to its
full path.

---

## 3. Install the dependencies

```bat
pip install -r requirements.txt
```

33 packages, no PyTorch, no model download. A minute or so.

---

## 4. Verify the install

```bat
pip check
python -c "from openai import AzureOpenAI; from lxml import etree; import docx; print('OK')"
```

Expect `No broken requirements found.` and `OK`.

**This check earns its place on a corporate laptop.** Two pins exist only to
survive Windows Application Control, which blocks the native DLLs shipped by the
newest builds of two dependencies:

| Pin | Without it |
|---|---|
| `jiter<0.17` | `import openai` dies with *"DLL load failed while importing jiter: An Application Control policy has blocked this file"* — and takes **every Azure call** with it, embeddings and chat alike |
| `lxml<6` | The same error on `_elementpath`, taking Tier-2 document parsing with it |

Both read like a corrupt install rather than a policy decision, which is what
makes them expensive to diagnose. If you hit either, you have a version above
the pin: `pip install "jiter<0.17"` or `pip install "lxml<6"`.

---

## 5. Configure `.env`

```bat
copy .env.example .env
notepad .env
```

`.env` is gitignored and must never be committed. The resolver reads it from the
working directory or the project root, and **real environment variables always
win** — so a UiPath robot with machine-level variables set does not need the
file at all.

There are **two separate Azure deployments**, with their own endpoint and key,
so one can be rotated or fail without touching the other. Both are required —
the local encoder is disabled, so a missing embedding key means no ticket
resolves at all. Fill in four values:

### Embeddings — required

| Variable | Value |
|---|---|
| `AZURE_EMBEDDING_ENDPOINT` | `https://<your-resource>.openai.azure.com/` |
| `AZURE_EMBEDDING_API_KEY` | the key for that resource |
| `AZURE_EMBEDDING_DEPLOYMENT` | **`text-embedding-3-large`** — the deployment *name*, not the model name, if they differ |

If any one of the three is missing, the run ends with `INFRASTRUCTURE_ERROR`
and **exit 2** — alert a human, do not retry, because no number of retries
produces an API key. A transient failure (network, timeout, throttling) is
`INFRASTRUCTURE_ERROR` with **exit 1** instead, which is worth retrying.

### Chat (the Actor/Judge loop) — required

| Variable | Value |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | `https://<your-resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | the key for that resource |
| `AZURE_OPENAI_DEPLOYMENT` | your chat deployment, e.g. `gpt-4o` |

Leave `AZURE_OPENAI_API_VERSION=2024-10-21` alone unless you have a reason:
it is the first GA version with `json_schema` structured outputs. Older versions
silently fall back to JSON mode plus client-side validation, which works but is
weaker.

### Thresholds — both live ones are guesses

`AZURE_EMBEDDING_THRESHOLD=0.50` and `TIER2_AZURE_THRESHOLD=0.35` are the two
that now apply, and **neither has ever been measured**. The measured pair
(0.89 / 0.62) belonged to the local encoder and is dormant.

[Step 7](#7-check-the-azure-deployment) gives a first reading in about ten
seconds; [step 9](#9-run-the-batch-evaluator) turns it into a distribution.

---

## 6. Run the test suite

```bat
python -m pytest -q
```

Expect **219 passed, 1 skipped** in about five seconds. Nothing here needs a
server, an Azure key or a model download.

The skip is `tests/test_real_embedder.py`, which exercises the disabled local
encoder. It needs torch and sentence-transformers reinstalled and is not part of
the current pipeline — leave it skipped.

---

## 7. Check the Azure deployment

Before any ticket, confirm the encoder works and see what its numbers look like:

```bat
python -m scripts.verify_embedder
```

Its first two lines tell you whether it found your `.env` at all:

```
Checking the Azure embedding deployment.
  .env: D:\PRB\SPS\.env
```

`not found -- using the environment only` there means the file is missing or
sits somewhere other than the working directory or the project root. Everything
below it will then report missing credentials no matter what you typed.

It goes on to check reachability, vector dimension, unit length and batch
ordering, then scores real SPS text against both live gates and ends with a
suggested range:

```
  Suggested gates from this run: Tier 1 between 0.1xxx and 0.8xxx,
  Tier 2 between 0.2xxx and 0.7xxx.
```

Two checks matter more than the rest. **"unrelated record is blocked by the 0.50
gate"** — if that FAILS the gate admits everything and is not a gate, which
usually means the deployment is `ada-002` rather than `text-embedding-3-large`.
And the reported dimension: **3072** confirms `-3-large`; 1536 means `-3-small`
or `ada-002`.

One sample each, so treat it as a smoke test with a hint attached, not a
calibration. [Step 9](#9-run-the-batch-evaluator) is the calibration.

---

## 8. Prove the pipeline runs, before any real data

### 8a. Through the wrapper UiPath actually calls

```bat
scripts\run_resolver.cmd samples\sample_ticket.csv samples\sample_history.csv smoke
echo %ERRORLEVEL%
```

Expect `%ERRORLEVEL%` of **0**, `Status` of `PASS` and `Status_Code` of
`SUCCESS_HISTORICAL` in `status.xlsx`, and a second workbook
`smoke\output.xlsx` with `Resolution_Source` of `HISTORICAL_DATA`.
`Embedding_Model` reads `azure:text-embedding-3-large` and `Reason` ends
`[Azure]`.

If it fails, the exit code says which kind of problem it is:

| Exit | Meaning |
|---|---|
| **2** with `INFRASTRUCTURE_ERROR` | A credential is missing. The `Reason` names exactly which variable. |
| **1** with `INFRASTRUCTURE_ERROR` | The deployment was reachable but the call failed — wrong deployment name, throttling, network. |

There is no longer an offline mode: with the local encoder disabled, nothing
resolves without a working embedding deployment.

---

## 9. Run the batch evaluator

This is the tool for calibrating `AZURE_EMBEDDING_THRESHOLD` and
`TIER2_AZURE_THRESHOLD` — the two numbers that gate everything, neither of which
has ever been measured against a real deployment.

It runs every case in **one process**, and records the raw similarity on
**every** row — including the cases the gate rejected, which are precisely the
ones that tell you whether the threshold sits in the right place. A run that
reported only its successes would hide exactly the scores you need.

```bat
scripts\run_eval.cmd samples\eval_cases eval_out samples\sample_history.csv
```

Six cases. It writes `eval_out\eval_results.xlsx` and prints a digest ending in
a score distribution:

```
  score distribution over 4 scored case(s):
    min 0.xxxx   p25 0.xxxx   median 0.xxxx
    p75 0.xxxx   p90 0.xxxx   max 0.xxxx
```

**The shape matters, not the values.** What you want is genuine matches bunched
high, the deliberately-unrelated case (`case04`, a hydraulic pump problem filed
against a bracket part) well below them, and a wide gap in between for the
threshold to sit in. On the retired local encoder that gap ran from 0.5658 to
0.9633; your deployment will produce different numbers, and finding out what
they are is the entire point of this step.

Set `SPS_CONFIDENCE_THRESHOLD=0.99` first to gate every case before the LLM.
That still costs one embedding call per case — embeddings are what you are
measuring — but no generation, so it is the cheap way to collect the
distribution.

### Your own test set

Drop files into a directory, named in pairs:

```
test01_ticket.xlsx      test01_history.xlsx    (optional, per-case history)
test02_ticket.xlsx
test03_ticket.csv
```

A case with no history of its own uses the one passed as the third argument —
the usual shape, where fifty tickets share a single extract. `.csv` and `.xlsx`
mix freely. Prefer `.csv` for a large history: the same 300k rows take ~1.5 s as
CSV and ~40 s as `.xlsx`, because openpyxl parses XML per row.

For anything more structured, use a run list instead:

```bat
python -m scripts.run_eval_batch --run-list cases\runs.csv --output-dir eval_out
```

with columns `Test_ID`, `Ticket_File`, `History_File` — paths resolved relative
to the list itself, so the set stays portable between machines.

### Reading the results

`eval_results.xlsx` gives you `Confidence_Score`, `Threshold_Applied` and
`Cleared_Threshold` per case. Add your own judgement column — *should* this case
have matched? — sort by score, and the threshold goes in the gap: above every
case you judged wrong, below every case you judged right. If there is no gap,
the threshold is not the problem and no value will save it.

`Tier2_Score` does the same job for Tier 2, and is populated only on the rows
where Tier 2 actually ran.

Two things to watch for:

- **Check `Embedding_Model` is `azure:...` on every row.** It should be — there
  is only one encoder now — so anything else means a row took a path you did not
  intend, and a distribution pooled across two embedding spaces describes
  neither. The digest prints a `WARNING` if it happens.
- Set the number in `.env`, not in code:
  `AZURE_EMBEDDING_THRESHOLD` for Tier 1, `TIER2_AZURE_THRESHOLD` for Tier 2.
  `SPS_CONFIDENCE_THRESHOLD` and `SPS_TIER2_THRESHOLD` override each tier
  regardless of encoder. The `LOCAL_*` pair is dormant and changing it does
  nothing while the local encoder is disabled.

---

## 10. Load the 0250 standards (Tier 2)

Tier 2 answers from the engineering standards when the historical records
produce nothing usable. **It is optional and it ships switched off**, in the
sense that `data\0250_docs\` is empty and an empty folder means Tier 2 never
runs. Every ticket then behaves exactly as it did before Tier 2 existed.

### Try it on the demo corpus first

Three invented standards ship in `samples\0250_docs\`, so you can see the whole
path work before touching a real document:

```bat
python -m scripts.run_resolver --ticket-file samples\sample_ticket.csv --history-file samples\sample_history.csv --output-dir smoke --docs-dir samples\0250_docs --threshold 0.99 --tier2-threshold 0.99
```

Forcing *both* gates to 0.99 stops before the LLM, so this costs two embedding
calls and no generation. Expect exit **0**, `BELOW_CONFIDENCE_THRESHOLD`, and a
Reason naming a score from each tier:

```
Best historical match 0.xxxx is below the 0.99 threshold across 3 candidate(s).
Best 0250 match 0.xxxx is below the 0.99 threshold across 8 chunk(s). [Azure]
```

`8 chunk(s)` proves the documents parsed, embedded and cached. The scores
themselves depend on your deployment, which is the point of running it. Drop
`--tier2-threshold 0.99` and Tier 2 will retrieve its top sections and generate
for real.

### Then load the real ones

```bat
copy "\\your-share\standards\0250-*.docx" data\0250_docs\
```

That is the whole procedure — no index to build, no command to run. The first
ticket afterwards parses and embeds; every ticket after that loads the cache in
milliseconds. Editing, adding, removing or renaming a document rebuilds it
automatically, because the cache is keyed on a SHA-256 of the folder's contents.

Three things to check before you trust it:

- **`.docx` only.** A legacy `.doc` is skipped with a warning naming the file.
  Open it in Word and *Save As* `.docx`. (`.doc` is not a zip container; reading
  one needs Word via COM automation, which hangs a headless robot on a modal
  dialog instead of failing it.)
- **Heading styles must be real headings.** Chunks are cut at `Heading 1` /
  `Heading 2`, and the heading becomes the citation the supplier sees. A
  document whose section titles are bold body text parses as one block and cites
  only the filename.
- **Measure the threshold.** `TIER2_AZURE_THRESHOLD=0.35` is the one that
  gates your runs, and it is a guess derived from another guess — never measured
  against any deployment. (`TIER2_LOCAL_THRESHOLD=0.62` *was* measured, but on
  the retired encoder.) A larger corpus also gives irrelevant sections more
  chances to score highly, so the headroom shrinks as you load more documents.
  Run the batch evaluator and read `Tier2_Score` before trusting it.

---

## Exit codes

Both wrappers propagate these verbatim, so the UiPath state machine can branch
without opening a workbook.

| Code | Meaning | What the robot should do |
|---|---|---|
| `0` | The run completed. `PASS`, or a legitimate `FAIL` — gated ticket, unknown part, wrong file type. | Log a business exception, routed on `Status_Code`. **Do not retry.** |
| `1` | Infrastructure fault: Azure unreachable, unhandled error. | Retry. |
| `2` | The ticket or history file could not be read. | Alert a human. |

Within exit 0, branch on `Status_Code`: `SUCCESS_HISTORICAL` and
`SUCCESS_0250_DOC` go to admin review, `NO_MATCHES` to Master Data,
`BELOW_CONFIDENCE_THRESHOLD` to a Reliability Engineer, `LLM_AUDIT_REJECTED` to
a human reviewer, and `INVALID_INPUT` faults the item.

`status.xlsx` is written **always**, including on an early abort or an unhandled
exception. `output.xlsx` appears whenever a verdict was reached, including
"no solution" — so there is one row per ticket to merge, and a missing row means
the run did not finish rather than that it found nothing. Both are deleted before work
starts, so a process killed outright leaves neither: "`status.xlsx` missing" is
unambiguous.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `An Application Control policy has blocked this file` on `import openai` (via `jiter`) | jiter 0.17+ ships a native DLL your machine's policy blocks. **This kills every Azure call, embeddings and chat.** | `pip install "jiter<0.17"` |
| `An Application Control policy has blocked this file` on `from lxml import etree` | Same, for lxml 6.x. Kills Tier-2 document parsing. | `pip install "lxml<6"` |
| `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` | httpx 0.28+ removed `proxies`, which openai 1.40.0 still passes | `pip install "httpx>=0.27,<0.28"` |
| `Python interpreter not found` from a `.cmd` | No `venv\` or `env\` in the project root | Redo step 2, or set `SPS_PYTHON` to the interpreter's full path |
| `pip` SSL errors on install | TLS interception | Point `pip` at your internal mirror via `PIP_INDEX_URL` — ask IT rather than disabling verification |
| `INVALID_INPUT`, exit 0, `Reason` names the file | A `.pdf` / `.docx` / `.xls` was attached | Only `.csv`, `.xlsx` and `.xlsm` are accepted. This is deliberate: a wrong attachment is a business problem, faulted rather than retried |
| `INVALID_INPUT`, exit 2 | The file is the right type but missing or unreadable | Genuine I/O trouble — check the path the robot passed |
| `Problem description is N characters; at least 10 required.` | The description column is blank or nearly so | Check the ticket's column name — `Problem_Description`, matched case-insensitively |
| `NO_MATCHES` on a part you know exists | Part numbers differ after normalisation (`.strip().upper()`, invisible characters removed) | `Reason` reports how many rows matched out of how many scanned — usually a stray character in one source |
| `INFRASTRUCTURE_ERROR`, exit **2**, on every ticket | One of the three `AZURE_EMBEDDING_*` values is missing | The `Reason` names exactly which. There is no local fallback to absorb it any more |
| Credentials are set but everything still reports them missing | `.env` is not where the tools look: the working directory, or the project root | `verify_embedder`'s second line prints the file it read, or `not found`. `copy .env.example .env` in the project root |
| `INFRASTRUCTURE_ERROR`, exit **1**, on every ticket | The deployment name is wrong, or the endpoint is unreachable | Run `python -m scripts.verify_embedder` — it isolates reachability from configuration |
| Everything is gated, nothing resolves | 0.50 is a guess and may be far from right for your deployment | `python -m scripts.verify_embedder` suggests a range; set `SPS_CONFIDENCE_THRESHOLD` |
| Nothing is gated, everything resolves | Same, in the other direction — likely an `ada-002` deployment, where even unrelated text scores above 0.7 | Same check. If the "unrelated record is blocked" line FAILS, raise the threshold |
| Tier 2 never runs | `data\0250_docs\` holds no `.docx` | Expected until the standards are loaded. Check with `dir data\0250_docs\*.docx` |
| A 0250 document is ignored | It is a legacy `.doc`, or a `~$` lock file | The warning names it. Re-save as `.docx`; close the document if Word has it open |
| Every 0250 citation is just the filename | The document's section titles are bold body text, not `Heading 2` | Apply Word's heading styles, then delete `data\0250_docs\0250_cache_*.npz` |
| Tier 2 finds nothing on a defect you know is covered | 0.35 is a guess derived from another guess | Run the batch evaluator and read `Tier2_Score`, then set `SPS_TIER2_THRESHOLD` |
| `pytest` reports 1 skipped | `tests/test_real_embedder.py`, which exercises the disabled local encoder | Expected. Leave it skipped |

---

## What to read next

- [`README.md`](README.md) — architecture, the constrained-RAG contract, measured
  latency, and why there are two thresholds.
- [`.env.example`](.env.example) — every setting, with the reasoning inline.
