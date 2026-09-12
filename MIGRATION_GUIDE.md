# Migration guide — standing this up on a fresh Windows laptop

Copy-paste, top to bottom, in a **Command Prompt** (`cmd.exe`) opened in the
folder you want the project in. Roughly 15 minutes, nearly all of it downloads.

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

## 0. Prerequisites

| Need | Check | Notes |
|---|---|---|
| Python 3.11 or 3.12 | `python --version` | Built and tested on 3.12.0. 3.13 is untested and torch 2.3.1 has no 3.13 wheel. |
| Git | `git --version` | Or download the repo as a zip. |
| Network to `pypi.org`, `download.pytorch.org`, `huggingface.co` | | The first resolver run downloads ~130 MB of model weights from Hugging Face. |
| Your Azure OpenAI endpoint and keys | | Two deployments: one chat, one embedding. See [step 6](#6-configure-env). |
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

## 3. Install PyTorch first, from PyTorch's own index

```bat
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

This has to come **before** `requirements.txt`. The CPU wheel is only published
on PyTorch's index; plain PyPI `torch==2.3.1` on Windows resolves to the
CUDA-bundled build, which is a far larger download and about 2.4 GB unpacked for
no benefit on a CPU-only host.

**Check:**

```bat
python -c "import torch; print(torch.__version__)"
```

Expect `2.3.1+cpu`. A bare `2.3.1` means the CUDA build landed — see
[Troubleshooting](#troubleshooting).

---

## 4. Install everything else

```bat
pip install -r requirements.txt
```

torch is already satisfied, so it is not refetched.

---

## 5. Verify the pinned set landed together

```bat
pip check
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
python -c "from lxml import etree; import docx; print('docx OK', etree.__version__)"
```

Expect `No broken requirements found.`, `2.3.1+cpu 1.26.4`, and `docx OK 5.3.0`.

That third line is not decoration. `lxml` is pinned below 6 because its 6.x
Windows wheel ships a native DLL that **Windows Application Control blocks on a
managed machine** — the import dies with *"An Application Control policy has
blocked this file"*, which reads like a corrupt install rather than a policy
decision. If you see that error, you have an lxml 6.x: `pip install "lxml<6"`.

This check earns its place. torch, numpy, scipy, scikit-learn and httpx are
pinned as **one coupled set**: torch 2.3.1 is built against the NumPy 1.x C ABI,
so with NumPy 2.x installed the import still succeeds but every `tensor.numpy()`
call raises `RuntimeError: Numpy is not available` — at ranking time, on a real
ticket, not here.

---

## 6. Configure `.env`

```bat
copy .env.example .env
notepad .env
```

`.env` is gitignored and must never be committed. The resolver reads it from the
working directory or the project root, and **real environment variables always
win** — so a UiPath robot with machine-level variables set does not need the
file at all.

There are **two separate Azure deployments**, with their own endpoint and key,
so one can be rotated or fail without touching the other. Fill in four values:

### Embeddings — required

| Variable | Value |
|---|---|
| `AZURE_EMBEDDING_ENDPOINT` | `https://<your-resource>.openai.azure.com/` |
| `AZURE_EMBEDDING_API_KEY` | the key for that resource |
| `AZURE_EMBEDDING_DEPLOYMENT` | **`text-embedding-3-large`** — the deployment *name*, not the model name, if they differ |

If any one of the three is missing, the resolver logs
`AZURE_EMBEDDING_FAILED_FALLING_BACK` and uses the local model for the **whole**
run. It never mixes vectors from two models in one run, so a partial
configuration degrades cleanly rather than producing meaningless scores.

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

### Thresholds — leave as shipped, for now

`LOCAL_EMBEDDING_THRESHOLD=0.89` is measured against `bge-small`.
`AZURE_EMBEDDING_THRESHOLD=0.50` is **provisional and has never been measured**.
[Step 9](#9-run-the-batch-evaluator) is how you replace that guess with a number.

---

## 7. Run the test suite

```bat
python -m pytest -q
```

Expect **216 passed, 1 skipped** in about two minutes. Nothing here needs a
server, an Azure key or a model download — the skip is the real-model module,
which is opt-in:

```bat
set SPS_MODEL_TESTS=1
python -m pytest -q
set SPS_MODEL_TESTS=
```

That is **230 passed**, and it downloads ~130 MB of weights the first time. It
pins the properties the ranking maths assumes: 384 dimensions, unit-length
vectors (so the NumPy matmul *is* the cosine the 0.89 gate is calibrated on),
and the query instruction applied to queries but never to history passages.

---

## 8. Prove the pipeline runs, before any real data

### 8a. Everything except Azure

`samples\` ships a ticket and a five-row history. Forcing the threshold to 0.99
gates the run *before* the LLM is called, so this works with no keys at all:

```bat
python -m scripts.run_resolver --ticket-file samples\sample_ticket.csv --history-file samples\sample_history.csv --output-dir smoke --threshold 0.99
echo %ERRORLEVEL%
```

The first run downloads the local model, so allow a minute. Expect
`%ERRORLEVEL%` of **0** and, in `smoke\status.xlsx`:

| Column | Value |
|---|---|
| `Status` | `FAIL` |
| `Status_Code` | `BELOW_CONFIDENCE_THRESHOLD` |
| `Reason` | `Best historical match 0.9641 is below the 0.99 threshold across 3 candidate(s). No 0250 documents found in data\0250_docs. [Local]` |
| `Embedding_Model` | `local:BAAI/bge-small-en-v1.5` |

A `FAIL` is the *expected* result here — you asked for an impossible threshold.
The Reason names both tiers because both were tried: Tier 1 gated, and Tier 2
had no documents to search yet. The code is Tier 1's own, because Tier 2
retrieved nothing to improve on it.
What it proves is everything underneath: the file reader, part-number matching,
the torch/numpy stack, the embedding model, cosine ranking and the atomic Excel
write all work on this machine. `0.9641` is the real similarity between the
sample ticket and its closest historical match.

### 8b. With Azure, through the wrapper UiPath actually calls

```bat
scripts\run_resolver.cmd samples\sample_ticket.csv samples\sample_history.csv smoke
echo %ERRORLEVEL%
```

Expect `%ERRORLEVEL%` of **0**, `Status` of `PASS` and `Status_Code` of
`SUCCESS_HISTORICAL` in `status.xlsx`, and a second workbook
`smoke\output.xlsx` with `Resolution_Source` of `HISTORICAL_DATA`. With Azure
embeddings configured, `Embedding_Model` reads `azure:text-embedding-3-large`
and `Reason` ends `[Azure]`.

`INFRASTRUCTURE_ERROR` and exit 1 here means the credentials did not work — the
`Reason` column names which variable or which call failed.

---

## 9. Run the batch evaluator

This is the tool for calibrating `AZURE_EMBEDDING_THRESHOLD`. It runs every
case in **one process**, so the model is loaded once for the whole batch rather
than once per case, and it records the raw similarity on **every** row —
including the cases the gate rejected, which are precisely the ones that tell
you whether the threshold sits in the right place.

```bat
scripts\run_eval.cmd samples\eval_cases eval_out samples\sample_history.csv
```

Six cases, about 15 seconds. It writes `eval_out\eval_results.xlsx` and prints a
digest ending in something like:

```
  score distribution over 4 scored case(s):
    min 0.5658   p25 0.5658   median 0.9633
    p75 0.9641   p90 0.9641   max 0.9656
```

That spread is the shape you are looking for: genuine matches bunched at
0.96, the deliberately-unrelated case (`case04`, a hydraulic pump problem filed
against a bracket part) down at 0.5658, and a wide gap between them for the
threshold to sit in.

To run it without Azure, set `SPS_CONFIDENCE_THRESHOLD=0.99` first — every case
then gates before the LLM and the batch exits 0.

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

Two things to watch for:

- **Check the `Embedding_Model` column is all `azure:...` before trusting the
  numbers.** A threshold belongs to one embedding space and does not survive a
  change of encoder. If some rows fell back to `local:...` the digest prints a
  `WARNING` and the distribution is pooled across two models, describing
  neither.
- Set the number in `.env`, not in code. `AZURE_EMBEDDING_THRESHOLD` and
  `LOCAL_EMBEDDING_THRESHOLD` override one encoder each;
  `SPS_CONFIDENCE_THRESHOLD` overrides both.

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

Forcing *both* gates to 0.99 keeps this offline. Expect exit **0**,
`BELOW_CONFIDENCE_THRESHOLD`, and a Reason naming a score from each tier:

```
Best historical match 0.9641 is below the 0.99 threshold across 3 candidate(s).
Best 0250 match 0.7462 is below the 0.99 threshold across 8 chunk(s). [Local]
```

`8 chunk(s)` proves the documents parsed, embedded and cached. Drop
`--tier2-threshold 0.99` and Tier 2 will retrieve 5 sections and call Azure for
real.

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
- **Re-measure the threshold.** `TIER2_LOCAL_THRESHOLD=0.62` was measured on the
  three-document demo corpus, where a defect nothing covers peaks at 0.5944 —
  only 0.026 of headroom. A larger corpus gives irrelevant sections more chances
  to score highly, so run the batch evaluator and read the `Tier2_Score` column
  before trusting it.

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
exception. `output.xlsx` appears only on `PASS`. Both are deleted before work
starts, so a process killed outright leaves neither: "`status.xlsx` missing" is
unambiguous.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Numpy is not available` | NumPy 2.x got pulled in past the pin | `pip install "numpy>=1.26,<2"` then `pip check` |
| `python -c "import torch..."` prints `2.3.1`, not `2.3.1+cpu` | The CUDA build installed — step 3 ran after step 4, or without `--index-url` | `pip uninstall -y torch` then redo step 3 |
| `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` | httpx 0.28+ removed `proxies`, which openai 1.40.0 still passes | `pip install "httpx>=0.27,<0.28"` |
| `Python interpreter not found` from a `.cmd` | No `venv\` or `env\` in the project root | Redo step 2, or set `SPS_PYTHON` to the interpreter's full path |
| Hugging Face download fails or hangs | Corporate proxy or blocked host | Set `HTTPS_PROXY`, or copy an existing `%USERPROFILE%\.cache\huggingface` folder from a machine that has one |
| `pip` SSL errors on install | TLS interception | Point `pip` at your internal mirror via `PIP_INDEX_URL` — ask IT rather than disabling verification |
| `INVALID_INPUT`, exit 0, `Reason` names the file | A `.pdf` / `.docx` / `.xls` was attached | Only `.csv`, `.xlsx` and `.xlsm` are accepted. This is deliberate: a wrong attachment is a business problem, faulted rather than retried |
| `INVALID_INPUT`, exit 2 | The file is the right type but missing or unreadable | Genuine I/O trouble — check the path the robot passed |
| `Problem description is N characters; at least 10 required.` | The description column is blank or nearly so | Check the ticket's column name — `Problem_Description`, matched case-insensitively |
| `NO_MATCHES` on a part you know exists | Part numbers differ after normalisation (`.strip().upper()`, invisible characters removed) | `Reason` reports how many rows matched out of how many scanned — usually a stray character in one source |
| Everything falls back to `local:` | One of the three `AZURE_EMBEDDING_*` values is missing or wrong | The `AZURE_EMBEDDING_FAILED_FALLING_BACK` log line names exactly which |
| `An Application Control policy has blocked this file` on `from lxml import etree` | lxml 6.x's native DLL is blocked on a managed Windows machine | `pip install "lxml<6"` |
| Tier 2 never runs | `data\0250_docs\` holds no `.docx` | Expected until the standards are loaded. Check with `dir data\0250_docs\*.docx` |
| A 0250 document is ignored | It is a legacy `.doc`, or a `~$` lock file | The warning names it. Re-save as `.docx`; close the document if Word has it open |
| Every 0250 citation is just the filename | The document's section titles are bold body text, not `Heading 2` | Apply Word's heading styles, then delete `data\0250_docs\0250_cache_*.npz` |
| Tier 2 finds nothing on a defect you know is covered | 0.62 is calibrated for the demo corpus, not yours | Run the batch evaluator and read `Tier2_Score`, then set `SPS_TIER2_THRESHOLD` |

---

## What to read next

- [`README.md`](README.md) — architecture, the constrained-RAG contract, measured
  latency, and why there are two thresholds.
- [`.env.example`](.env.example) — every setting, with the reasoning inline.
