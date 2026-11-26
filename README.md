# RealData: decision models on retrieval rankings

Runs Elephant-1, Clef and Drex Max (general decision models) and BioDecision v2 4B, MedDecider-27B and MedDecider-31B
(medical decision models) as ICD-10-CA code pickers. No clinical notes, bare codes as options.
The context is the entity text plus its expansion from `query_text`. Each request gives a model **every code in one
source's ranking** (or its first K with `--top K`); top-k is then read from the model's own probabilities
(hit@1, 3, 5, 10, ... up to K, and MRR) and compared with the retriever's ranking of the same list. 7 ranking sources, in
original and shuffled option order.

How many options a model can score is fixed by its own code (`max_options` in `lh_core.MODELS_INFO`):

| Model | Most options per request | Why |
|---|---|---|
| Elephant-1, Clef, Drex Max | 255 | built for long option lists |
| BioDecision v2 4B | 24 | its letters A-X (`biodecision_server.LETTERS`) |
| MedDecider-27B, MedDecider-31B | 10 | its letters A-J (`meddecider_infer.LETTERS`; model cards: "2 to 10 answer options") |

Every model runs on every folder. The build reads at most 255 codes per list; a model with a lower limit gets the
retriever's first N codes of each longer list (N = its limit), in the same order and with the same shuffle a `--top N`
build would give, so its answers match a `--top N` run. In the report its hit@k above N is `N/A`, a reference code
ranked below N is a miss for it, and its MRR and Acc are over those N codes; the report says which models were cut.
The medical models share one env, `realdata-med`
(torch 2.8, transformers 5.22.2, peft 0.21.2, the versions of GeminiOps/MedDecision); the MedDecider adapters also need
their base models (`Qwen/Qwen3.8-27B`, `google/gemma-4-31b-it`) in `HF_CACHE`.

What the models read, for a list E119, E1165, R739 (each option's key is the code, with no description):

| Elephant-1 | Drex Max | Clef |
|---|---|---|
| `A. E119` `B. E1165` `C. R739` | `A: E119` `B: E1165` `C: R739` | `OPTION 1: {"option_id":"E1165"}` ... |

Clef sorts options by key, so it always sees them in code order: its `original` and `shuf1` inputs are identical, and so
should its probabilities be. `report` checks this (`Clef order check: N pairs compared, 0 with different probabilities`)
and says to investigate if any differ. Its hit@k can still differ slightly between the orders on an exact tie, since a
tie goes to the earlier position.

How the code works, from the big picture down to individual functions: [IMPLEMENTATION.md](IMPLEMENTATION.md).
Problems met on a real cluster and their fixes: [Issues](#issues) at the end.

Setup downloads software (conda and pip packages, three GitHub runtimes) once. After that everything runs offline
(`HF_HUB_OFFLINE=1`), and no command sends data anywhere. Check that installing user software from the internet is
allowed on your server before step 1; the sources are listed under [What setup downloads](#what-setup-downloads).

## Quick start: two commands

    cp realdata.conf.example realdata.conf     # once: set DATA, INDEX, HF_CACHE (and RESULTS_DIR, GPU if needed)
    bash realdata_setup.sh                     # once per server: checks everything, installs only what is missing
    bash realdata_run.sh --desc --top 20       # build, smoke test, full run (background), report

`realdata_setup.sh` checks the GPU and driver, conda, and for each model its env, pinned core packages, imports,
checkpoint and runtime. Anything already in place is skipped. With `HF_CACHE` set, checkpoints are verified and linked,
never downloaded. If something is missing, it lists what it would download and asks first. `--check-only` never
installs anything. It ends with `READY` or `NOT READY: <reasons>`.

`realdata_run.sh` takes the options of steps 3-9 below: nothing (bare codes, whole lists), `--desc`, `--top K`,
`--models "DREX CLEF"`, `--gpu N`, `--name FOLDER`. It names the folder under `RESULTS_DIR` (`real_top20_desc` etc.), builds or
reuses the requests (only if they come from the same data, index and K), smoke-tests each model, runs the full set in
the background and writes the report plus `failure_summary_<MODEL>.txt`. Rerunning the same command resumes.
`--sources "rrf_csv+fsn rrf_csv+fsn+syn"` runs and reports only those retrieval sources (default: all 7). The build
still holds every source, so the same folder is used: a later run without `--sources` (or with other sources) adds the
rest and skips the requests already done. The report pools only the chosen sources and says which ones they are; the
per-source counts table lists all 7, with `-` for those left out. Run `--report-only` without `--sources` for every source.
`--orders original` runs and reports only the retriever's option order (default: original and shuf1), which halves the
GPU time; a later run without it adds the shuf1 requests to the same folder.
`--ks "1,3,5,10,20,30,40"` sets the report's hit@k cutoffs (default: 1, 3, 5, 10, 20, 50, ... up to K). Only the report
uses them, so `--report-only --foreground --ks "..."` redoes the tables without running a model. A k above the longest list is dropped
with a warning (it would repeat the last column).
`--report-only` rebuilds the tables from an existing folder. Quick check that everything runs: `bash realdata_run.sh --desc --top 20 --smoke-only --smoke-n 2 --name check` (a few requests per model, then stops). The sections below are the manual steps the scripts run.

## Needs

- A GPU with enough free memory: 22 GB (Elephant-1), 60 GB (Clef), 58 GB (Drex); models run one at a time. An 80-96 GB
  A100 or H100 runs all three. A 16 GB card (for example a Tesla T4) runs none of them.
- NVIDIA driver 580 or newer (Clef's and Drex's torch builds are CUDA 13.0 and 13.2; Elephant's is 12.8).
- conda, git, and any Python 3.10+ to run the commands (the system `python3` is enough; no packages needed).
- Disk: about 40 GB for conda, the three envs and caches, plus about 120 GB of weights unless they are in a shared cache.
- Internet during setup only. Nothing else: setup builds everything from this folder on a new server.

## 0. Prepare the server (only if needed)

Skip this if `conda --version`, `git --version` and `nvidia-smi` work and conda can download packages. On the Gemini HPC
cluster (`gpu02`/`gpu03`) none of that was true out of the box; these steps fixed it. The layout keeps everything next to the
repo and outside git:

```
~/DecisionModels/
├── RealData/        this repo
├── miniforge3/      conda, its package cache and the 3 model envs
├── models/          links to the checkpoints, the GitHub runtimes, setup logs   ($MODELS)
├── cache/           pip and Triton caches
└── env.sh           source it in every new terminal
```

**0a. Install Miniforge next to the repo** (the cluster's `conda/24.3.0` module was broken; no admin rights needed):

```bash
cd ~/DecisionModels
wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p ~/DecisionModels/miniforge3      # -b: no questions, ~/.bashrc untouched
rm Miniforge3.sh
```

**0b. Create `env.sh`** (once):

```bash
cat > ~/DecisionModels/env.sh <<'EOF'
# source ~/DecisionModels/env.sh : a clean shell for RealData
module purge 2>/dev/null                    # the cluster's python/R modules set PYTHONHOME and library paths
unset PYTHONHOME PYTHONPATH
export LD_LIBRARY_PATH=                      # conda envs use their own libraries
export PATH=$HOME/DecisionModels/miniforge3/bin:$PATH
export CONDA_ENVS_PATH=$HOME/DecisionModels/miniforge3/envs
export CONDA_PKGS_DIRS=$HOME/DecisionModels/miniforge3/pkgs
# the network re-signs TLS traffic: trust the system certificates (the same ones curl, wget and git use)
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
export PIP_CERT=$SSL_CERT_FILE
export CONDA_SSL_VERIFY=$SSL_CERT_FILE
export TORCH_CUDA_ARCH_LIST="8.0;9.0"        # build Elephant's kernel for A100 (8.0) and H100 (9.0)
export PIP_CACHE_DIR=$HOME/DecisionModels/cache/pip
export TRITON_CACHE_DIR=$HOME/DecisionModels/cache/triton
export PY=python3
export MODELS=$HOME/DecisionModels/models
cd $HOME/DecisionModels/RealData
EOF
source ~/DecisionModels/env.sh
```

**0c. mamba ignores `CONDA_SSL_VERIFY`**, so put the certificate setting in Miniforge's own config file (once):

```bash
conda config --file ~/DecisionModels/miniforge3/.condarc --set ssl_verify /etc/ssl/certs/ca-certificates.crt
```

**0d. Check** (in a new terminal):

```bash
source ~/DecisionModels/env.sh
echo "PYTHONHOME=[$PYTHONHOME] LD_LIBRARY_PATH=[$LD_LIBRARY_PATH]"      # both empty
which conda python3 git                                                  # conda, python3 under ~/DecisionModels/miniforge3/bin
conda --version
curl -sSL -o /dev/null -w "%{http_code}\n" https://prefix.dev/conda-forge/noarch/repodata.json.zst   # 200
mamba create --dry-run -n test-ssl python=3.11 2>&1 | tail -1           # "Dry run. Not executing the transaction."
python3 -m pip download --no-deps -d /tmp/piptest six 2>&1 | tail -1     # "Saved ..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv
```

From here on, every new terminal starts with `source ~/DecisionModels/env.sh`, which also sets `PY` and `MODELS`.

## 1. Setup (once; 30-60 minutes)

    cd RealData
    PY=python3                     # base Python (already set by env.sh)
    MODELS=/path/to/models         # where links, runtimes and logs go (already set by env.sh)
    $PY main.py --project-dir $MODELS setup-envs

If the weights are already in a shared Hugging Face cache (folders named `models--baselabs--elephant-1`,
`models--Cloudflare--clef`, `models--nace-ai--drex-max-v1`), point setup at it instead of downloading:

    nohup $PY main.py --project-dir $MODELS setup-envs --hf-cache /mnt/nfs/projects/shared_llms/hf_cache/hub > setup.log 2>&1 &
    tail -f setup.log

It ends with three `environment ready` lines. Rerunning it continues where it stopped (existing envs are reused).

Each checkpoint is checked file by file against the pinned version (size and hash, `setup/checkpoint_files.json`), then
`$MODELS/<model>` becomes a link to the cache snapshot. Nothing is downloaded or written into the cache. A newer commit is
accepted if every file the model uses is identical (on the Hub, Clef and Drex have newer commits that change only
documentation). If a checkpoint is missing or differs, setup stops and names the files. `$MODELS/setup_logs/<model>_source.json`
records which snapshot was used. Conda envs, packages and the GitHub clones (OneJev, decision-index, drex-decision-models)
are installed as usual.

Package versions: Elephant uses its checkpoint's own `constraints.txt` and the CUDA 12.8 toolkit; Clef and Drex are pinned with
`setup/tested/<model>.txt`, the package lists of the envs that produced the recorded results. At the end each script lists any
package that differs from that list and stops if torch, transformers or a kernel package differs.

### What setup downloads

All incoming, public, open-source software; nothing is uploaded:

| Source | What |
|---|---|
| `prefix.dev`, `conda.anaconda.org` (conda-forge) | Python, build tools |
| `conda.anaconda.org/nvidia` | CUDA 12.8 toolkit (Elephant's kernel compile) |
| `pypi.org`, `files.pythonhosted.org` | transformers and other Python packages |
| `download.pytorch.org` | PyTorch for Elephant |
| `github.com` | OneJev, decision-index, drex-decision-models (pinned commits) |
| `huggingface.co` | the checkpoints, only without `--hf-cache` |

Some conda packages run a small install script; mamba prints "This transaction includes executing package scripts". That is
normal for conda-forge and NVIDIA packages (they run as you, inside the env).

## 2. Check the data

    $PY inspect_rankings.py /path/to/data.json

Prints counts and at most 5 unusual codes per list; no entity or query text.

## 3. Build requests

    RUN=results/real_whole_list
    $PY decision_eval.py build --src /path/to/data.json --run-dir $RUN

Checks the whole file first and writes nothing if any entity has a problem (missing keys, duplicate or empty `entity_id`, empty
`entity_text`, not exactly one ground-truth code, a missing source, fewer than 2 codes, a code listed twice, more than 255 codes). The
messages name entity ids and codes only. It prints, per source, the list lengths and the retriever's hit@1/5/10/20 and
share of lists holding the reference code, and stops if no reference code appears in any list (usually a code-format
mismatch; `inspect_rankings.py` shows the formats). It also refuses a run folder that already has requests or predictions.
One entity gives 14 requests (7 sources x 2 orders): 25,732 per model for 1,838 entities.

If the entities have `note_id`, `start` and `end` (the fused ranking files do), the build copies them, with
`ground_truth_codes`, into every request, and the workers copy them into every prediction row. The models never see
them. They let the LLM picker find each entity's clinical note (step 10). The fields must be on all entities or on
none. A run built before this lacks them; the picker then looks them up in the DM's input file (see step 10).

**With the clinical note** (`--notes notes.jsonl`, or `realdata_run.sh --note` with `NOTES` in `realdata.conf`). The
DM's context becomes the note's first 3000 characters (`--max-note-chars N`), then the entity text as before:

    Clinical note:
    <the note's first 3000 characters>

    Diagnosis: <entity_text (expanded: query_text)>

The question becomes "Based on the clinical note as context, which ICD-10-CA code best matches the diagnosis?". The
options, their order and everything else stay the same as in the build without the note. Each request also keeps the
entity text alone as `entity_state`, for the LLM picker. The build needs `note_id`/`start`/`end` on every entity and
every note in notes.jsonl. It stops when 5% or more of the mentions are not at `start:end` of their note (a notes file
that does not belong to the data), and prints how many mentions lie past the cut. `realdata_run.sh` names the folder
`..._note3000`, and reuses a folder only when it was built with the same notes file and cut.

    bash realdata_run.sh --desc --note --name drex_med_desc_note3000 --models "DREX MEDDECIDER31"

## 4. Smoke test (about 260 requests per model)

    for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run $m --gpu 0 --run-dir $RUN --weights-dir $MODELS --stride 100; done

Use the `--gpu` index of a free GPU. Each model should end with `finished, N/N ok`. If some requests failed, the last line
says how many and why; rerunning the same command retries only those.

Report of the smoke test (prints a summary table; writes `ranking_tables_smoke.md` and `ranking_results_smoke.csv`):

    $PY decision_eval.py report --run-dir $RUN --smoke

The sample is about 260 requests per model, so the per-source tables hold only about 18 requests per cell. Use it to
check that the numbers are sensible and to time the full run (each model logs its seconds per request), not as results.

## 5. Full run (25,732 requests per model for 1,838 entities)

    nohup bash -c "for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run \$m --gpu 0 --run-dir $RUN --weights-dir $MODELS; done; echo ALL_DONE" > real_run.log 2>&1 &
    tail -f real_run.log | tr '\r' '\n' | grep --line-buffered -v "Loading weights"

Ctrl-C only stops `tail`. If interrupted, rerun the `nohup` line and it resumes. Each predictions file records which
`requests.jsonl` it belongs to (`predictions_<MODEL>.jsonl.meta.json`); a worker refuses to resume, and `report` skips the file,
if `requests.jsonl` has changed. Only one worker per model and run folder can run at a time.

How long it takes depends on how long the lists are. Measured on an A100 with synthetic lists: about 0.2 s (Elephant),
0.25 s (Drex) and 0.8 s (Clef) per request at about 70 codes, and 1.0 / 1.4 / 2.4 s at 255 codes. Estimate yours from the
smoke test: total hours = 25,732 x (sum of the three s/request) / 3600. On a cluster, check that the session will last
that long (for example `echo $SLURM_JOB_ID`, `squeue -u $USER`); a time-limited job stops the run, and rerunning resumes it.

## 6. Results (after ALL_DONE)

    $PY decision_eval.py report --run-dir $RUN

Prints a summary (model x order, all sources pooled) and writes `ranking_tables.md` and `ranking_results.csv` (per
source) to the run folder. It also writes `ranking_comparison.md` / `.csv`: for every source, the retriever baseline
and each model x order with top-1/5/10/20 (the csv also has MRR and the differences from the baseline). For a run reported before
this existed, make it from the existing csv (no rerun):

    $PY decision_eval.py compare --run-dir $RUN

- `hit@k`: the reference code is among the model's k most likely codes (ties go to the earlier position). Lists without
  the reference code count as misses. hit@1 is the accuracy of the model's answer.
- `MRR`: mean of 1 / rank of the reference code (0 if it is not in the list).
- `retriever @k`, `retriever MRR`: the same for the source's own ranking on the same requests: the baseline to beat.
- `In list`: share of lists holding the reference code (the ceiling for every hit@k).
- `Acc | in list`, Brier, ECE: only requests whose list holds the reference code.

**Per-request results.** Every run (smoke or full) ends by writing three files to `$RUN/option_probs/`, all named
`<MODEL>[_smoke]_<run folder>_<UTC time>`:

- `.jsonl`: one self-contained record per request: the model input (`state`, `instructions`, `options`: the codes in
  the order shown), `ground_truth` (code, position, retriever rank, whether it is in the list), `prediction` (code,
  correct), and `probabilities`: every option as `{rank, key, code, probability, retriever_rank}`, sorted from high to low. Requests without a
  result are included with `status` `error` (and the error type) or `not_run`. **It contains the entity text**; keep it
  with the data.
- `.csv`: the same probabilities, one row per request and option, ids and codes only (no entity text):

| column | meaning |
|---|---|
| `request_id`, `entity_id`, `source`, `n_options`, `order`, `model` | which request |
| `position`, `option_key`, `option_code` | where the option was shown (1 = first) and its code (the key is the code) |
| `probability` | the model's normalized probability for this option, full precision |
| `prob_rank` | 1 = highest; ties go to the earlier position, the same rule that picks the answer |
| `retriever_rank` | this option's rank in the source's original ranking (for the reference code: the baseline) |
| `is_prediction`, `is_gt`, `answerable` | picked by the runner / the reference code / reference code is among the options |
| `n_tied_top` | how many options share the highest probability (1 = no tie) |

- `.json`: what the export holds: model revision, the run's settings, hashes of `requests.jsonl` and the predictions
  file, and counts (including requests with a tie for the top).

Each export is a new set of files, so earlier ones are kept. To write one by hand (for example for an interrupted run,
or a run made before this export existed): `$PY run_decision_model.py export DREX --run-dir $RUN` (add `--smoke` for
the smoke-test predictions). A full run gives 25,732 records per model, and one CSV row per code in every list.

The worker's own file, `predictions_<MODEL>[_smoke].jsonl`, is what makes runs resumable: it holds the request id, the
picked option and code, and the probabilities keyed by option. The input, the options and the ground truth for each
`request_id` are in `requests.jsonl`; the export joins the two.

Run folders from the earlier top-k version (5/10/20 options per request, lettered A-T) keep working with `export`;
`report` refuses them, since their design differs.

Ties: Drex can give two options exactly the same probability, apparently because its scores are computed at reduced
precision (frequent on synthetic codes, none in the first real smoke test). The report's `tied` column and
`requests_with_tied_top` in the `.json` count them.

## 7. Before a run with code descriptions: check the index (CPU, reads only)

    $PY check_descriptions.py --data "$DATA" --index /path/to/icd_snomed_index.jsonl \
        --weights-dir $MODELS --bare-run $RUN --out desc_check.json [--show-examples]

PASS/WARN/FAIL per check; exit code 1 on any FAIL. It checks:
- **A, the index:** it parses, has no duplicate codes, no empty descriptions or odd characters, and shows the code format.
- **B, coverage:** how many list codes and reference codes have a description, matched exactly, after normalising
  dots/case/spaces, or only through a hierarchy parent. Missing codes are printed (bare codes only).
- **C, duplicates:** lists where two codes share one description.
- **D, rendering:** each model's exact input for a made-up request (only with `--show-examples`).
- **E, tokens:** tokens per request, bare and with descriptions, counted with each model's own prompt code in its
  conda env on CPU, against the limits (Elephant 32,768; Drex and Clef 16,000).
- **F, consistency and time:** the data matches the bare-code run, plus a rough run-time estimate from its latencies.
- **G, exact matches:** entities whose text equals their reference description.

Entity text is never printed or saved; `desc_check.json` holds counts only. `--stride N` counts tokens for every Nth
request only.

## 8. Run with code descriptions

Same steps 3-6 in a new folder. The build adds `--descriptions`. It stops if any list code has no description, and
it records both input files' sha256 in `build_meta.json`. Same entities, rankings and option orders as the bare run,
so the two compare directly.

    export INDEX=/path/to/icd_snomed_index.jsonl
    export RUN_DESC=results/real_whole_list_desc
    $PY decision_eval.py build --src "$DATA" --run-dir $RUN_DESC --descriptions "$INDEX"     # prints "options: code + long_description"
    for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run $m --gpu 0 --run-dir $RUN_DESC --weights-dir $MODELS --stride 100; done
    $PY decision_eval.py report --run-dir $RUN_DESC --smoke
    nohup bash -c "for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run \$m --gpu 0 --run-dir $RUN_DESC --weights-dir $MODELS; done; echo ALL_DONE" > desc_run.log 2>&1 &
    $PY decision_eval.py report --run-dir $RUN_DESC

## 9. Run on the retriever's top K only

Same as step 8 with `--top K`: each request holds only the first K codes of the source's ranking (lists shorter than K
are kept whole). A reference code ranked below K is not an option and counts as a miss, so the model's top-K can at
most equal the retriever's hit@K. Choose K on the command line; use one folder per K. `--top` works with or without
`--descriptions`, and the shuf1 orders are seeded per K (without `--top`, the orders are those of the whole-list runs).

    export K=20
    export RUN_TOP=results/real_top${K}_desc
    $PY decision_eval.py build --src "$DATA" --run-dir $RUN_TOP --descriptions "$INDEX" --top $K     # prints "top 20 per list"
    for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run $m --gpu 0 --run-dir $RUN_TOP --weights-dir $MODELS --stride 100; done
    $PY decision_eval.py report --run-dir $RUN_TOP --smoke
    nohup bash -c "for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run \$m --gpu 0 --run-dir $RUN_TOP --weights-dir $MODELS; done; echo ALL_DONE" > top${K}_run.log 2>&1 &
    tail -f top${K}_run.log
    $PY decision_eval.py report --run-dir $RUN_TOP

## 10. LLM picker on a decision model's reranking

The decision model (DM) reranks each retriever list. The LLM picker (Gemma 4 31B, `llm_picker.py`, the RealData copy
of `GeminiOps/run_rag_picker.py`) then sees the DM's top K codes with their descriptions and picks one. It sees the same
entity text as the DM. By default it sees no clinical note; with `--note first` it also sees the note (below). It
needs a finished DM run built with `--desc`.

    cp picker.conf.example picker.conf      # once: LLM_MODEL, LLM_ENV, DM_RESULTS_DIR, PICKER_RESULTS_DIR, CONDA, GPU
    bash picker_run.sh --dm-run real_whole_list_desc --dms DREX --sources rrf_fsn+syn --ks 3 --limit 2 --foreground   # smoke
    bash picker_run.sh --dm-run real_whole_list_desc --dms "DREX MEDDECIDER31" \
         --sources "rrf_csv+fsn rrf_fsn+syn rrf_csv+fsn+syn" --ks "3,5,10,20"

The pipeline has three steps:

- `picker_build.py`: for each source × DM, it sorts the codes the DM read by the DM's probabilities.
  - It writes the picker's own input (`--format native`, the default): a consolidated JSON in retriever order, plus a reranked JSON in DM order.
  - Or it writes a JSONL with the candidates already in DM order (`--format requests`).
  - It writes one config per K.
  - A K above what the DM read (MedDecider-31B reads 10) is not built and reads N/A.
  - Entities the DM has no result for are left out and counted.
- `llm_picker.py`: runs every config in one model load and skips the configs that are finished.
- `picker_report.py`: compares the LLM's top-1 with the DM's top-1 and with the most it can reach at K (the DM's hit@K).
  It also lists the fixed, broke, no_answer and not_in_list counts.

The report is saved as `report_<UTC time>.md/.csv`. `--report-only` rebuilds it, and old reports are kept.
`--batch-size N` answers N entities per generate call. With 6 entities it was 3–4× faster.
bf16 results depend on the batch shape, so a borderline pick can differ between batch sizes. Use one batch size for
every run you compare. Each prediction file's `.meta.json` records the size, and the report warns when sizes are mixed.
**With the clinical note.** `--note first` shows the first 4000 characters of each entity's note
(`--max-note-chars N` to change), as the original picker does, in its original prompt with the note section. Set
`NOTES` in `picker.conf` to a notes.jsonl with `note_id` and `note_text`. The note link (`note_id`, `start`, `end`) comes
from `requests.jsonl` when the DM run has it. Otherwise it comes from the DM's input file, the one named in the DM
`build_meta.json`; the build stops if that file has changed since (sha256). It also stops if an entity has no note link
or its note is missing. The build and the report count mentions that end past the cut (the LLM sees the note without
the mention), and warn when the note text at `start:end` does not match the entity text (a sign of wrong note ids).
Folders are named `<dm run>__llm_note4000`.

    bash picker_run.sh --dm-run real_whole_list_desc --dms "DREX MEDDECIDER31" --sources rrf_fsn+syn --ks 3 \
         --note first --limit 5 --batch-size 8 --foreground                      # smoke

**Retriever baseline (no DM).** `--candidates retriever` gives the LLM the retriever's own top K from the fused ranking
file, in retriever order, with no reranked JSON: the original RAG picker's input. Leave out `--dms`. Its rows are named
RETRIEVER (folder `<dm run>__llm_retriever...`). Use it to reproduce the RAG picker's result with the same entities,
entity text and note. `picker_compare.py --ours <our predictions .jsonl> --original <run_rag_picker.py output>` matches
two prediction files on (note_id, start, end) and counts same reference code, entity text, candidates and pick, with
both top-1s (counts and codes only).

    bash picker_run.sh --dm-run real_whole_list_desc --candidates retriever --sources rrf_csv+fsn+syn --ks 20 --note first --batch-size 8

**The RAG pipeline's picker settings.** That picker shows the mention alone (`entity_text` from the fused ranking
file, not the expanded `query_text`), the note's first 3000 characters, and allows 8192 new tokens. `--entity-text
mention` gives the mention alone (read from the DM's input file, sha256-checked; folder suffix `_mention`); the default
`expanded` keeps the DM's input, `entity_text (expanded: query_text)`. `--max-new-tokens N` (default 512; suffix
`_mnt<N>`). It ran Gemma in 4-bit; ours runs in bf16.

    bash picker_run.sh --dm-run real_whole_list_desc --candidates retriever --sources snomed_syn --ks 20 --note first \
         --max-note-chars 3000 --entity-text mention --max-new-tokens 8192 --batch-size 8

The report also gives `llm_cat3` (the pick has the reference code's first 3 characters) and `llm_chapter` (the same
ICD-10 chapter, by the WHO chapter blocks).

`--stop-tokens tokenizer_eos` gives the uploaded script's generation (folder suffix `_tokeos`). There, Gemma does not stop at the end of its
turn, and its last answer is scored. Picker folders hold entity text (and, with `--note`, the notes), so keep them on the server.

## Data handling

`requests.jsonl` and the `option_probs/*.jsonl` exports in the run folder contain entity and query text, so keep the run folder with the data. Logs hold only counts and error types.
The report tables hold metrics and source names only. `.gitignore` keeps `results/`, `data/` and every `*.jsonl` out of git.

Option text is bare codes in the first run (step 5), so scores show whether a model can pick a code without its
description. The run in step 8 adds each code's long description from the index file (reference data, read on the server only).

## Issues

Met while setting up on the Gemini HPC cluster (Ubuntu 24.04, Lmod modules, hospital network), in the order they appeared.

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` (Ubuntu's `command-not-found` traceback) | no conda on the node | Miniforge next to the repo (step 0a) |
| `ModuleNotFoundError: No module named 'numpy'` from the system Python | system Python has no packages | none needed now: RealData runs on a plain `python3` |
| `module load conda/24.3.0`: `The following module(s) are unknown: "python/3.12.2"` | the module depends on a Python version the cluster no longer has | don't use the module; use Miniforge (step 0a) |
| `conda --version`: `Fatal Python error: Failed to import encodings module`, `PYTHONHOME = '/s/modules/pkgs/...'` | the auto-loaded `python` module sets `PYTHONHOME`, so Miniforge's Python reads the wrong standard library | `module purge` and `unset PYTHONHOME PYTHONPATH` in `env.sh` (step 0b) |
| `nvidia-smi` shows a Tesla T4, 15 GB | node too small for any of the models | use a node with an 80-96 GB A100/H100 (`gpu03` has an H100 NVL) |
| setup: `SSL peer certificate ... unable to get local issuer certificate` for `prefix.dev` | the network re-signs TLS; Miniforge, pip and Hugging Face ship their own certificate lists | point them at `/etc/ssl/certs/ca-certificates.crt` (step 0b); `curl` returning 200 or 303 shows the system list works |
| same SSL error from `mamba` while `conda create --dry-run` works | mamba ignores `CONDA_SSL_VERIFY` | `ssl_verify` in `miniforge3/.condarc` (step 0c) |
| `Security warning: This transaction includes executing package scripts` | conda packages with post-link scripts | normal; no action |
| `STOPPED: nvcc is not CUDA 12.8` (nvcc reports 12.9) | conda 26 / mamba 2 took a newer `cuda-toolkit` from conda-forge | fixed in `setup_elephant.sh` (`cuda-toolkit=12.8 cuda-version=12.8`); `git pull` and rerun setup |
| `Hugging Face token file: MISSING` | no `hf auth login` | harmless: the repos are public, and `--hf-cache` downloads nothing |
| `git push`: `Password authentication is not supported` | GitHub needs a personal access token, not the account password | paste a token at the password prompt |

Future risk: while building OneJev, setuptools warns that `project.license` as a TOML table is deprecated and must be changed
"by 2027-Feb-18". Installed envs are not affected. A fresh setup after that date may fail at Elephant's `setup.sh` if pip's
newest setuptools has dropped the format and OneJev (pinned to commit 81ce62f) has not changed its `pyproject.toml`. Fix: put
`setuptools<NN` (NN = the first release without support) in a file and rerun setup with `PIP_CONSTRAINT=/path/to/that/file`,
which pip also applies to its build environments.

If `causal-conv1d` was compiled before `TORCH_CUDA_ARCH_LIST` included `9.0`, Elephant on an H100 fails with "no kernel image
is available". Fix: `source ~/DecisionModels/env.sh`, then `conda env remove -y -n realdata-elephant` and rerun setup.
