# RealData: decision models on retrieval rankings

Runs Elephant-1, Clef and Drex Max as ICD-10-CA code pickers. No clinical notes, bare codes as options.
The context is the entity text plus its expansion from `query_text`. Compares 7 ranking sources at top-5, 10 and 20, in original and shuffled option order.

How the code works, from the big picture down to individual functions: [IMPLEMENTATION.md](IMPLEMENTATION.md).
Problems met on a real cluster and their fixes: [Issues](#issues) at the end.

Setup downloads software (conda and pip packages, three GitHub runtimes) once. After that everything runs offline
(`HF_HUB_OFFLINE=1`), and no command sends data anywhere. Check that installing user software from the internet is
allowed on your server before step 1; the sources are listed under [What setup downloads](#what-setup-downloads).

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

    RUN=results/real_no_notes_bare_codes
    $PY decision_eval.py build --src /path/to/data.json --run-dir $RUN

Checks the whole file first and writes nothing if any entity has a problem (missing keys, duplicate or empty `entity_id`, empty
`entity_text`, not exactly one ground-truth code, a missing source, fewer than 20 unique string codes). The messages name entity ids
and codes only. It prints recall per source and stops if no reference code appears in any list (usually a code-format mismatch;
`inspect_rankings.py` shows the formats). It also refuses a run folder that already has requests or predictions.

## 4. Smoke test (about 190 requests per model)

    for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run $m --gpu 0 --run-dir $RUN --weights-dir $MODELS --stride 400; done

Use the `--gpu` index of a free GPU. Each model should end with `finished, N/N ok`. If some requests failed, the last line
says how many and why; rerunning the same command retries only those.

## 5. Full run (77,196 requests per model for 1,838 entities; roughly 15 hours in total)

    nohup bash -c "for m in ELEPHANT DREX CLEF; do $PY -u run_decision_model.py run \$m --gpu 0 --run-dir $RUN --weights-dir $MODELS; done; echo ALL_DONE" > real_run.log 2>&1 &
    tail -f real_run.log | tr '\r' '\n' | grep --line-buffered -v "Loading weights"

Ctrl-C only stops `tail`. If interrupted, rerun the `nohup` line and it resumes. Each predictions file records which
`requests.jsonl` it belongs to (`predictions_<MODEL>.jsonl.meta.json`); a worker refuses to resume, and `report` skips the file,
if `requests.jsonl` has changed. Only one worker per model and run folder can run at a time.

On a cluster, check that the session will last about 15 hours (for example `echo $SLURM_JOB_ID`, `squeue -u $USER`); a
time-limited job stops the run, and rerunning resumes it.

## 6. Results (after ALL_DONE)

    $PY decision_eval.py report --run-dir $RUN

Writes `ranking_tables.md` and `ranking_results.csv` to the run folder.

**Per-option probabilities.** Every run (smoke or full) ends by writing
`$RUN/option_probs/<MODEL>[_smoke]_<run folder>_<UTC time>.csv` with one row per request and option:

| column | meaning |
|---|---|
| `request_id`, `entity_id`, `source`, `k`, `order`, `model` | which request |
| `position`, `option_letter`, `option_code` | where the option was shown (A = 1) and its code |
| `probability` | the model's normalized probability for this option, full precision |
| `prob_rank` | 1 = highest; ties go to the earlier position, the same rule that picks the answer |
| `is_prediction`, `is_gt`, `answerable` | picked by the runner / the reference code / reference code is among the options |
| `n_tied_top` | how many options share the highest probability (1 = no tie) |

A `.json` file of the same name records the model revision, the run's settings, the hashes of `requests.jsonl` and the
predictions file, and counts (including requests with a tie for the top). Each export is a new file, so earlier ones are
kept. To write one by hand (for example after an interrupted run): `$PY run_decision_model.py export DREX --run-dir $RUN`
(add `--smoke` for the smoke-test predictions). A full run gives about 900,000 rows (roughly 90 MB) per model.

Ties: in tests, Drex often gave two options exactly the same probability (52 of 210 synthetic requests; Elephant and Clef
none), apparently because its scores are computed at reduced precision. Check `requests_with_tied_top` in the `.json`.

## Data handling

`requests.jsonl` in the run folder contains entity and query text, so keep the run folder with the data. Logs hold only counts and error types.
The report tables hold metrics and source names only. `.gitignore` keeps `results/`, `data/` and every `*.jsonl` out of git.

Option text is bare codes in this run, so scores show whether a model can pick a code without its description. Descriptions are planned as a later run.

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
