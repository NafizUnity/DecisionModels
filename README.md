# RealData: decision models on retrieval rankings

Runs Elephant-1, Clef and Drex Max as ICD-10-CA code pickers. No clinical notes, bare codes as options.
The context is the entity text plus its expansion from `query_text`. Compares 7 ranking sources at top-5, 10 and 20, in original and shuffled option order.

How the code works, from the big picture down to individual functions: [IMPLEMENTATION.md](IMPLEMENTATION.md).

Everything runs locally. Models are downloaded once during setup, and inference runs offline (`HF_HUB_OFFLINE=1`).

## Needs

- GPU with free VRAM: 22 GB (Elephant-1), 60 GB (Clef), 58 GB (Drex). Models run one at a time.
- Disk: 150-200 GB (about 120 GB of weights, plus three conda envs).
- conda, git, an NVIDIA driver 580 or newer (Clef's and Drex's torch builds are CUDA 13.0 and 13.2; Elephant's is 12.8),
  and a base Python with numpy and pandas.
- Internet for setup only: Hugging Face (run `hf auth login` if needed), GitHub, PyTorch package index, conda channels.
- Nothing else: setup starts from scratch on a new server (conda envs `realdata-elephant`, `realdata-clef`, `realdata-drex`,
  packages, GitHub clones and checkpoints), using only this folder. `setup-envs` needs `--project-dir`; runs need `--weights-dir`.

## 1. Setup (once)

    cd RealData
    PY=python3                     # base Python
    MODELS=/path/to/models         # where weights will be stored
    $PY main.py --project-dir $MODELS setup-envs

If the weights are already in a shared Hugging Face cache (folders named `models--baselabs--elephant-1`,
`models--Cloudflare--clef`, `models--nace-ai--drex-max-v1`), point setup at it instead of downloading:

    $PY main.py --project-dir $MODELS setup-envs --hf-cache /mnt/nfs/projects/shared_llms/hf_cache/hub

Each checkpoint is checked file by file against the pinned version (size and hash, `setup/checkpoint_files.json`), then
`$MODELS/<model>` becomes a link to the cache snapshot. Nothing is downloaded or written into the cache. A newer commit is
accepted if every file the model uses is identical (on the Hub, Clef and Drex have newer commits that change only
documentation). If a checkpoint is missing or differs, setup stops and names the files. `$MODELS/setup_logs/<model>_source.json`
records which snapshot was used. Conda envs, packages and the GitHub clones (OneJev, decision-index, drex-decision-models)
are installed as usual.

Package versions: Elephant uses its checkpoint's own `constraints.txt`; Clef and Drex are pinned with
`setup/tested/<model>.txt`, the package lists of the envs that produced the recorded results. At the end each script lists any
package that differs from that list and stops if torch, transformers or a kernel package differs.

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

## 6. Results (after ALL_DONE)

    $PY decision_eval.py report --run-dir $RUN

Writes `ranking_tables.md` and `ranking_results.csv` to the run folder.

## Data handling

`requests.jsonl` in the run folder contains entity and query text, so keep the run folder with the data. Logs hold only counts and error types.
The report tables hold metrics and source names only.

Option text is bare codes in this run, so scores show whether a model can pick a code without its description. Descriptions are planned as a later run.
