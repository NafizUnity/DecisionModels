#!/usr/bin/env bash
# realdata_run.sh: build the requests, smoke-test, run the decision models and write the report, in one command.
#
#   bash realdata_run.sh                       # whole lists, bare codes        -> RESULTS_DIR/real_whole_list
#   bash realdata_run.sh --desc                # whole lists + descriptions     -> RESULTS_DIR/real_whole_list_desc
#   bash realdata_run.sh --desc --top 20       # retriever's top 20 + desc.     -> RESULTS_DIR/real_top20_desc
#   bash realdata_run.sh --desc --top 20 --smoke-only --smoke-n 2 --name check   # quick check: about 2 requests per model
#   bash realdata_run.sh --desc --top 10 --sources "rrf_csv+fsn rrf_csv+fsn+syn"   # only these retrieval sources
#   Models that cannot score the lists are skipped: MedDecider-27B/31B take at most 10 options, BioDecision 24.
#   --sources: the build still holds all 7 sources; only the chosen ones are run and reported. Rerun later without it
#   (same folder) to fill in the rest; finished requests are kept.
#   options: --models "DREX CLEF"  --sources "A B"  --gpu 1  --name my_run (under RESULTS_DIR, or a full path)  --conf FILE  --no-smoke
#            --foreground  --report-only  --smoke-only (stop after the smoke test)  --smoke-n N (smoke sample of at least N)
#
# Paths (data, index, models, RESULTS_DIR) come from realdata.conf (run realdata_setup.sh first). The build runs in this terminal; the model runs then
# continue in the background (nohup) and the script prints the `tail -f` command for its log. Rerunning the same
# command resumes: the build is reused when its inputs are unchanged, finished requests are skipped, and a model whose
# full run is complete is not smoke-tested again. It ends with "DONE: <run folder>" or "STOPPED: <reason>".
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
CONF="$HERE/realdata.conf"; DESC=0; TOP=""; NAME=""; GPU_ARG=""; MODELS_ARG=""; SMOKE=1; FOREGROUND=0; REPORT_ONLY=0
SMOKE_ONLY=0; SMOKE_N=""; SOURCES_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --desc) DESC=1; shift ;;
        --top) TOP="${2:?--top needs K}"; shift 2 ;;
        --models) MODELS_ARG="${2:?--models needs a list}"; shift 2 ;;
        --sources) SOURCES_ARG="${2:?--sources needs a list}"; shift 2 ;;
        --gpu) GPU_ARG="${2:?--gpu needs an index}"; shift 2 ;;
        --name) NAME="${2:?--name needs a folder}"; shift 2 ;;
        --conf) CONF="${2:?--conf needs a file}"; shift 2 ;;
        --no-smoke) SMOKE=0; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --report-only) REPORT_ONLY=1; shift ;;
        --smoke-only) SMOKE_ONLY=1; FOREGROUND=1; shift ;;
        --smoke-n) SMOKE_N="${2:?--smoke-n needs a number}"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
done
stop() { echo "[$(date +%H:%M:%S)] STOPPED: $*"; exit 1; }
say()  { echo "[$(date +%H:%M:%S)] $*"; }
placeholder() { [[ -z "$1" || "$1" == /path/to/* ]]; }

[ -f "$CONF" ] || stop "$CONF not found: run bash realdata_setup.sh first (it creates it from realdata.conf.example)"
# shellcheck disable=SC1090
source "$CONF"
: "${BASE:?BASE is not set in $CONF}"; MODELS="${MODELS:-$BASE/models}"
GPU="${GPU_ARG:-${GPU:-0}}"; RUN_MODELS="${MODELS_ARG:-${RUN_MODELS:-ELEPHANT DREX CLEF BIODECISION MEDDECIDER27 MEDDECIDER31}}"; RUN_MODELS="${RUN_MODELS^^}"
SMOKE_STRIDE="${SMOKE_STRIDE:-100}"; DATA="${DATA:-}"; INDEX="${INDEX:-}"
if [ -n "$TOP" ] && ! [[ "$TOP" =~ ^[0-9]+$ && "$TOP" -ge 2 ]]; then stop "--top must be a whole number of 2 or more"; fi
if [ -n "$SMOKE_N" ] && ! [[ "$SMOKE_N" =~ ^[0-9]+$ && "$SMOKE_N" -ge 1 ]]; then stop "--smoke-n must be a whole number of 1 or more"; fi
if [ "$SMOKE_ONLY" = 1 ] && { [ "$SMOKE" = 0 ] || [ "$REPORT_ONLY" = 1 ]; }; then stop "--smoke-only cannot be combined with --no-smoke or --report-only"; fi
for m in $RUN_MODELS; do
    [[ "$m" =~ ^(ELEPHANT|CLEF|DREX|BIODECISION|MEDDECIDER27|MEDDECIDER31)$ ]] \
        || stop "unknown model $m (ELEPHANT, CLEF, DREX, BIODECISION, MEDDECIDER27, MEDDECIDER31)"
done

# ---- the same clean shell as realdata_setup.sh, offline ---------------------------------------------------------------
if declare -F module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; fi
unset PYTHONHOME PYTHONPATH
keep=""; IFS=: read -r -a _parts <<<"${LD_LIBRARY_PATH:-}"
for p in "${_parts[@]}"; do if [[ "$p" == *nvidia* || "$p" == *cuda* ]]; then keep="${keep:+$keep:}$p"; fi; done
export LD_LIBRARY_PATH="$keep" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
for c in "$BASE/miniforge3/bin/conda" "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" "$HOME/miniforge3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then export PATH="$(dirname "$c"):$PATH"; break; fi
done
command -v conda >/dev/null || stop "conda not found: run bash realdata_setup.sh"
PY="$(dirname "$(command -v conda)")/python"; [ -x "$PY" ] || PY=python3

# ---- run folder -------------------------------------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
if [ -z "$NAME" ]; then
    NAME="real_$([ -n "$TOP" ] && echo "top$TOP" || echo whole_list)$([ "$DESC" = 1 ] && echo _desc || true)"
fi
case "$NAME" in /*) RUN="$NAME" ;; *) RUN="$RESULTS_DIR/$NAME" ;; esac   # --name: a full path, or a name under RESULTS_DIR
cd "$HERE"

# ---- --sources: check the names, pass them on as one space-separated list -----------------------------------------------
SRC_OPT=()
if [ -n "$SOURCES_ARG" ]; then
    SOURCES_ARG="$("$PY" - "$SOURCES_ARG" <<'PY'
import sys
sys.path.insert(0, ".")
from decision_eval import SOURCES
from run_decision_model import parse_sources
names = parse_sources(sys.argv[1]) or ()
bad = [n for n in names if n not in SOURCES]
if bad or not names:
    sys.exit(f"unknown source(s) {bad or sys.argv[1]!r}; choose from: {' '.join(SOURCES)}")
print(" ".join(names))
PY
    )" || stop "--sources: see the message above"
    SRC_OPT=(--sources "$SOURCES_ARG")
fi

# ======================================================================================================================
# Stage 1 (this terminal): checks and build. Stage 2 (background): smoke test, full runs, report.
if [ "${REALDATA_STAGE:-1}" = 1 ]; then
    [ "$REPORT_ONLY" = 1 ] && say "Report only: $RUN | models $RUN_MODELS" || say "Run folder $RUN | options: $([ "$DESC" = 1 ] && echo "code + description" || echo "bare codes"); $([ -n "$TOP" ] && echo "top $TOP" || echo "whole lists") | models $RUN_MODELS | sources ${SOURCES_ARG:-all} | GPU $GPU"
    [ -d "$MODELS" ] || stop "models folder $MODELS not found: run bash realdata_setup.sh"
    if [ "$REPORT_ONLY" = 0 ]; then
        placeholder "$DATA" && stop "DATA is not set in $CONF"
        [ -f "$DATA" ] || stop "DATA $DATA not found"
        if [ "$DESC" = 1 ]; then placeholder "$INDEX" && stop "--desc needs INDEX in $CONF"; [ -f "$INDEX" ] || stop "INDEX $INDEX not found"; fi
        if [ -f "$RUN/requests.jsonl" ]; then
            # reuse the build only if it was made from the same data, index and K
            if ! verdict="$(DATA="$DATA" INDEX="$([ "$DESC" = 1 ] && echo "$INDEX" || true)" TOP="$TOP" "$PY" - "$RUN/build_meta.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
meta_path = Path(sys.argv[1])
if not meta_path.exists():
    print("no build_meta.json (built before it existed); reusing requests.jsonl as is"); sys.exit(0)
meta = json.loads(meta_path.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
index, top = os.environ["INDEX"] or None, int(os.environ["TOP"]) if os.environ["TOP"] else None
problems = []
if meta.get("src_sha256") != sha(os.environ["DATA"]): problems.append("DATA differs from the file it was built from")
if (meta.get("descriptions_sha256") is None) != (index is None): problems.append("built " + ("without" if index else "with") + " descriptions")
elif index and meta.get("descriptions_sha256") != sha(index): problems.append("INDEX differs from the one it was built with")
if meta.get("top") != top: problems.append(f"built with top {meta.get('top')}, asked for top {top}")
print("; ".join(problems) if problems else f"same inputs ({meta.get('requests')} requests)")
sys.exit(1 if problems else 0)
PY
            )"; then stop "$RUN already holds requests from other inputs: $verdict. Use --name for a new folder."; fi
            say "Build: reusing $RUN/requests.jsonl ($verdict)"
        else
            build=(--src "$DATA" --run-dir "$RUN")
            [ "$DESC" = 1 ] && build+=(--descriptions "$INDEX")
            [ -n "$TOP" ] && build+=(--top "$TOP")
            say "Build: decision_eval.py build ${build[*]}"
            "$PY" decision_eval.py build "${build[@]}" || stop "build failed (see above; nothing was written)"
        fi
    else
        [ -f "$RUN/requests.jsonl" ] || stop "--report-only: $RUN has no requests.jsonl"
    fi
    if [ "$REPORT_ONLY" = 0 ]; then
        # Skip every model that cannot score this folder's longest list (lh_core.MODELS_INFO max_options: MedDecider 10,
        # BioDecision 24, the general models 255). --top 10 runs all six; --top 11..24 drops MedDecider; above 24 (or
        # whole lists) only the general models run.
        kept="$(SOURCES_ARG="$SOURCES_ARG" "$PY" - "$RUN/requests.jsonl" $RUN_MODELS <<'PY'
import json, os, sys
sys.path.insert(0, ".")
import lh_core
from run_decision_model import parse_sources, select_requests
rows = select_requests(list(map(json.loads, open(sys.argv[1], encoding="utf8"))), 1, parse_sources(os.environ["SOURCES_ARG"]))
largest = max(r["n_options"] for r in rows)
keep = []
for m in sys.argv[2:]:
    limit = lh_core.MODELS_INFO[m]["max_options"]
    if limit >= largest:
        keep.append(m)
    else:
        print(f"[skip] {m}: scores at most {limit} options per request; this run's lists hold up to {largest} "
              f"(use --top {limit} or less to include it)", file=sys.stderr)
print(" ".join(keep))
PY
        )" || stop "could not read $RUN/requests.jsonl"
        [ -n "$kept" ] || stop "none of the models ($RUN_MODELS) can score lists this long; use a smaller --top"
        [ "$kept" = "$RUN_MODELS" ] || say "Models to run: $kept"
        RUN_MODELS="$kept"; export REALDATA_RUN_MODELS="$kept"
    fi
    if [ "$FOREGROUND" = 1 ]; then
        export REALDATA_STAGE=2
    else
        LOG="$RUN/realdata_run_$(date +%Y%m%d_%H%M%S).log"
        REALDATA_STAGE=2 nohup bash "$0" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
        say "Started in the background (pid $!). Follow it with:"
        echo "    tail -f $LOG | tr '\\r' '\\n' | grep --line-buffered -v 'Loading weights'"
        echo "Ctrl-C stops only tail. The log ends with DONE or STOPPED; rerun the same command to resume."
        exit 0
    fi
fi

# ---- Stage 2 ----------------------------------------------------------------------------------------------------------
exec 9>"$RUN/.realdata_run.lock"
flock -n 9 || stop "another realdata_run.sh is already working on $RUN"
RUN_MODELS="${REALDATA_RUN_MODELS:-$RUN_MODELS}"   # stage 1 dropped the models that cannot score these lists
say "Stage 2 on $RUN (pid $$, GPU $GPU, models $RUN_MODELS)"
partial=()

# run_model MODEL [--stride N]: one attempt plus one retry for requests without a result. Returns 0 only if all are ok.
run_model() {
    local m="$1"; shift
    local attempt out rc
    for attempt in 1 2; do
        out="$(mktemp)"
        set +e
        "$PY" -u run_decision_model.py run "$m" --gpu "$GPU" --run-dir "$RUN" --weights-dir "$MODELS" ${SRC_OPT[@]+"${SRC_OPT[@]}"} "$@" 2>&1 | tee "$out"
        rc=${PIPESTATUS[0]}
        set -e
        if [ "$rc" -ne 0 ]; then rm -f "$out"; say "$m: stopped with exit code $rc (see above)"; return 2; fi
        # "<MODEL>: finished, N/N ok" (awk, not a grep backreference: some servers alias grep to ugrep)
        if grep -q "$m: finished, 0/0 ok" "$out"; then
            rm -f "$out"; say "$m: no requests selected (with --stride, fewer requests than the stride: lower SMOKE_STRIDE or use --no-smoke)"; return 2
        fi
        if awk -v key="$m: finished, " 'index($0, key) { split(substr($0, index($0, key) + length(key)), a, "[/ ]")
                                                          if (a[1] != "" && a[1] == a[2]) found = 1 } END { exit !found }' "$out"; then
            rm -f "$out"; return 0
        fi
        rm -f "$out"
        [ "$attempt" = 1 ] && say "$m: some requests have no result; retrying them once"
    done
    return 1
}

if [ "$REPORT_ONLY" = 0 ]; then
    if [ "$SMOKE" = 1 ]; then
        if [ -n "$SMOKE_N" ]; then   # the largest stride whose sample (a hash of the request id, as in the worker) has >= N requests
            SMOKE_STRIDE="$(SOURCES_ARG="$SOURCES_ARG" "$PY" - "$RUN/requests.jsonl" "$SMOKE_N" <<'PY'
import hashlib, json, os, sys
sys.path.insert(0, ".")
from run_decision_model import parse_sources, select_requests
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf8") if line.strip()]
ids = [r["request_id"] for r in select_requests(rows, 1, parse_sources(os.environ["SOURCES_ARG"]))]   # within --sources
n, hashes = int(sys.argv[2]), [int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) for i in ids]
s = max(2, len(ids) // n)
while s > 2 and sum(h % s == 0 for h in hashes) < n:
    s -= 1
print(s)
PY
            )"
        fi
        smoked=0
        for m in $RUN_MODELS; do
            if [ -f "$RUN/predictions_${m}.jsonl" ]; then say "Smoke test: $m skipped (its full run has started before)"; continue; fi
            meta="$RUN/predictions_${m}_smoke.jsonl.meta.json"
            if [ -f "$meta" ]; then
                old="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stride"))' "$meta")"
                [ "$old" = "$SMOKE_STRIDE" ] || stop "$m already has a smoke test with stride $old in $RUN; use the same SMOKE_STRIDE, or --name for a new folder"
            fi
            say "Smoke test: $m (every ${SMOKE_STRIDE}th request${SMOKE_N:+, chosen for at least $SMOKE_N})"
            run_model "$m" --stride "$SMOKE_STRIDE" || stop "smoke test of $m did not finish cleanly (see above); fix it, then rerun"
            smoked=1
        done
        if [ "$smoked" = 1 ]; then
            say "Smoke report"
            "$PY" decision_eval.py report --run-dir "$RUN" --smoke ${SRC_OPT[@]+"${SRC_OPT[@]}"} || stop "smoke report failed (see above)"
        fi
        if [ "$SMOKE_ONLY" = 1 ]; then say "DONE (smoke test only): $RUN"; exit 0; fi
    fi
    partial=()
    for m in $RUN_MODELS; do
        say "Full run: $m"
        if run_model "$m"; then :; else
            rc=$?; [ "$rc" = 2 ] && stop "$m failed; fix it and rerun the same command (finished requests are kept)"
            partial+=("$m")
        fi
    done
fi

say "Report${SOURCES_ARG:+ (sources: $SOURCES_ARG)}"
"$PY" decision_eval.py report --run-dir "$RUN" ${SRC_OPT[@]+"${SRC_OPT[@]}"} || stop "report failed (see above)"
for m in $RUN_MODELS; do
    if [ -f "$RUN/predictions_${m}.jsonl" ]; then
        "$PY" show_predictions.py --run-dir "$RUN" --model "$m" --limit 0 > "$RUN/failure_summary_${m}.txt" 2>&1 \
            && say "Failure summary: $RUN/failure_summary_${m}.txt" || say "WARN: show_predictions.py failed for $m (see $RUN/failure_summary_${m}.txt)"
    fi
done
echo
[ "${#partial[@]}" -gt 0 ] 2>/dev/null && say "WARN: ${partial[*]} still have requests without a result (rerun the same command to retry them)"
say "Tables: $RUN/ranking_tables.md, $RUN/ranking_comparison.md, $RUN/ranking_results.csv"
say "DONE: $RUN"
