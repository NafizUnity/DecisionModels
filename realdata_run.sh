#!/usr/bin/env bash
# realdata_run.sh: build the requests, smoke-test, run the decision models and write the report, in one command.
#
#   bash realdata_run.sh                       # whole lists, bare codes        -> results/real_whole_list
#   bash realdata_run.sh --desc                # whole lists + descriptions     -> results/real_whole_list_desc
#   bash realdata_run.sh --desc --top 20       # retriever's top 20 + desc.     -> results/real_top20_desc
#   options: --models "DREX CLEF"  --gpu 1  --name results/my_run  --conf FILE  --no-smoke  --foreground  --report-only
#
# Paths come from realdata.conf (run realdata_setup.sh first). The build runs in this terminal; the model runs then
# continue in the background (nohup) and the script prints the `tail -f` command for its log. Rerunning the same
# command resumes: the build is reused when its inputs are unchanged, finished requests are skipped, and a model whose
# full run is complete is not smoke-tested again. It ends with "DONE: <run folder>" or "STOPPED: <reason>".
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
CONF="$HERE/realdata.conf"; DESC=0; TOP=""; NAME=""; GPU_ARG=""; MODELS_ARG=""; SMOKE=1; FOREGROUND=0; REPORT_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --desc) DESC=1; shift ;;
        --top) TOP="${2:?--top needs K}"; shift 2 ;;
        --models) MODELS_ARG="${2:?--models needs a list}"; shift 2 ;;
        --gpu) GPU_ARG="${2:?--gpu needs an index}"; shift 2 ;;
        --name) NAME="${2:?--name needs a folder}"; shift 2 ;;
        --conf) CONF="${2:?--conf needs a file}"; shift 2 ;;
        --no-smoke) SMOKE=0; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --report-only) REPORT_ONLY=1; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
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
GPU="${GPU_ARG:-${GPU:-0}}"; RUN_MODELS="${MODELS_ARG:-${RUN_MODELS:-ELEPHANT DREX CLEF}}"; RUN_MODELS="${RUN_MODELS^^}"
SMOKE_STRIDE="${SMOKE_STRIDE:-100}"; DATA="${DATA:-}"; INDEX="${INDEX:-}"
if [ -n "$TOP" ] && ! [[ "$TOP" =~ ^[0-9]+$ && "$TOP" -ge 2 ]]; then stop "--top must be a whole number of 2 or more"; fi
for m in $RUN_MODELS; do [[ "$m" =~ ^(ELEPHANT|CLEF|DREX)$ ]] || stop "unknown model $m (ELEPHANT, CLEF, DREX)"; done

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
if [ -z "$NAME" ]; then
    NAME="results/real_$([ -n "$TOP" ] && echo "top$TOP" || echo whole_list)$([ "$DESC" = 1 ] && echo _desc || true)"
fi
case "$NAME" in /*) RUN="$NAME" ;; *) RUN="$HERE/$NAME" ;; esac
cd "$HERE"

# ======================================================================================================================
# Stage 1 (this terminal): checks and build. Stage 2 (background): smoke test, full runs, report.
if [ "${REALDATA_STAGE:-1}" = 1 ]; then
    [ "$REPORT_ONLY" = 1 ] && say "Report only: $RUN | models $RUN_MODELS" || say "Run folder $RUN | options: $([ "$DESC" = 1 ] && echo "code + description" || echo "bare codes"); $([ -n "$TOP" ] && echo "top $TOP" || echo "whole lists") | models $RUN_MODELS | GPU $GPU"
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
say "Stage 2 on $RUN (pid $$, GPU $GPU)"
partial=()

# run_model MODEL [--stride N]: one attempt plus one retry for requests without a result. Returns 0 only if all are ok.
run_model() {
    local m="$1"; shift
    local attempt out rc
    for attempt in 1 2; do
        out="$(mktemp)"
        set +e
        "$PY" -u run_decision_model.py run "$m" --gpu "$GPU" --run-dir "$RUN" --weights-dir "$MODELS" "$@" 2>&1 | tee "$out"
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
        smoked=0
        for m in $RUN_MODELS; do
            if [ -f "$RUN/predictions_${m}.jsonl" ]; then say "Smoke test: $m skipped (its full run has started before)"; continue; fi
            say "Smoke test: $m (every ${SMOKE_STRIDE}th request)"
            run_model "$m" --stride "$SMOKE_STRIDE" || stop "smoke test of $m did not finish cleanly (see above); fix it, then rerun"
            smoked=1
        done
        if [ "$smoked" = 1 ]; then
            say "Smoke report"
            "$PY" decision_eval.py report --run-dir "$RUN" --smoke || stop "smoke report failed (see above)"
        fi
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

say "Report"
"$PY" decision_eval.py report --run-dir "$RUN" || stop "report failed (see above)"
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
