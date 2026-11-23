#!/usr/bin/env bash
# picker_run.sh: the LLM picker on a finished decision-model (DM) run: build, run the LLM, report, in one command.
#
#   bash picker_run.sh --dm-run real_whole_list_desc --dms "DREX MEDDECIDER31" \
#        --sources "rrf_csv+fsn rrf_fsn+syn rrf_csv+fsn+syn" --ks "3,5,10,20"
#   bash picker_run.sh --dm-run real_whole_list_desc --dms DREX --sources rrf_fsn+syn --ks 3 --limit 2   # smoke test
#
#   The LLM sees each DM's top K codes (sorted by the DM's probabilities), without the clinical note. A K above what a
#   DM read (MedDecider-31B: 10) reads N/A. Every source x DM x K is one picker config; all run in one model load.
#   options: --dm-run NAME (a folder under DM_RESULTS_DIR, or a full path; required)  --dms "DREX MEDDECIDER31"
#            --sources "A B" (default: all)  --ks "3,5,10,20"  --format native|requests (default native)
#            --stop-tokens tokenizer_eos (the original script's generation; default: the model's own end tokens)
#            --batch-size N (entities per generate call; default 1)
#            --limit N (first N entities: smoke test)  --name NAME (picker folder under PICKER_RESULTS_DIR, or a full path)
#            --gpu N  --conf FILE  --foreground  --report-only  --validate-only (build and check, no model)
#
# Paths come from picker.conf (cp picker.conf.example picker.conf). Build and checks run in this terminal; the LLM runs
# continue in the background (nohup) unless --foreground, and the script prints the `tail -f` command for the log.
# Rerunning the same command resumes: the build is reused when its settings are unchanged and finished configs are
# skipped. The report is printed and saved as report_<UTC time>.md/.csv (never overwritten).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$HERE/picker.conf"; DM_RUN=""; DMS="DREX MEDDECIDER31"; SOURCES_ARG=""; KS_ARG="3,5,10,20"; FORMAT="native"
STOP_TOKENS="generation_config"; LIMIT=""; NAME=""; GPU_ARG=""; FOREGROUND=0; REPORT_ONLY=0; VALIDATE_ONLY=0; BATCH=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dm-run) DM_RUN="${2:?--dm-run needs a folder}"; shift 2 ;;
        --dms) DMS="${2:?--dms needs a list}"; shift 2 ;;
        --sources) SOURCES_ARG="${2:?--sources needs a list}"; shift 2 ;;
        --ks) KS_ARG="${2:?--ks needs a list such as 3,5,10,20}"; shift 2 ;;
        --format) FORMAT="${2:?--format needs native or requests}"; shift 2 ;;
        --stop-tokens) STOP_TOKENS="${2:?--stop-tokens needs generation_config or tokenizer_eos}"; shift 2 ;;
        --limit) LIMIT="${2:?--limit needs a number}"; shift 2 ;;
        --batch-size) BATCH="${2:?--batch-size needs a number}"; shift 2 ;;
        --name) NAME="${2:?--name needs a folder}"; shift 2 ;;
        --gpu) GPU_ARG="${2:?--gpu needs an index}"; shift 2 ;;
        --conf) CONF="${2:?--conf needs a file}"; shift 2 ;;
        --foreground) FOREGROUND=1; shift ;;
        --report-only) REPORT_ONLY=1; shift ;;
        --validate-only) VALIDATE_ONLY=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
done
stop() { echo "[$(date +%H:%M:%S)] STOPPED: $*"; exit 1; }
say()  { echo "[$(date +%H:%M:%S)] $*"; }
placeholder() { [[ -z "$1" || "$1" == /path/to/* ]]; }

[ -f "$CONF" ] || stop "$CONF not found: cp picker.conf.example picker.conf and edit its paths"
# shellcheck disable=SC1090
source "$CONF"
for v in LLM_MODEL LLM_ENV DM_RESULTS_DIR PICKER_RESULTS_DIR; do
    placeholder "${!v:-}" && stop "$v is not set in $CONF"
done
GPU="${GPU_ARG:-${GPU:-0}}"
[ -n "$DM_RUN" ] || stop "--dm-run is required (a DM run folder under $DM_RESULTS_DIR)"
if [ -n "$LIMIT" ] && ! [[ "$LIMIT" =~ ^[0-9]+$ && "$LIMIT" -ge 1 ]]; then stop "--limit must be a whole number of 1 or more"; fi
[[ "$FORMAT" =~ ^(native|requests)$ ]] || stop "--format must be native or requests"
[[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -ge 1 ]] || stop "--batch-size must be a whole number of 1 or more"
[[ "$STOP_TOKENS" =~ ^(generation_config|tokenizer_eos)$ ]] || stop "--stop-tokens must be generation_config or tokenizer_eos"

# ---- the same clean, offline shell as realdata_run.sh -------------------------------------------------------------------
if declare -F module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; fi
unset PYTHONHOME PYTHONPATH
keep=""; IFS=: read -r -a _parts <<<"${LD_LIBRARY_PATH:-}"
for p in "${_parts[@]}"; do if [[ "$p" == *nvidia* || "$p" == *cuda* ]]; then keep="${keep:+$keep:}$p"; fi; done
export LD_LIBRARY_PATH="$keep" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONDA_BIN=""   # CONDA in picker.conf, else the conda that is active or on PATH
for c in "${CONDA:-}" "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then CONDA_BIN="$c"; break; fi
done
[ -n "$CONDA_BIN" ] || stop "conda not found: set CONDA in $CONF (the conda that realdata_setup.sh installed)"
CONDA="$CONDA_BIN"
PY="$("$CONDA" env list --json | python3 -c 'import json,os,sys
n = sys.argv[1]
print(next((e + "/bin/python" for e in json.load(sys.stdin)["envs"] if os.path.basename(e) == n and os.path.exists(e + "/bin/python")), ""))' "$LLM_ENV")"
[ -n "$PY" ] || stop "conda env $LLM_ENV not found (LLM_ENV in $CONF); for realdata-med run realdata_setup.sh with MEDDECIDER31"
cd "$HERE"

case "$DM_RUN" in /*) DM_DIR="$DM_RUN" ;; *) DM_DIR="$DM_RESULTS_DIR/$DM_RUN" ;; esac
[ -f "$DM_DIR/requests.jsonl" ] || stop "$DM_DIR has no requests.jsonl (is --dm-run a DM run folder?)"
if [ -z "$NAME" ]; then
    NAME="$(basename "$DM_DIR")__llm$([ "$FORMAT" = requests ] && echo _requests || true)$([ -n "$LIMIT" ] && echo "_smoke$LIMIT" || true)"
fi
case "$NAME" in /*) OUT="$NAME" ;; *) OUT="$PICKER_RESULTS_DIR/$NAME" ;; esac

if [ "$REPORT_ONLY" = 1 ]; then
    [ -f "$OUT/build_meta.json" ] || stop "--report-only: $OUT has no build"
    "$PY" picker_report.py --picker-dir "$OUT" --save; exit 0
fi

# ---- stage 1 (this terminal): build and validate -----------------------------------------------------------------------
say "DM run $DM_DIR | DMs $DMS | sources ${SOURCES_ARG:-all} | K $KS_ARG | format $FORMAT | no clinical note | GPU $GPU"
build=("$PY" picker_build.py --dm-run "$DM_DIR" --out "$OUT" --llm-model "$LLM_MODEL" --dms "$DMS" --ks "$KS_ARG"
       --format "$FORMAT" --stop-tokens "$STOP_TOKENS")
[ -n "$SOURCES_ARG" ] && build+=(--sources "$SOURCES_ARG")
[ -n "$LIMIT" ] && build+=(--limit "$LIMIT")
"${build[@]}" || stop "build failed (see above)"
mapfile -t CONFIGS < <(ls "$OUT"/configs/*.yaml 2>/dev/null)
[ "${#CONFIGS[@]}" -gt 0 ] || stop "the build wrote no picker configs"
"$PY" llm_picker.py --validate-only --config "${CONFIGS[@]}" | grep -c "validated" | xargs -I{} echo "[$(date +%H:%M:%S)] {} configs validated"
todo=0; for c in "${CONFIGS[@]}"; do f="$OUT/predictions/$(basename "${c%.yaml}").jsonl"; [ -f "$f" ] || todo=$((todo + 1)); done
say "$todo of ${#CONFIGS[@]} configs to run ($(( ${#CONFIGS[@]} - todo )) done before)"
[ "$VALIDATE_ONLY" = 1 ] && { say "DONE (validate only): $OUT"; exit 0; }

# ---- stage 2: the LLM runs (one model load for all configs) and the report ---------------------------------------------
LOG="$OUT/picker_$(date -u +%Y%m%dT%H%M%SZ).log"
stage2=(env CUDA_VISIBLE_DEVICES="$GPU" "$PY" llm_picker.py --batch-size "$BATCH" --config "${CONFIGS[@]}")
report=("$PY" picker_report.py --picker-dir "$OUT" --save)
if [ "$FOREGROUND" = 1 ]; then
    "${stage2[@]}" 2>&1 | tee "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] || stop "llm_picker.py failed (see $LOG)"
    "${report[@]}" | tee -a "$LOG"; say "DONE: $OUT"
else
    nohup bash -c "$(printf '%q ' "${stage2[@]}") && $(printf '%q ' "${report[@]}") && echo DONE: $(printf '%q' "$OUT") || echo STOPPED: see above" \
        > "$LOG" 2>&1 < /dev/null &
    say "LLM runs continue in the background (pid $!). Follow them with: tail -f $LOG"
fi
