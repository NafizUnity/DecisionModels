#!/usr/bin/env bash
# realdata_setup.sh: get this server ready to run RealData. Run once per server; safe to rerun.
#
#   bash realdata_setup.sh                 # check everything, install only what is missing (asks before downloading)
#   bash realdata_setup.sh --check-only    # check only; never installs or downloads anything
#   bash realdata_setup.sh --yes           # do not ask (for a non-interactive shell)
#   bash realdata_setup.sh --conf FILE     # settings file (default: realdata.conf next to this script)
#
# Every item is checked first and skipped when it is already in place:
#   GPU and driver, conda (Miniforge), and per model: its conda env, pinned core packages, imports, the checkpoint
#   (a verified link into HF_CACHE, never a download when HF_CACHE is set) and its GitHub runtime.
# Only when something is missing does it list what it would download, ask, check the network (certificates, mamba, pip)
# and run `main.py setup-envs` for the missing models. It ends with the data checks and one line: READY or NOT READY.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$HERE/realdata.conf"; YES=0; CHECK_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --conf) CONF="${2:?--conf needs a file}"; shift 2 ;;
        --yes) YES=1; shift ;;
        --check-only) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
done

if [ ! -f "$CONF" ]; then
    cp "$HERE/realdata.conf.example" "$CONF"
    echo "Created $CONF from the example. Edit its paths (BASE, HF_CACHE, DATA, INDEX, GPU), then rerun."
    exit 1
fi
# shellcheck disable=SC1090
source "$CONF"
: "${BASE:?BASE is not set in $CONF}"; MODELS="${MODELS:-$BASE/models}"
HF_CACHE="${HF_CACHE:-}"; DATA="${DATA:-}"; INDEX="${INDEX:-}"
RUN_MODELS="${RUN_MODELS:-ELEPHANT DREX CLEF BIODECISION MEDDECIDER27 MEDDECIDER31}"; RUN_MODELS="${RUN_MODELS^^}"
mkdir -p "$MODELS/setup_logs"
LOG="$MODELS/setup_logs/realdata_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "  ok    $*"; }
bad()  { echo "  MISSING  $*"; }
NOT_READY=()
not_ready() { NOT_READY+=("$*"); echo "  NOT READY  $*"; }
finish() {
    echo
    if [ ${#NOT_READY[@]} -eq 0 ]; then
        say "READY. Next: bash realdata_run.sh [--desc] [--top K]   (log: $LOG)"; exit 0
    fi
    say "NOT READY:"; printf '  - %s\n' "${NOT_READY[@]}"; echo "Log: $LOG"; exit 1
}
confirm() {
    [ "$YES" = 1 ] && return 0
    [ -t 0 ] || { echo "Confirmation needed: rerun in a terminal, or with --yes."; return 1; }
    local a; read -r -p "Proceed? Type yes to continue: " a; [ "$a" = yes ]
}
placeholder() { [[ -z "$1" || "$1" == /path/to/* ]]; }

# ---- A clean shell: what ~/DecisionModels/env.sh does on the Gemini cluster ------------------------------------------
clean_shell() {
    if declare -F module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; fi   # cluster python/R modules
    unset PYTHONHOME PYTHONPATH
    # conda envs bring their own libraries; keep only driver/CUDA folders (containers put libcuda there)
    local keep="" p; IFS=: read -r -a _parts <<<"${LD_LIBRARY_PATH:-}"
    for p in "${_parts[@]}"; do [[ "$p" == *nvidia* || "$p" == *cuda* ]] && keep="${keep:+$keep:}$p"; done
    export LD_LIBRARY_PATH="$keep"
    CERT=/etc/ssl/certs/ca-certificates.crt   # the network re-signs TLS: trust the system certificates
    if [ -f "$CERT" ]; then
        export SSL_CERT_FILE="$CERT" REQUESTS_CA_BUNDLE="$CERT" CURL_CA_BUNDLE="$CERT" PIP_CERT="$CERT" CONDA_SSL_VERIFY="$CERT"
    else CERT=""; fi
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"   # Elephant's kernel for A100 (8.0) and H100 (9.0)
    export PIP_CACHE_DIR="$BASE/cache/pip" TRITON_CACHE_DIR="$BASE/cache/triton" PYTHONNOUSERSITE=1
}
find_conda() {
    local c
    for c in "$BASE/miniforge3/bin/conda" "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" \
             "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda"; do
        [ -n "$c" ] && [ -x "$c" ] && "$c" --version >/dev/null 2>&1 && { echo "$c"; return 0; }
    done
    return 1
}

# as lh_core.MODELS_INFO; the three medical models share one env, realdata-med (setup/setup_med.sh)
declare -A ENV_OF=([ELEPHANT]=realdata-elephant [CLEF]=realdata-clef [DREX]=realdata-drex
                   [BIODECISION]=realdata-med [MEDDECIDER27]=realdata-med [MEDDECIDER31]=realdata-med)
declare -A WDIR=([ELEPHANT]=elephant-1 [CLEF]=clef [DREX]=drex-max-v1
                 [BIODECISION]=biodecision-v2-4b [MEDDECIDER27]=meddecider-27b [MEDDECIDER31]=meddecider-31b)
declare -A REPO=([ELEPHANT]=baselabs/elephant-1 [CLEF]=Cloudflare/clef [DREX]=nace-ai/drex-max-v1
                 [BIODECISION]=lighteternal/biodecision-v2-4b [MEDDECIDER27]=lion-ai/MedDecider-27B [MEDDECIDER31]=lion-ai/MedDecider-31B)
declare -A BASE_REPO=([MEDDECIDER27]=Qwen/Qwen3.8-27B [MEDDECIDER31]=google/gemma-4-31b-it)   # MedDecider: LoRA adapter + base model
declare -A BASE_DIR=([MEDDECIDER27]=qwen3.8-27b [MEDDECIDER31]=gemma-4-31b-it)
GENERAL_MODELS="ELEPHANT CLEF DREX"; ALL_MODELS="$GENERAL_MODELS BIODECISION MEDDECIDER27 MEDDECIDER31"
lock() { "$PYB" -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]].get(sys.argv[3], ""))' \
           "$HERE/revisions.lock.json" "$1" "$2"; }
env_python() {
    "$CONDA" env list --json | "$PYB" -c 'import json,os,sys
n = sys.argv[1]
print(next((e + "/bin/python" for e in json.load(sys.stdin)["envs"] if os.path.basename(e) == n and os.path.exists(e + "/bin/python")), ""))' "$1"
}

# Is model $1 ready? Prints one line per check; returns 1 at the first missing piece.
model_ready() {
    local m="$1" py rev cache_arg=()
    py="$(env_python "${ENV_OF[$m]}")"
    if [ -z "$py" ]; then bad "$m: conda env ${ENV_OF[$m]}"; return 1; fi
    ok "$m: conda env ${ENV_OF[$m]}"
    rev="$(lock "$m" revision)"
    [ -n "$HF_CACHE" ] && cache_arg=(--cache "$HF_CACHE")
    if ! "$PYB" "$HERE/setup/shared_cache.py" check --repo "${REPO[$m]}" --revision "$rev" --dest "$MODELS/${WDIR[$m]}" "${cache_arg[@]}" \
         | sed 's/^/        /'; then
        bad "$m: checkpoint ${REPO[$m]}@${rev:0:8} at $MODELS/${WDIR[$m]}"; return 1
    fi
    ok "$m: checkpoint ${REPO[$m]}@${rev:0:8}"
    if [ -n "${BASE_REPO[$m]:-}" ]; then
        local brev; brev="$(lock "$m" base_revision)"
        if ! "$PYB" "$HERE/setup/shared_cache.py" check --repo "${BASE_REPO[$m]}" --revision "$brev" --dest "$MODELS/${BASE_DIR[$m]}" \
             "${cache_arg[@]}" | sed 's/^/        /'; then
            bad "$m: base model ${BASE_REPO[$m]}@${brev:0:8} at $MODELS/${BASE_DIR[$m]}"; return 1
        fi
        ok "$m: base model ${BASE_REPO[$m]}@${brev:0:8}"
    fi
    if [ "$m" = DREX ]; then
        local want have; want="$(lock DREX runtime_revision)"
        have="$(git -C "$MODELS/drex-decision-models" rev-parse HEAD 2>/dev/null || true)"
        if [ "$have" != "$want" ]; then bad "$m: runtime drex-decision-models at ${want:0:8} (found: ${have:-none})"; return 1; fi
        ok "$m: runtime drex-decision-models @ ${want:0:8}"
    fi
    local tested="${m,,}"; [ "${ENV_OF[$m]}" = realdata-med ] && tested=med
    if ! MODEL="$m" MODELS_DIR="$MODELS" TESTED="$HERE/setup/tested/$tested.txt" PYTHONDONTWRITEBYTECODE=1 \
         "$py" - <<'PY' 2>&1 | sed 's/^/        /'
import os, re, subprocess, sys
m, models = os.environ["MODEL"], os.environ["MODELS_DIR"]
def pins(lines):
    out = {}
    for line in lines:
        hit = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if hit:
            out[re.sub(r"[-_.]+", "-", hit.group(1)).lower()] = hit.group(2)
    return out
tested = pins(open(os.environ["TESTED"]))
installed = pins(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines())
core = {"torch", "torchvision", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d", "accelerate", "safetensors", "tokenizers"}
diff = [(n, v, installed.get(n, "missing")) for n, v in sorted(tested.items()) if installed.get(n) != v]
core_diff = [d for d in diff if d[0] in core]
for n, want, got in core_diff:
    print(f"core package {n}: tested {want}, installed {got}")
print(f"{len(tested) - len(diff)} of {len(tested)} tested package versions match")
if core_diff:
    sys.exit(1)
import torch
assert torch.cuda.is_available(), "torch sees no GPU (driver too old for this build, or no GPU visible)"
if m == "ELEPHANT":
    from qev.engine import DecisionEngine  # noqa: F401  (OneJev runtime)
elif m == "CLEF":
    sys.path.insert(0, os.path.join(models, "clef"))
    from joint_schema_model import load_release_model  # noqa: F401
elif m == "DREX":
    sys.path.insert(0, os.path.join(models, "drex-decision-models"))
    from drex_max.model import DecisionModel  # noqa: F401
else:   # BIODECISION, MEDDECIDER27, MEDDECIDER31: the model repo's own inference module
    import peft, fla  # noqa: F401,E401
    folder = {"BIODECISION": "biodecision-v2-4b", "MEDDECIDER27": "meddecider-27b", "MEDDECIDER31": "meddecider-31b"}[m]
    sys.path.insert(0, os.path.join(models, folder))
    __import__("biodecision_server" if m == "BIODECISION" else "meddecider_infer")
print("imports ok")
PY
    then bad "$m: packages/imports in ${ENV_OF[$m]}"; return 1; fi
    ok "$m: pinned core packages and imports"
}

net_checks() {
    say "Checking the network (certificates, mamba, pip)"
    local code tmp
    code="$(curl -sSL -o /dev/null -w '%{http_code}' https://prefix.dev/conda-forge/noarch/repodata.json.zst 2>/dev/null || true)"
    case "$code" in 200|30?) ok "https to prefix.dev ($code)" ;; *) bad "https to prefix.dev failed (code '$code'): certificates or proxy"; return 1 ;; esac
    if [ -n "$CONDA" ]; then
        if "$INSTALLER" create --dry-run -n realdata-net-test python=3.11 >/dev/null 2>&1; then ok "$(basename "$INSTALLER") can resolve packages"
        else bad "$(basename "$INSTALLER") create --dry-run failed (SSL? ssl_verify in $(dirname "$(dirname "$CONDA")")/.condarc)"; return 1; fi
    fi
    tmp="$(mktemp -d)"
    if "$PYB" -m pip download -q --no-deps -d "$tmp" six >/dev/null 2>&1; then ok "pip can download"; rm -rf "$tmp"
    else rm -rf "$tmp"; bad "pip download failed (PIP_CERT / proxy)"; return 1; fi
}

# ======================================================================================================================
clean_shell
say "RealData setup check (settings: $CONF; models folder: $MODELS; log: $LOG)"

say "GPU and driver"
if ! command -v nvidia-smi >/dev/null; then not_ready "nvidia-smi not found: no NVIDIA driver on this node"; finish; fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/        /'
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
MAXMEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)"
if [ "${DRIVER:-0}" -ge 580 ]; then ok "driver $DRIVER (580+ needed)"; else not_ready "NVIDIA driver $DRIVER is older than 580 (Clef and Drex need CUDA 13)"; fi
if [ "${MAXMEM:-0}" -lt 22000 ]; then not_ready "largest GPU has ${MAXMEM} MiB; every model needs 22-60 GB (use an A100/H100 node)"
elif [ "$MAXMEM" -lt 60000 ]; then echo "  WARN  largest GPU has ${MAXMEM} MiB: enough for Elephant only (Clef 60 GB, Drex 58 GB)"
else ok "largest GPU ${MAXMEM} MiB"; fi

say "conda"
CONDA="$(find_conda || true)"
if [ -n "$CONDA" ]; then ok "$CONDA ($("$CONDA" --version))"; NEED_MINIFORGE=0
else bad "conda (Miniforge would go to $BASE/miniforge3)"; NEED_MINIFORGE=1; fi
if [ -n "$CONDA" ]; then
    export PATH="$(dirname "$CONDA"):$PATH"; PYB="$(dirname "$CONDA")/python"
    INSTALLER="$(dirname "$CONDA")/mamba"; [ -x "$INSTALLER" ] || INSTALLER="$CONDA"
else PYB="$(command -v python3)"; INSTALLER=""; fi
[ -n "$HF_CACHE" ] && { [ -d "$HF_CACHE" ] && ok "shared Hugging Face cache $HF_CACHE" || not_ready "HF_CACHE $HF_CACHE is not a folder"; }

say "Models: ${RUN_MODELS}"
TODO=()
for m in $RUN_MODELS; do
    [ -n "${ENV_OF[$m]:-}" ] || { not_ready "unknown model $m in RUN_MODELS ($ALL_MODELS)"; continue; }
    if [ -n "$CONDA" ] && model_ready "$m"; then :; else TODO+=("$m"); fi
done

if [ "$NEED_MINIFORGE" = 1 ] || [ ${#TODO[@]} -gt 0 ]; then
    echo
    say "To install: $([ "$NEED_MINIFORGE" = 1 ] && echo -n "Miniforge; ")models ${TODO[*]:-none}"
    echo "  This downloads public open-source software only (nothing is uploaded):"
    [ "$NEED_MINIFORGE" = 1 ] && echo "    github.com/conda-forge/miniforge      Miniforge installer -> $BASE/miniforge3"
    echo "    prefix.dev, conda.anaconda.org          Python, build tools, CUDA 12.8 toolkit (Elephant)"
    echo "    pypi.org, download.pytorch.org          PyTorch, transformers and other pinned packages"
    echo "    github.com                              OneJev (Elephant), drex-decision-models (Drex), pinned commits"
    if [ -n "$HF_CACHE" ]; then echo "    (no model weights: checkpoints are linked from $HF_CACHE after verification)"
    else echo "    huggingface.co                          the pinned checkpoints into $MODELS (about 120 GB general, 128 GB medical)"; fi
fi
if [ "$CHECK_ONLY" = 1 ] && { [ "$NEED_MINIFORGE" = 1 ] || [ ${#TODO[@]} -gt 0 ]; }; then
    not_ready "missing: $([ "$NEED_MINIFORGE" = 1 ] && echo -n "Miniforge ")${TODO[*]:-} (rerun without --check-only to install)"
elif [ "$NEED_MINIFORGE" = 1 ] || [ ${#TODO[@]} -gt 0 ]; then
    confirm || { not_ready "installation declined"; finish; }

    if [ "$NEED_MINIFORGE" = 1 ]; then
        say "Installing Miniforge into $BASE/miniforge3"
        tmp="$(mktemp -d)"; mkdir -p "$BASE"
        curl -fsSL -o "$tmp/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
            || { rm -rf "$tmp"; not_ready "Miniforge download failed (certificates or proxy)"; finish; }
        bash "$tmp/Miniforge3.sh" -b -p "$BASE/miniforge3"; rm -rf "$tmp"   # -b: no questions, ~/.bashrc untouched
        CONDA="$BASE/miniforge3/bin/conda"; export PATH="$BASE/miniforge3/bin:$PATH"; PYB="$BASE/miniforge3/bin/python"
        INSTALLER="$BASE/miniforge3/bin/mamba"; [ -x "$INSTALLER" ] || INSTALLER="$CONDA"
        [ ${#TODO[@]} -gt 0 ] || TODO=($RUN_MODELS)
    fi
    # mamba ignores CONDA_SSL_VERIFY: put the certificate in the conda root's own config (Gemini cluster fix)
    CONDARC="$(dirname "$(dirname "$CONDA")")/.condarc"
    if [ -n "$CERT" ] && ! grep -qs '^ssl_verify:' "$CONDARC"; then
        "$CONDA" config --file "$CONDARC" --set ssl_verify "$CERT"; ok "ssl_verify -> $CERT in $CONDARC"
    fi
    net_checks || { not_ready "network check failed (see above)"; finish; }

    say "Installing models: ${TODO[*]}"
    cache_arg=(); [ -n "$HF_CACHE" ] && cache_arg=(--hf-cache "$HF_CACHE")
    # one model at a time, so a model that cannot be set up (e.g. a base model missing from HF_CACHE) does not stop the rest
    for m in "${TODO[@]}"; do
        say "Setting up $m"
        if ! "$PYB" -u "$HERE/main.py" --project-dir "$MODELS" setup-envs --models "${m,,}" "${cache_arg[@]}"; then
            not_ready "$m: setup stopped (see above; its log is in $MODELS/setup_logs); the other models continue"; continue
        fi
        model_ready "$m" || not_ready "$m is still not ready after setup (see above)"
    done
fi

say "Data"
if placeholder "$DATA"; then echo "  WARN  DATA is not set in $CONF (needed by realdata_run.sh)"
elif [ ! -f "$DATA" ]; then not_ready "DATA $DATA not found"
elif "$PYB" "$HERE/inspect_rankings.py" "$DATA" | sed 's/^/        /'; then ok "DATA $DATA (counts and codes only above)"
else not_ready "inspect_rankings.py could not read $DATA"; fi
if placeholder "$INDEX"; then echo "  note  INDEX not set: bare-code runs only (realdata_run.sh --desc needs it)"
elif [ ! -f "$INDEX" ]; then not_ready "INDEX $INDEX not found"
elif [ ${#NOT_READY[@]} -gt 0 ] || placeholder "$DATA"; then echo "  skip  description check (fix the items above first)"
else
    say "Checking the description index (coverage, duplicates, tokens per model; CPU only)"
    # token counts use each general model's own prompt code; the medical models' prompts are short and checked at load
    desc_models=""; for m in $RUN_MODELS; do [[ " $GENERAL_MODELS " == *" $m "* ]] && desc_models+="$m "; done
    [ -n "$desc_models" ] || desc_models="$GENERAL_MODELS"
    # shellcheck disable=SC2086
    if "$PYB" "$HERE/check_descriptions.py" --data "$DATA" --index "$INDEX" --weights-dir "$MODELS" --models $desc_models \
         --stride 20 --out "$MODELS/setup_logs/desc_check.json"; then ok "INDEX $INDEX"
    else not_ready "check_descriptions.py reported a FAIL (see above; counts in $MODELS/setup_logs/desc_check.json)"; fi
fi
finish
