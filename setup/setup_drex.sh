#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

PROJECT_ARG="${1:?usage: bash setup_drex.sh MODELS_DIR (main.py setup-envs passes it)}"
mkdir -p "$PROJECT_ARG"
PROJECT_DIR="$(cd "$PROJECT_ARG" && pwd)"
# Exact revisions and the env name come from RealData's revisions.lock.json and lh_core.MODELS_INFO (main.py setup-envs exports them).
: "${CHECKPOINT_REVISION:?CHECKPOINT_REVISION is required}"
: "${RUNTIME_REVISION:?RUNTIME_REVISION is required}"
ENV_NAME="${ENV_NAME:-realdata-drex}"
PYTHON_VERSION="3.12"
DREX_REPO="$PROJECT_DIR/drex-decision-models"
LOG_DIR="$PROJECT_DIR/setup_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/drex_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
on_error() { local rc=$?; echo "[$(date +%H:%M:%S)] FAILED (exit $rc) at line $1: $2"; echo "Full log: $LOG"; }
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
step() { echo; echo "[$(date +%H:%M:%S)] STEP: $*"; }
die() { echo "[$(date +%H:%M:%S)] STOPPED: $*"; echo "Full log: $LOG"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/checkpoint.sh"

step "Drex Max environment setup (project: $PROJECT_DIR, env: $ENV_NAME, log: $LOG)"

step "Checking prerequisites"
command -v git >/dev/null || die "git is required"
command -v conda >/dev/null || die "conda is required (Miniforge or Miniconda)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; an NVIDIA driver is required"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if command -v mamba >/dev/null; then CONDA_INSTALLER=mamba; else CONDA_INSTALLER=conda; fi
eval "$(conda shell.bash hook)"

step "Creating or reusing conda env $ENV_NAME (Python $PYTHON_VERSION)"
ENV_LIST="$(conda env list | awk '{print $1}')"   # captured first: grep -q closing the pipe early would trip pipefail
if grep -qx "$ENV_NAME" <<<"$ENV_LIST"; then
    echo "Env $ENV_NAME already exists; reusing it"
else
    "$CONDA_INSTALLER" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
set +u
conda activate "$ENV_NAME"
set -u
echo "python: $(command -v python)"
python --version

step "Cloning the Drex repository into $DREX_REPO"
if [ ! -d "$DREX_REPO/.git" ]; then
    git clone https://github.com/nace-ai/drex-decision-models.git "$DREX_REPO"
fi
git -C "$DREX_REPO" checkout --quiet "$RUNTIME_REVISION"
git -C "$DREX_REPO" log -1 --format="Drex repository commit %h (%cs)"

step "Installing Drex Max's own requirements.txt (torch>=2.14, transformers 5.17.0, flash-linear-attention)"
python -m pip install --upgrade pip
python -m pip install -c "$TESTED" -r "$DREX_REPO/models/drex-max-v1/requirements.txt"   # its own >= ranges, pinned to the tested versions

python -m pip install -c "$TESTED" "huggingface_hub>=0.34"
get_checkpoint nace-ai/drex-max-v1 drex-max-v1   # about 52 GB

step "Installing helper packages"
python -m pip install -c "$TESTED" pandas matplotlib tokenizers

step "Verifying imports in a fresh interpreter"
DREX_REPO="$DREX_REPO" python - <<'PY'
import os, sys
import torch, transformers
major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert major_minor >= (2, 14), torch.__version__
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.cuda.is_available(), "driver too old for this PyTorch build or GPU not visible"
sys.path.insert(0, os.environ["DREX_REPO"])
from drex_max.model import DecisionModel, answer
import fla, pandas, matplotlib, tokenizers
print("OK: torch", torch.__version__, "| transformers", transformers.__version__)
PY

compare_tested

step "Done: env $ENV_NAME is ready; checkpoint at $PROJECT_DIR (pass it as --weights-dir)."
