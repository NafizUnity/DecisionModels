#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

PROJECT_ARG="${1:?usage: bash setup_clef.sh MODELS_DIR (main.py setup-envs passes it)}"
mkdir -p "$PROJECT_ARG"
PROJECT_DIR="$(cd "$PROJECT_ARG" && pwd)"
# Exact revisions and the env name come from RealData's revisions.lock.json and lh_core.MODELS_INFO (main.py setup-envs exports them).
: "${CHECKPOINT_REVISION:?CHECKPOINT_REVISION is required}"
ENV_NAME="${ENV_NAME:-realdata-clef}"
PYTHON_VERSION="3.11"
LOG_DIR="$PROJECT_DIR/setup_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/clef_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
on_error() { local rc=$?; echo "[$(date +%H:%M:%S)] FAILED (exit $rc) at line $1: $2"; echo "Full log: $LOG"; }
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
step() { echo; echo "[$(date +%H:%M:%S)] STEP: $*"; }
die() { echo "[$(date +%H:%M:%S)] STOPPED: $*"; echo "Full log: $LOG"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/checkpoint.sh"

step "Clef environment setup (project: $PROJECT_DIR, env: $ENV_NAME, log: $LOG)"

step "Checking prerequisites"
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

step "Installing PyTorch 2.11 and Transformers 5.10.2 (Clef's documented versions)"
python -m pip install --upgrade pip
# torchvision: required by the Qwen3-VL processor Clef loads
python -m pip install -c "$TESTED" "torch==2.11.*" "torchvision==0.26.*" "transformers==5.10.2" accelerate safetensors "huggingface_hub>=0.34" tokenizers
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu available', torch.cuda.is_available()); assert torch.__version__.startswith('2.11.'); assert torch.cuda.is_available(), 'driver too old for this PyTorch build or GPU not visible'"

step "Installing helper packages"
python -m pip install -c "$TESTED" pandas matplotlib

get_checkpoint Cloudflare/clef clef   # about 55 GB

step "Verifying imports in a fresh interpreter"
# no __pycache__ inside the checkpoint folder (it may be a link into a shared cache)
CLEF_DIR="$PROJECT_DIR/clef" PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import os, sys
import torch, torchvision, transformers
assert torch.__version__.startswith("2.11."), torch.__version__
assert transformers.__version__ == "5.10.2", transformers.__version__
assert torch.cuda.is_available()
sys.path.insert(0, os.environ["CLEF_DIR"])
from joint_schema_model import encode_record, collate_records, load_release_model
import pandas, matplotlib, tokenizers
print("OK: torch", torch.__version__, "| transformers", transformers.__version__)
PY

compare_tested

step "Done: env $ENV_NAME is ready; checkpoint at $PROJECT_DIR (pass it as --weights-dir)."
