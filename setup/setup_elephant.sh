#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

PROJECT_ARG="${1:?usage: bash setup_elephant.sh MODELS_DIR (main.py setup-envs passes it)}"
mkdir -p "$PROJECT_ARG"
PROJECT_DIR="$(cd "$PROJECT_ARG" && pwd)"
# Exact revisions and the env name come from RealData's revisions.lock.json and lh_core.MODELS_INFO (main.py setup-envs exports them).
: "${CHECKPOINT_REVISION:?CHECKPOINT_REVISION is required}"
ENV_NAME="${ENV_NAME:-realdata-elephant}"
PYTHON_VERSION="3.11"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.8.1}"
TORCH_PINS=("torch==2.8.0" "torchvision==0.23.0")
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
LOG_DIR="$PROJECT_DIR/setup_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/elephant_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
on_error() { local rc=$?; echo "[$(date +%H:%M:%S)] FAILED (exit $rc) at line $1: $2"; echo "Full log: $LOG"; }
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
step() { echo; echo "[$(date +%H:%M:%S)] STEP: $*"; }
die() { echo "[$(date +%H:%M:%S)] STOPPED: $*"; echo "Full log: $LOG"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/checkpoint.sh"

step "Elephant-1 environment setup (project: $PROJECT_DIR, env: $ENV_NAME, log: $LOG)"

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
echo "pip: $(command -v pip)"

step "Installing the CUDA 12.8 toolkit into the env (nvcc is needed to compile causal-conv1d)"
# pin the version as well as the channel: newer conda/mamba otherwise take a newer toolkit from conda-forge
"$CONDA_INSTALLER" install -y -c "nvidia/label/$CUDA_LABEL" "cuda-toolkit=12.8" "cuda-version=12.8"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
NVCC_VERSION="$(nvcc --version)"
echo "$NVCC_VERSION"
[[ "$NVCC_VERSION" == *"release 12.8"* ]] || die "nvcc is not CUDA 12.8; set CUDA_LABEL to a cuda-12.8.x label"

step "Installing PyTorch 2.8 for CUDA 12.8 (the version in Elephant's reference image)"
python -m pip install --upgrade pip
python -m pip install "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu available', torch.cuda.is_available()); assert torch.__version__.startswith('2.8.'); assert torch.cuda.is_available(), 'driver too old for CUDA 12.8 or GPU not visible'"

step "Installing build helpers for compiled kernels"
python -m pip install ninja packaging wheel "huggingface_hub>=0.34"

get_checkpoint baselabs/elephant-1 elephant-1   # about 19 GB

step "Running the official setup.sh (OneJev + serve.patch, flash-linear-attention, causal-conv1d) in $PROJECT_DIR"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
bash "$PROJECT_DIR/elephant-1/setup.sh" "$PROJECT_DIR"

step "Installing helper packages within Elephant's pinned constraints"
python -m pip install -c "$PROJECT_DIR/elephant-1/constraints.txt" pandas matplotlib tokenizers

step "Verifying imports in a fresh interpreter"
python - <<'PY'
import torch
assert torch.__version__.startswith("2.8."), torch.__version__
assert torch.cuda.is_available()
from qev.engine import DecisionEngine
from qev.schema import SystemOneRequest
import qev, pandas, matplotlib, tokenizers
print("OK: torch", torch.__version__, "| qev from", qev.__file__)
PY

compare_tested

step "Done: env $ENV_NAME is ready; checkpoint at $PROJECT_DIR (pass it as --weights-dir)."
