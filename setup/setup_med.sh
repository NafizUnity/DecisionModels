#!/usr/bin/env bash
# The realdata-med environment (BioDecision v2 4B, MedDecider-27B, MedDecider-31B) and one model's checkpoints.
# Called through setup_biodecision.sh / setup_meddecider27.sh / setup_meddecider31.sh, which set MED_MODEL; main.py
# setup-envs passes CHECKPOINT_REVISION (and BASE_REVISION for a MedDecider adapter's base model).
# Package versions are those of GeminiOps/MedDecision's longhealth-med-hf env that ran these models on LongHealth
# (setup/tested/med.txt): torch 2.8.0 (CUDA 12.8), transformers 5.22.2, peft 0.21.2, flash-linear-attention 0.5.2.
# causal-conv1d is optional (fast path for the Qwen-based models); without it transformers uses its torch fallback.
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

PROJECT_ARG="${1:?usage: bash setup_<model>.sh MODELS_DIR (main.py setup-envs passes it)}"
mkdir -p "$PROJECT_ARG"
PROJECT_DIR="$(cd "$PROJECT_ARG" && pwd)"
: "${MED_MODEL:?MED_MODEL is required (setup_biodecision.sh / setup_meddecider27.sh / setup_meddecider31.sh set it)}"
: "${CHECKPOINT_REVISION:?CHECKPOINT_REVISION is required}"
ENV_NAME="${ENV_NAME:-realdata-med}"
PYTHON_VERSION="3.12"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
LOG_DIR="$PROJECT_DIR/setup_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/med_${MED_MODEL,,}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
on_error() { local rc=$?; echo "[$(date +%H:%M:%S)] FAILED (exit $rc) at line $1: $2"; echo "Full log: $LOG"; }
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
step() { echo; echo "[$(date +%H:%M:%S)] STEP: $*"; }
die() { echo "[$(date +%H:%M:%S)] STOPPED: $*"; echo "Full log: $LOG"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/checkpoint.sh"   # TESTED=setup/tested/med.txt

step "Medical decision model environment for $MED_MODEL (project: $PROJECT_DIR, env: $ENV_NAME, log: $LOG)"

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

step "Installing PyTorch 2.8.0 for CUDA 12.8 and what meddecider_infer.py names (transformers>=5.18, peft, flash-linear-attention)"
python -m pip install --upgrade pip
python -m pip install -c "$TESTED" "torch==2.8.0" --index-url "$TORCH_INDEX"
python -m pip install -c "$TESTED" "transformers==5.22.2" "peft==0.21.2" "accelerate==1.15.0" "flash-linear-attention==0.5.2" \
    safetensors tokenizers "huggingface_hub>=0.34" packaging ninja wheel --extra-index-url "$TORCH_INDEX"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu available', torch.cuda.is_available()); assert torch.__version__.startswith('2.8.'); assert torch.cuda.is_available(), 'driver too old for CUDA 12.8 or GPU not visible'"

step "Optional: causal-conv1d 1.7.0 (fast path for the Qwen-based models)"
if python -c "import causal_conv1d" 2>/dev/null; then
    echo "causal-conv1d already installed"
elif python -m pip install --no-build-isolation "causal-conv1d==1.7.0"; then
    echo "causal-conv1d installed"
else
    echo "WARNING: causal-conv1d could not be installed; the models run with transformers' torch fallback (slower, same maths up to bf16 rounding)."
fi

case "$MED_MODEL" in
    BIODECISION)
        get_checkpoint lighteternal/biodecision-v2-4b biodecision-v2-4b "" --exclude "assets/*" --exclude "*.png" --exclude "*.mp4"   # about 9 GB
        ;;
    MEDDECIDER27)
        : "${BASE_REVISION:?BASE_REVISION is required for the MedDecider-27B base model}"
        get_checkpoint lion-ai/MedDecider-27B meddecider-27b                     # LoRA adapter, 0.2 GB
        get_checkpoint Qwen/Qwen3.8-27B qwen3.8-27b "$BASE_REVISION"             # base model, about 52 GB
        ;;
    MEDDECIDER31)
        : "${BASE_REVISION:?BASE_REVISION is required for the MedDecider-31B base model}"
        get_checkpoint lion-ai/MedDecider-31B meddecider-31b                     # LoRA adapter, 0.3 GB
        get_checkpoint google/gemma-4-31b-it gemma-4-31b-it "$BASE_REVISION"     # base model, about 59 GB
        ;;
    *) die "unknown MED_MODEL $MED_MODEL" ;;
esac

step "Verifying imports in a fresh interpreter"
# no __pycache__ inside a checkpoint folder (it may be a link into a shared cache)
MED_MODEL="$MED_MODEL" PROJECT_DIR="$PROJECT_DIR" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python - <<'PY'
import os, sys
import torch, transformers, peft, fla  # noqa: F401
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.18"), transformers.__version__
assert torch.cuda.is_available()
m, project = os.environ["MED_MODEL"], os.environ["PROJECT_DIR"]
folder = {"BIODECISION": "biodecision-v2-4b", "MEDDECIDER27": "meddecider-27b", "MEDDECIDER31": "meddecider-31b"}[m]
sys.path.insert(0, os.path.join(project, folder))
mod = __import__("biodecision_server" if m == "BIODECISION" else "meddecider_infer")
print("OK: torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "|", mod.__name__, "letters", len(mod.LETTERS))
PY

compare_tested

step "Done: env $ENV_NAME is ready for $MED_MODEL; checkpoints at $PROJECT_DIR (pass it as --weights-dir)."
