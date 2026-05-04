#!/bin/bash
#SBATCH --job-name="dtgpt-install-unsloth"
#SBATCH --partition=cpu-2g
#SBATCH --account=cpu-2g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/install_dtgpt_unsloth_%j.out
#SBATCH --error=logs/install_dtgpt_unsloth_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="/share/home/r15543056/trajectory_forecast/DT-GPT"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"
cd "${REPO_ROOT}"
mkdir -p logs

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.1 || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

CONDA_ENV_NAME="${DTGPT_CONDA_ENV:-dtgpt-unsloth}"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PYTHON_BIN="${DTGPT_PYTHON_BIN:-python}"

echo "Conda env: ${CONDA_ENV_NAME}"
echo "Python: $("${PYTHON_BIN}" --version)"
echo "Python path: $(command -v "${PYTHON_BIN}")"
echo "Working directory: $(pwd)"
echo "Requirements: ${REQUIREMENTS_FILE}"

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "Requirements file not found: ${REQUIREMENTS_FILE}"
    exit 1
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel packaging
"${PYTHON_BIN}" -m pip install --prefer-binary -r "${REQUIREMENTS_FILE}"
"${PYTHON_BIN}" -m pip check

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import inspect
import sys

packages = [
    "torch",
    "triton",
    "unsloth",
    "unsloth_zoo",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "numpy",
    "pandas",
]

print("Python executable:", sys.executable)
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "NOT_INSTALLED"
    print(f"{package}: {version}")

import torch

print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

import unsloth
from peft import LoraConfig

if "use_dora" not in inspect.signature(LoraConfig).parameters:
    raise SystemExit("PEFT LoraConfig does not expose use_dora")

print("Unsloth import: OK")
print("PEFT DoRA support: OK")
PY

echo "dtgpt-unsloth dependency installation completed."
