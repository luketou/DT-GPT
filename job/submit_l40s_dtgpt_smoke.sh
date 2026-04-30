#!/bin/bash
#SBATCH --job-name="dtgpt-l40s-smoke"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/l40s_smoke_%j.out
#SBATCH --error=logs/l40s_smoke_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

mkdir -p logs

if command -v module >/dev/null 2>&1; then
    module load cuda/12.1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-.}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt}"
export DTGPT_BIOMISTRAL_MODEL_PATH="${DTGPT_BIOMISTRAL_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_TOKENIZER_MODEL_PATH="${DTGPT_TOKENIZER_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export HF_HOME="${HF_HOME:-/tmp/dtgpt_runtime_cache/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export DTGPT_RUN_COMPILEALL="${DTGPT_RUN_COMPILEALL:-0}"
mkdir -p "${HF_HOME}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${DTGPT_CONDA_ENV}"
else
    echo "[ERROR] conda is not available on this node."
    exit 1
fi

echo "[ENV] python=$(which python)"
echo "[ENV] conda_env=${DTGPT_CONDA_ENV}"
echo "[ENV] hostname=$(hostname)"
echo "[ENV] model_path=${DTGPT_BIOMISTRAL_MODEL_PATH}"
echo "[ENV] tokenizer_path=${DTGPT_TOKENIZER_MODEL_PATH}"
echo "[ENV] hf_home=${HF_HOME}"

if [ "${DTGPT_RUN_COMPILEALL}" = "1" ]; then
    python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv >/dev/null
fi

python - <<'PY'
import os
import json
from pathlib import Path

import torch

model_path = Path(os.environ["DTGPT_BIOMISTRAL_MODEL_PATH"])
tokenizer_path = Path(os.environ["DTGPT_TOKENIZER_MODEL_PATH"])

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

print(f"Model path exists: {model_path.exists()}")
print(f"Tokenizer path exists: {tokenizer_path.exists()}")

config_path = model_path / "config.json"
tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
weights_index_path = model_path / "model.safetensors.index.json"

with config_path.open("r", encoding="utf-8") as fh:
    config = json.load(fh)
with tokenizer_config_path.open("r", encoding="utf-8") as fh:
    tokenizer_config = json.load(fh)

print(f"Config model_type: {config.get('model_type')}")
print(f"Tokenizer class: {tokenizer_config.get('tokenizer_class')}")
print(f"Weights index exists: {weights_index_path.exists()}")

print("L40S smoke check passed.")
PY
