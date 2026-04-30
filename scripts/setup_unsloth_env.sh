#!/bin/bash
# setup_unsloth_env.sh — Create a conda environment for Unsloth + DoRA training.
#
# Usage:
#   bash scripts/setup_unsloth_env.sh
#
# This creates a *separate* environment called 'dtgpt-unsloth' so that the
# existing 'dtgpt' environment (Python 3.8) remains untouched.

set -euo pipefail

ENV_NAME="${DTGPT_UNSLOTH_ENV_NAME:-dtgpt-unsloth}"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.1"

echo "=== Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} ==="

# Create a fresh environment
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

# Activate it inside this script
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch with CUDA ${CUDA_VERSION} ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

echo "=== Installing Unsloth ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes

echo "=== Installing project dependencies ==="
# Core dependencies that the DT-GPT pipeline needs (aligned with requirements.txt
# but at versions compatible with the newer Python and PyTorch).
pip install \
    "transformers>=4.41" \
    "datasets>=2.16" \
    "peft>=0.11" \
    "trl>=0.8" \
    "accelerate>=0.27" \
    "bitsandbytes>=0.43" \
    "safetensors>=0.4" \
    "sentencepiece>=0.1.99" \
    "wandb>=0.16" \
    "pandas>=1.5,<2" \
    "numpy>=1.22,<2" \
    "scipy>=1.7" \
    "scikit-learn>=1.2" \
    "matplotlib>=3.7" \
    "plotnine>=0.10" \
    "GPUtil>=1.4" \
    "evaluate>=0.4" \
    "flash-attn>=2.5" \
    "huggingface-hub>=0.19" \
    "tokenizers>=0.15"

echo "=== Installing DT-GPT in editable mode ==="
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pip install -e "${REPO_ROOT}"

echo ""
echo "=== Done! ==="
echo "Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Run training with --use-unsloth flag, e.g.:"
echo "  python 1_experiments/.../2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py --use-unsloth --use-dora"
