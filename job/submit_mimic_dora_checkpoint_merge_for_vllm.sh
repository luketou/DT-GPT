#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-merge"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/mimic_dora_merge_%j.out
#SBATCH --error=logs/mimic_dora_merge_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.1
fi

export TOKENIZERS_PARALLELISM=false
export DTGPT_BIOMISTRAL_MODEL_PATH="${DTGPT_BIOMISTRAL_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_RUNTIME_CACHE_ROOT="${DTGPT_RUNTIME_CACHE_ROOT:-/tmp/dtgpt_runtime_cache}"
export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-eager}"
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
unset TRANSFORMERS_CACHE
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate dtgpt-vllm
    PYTHON_BIN="$(command -v python)"
else
    echo "conda is not available in this job environment."
    exit 1
fi
unset TRANSFORMERS_CACHE

# CHECKPOINT_PATH="${DTGPT_EVAL_MODEL_PATH:-${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700}"
# MERGED_MODEL_PATH="${DTGPT_VLLM_FULL_MODEL_PATH:-${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700-merged-vllm}"
CHECKPOINT_PATH="${DTGPT_EVAL_MODEL_PATH:-${REPO_ROOT}/3_results/checkpoint/checkpoint-5602-4epoch}"
MERGED_MODEL_PATH="${DTGPT_VLLM_FULL_MODEL_PATH:-${REPO_ROOT}/3_results/checkpoint/merge_for_vllm/checkpoint-5602-4epoch-merged-vllm}"
echo "Python binary: ${PYTHON_BIN}"
echo "Base model: ${DTGPT_BIOMISTRAL_MODEL_PATH}"
echo "Adapter checkpoint: ${CHECKPOINT_PATH}"
echo "Merged model output: ${MERGED_MODEL_PATH}"

"${PYTHON_BIN}" -m pipeline.merge_lora_adapter \
    --base-model "${DTGPT_BIOMISTRAL_MODEL_PATH}" \
    --adapter-path "${CHECKPOINT_PATH}" \
    --output-path "${MERGED_MODEL_PATH}" \
    --cache-dir "${REPO_ROOT}/3_cache"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
