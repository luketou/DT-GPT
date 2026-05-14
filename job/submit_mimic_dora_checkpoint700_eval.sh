#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-eval700"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:0
#SBATCH --output=logs/mimic_dora_eval700_%j.out
#SBATCH --error=logs/mimic_dora_eval700_%j.err
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
export DTGPT_TOKENIZER_MODEL_PATH="${DTGPT_TOKENIZER_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_MIMIC_DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/1_experiments/2024_02_08_mimic_iv/1_data}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
export DTGPT_MIMIC_RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
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
    conda activate dtgpt-unsloth
    PYTHON_BIN="$(command -v python)"
else
    echo "conda is not available in this job environment."
    exit 1
fi
unset TRANSFORMERS_CACHE

EVAL_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py"
CHECKPOINT_PATH="${DTGPT_EVAL_MODEL_PATH:-${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700}"

echo "Python binary: ${PYTHON_BIN}"
echo "Working directory: $(pwd)"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}"
echo "Attention implementation: ${DTGPT_ATTN_IMPLEMENTATION}"
echo "Eval backend: ${DTGPT_EVAL_BACKEND:-hf}"
echo "Eval shard: ${DTGPT_EVAL_SHARD_INDEX:-0} / ${DTGPT_EVAL_NUM_SHARDS:-1}"
echo "HF home: ${HF_HOME}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --eval-model-path "${CHECKPOINT_PATH}" \
    --validation-batch-size "${DTGPT_VALIDATION_BATCH_SIZE:-1}" \
    --seq-max-len "${DTGPT_SEQ_MAX_LEN:-6000}" \
    --num-samples-to-generate "${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}" \
    --max-new-tokens-to-generate "${DTGPT_MAX_NEW_TOKENS:-256}" \
    --eval-backend "${DTGPT_EVAL_BACKEND:-hf}" \
    --eval-shard-index "${DTGPT_EVAL_SHARD_INDEX:-0}" \
    --eval-num-shards "${DTGPT_EVAL_NUM_SHARDS:-1}" \
    --prediction-url "${DTGPT_PREDICTION_URL:-http://127.0.0.1:18101/v1/}" \
    --vllm-model-name "${DTGPT_VLLM_MODEL_NAME:-}" \
    --max-concurrent-requests "${DTGPT_MAX_CONCURRENT_REQUESTS:-16}" \
    --vllm-temperature "${DTGPT_VLLM_TEMPERATURE:-1.0}" \
    --vllm-top-p "${DTGPT_VLLM_TOP_P:-0.9}"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
