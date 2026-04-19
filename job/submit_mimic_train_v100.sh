#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-train"
#SBATCH --partition=v100-32g
#SBATCH --account=v100-32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
#SBATCH --time=1-0:0
#SBATCH --output=logs/mimic_train_%j.out
#SBATCH --error=logs/mimic_train_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT
###SBATCH --test-only

set -euo pipefail

mkdir -p logs

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

module load cuda/12.1

export TOKENIZERS_PARALLELISM=false
export DTGPT_BIOMISTRAL_MODEL_PATH="${DTGPT_BIOMISTRAL_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_TOKENIZER_MODEL_PATH="${DTGPT_TOKENIZER_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_EXPERIMENT_ROOT="${DTGPT_EXPERIMENT_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPT}"
export DTGPT_RUNTIME_CACHE_ROOT="${DTGPT_RUNTIME_CACHE_ROOT:-/tmp/dtgpt_runtime_cache}"
export DTGPT_MIMIC_DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/1_experiments/2024_02_08_mimic_iv/1_data}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
export DTGPT_MIMIC_RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export PYTHONPATH="${PYTHONPATH:-.}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if [ -z "${DTGPT_PYTHON_BIN:-}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # Load conda into this non-interactive shell so batch jobs use the expected env.
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${DTGPT_CONDA_ENV:-dtgpt}"
    PYTHON_BIN="python"
else
    PYTHON_BIN="${DTGPT_PYTHON_BIN:-python3}"
fi

# The original experiment notes assume an 80GB GPU.
# These defaults are adjusted to be more realistic on a single V100-32GB.
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2048}"
VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-10}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-512}"

echo "MIMIC data root: ${DTGPT_MIMIC_DATA_ROOT}"
echo "MIMIC raw events dir: ${DTGPT_MIMIC_RAW_EVENTS_DIR}"
echo "MIMIC raw stats path: ${DTGPT_MIMIC_RAW_STATS_PATH}"
"${PYTHON_BIN}" 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py

echo "Training mode: LoRA"
"${PYTHON_BIN}" 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py \
    --use-lora \
    --lora-r "${DTGPT_LORA_R:-16}" \
    --lora-alpha "${DTGPT_LORA_ALPHA:-32}" \
    --lora-dropout "${DTGPT_LORA_DROPOUT:-0.05}" \
    --gradient-checkpointing \
    --train-batch-size 1 \
    --validation-batch-size "${VALIDATION_BATCH_SIZE}" \
    --seq-max-len "${SEQ_MAX_LEN}" \
    --num-train-epochs 5 \
    --num-samples-to-generate "${NUM_SAMPLES_TO_GENERATE}" \
    --max-new-tokens-to-generate "${MAX_NEW_TOKENS}"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
