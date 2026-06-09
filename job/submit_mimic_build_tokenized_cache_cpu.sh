#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-token-cache"
#SBATCH --partition=cpu-2g
#SBATCH --account=cpu-2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/mimic_build_tokenized_cache_%j.out
#SBATCH --error=logs/mimic_build_tokenized_cache_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

export TOKENIZERS_PARALLELISM=false
export DTGPT_BIOMISTRAL_MODEL_PATH="${DTGPT_BIOMISTRAL_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_TOKENIZER_MODEL_PATH="${DTGPT_TOKENIZER_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_EXPERIMENT_ROOT="${DTGPT_EXPERIMENT_ROOT:-${REPO_ROOT}/3_results/raw_experiments/DT-GPT}"
export DTGPT_RUNTIME_CACHE_ROOT="${DTGPT_RUNTIME_CACHE_ROOT:-/tmp/dtgpt_runtime_cache}"
export DTGPT_DATASET_CACHE_ROOT="${DTGPT_DATASET_CACHE_ROOT:-${REPO_ROOT}/3_cache/dataset_cache}"
export DTGPT_RUN_TIMESTAMP="${DTGPT_RUN_TIMESTAMP:-mimic_tokenized_cache_build_$(date '+%Y_%m_%d___%H_%M_%S')}"
export DTGPT_MIMIC_DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-${REPO_ROOT}/1_experiments/2024_02_08_mimic_iv/1_data}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
export DTGPT_MIMIC_RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1.0}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2048}"
export DTGPT_DECIMAL_PRECISION="${DTGPT_DECIMAL_PRECISION:-1}"
export DTGPT_DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
export DTGPT_SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
export DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE="${DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE:-256}"
export DTGPT_DATASET_CACHE_NAME="${DTGPT_DATASET_CACHE_NAME:-}"
export DTGPT_USE_UNSLOTH=0
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
unset TRANSFORMERS_CACHE

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}" "${DTGPT_DATASET_CACHE_ROOT}"

if command -v conda >/dev/null 2>&1; then
    CONDA_ENV_NAME="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    PYTHON_BIN="$(command -v python)"
else
    PYTHON_BIN="${DTGPT_PYTHON_BIN:-python3}"
fi

TRAIN_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py"

CACHE_NAME_FLAG=()
if [ -n "${DTGPT_DATASET_CACHE_NAME}" ]; then
    CACHE_NAME_FLAG=(--dataset-cache-name "${DTGPT_DATASET_CACHE_NAME}")
fi

DATA_LIMIT_FLAGS=()
if [ -n "${DTGPT_TRAIN_MAX_PATIENTS:-}" ]; then
    DATA_LIMIT_FLAGS+=(--train-max-patients "${DTGPT_TRAIN_MAX_PATIENTS}")
fi
if [ -n "${DTGPT_VALIDATION_MAX_PATIENTS:-}" ]; then
    DATA_LIMIT_FLAGS+=(--validation-max-patients "${DTGPT_VALIDATION_MAX_PATIENTS}")
fi
if [ -n "${DTGPT_TRAIN_MAX_SAMPLES:-}" ]; then
    DATA_LIMIT_FLAGS+=(--train-max-samples "${DTGPT_TRAIN_MAX_SAMPLES}")
fi
if [ -n "${DTGPT_VALIDATION_MAX_SAMPLES:-}" ]; then
    DATA_LIMIT_FLAGS+=(--validation-max-samples "${DTGPT_VALIDATION_MAX_SAMPLES}")
fi

cat <<CONFIG
MIMIC tokenized dataset cache CPU build
  Partition: cpu-2g
  CPUs: ${SLURM_CPUS_PER_TASK:-32}
  Memory request: 300G
  Repo root: ${REPO_ROOT}
  Conda env: ${CONDA_ENV_NAME:-none}
  Python: ${PYTHON_BIN}
  Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}
  Sequence max length: ${DTGPT_SEQ_MAX_LEN}
  Decimal precision: ${DTGPT_DECIMAL_PRECISION}
  DF conversion workers: ${DTGPT_DF_CONVERSION_N_JOBS}
  SFT dataset num proc: ${DTGPT_SFT_DATASET_NUM_PROC}
  Build chunk size: ${DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE}
  Dataset cache root: ${DTGPT_DATASET_CACHE_ROOT}
  Dataset cache name: ${DTGPT_DATASET_CACHE_NAME:-manifest default}
CONFIG

"${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
    --dataset-cache-mode build-only \
    "${CACHE_NAME_FLAG[@]}" \
    --dataset-cache-build-chunk-size "${DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE}" \
    "${DATA_LIMIT_FLAGS[@]}" \
    --seq-max-len "${DTGPT_SEQ_MAX_LEN}" \
    --decimal-precision "${DTGPT_DECIMAL_PRECISION}" \
    --sft-dataset-num-proc "${DTGPT_SFT_DATASET_NUM_PROC}" \
    --df-conversion-n-jobs "${DTGPT_DF_CONVERSION_N_JOBS}" \
    --skip-eval \
    --debug

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
