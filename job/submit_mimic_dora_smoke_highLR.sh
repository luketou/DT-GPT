#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-smoke-highLR"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --nodelist=node-201
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/mimic_dora_smoke_highLR_%j.out
#SBATCH --error=logs/mimic_dora_smoke_highLR_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Small-data smoke test for the resume path. It must finish/produce a concrete
# traceback before the full ~30k-patient resume job is attempted again.
REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

export DTGPT_RESUME_FROM_CHECKPOINT=""
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-20}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-64,128,1,1,1e-4}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-1024}"
export DTGPT_NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-128}"
export DTGPT_TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
export DTGPT_GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
export DTGPT_USE_DORA="${DTGPT_USE_DORA:-1}"
export DTGPT_USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
export DTGPT_USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-0}"
export DTGPT_USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
export DTGPT_NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-1}"
export DTGPT_DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"
export DTGPT_RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
export DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-1}"
export DTGPT_DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
export DTGPT_SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
export DTGPT_LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-1}"
export DTGPT_TRAIN_MAX_PATIENTS="${DTGPT_TRAIN_MAX_PATIENTS:-80}"
export DTGPT_VALIDATION_MAX_PATIENTS="${DTGPT_VALIDATION_MAX_PATIENTS:-20}"
export DTGPT_TEST_MAX_PATIENTS="${DTGPT_TEST_MAX_PATIENTS:-20}"
export DTGPT_TRAIN_MAX_SAMPLES="${DTGPT_TRAIN_MAX_SAMPLES:-32}"
export DTGPT_VALIDATION_MAX_SAMPLES="${DTGPT_VALIDATION_MAX_SAMPLES:-8}"
export DTGPT_SKIP_EVAL="${DTGPT_SKIP_EVAL:-1}"
export DTGPT_DEBUG="${DTGPT_DEBUG:-1}"

if [ -n "${DTGPT_RESUME_FROM_CHECKPOINT}" ] && [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

if [ -n "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Submitting small-data smoke resume from checkpoint ${DTGPT_RESUME_FROM_CHECKPOINT}"
else
    echo "Submitting small-data smoke training from scratch"
fi
echo "Smoke target max global step: ${DTGPT_MAX_STEPS}"
echo "Smoke train patients/samples: ${DTGPT_TRAIN_MAX_PATIENTS}/${DTGPT_TRAIN_MAX_SAMPLES}"
echo "Smoke validation patients/samples: ${DTGPT_VALIDATION_MAX_PATIENTS}/${DTGPT_VALIDATION_MAX_SAMPLES}"
echo "Smoke eval is skipped: ${DTGPT_SKIP_EVAL}"

bash job/submit_mimic_dora.sh
