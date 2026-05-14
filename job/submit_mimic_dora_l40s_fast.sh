#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-fast"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_fast_%j.out
#SBATCH --error=logs/mimic_dora_fast_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

# Fast single-L40S baseline for running alongside an existing long sweep.
# These defaults keep the current Unsloth DoRA path but target a 7-day-safe run.
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1.0}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-16,32,32,1,8e-6}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-4096}"
export DTGPT_NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-128}"
export DTGPT_TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
export DTGPT_GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
export DTGPT_USE_DORA="${DTGPT_USE_DORA:-1}"
export DTGPT_USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
export DTGPT_USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-0}"
export DTGPT_USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
export DTGPT_RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
export DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-0}"

echo "Submitting fast MIMIC DoRA run on one L40S with:"
echo "  DTGPT_PATIENT_SPLIT_FRACTION=${DTGPT_PATIENT_SPLIT_FRACTION}"
echo "  DTGPT_SWEEP_CONFIGS=${DTGPT_SWEEP_CONFIGS}"
echo "  DTGPT_LORA_DROPOUT=${DTGPT_LORA_DROPOUT}"
echo "  DTGPT_SEQ_MAX_LEN=${DTGPT_SEQ_MAX_LEN}"
echo "  DTGPT_NUM_SAMPLES_TO_GENERATE=${DTGPT_NUM_SAMPLES_TO_GENERATE}"
echo "  DTGPT_MAX_NEW_TOKENS=${DTGPT_MAX_NEW_TOKENS}"

bash job/submit_mimic_dora.sh
