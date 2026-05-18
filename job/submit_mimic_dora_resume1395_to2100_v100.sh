#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-v100-1395to2100"
#SBATCH --partition=v100-32g
#SBATCH --account=v100-32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_v100_resume1395_to2100_%j.out
#SBATCH --error=logs/mimic_dora_v100_resume1395_to2100_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Single-V100 fallback for the L40S resume job.
# This intentionally keeps the existing Unsloth 4-bit QLoRA + DoRA path and
# does not try to combine multiple V100 GPUs. Multi-GPU Unsloth is rejected by
# job/submit_mimic_dora.sh in this repository.

export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_05___16_46_44_938674/models/checkpoint-1395}"
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-2100}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1.0}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-16,32,32,2,8e-6}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0}"

# V100-32G fallback default. If this OOMs, resubmit with:
#   sbatch --export=ALL,DTGPT_SEQ_MAX_LEN=3072 job/submit_mimic_dora_resume1395_to2100_v100.sh
# or, more conservatively:
#   sbatch --export=ALL,DTGPT_SEQ_MAX_LEN=2048 job/submit_mimic_dora_resume1395_to2100_v100.sh
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

if [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

echo "Resume checkpoint: ${DTGPT_RESUME_FROM_CHECKPOINT}"
echo "Conda env: ${DTGPT_CONDA_ENV}"
echo "Target max global step: ${DTGPT_MAX_STEPS}"
echo "GPU fallback: single V100-32G"
echo "Sequence max length: ${DTGPT_SEQ_MAX_LEN}"
echo "Distributed: ${DTGPT_USE_DISTRIBUTED}"
echo "DeepSpeed: ${DTGPT_USE_DEEPSPEED}"
echo "Training mode: Unsloth + DoRA single-process fallback"

bash job/submit_mimic_dora.sh
