#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-1395to4185"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --nodelist=node-201
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_resume1395_to4185_%j.out
#SBATCH --error=logs/mimic_dora_resume1395_to4185_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Targeted recovery wrapper for resuming the May 2026 MIMIC DoRA run on node-201.
# This path uses one L40S with Unsloth/4-bit adapter training to avoid the
# DeepSpeed ZeRO-3 CPU-offload resume fallback that previously hung at the
# first training step.
REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_05___16_46_44_938674/models/checkpoint-1395}"
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-4185}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
# One GPU with grad_acc=16 preserves the previous 2-GPU effective batch size:
# 2 GPUs * per_device_batch 1 * grad_acc 8 == 1 GPU * per_device_batch 1 * grad_acc 16.
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-16,32,16,3,8e-6}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2048}"
export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-sdpa}"
export DTGPT_NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-128}"
export DTGPT_TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
# Eval can use a larger micro-batch than training on one L40S to spend available VRAM.
export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-8}"
export DTGPT_GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
export DTGPT_USE_DORA="${DTGPT_USE_DORA:-1}"
export DTGPT_USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
export DTGPT_USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-0}"
export DTGPT_USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
export DTGPT_NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-1}"
export UNSLOTH_RETURN_LOGITS="${UNSLOTH_RETURN_LOGITS:-1}"
export DTGPT_DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"
export DTGPT_RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
export DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-0}"
export DTGPT_DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
export DTGPT_SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
export DTGPT_LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-1}"

if [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

if [ "${DTGPT_NPROC_PER_NODE}" != "1" ]; then
    echo "This recovery wrapper requests exactly 1 GPU; set DTGPT_NPROC_PER_NODE=1." >&2
    exit 1
fi

echo "Resume checkpoint: ${DTGPT_RESUME_FROM_CHECKPOINT}"
echo "Conda env: ${DTGPT_CONDA_ENV}"
echo "Target max global step: ${DTGPT_MAX_STEPS}"
echo "Sequence max length: ${DTGPT_SEQ_MAX_LEN}"
echo "Attention implementation: ${DTGPT_ATTN_IMPLEMENTATION}"
echo "Sweep configs: ${DTGPT_SWEEP_CONFIGS}"
echo "Logging steps: ${DTGPT_LOGGING_STEPS}"
echo "Unsloth return logits: ${UNSLOTH_RETURN_LOGITS}"
echo "Reserved node: node-201"
echo "GPUs requested: 1 L40S; distributed processes: ${DTGPT_NPROC_PER_NODE}"
echo "Training mode: Unsloth 4-bit DoRA adapter training; distributed/DeepSpeed disabled"
echo "Requested CPU memory: 48G, shared node; one L40S allocation"

bash job/submit_mimic_dora.sh
