#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-paper-r2"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_paper_r2_%j.out
#SBATCH --error=logs/mimic_dora_paper_r2_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Paper-R2 oriented fine-tuning wrapper for the MIMIC 3-variable task.
# Base file: job/submit_mimic_dora_r64_full_l40s.sh
# Default posture:
# - fresh adapter, no resume
# - full patient coverage (split fraction 1.0)
# - single L40S with Unsloth 4-bit DoRA
# - conservative 2-epoch r=64 / alpha=128 run
#
# Override examples:
#   Exp1:
#   DTGPT_SWEEP_CONFIGS='64,128,8,2,2e-5' DTGPT_LORA_DROPOUT='0.05' sbatch --export=ALL job/submit_mimic_dora_r64_paper_r2_l40s.sh
#   Exp2:
#   DTGPT_SWEEP_CONFIGS='64,128,8,2,5e-5' DTGPT_LORA_DROPOUT='0.10' sbatch --export=ALL job/submit_mimic_dora_r64_paper_r2_l40s.sh
#   Exp3:
#   DTGPT_SWEEP_CONFIGS='64,128,8,2,2e-5' DTGPT_LORA_DROPOUT='0.10' sbatch --export=ALL job/submit_mimic_dora_r64_paper_r2_l40s.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-}"
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:--1}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1.0}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-64,128,8,2,2e-5}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.10}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2048}"
export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-sdpa}"
export DTGPT_NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-256}"
export DTGPT_TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
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
export DTGPT_SKIP_EVAL="${DTGPT_SKIP_EVAL:-0}"
export DTGPT_PRESERVE_EPOCH_CHECKPOINTS="${DTGPT_PRESERVE_EPOCH_CHECKPOINTS:-1}"
export DTGPT_TRAIN_MAX_SAMPLES="${DTGPT_TRAIN_MAX_SAMPLES:-}"
export DTGPT_VALIDATION_MAX_SAMPLES="${DTGPT_VALIDATION_MAX_SAMPLES:-}"
export DTGPT_TEST_MAX_SAMPLES="${DTGPT_TEST_MAX_SAMPLES:-}"
export DTGPT_DATASET_CACHE_ROOT="${DTGPT_DATASET_CACHE_ROOT:-${REPO_ROOT}/3_cache/dataset_cache}"
export DTGPT_DATASET_CACHE_MODE="${DTGPT_DATASET_CACHE_MODE:-require}"
export DTGPT_DATASET_CACHE_NAME="${DTGPT_DATASET_CACHE_NAME:-}"
export DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE="${DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE:-256}"

if [ -n "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "This paper-R2 fine-tune job starts a fresh adapter; unset DTGPT_RESUME_FROM_CHECKPOINT." >&2
    exit 1
fi

if [ "${DTGPT_NPROC_PER_NODE}" != "1" ]; then
    echo "This wrapper requires exactly 1 GPU with Unsloth; set DTGPT_NPROC_PER_NODE=1." >&2
    exit 1
fi

cat <<CONFIG
Paper-R2 oriented r=64 DoRA fine-tune
  Base wrapper: job/submit_mimic_dora_r64_full_l40s.sh
  Rank / Alpha: 64 / 128
  Sweep configs: ${DTGPT_SWEEP_CONFIGS}
  LoRA dropout: ${DTGPT_LORA_DROPOUT}
  Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}
  Max steps: ${DTGPT_MAX_STEPS}  (-1 = use num_train_epochs from sweep config)
  Sequence max length: ${DTGPT_SEQ_MAX_LEN}
  Logging steps: ${DTGPT_LOGGING_STEPS}
  Train sample cap: ${DTGPT_TRAIN_MAX_SAMPLES:-none}
  Validation sample cap: ${DTGPT_VALIDATION_MAX_SAMPLES:-none}
  Test sample cap: ${DTGPT_TEST_MAX_SAMPLES:-none}
  Dataset cache root: ${DTGPT_DATASET_CACHE_ROOT}
  Dataset cache mode: ${DTGPT_DATASET_CACHE_MODE}
  Dataset cache name: ${DTGPT_DATASET_CACHE_NAME:-manifest default}
  Skip generation eval: ${DTGPT_SKIP_EVAL}
  Preserve epoch checkpoints: ${DTGPT_PRESERVE_EPOCH_CHECKPOINTS}
  Resume checkpoint: <fresh adapter; none>
  Training mode: Unsloth 4-bit DoRA adapter training
  Node/GPU request: any l40s node, 1 L40S, 48G CPU RAM
CONFIG

bash job/submit_mimic_dora.sh
