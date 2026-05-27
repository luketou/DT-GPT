#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-r32-full"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --nodelist=node-201
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=3-0:0
#SBATCH --output=logs/mimic_dora_r32_full_%j.out
#SBATCH --error=logs/mimic_dora_r32_full_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Full-validation r=32 DoRA probe.
# This starts a NEW adapter because a PEFT r=16 checkpoint cannot be expanded
# in-place to r=32. Compared with the earlier minival wrapper, this keeps the
# full validation set and enables dropout so the larger adapter is less likely
# to overfit before the first useful checkpoint.
REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

# Default r=32 candidate: double r=16 capacity, keep alpha/r=2, preserve the
# previous effective batch size with one L40S: 1 GPU * batch 1 * grad_acc 16.
export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-}"
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-500}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-32,64,16,3,8e-6}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.05}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2048}"
export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-sdpa}"
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
export UNSLOTH_RETURN_LOGITS="${UNSLOTH_RETURN_LOGITS:-1}"
export DTGPT_DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"
export DTGPT_RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
export DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-0}"
export DTGPT_DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
export DTGPT_SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
export DTGPT_LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-1}"
export DTGPT_SKIP_EVAL="${DTGPT_SKIP_EVAL:-0}"

# Full-validation probe: intentionally leave TRAIN/VALIDATION/TEST sample caps
# unset unless the submitter overrides them in the environment.
export DTGPT_TRAIN_MAX_SAMPLES="${DTGPT_TRAIN_MAX_SAMPLES:-}"
export DTGPT_VALIDATION_MAX_SAMPLES="${DTGPT_VALIDATION_MAX_SAMPLES:-}"
export DTGPT_TEST_MAX_SAMPLES="${DTGPT_TEST_MAX_SAMPLES:-}"

if [ -n "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "This r=32 full-validation job must start a fresh adapter; unset DTGPT_RESUME_FROM_CHECKPOINT." >&2
    exit 1
fi

if [ "${DTGPT_NPROC_PER_NODE}" != "1" ]; then
    echo "This r=32 full-validation wrapper requests exactly 1 GPU; set DTGPT_NPROC_PER_NODE=1." >&2
    exit 1
fi

cat <<CONFIG
Full-validation r=32 DoRA run
  Max steps: ${DTGPT_MAX_STEPS}
  Sweep configs: ${DTGPT_SWEEP_CONFIGS}
  LoRA dropout: ${DTGPT_LORA_DROPOUT}
  Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}
  Train sample cap: ${DTGPT_TRAIN_MAX_SAMPLES:-none}
  Validation sample cap: ${DTGPT_VALIDATION_MAX_SAMPLES:-none}
  Test sample cap: ${DTGPT_TEST_MAX_SAMPLES:-none}
  Skip generation eval: ${DTGPT_SKIP_EVAL}
  Sequence max length: ${DTGPT_SEQ_MAX_LEN}
  Logging steps: ${DTGPT_LOGGING_STEPS}
  Resume checkpoint: <fresh adapter; none>
  Node/GPU request: node-201, 1 L40S
  CPU/memory request: 8 CPUs, 48G, exclusive node to avoid memory contention
  Training mode: Unsloth 4-bit DoRA adapter training; distributed/DeepSpeed disabled
CONFIG

bash job/submit_mimic_dora.sh
