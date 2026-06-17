#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-paper-r2-cont4"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_paper_r2_continue_to4_%j.out
#SBATCH --error=logs/mimic_dora_paper_r2_continue_to4_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Continue the Paper-R2 r=64 DoRA run from the completed 2-epoch adapter.
#
# This intentionally uses the checkpoint as a trainable adapter initializer
# under Unsloth instead of resuming the old Trainer optimizer/scheduler state.
# The completed 2-epoch run ended with learning_rate ~= 0, so full scheduler
# resume would preserve the plateau rather than restart learning.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

DEFAULT_CHECKPOINT="${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_06_10___16_08_28_882856/models/checkpoint-11168"
export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

if [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

for required_checkpoint_file in adapter_model.safetensors trainer_state.json; do
    if [ ! -f "${DTGPT_RESUME_FROM_CHECKPOINT}/${required_checkpoint_file}" ]; then
        echo "Checkpoint is missing ${required_checkpoint_file}: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
        exit 1
    fi
done

read -r CHECKPOINT_GLOBAL_STEP CHECKPOINT_EPOCH TARGET_GLOBAL_STEP ADDITIONAL_STEPS < <(
    python - "${DTGPT_RESUME_FROM_CHECKPOINT}" "${DTGPT_TARGET_TOTAL_EPOCHS:-4}" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
target_epochs = float(sys.argv[2])
state = json.loads((checkpoint / "trainer_state.json").read_text())
global_step = int(state.get("global_step") or 0)
epoch = float(state.get("epoch") or 0)
if global_step <= 0 or epoch <= 0:
    raise SystemExit(f"Cannot infer epoch size from {checkpoint / 'trainer_state.json'}")
steps_per_epoch = global_step / epoch
target_global_step = int(round(steps_per_epoch * target_epochs))
additional_steps = target_global_step - global_step
if additional_steps <= 0:
    raise SystemExit(
        f"Target epochs {target_epochs:g} gives target step {target_global_step}, "
        f"which is not beyond checkpoint step {global_step}."
    )
print(global_step, epoch, target_global_step, additional_steps)
PY
)

export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-${TARGET_GLOBAL_STEP}}"

if [ "${DTGPT_MAX_STEPS}" -le "${CHECKPOINT_GLOBAL_STEP}" ]; then
    echo "DTGPT_MAX_STEPS (${DTGPT_MAX_STEPS}) must be greater than checkpoint global_step (${CHECKPOINT_GLOBAL_STEP})." >&2
    exit 1
fi

export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1.0}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-64,128,8,4,1e-5}"
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
export DTGPT_SKIP_EVAL="${DTGPT_SKIP_EVAL:-1}"
export DTGPT_PRESERVE_EPOCH_CHECKPOINTS="${DTGPT_PRESERVE_EPOCH_CHECKPOINTS:-1,2}"
export DTGPT_TRAIN_MAX_SAMPLES="${DTGPT_TRAIN_MAX_SAMPLES:-}"
export DTGPT_VALIDATION_MAX_SAMPLES="${DTGPT_VALIDATION_MAX_SAMPLES:-}"
export DTGPT_TEST_MAX_SAMPLES="${DTGPT_TEST_MAX_SAMPLES:-}"

REQUIRED_DATASET_CACHE_PATH="${REPO_ROOT}/3_cache/dataset_cache/mimic_tokenized_seq2048_split100_dp1_a8592e561769"
REQUIRED_DATASET_CACHE_ROOT="$(dirname "${REQUIRED_DATASET_CACHE_PATH}")"
REQUIRED_DATASET_CACHE_NAME="$(basename "${REQUIRED_DATASET_CACHE_PATH}")"
export DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE="${DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE:-256}"

for required_cache_file in \
    "${REQUIRED_DATASET_CACHE_PATH}/_SUCCESS" \
    "${REQUIRED_DATASET_CACHE_PATH}/manifest.json" \
    "${REQUIRED_DATASET_CACHE_PATH}/train/state.json" \
    "${REQUIRED_DATASET_CACHE_PATH}/validation/state.json"; do
    if [ ! -f "${required_cache_file}" ]; then
        echo "Required tokenized dataset cache is incomplete: ${required_cache_file}" >&2
        echo "Rebuild it first with job/submit_mimic_build_tokenized_cache_cpu.sh." >&2
        exit 1
    fi
done

if [ "${DTGPT_NPROC_PER_NODE}" != "1" ]; then
    echo "This wrapper requires exactly 1 GPU with Unsloth; set DTGPT_NPROC_PER_NODE=1." >&2
    exit 1
fi

cat <<CONFIG
Paper-R2 r=64 DoRA continuation to 4 epochs
  Adapter init checkpoint: ${DTGPT_RESUME_FROM_CHECKPOINT}
  Checkpoint epoch/global step: ${CHECKPOINT_EPOCH} / ${CHECKPOINT_GLOBAL_STEP}
  Target global step: ${DTGPT_MAX_STEPS}
  Additional optimizer steps: $((DTGPT_MAX_STEPS - CHECKPOINT_GLOBAL_STEP))
  Sweep configs: ${DTGPT_SWEEP_CONFIGS}
  LoRA dropout: ${DTGPT_LORA_DROPOUT}
  Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}
  Dataset cache path: ${REQUIRED_DATASET_CACHE_PATH}
  Skip generation eval: ${DTGPT_SKIP_EVAL}
  Preserve epoch checkpoints: ${DTGPT_PRESERVE_EPOCH_CHECKPOINTS}
  Training mode: Unsloth 4-bit DoRA adapter initialization with fresh optimizer/scheduler
CONFIG

DTGPT_DATASET_CACHE_ROOT="${REQUIRED_DATASET_CACHE_ROOT}" \
DTGPT_DATASET_CACHE_MODE="require" \
DTGPT_DATASET_CACHE_NAME="${REQUIRED_DATASET_CACHE_NAME}" \
bash job/submit_mimic_dora.sh
