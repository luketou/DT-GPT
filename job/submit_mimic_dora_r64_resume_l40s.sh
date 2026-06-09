#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-r64-resume"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --nodelist=node-201
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_r64_resume_%j.out
#SBATCH --error=logs/mimic_dora_r64_resume_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Resume/continue wrapper for the May 2026 MIMIC r=64 DoRA run.
#
# Default behavior:
#   - initialize from the latest completed r=64 checkpoint
#   - continue for two additional epoch-sized blocks of optimizer steps
#   - skip inline generation eval, because that requires a separate vLLM
#     OpenAI-compatible server on DTGPT_PREDICTION_URL.
#
# To finish an interrupted run instead of training two full blocks, override:
#   sbatch --export=ALL,DTGPT_RESUME_FROM_CHECKPOINT=/path/to/checkpoint-2529,DTGPT_MAX_STEPS=2801 job/submit_mimic_dora_r64_resume_l40s.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

DEFAULT_CHECKPOINT="${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_31___14_49_25_098295/models/checkpoint-2801"
export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

if [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

for required_checkpoint_file in adapter_model.safetensors trainer_state.json optimizer.pt scheduler.pt; do
    if [ ! -f "${DTGPT_RESUME_FROM_CHECKPOINT}/${required_checkpoint_file}" ]; then
        cat >&2 <<ERROR
Resume checkpoint is missing ${required_checkpoint_file}: ${DTGPT_RESUME_FROM_CHECKPOINT}

Use an original Trainer/LoRA checkpoint directory such as:
  ${DEFAULT_CHECKPOINT}

Do not use a merged vLLM export directory for training resume; merged-vllm
directories are for inference/evaluation and do not contain Trainer state.
ERROR
        exit 1
    fi
done

CHECKPOINT_GLOBAL_STEP="$(
    python - "${DTGPT_RESUME_FROM_CHECKPOINT}" <<'PY'
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1]) / "trainer_state.json"
if not state_path.exists():
    print(0)
else:
    print(int(json.loads(state_path.read_text()).get("global_step") or 0))
PY
)"

# r=64 full run used 2,801 optimizer steps per 0.5 patient-split epoch.
# Continue for two more epoch-sized blocks by default.
DEFAULT_ADDITIONAL_STEPS="${DTGPT_ADDITIONAL_STEPS:-5602}"
DEFAULT_TARGET_STEPS=$((CHECKPOINT_GLOBAL_STEP + DEFAULT_ADDITIONAL_STEPS))
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-${DEFAULT_TARGET_STEPS}}"

if [ "${DTGPT_MAX_STEPS}" -le "${CHECKPOINT_GLOBAL_STEP}" ]; then
    echo "DTGPT_MAX_STEPS (${DTGPT_MAX_STEPS}) must be greater than checkpoint global_step (${CHECKPOINT_GLOBAL_STEP})." >&2
    echo "For example, set DTGPT_MAX_STEPS=$((CHECKPOINT_GLOBAL_STEP + 5602)) to continue two more epoch-sized blocks." >&2
    exit 1
fi

export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-64,128,8,2,5e-5}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.05}"
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

# Inline generation eval needs a running vLLM server on DTGPT_PREDICTION_URL.
# Default to training-only continuation; run vLLM eval as a separate job after
# merging/exporting the adapter.
export DTGPT_SKIP_EVAL="${DTGPT_SKIP_EVAL:-1}"

export DTGPT_TRAIN_MAX_SAMPLES="${DTGPT_TRAIN_MAX_SAMPLES:-}"
export DTGPT_VALIDATION_MAX_SAMPLES="${DTGPT_VALIDATION_MAX_SAMPLES:-}"
export DTGPT_TEST_MAX_SAMPLES="${DTGPT_TEST_MAX_SAMPLES:-}"

if [ "${DTGPT_NPROC_PER_NODE}" != "1" ]; then
    echo "This r=64 resume wrapper requires exactly 1 GPU (Unsloth single-GPU); set DTGPT_NPROC_PER_NODE=1." >&2
    exit 1
fi

cat <<CONFIG
r=64 DoRA resume/continue run
  Resume checkpoint: ${DTGPT_RESUME_FROM_CHECKPOINT}
  Checkpoint global step: ${CHECKPOINT_GLOBAL_STEP}
  Target global step: ${DTGPT_MAX_STEPS}
  Additional optimizer steps: $((DTGPT_MAX_STEPS - CHECKPOINT_GLOBAL_STEP))
  Rank / Alpha: 64 / 128
  Grad accumulation: 8
  Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}
  Skip generation eval: ${DTGPT_SKIP_EVAL}
  Sequence max length: ${DTGPT_SEQ_MAX_LEN}
  Training mode: Unsloth 4-bit DoRA adapter training; distributed/DeepSpeed disabled
CONFIG

bash job/submit_mimic_dora.sh
