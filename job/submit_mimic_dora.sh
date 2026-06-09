#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-sweep"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_sweep_%j.out
#SBATCH --error=logs/mimic_dora_sweep_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT


set -euo pipefail

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"

mkdir -p logs

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.1
fi

export TOKENIZERS_PARALLELISM=false
export DTGPT_BIOMISTRAL_MODEL_PATH="${DTGPT_BIOMISTRAL_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_TOKENIZER_MODEL_PATH="${DTGPT_TOKENIZER_MODEL_PATH:-/home/r15543056/llm_model/BioMistral-7B-DARE}"
export DTGPT_EXPERIMENT_ROOT="${DTGPT_EXPERIMENT_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPT}"
export DTGPT_RUNTIME_CACHE_ROOT="${DTGPT_RUNTIME_CACHE_ROOT:-/tmp/dtgpt_runtime_cache}"
export DTGPT_DATASET_CACHE_ROOT="${DTGPT_DATASET_CACHE_ROOT:-${REPO_ROOT}/3_cache/dataset_cache}"
export DTGPT_RUN_TIMESTAMP="${DTGPT_RUN_TIMESTAMP:-$(date '+%Y_%m_%d___%H_%M_%S_%6N')}"
export DTGPT_MIMIC_DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/1_experiments/2024_02_08_mimic_iv/1_data}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
export DTGPT_MIMIC_RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
unset TRANSFORMERS_CACHE
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if command -v conda >/dev/null 2>&1; then
    CONDA_ENV_NAME="${DTGPT_CONDA_ENV:-dtgpt-unsloth}"
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    PYTHON_BIN="$(command -v python)"
else
    echo "conda is not available in this job environment."
    exit 1
fi
unset TRANSFORMERS_CACHE

# Keep torch distributed on the NCCL runtime bundled with the active Python env.
# A mismatched system libnccl can fail at torch.distributed.barrier() during
# SFTTrainer setup with "Invalid config blocking attribute value".
TORCH_NCCL_LIB_DIR="$("${PYTHON_BIN}" -c "import pathlib, sys; p = pathlib.Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'nvidia' / 'nccl' / 'lib'; print(p if p.is_dir() else '')")"
if [ -n "${TORCH_NCCL_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${TORCH_NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
# Unset GCC include path overrides that were set by the HPC module system.
# Having /usr/include in C_INCLUDE_PATH / CPLUS_INCLUDE_PATH / CPATH causes GCC
# to treat that path as a *user* directory instead of a system directory, which
# breaks the #include_next directive inside GCC's own <cstdlib> / <stdlib.h>
# and makes DeepSpeed's JIT compilation of cpu_adam fail with:
#   fatal error: stdlib.h: No such file or directory
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset CPATH
unset NCCL_COMM_BLOCKING
unset NCCL_BLOCKING_WAIT
unset TORCH_NCCL_BLOCKING_WAIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

"${PYTHON_BIN}" -c "import os, sys, torch; print('Python executable:', sys.executable); print('Python prefix:', sys.prefix); print('Conda env:', os.environ.get('CONDA_DEFAULT_ENV')); print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda); print('Torch NCCL:', torch.cuda.nccl.version()); print('CUDA available:', torch.cuda.is_available()); print('NCCL lib override:', os.environ.get('LD_LIBRARY_PATH', '').split(':')[0])"

# These defaults target 2x L40S with Unsloth 4-bit QLoRA + DoRA at r=16.
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-6000}"
VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-256}"
TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.05}"
DECIMAL_PRECISION="${DTGPT_DECIMAL_PRECISION:-1}"
LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-10}"
MAX_STEPS="${DTGPT_MAX_STEPS:--1}"
RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-}"
SAMPLE_MERGING_STRATEGY="${DTGPT_SAMPLE_MERGING_STRATEGY:-mean}"
GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
USE_DORA="${DTGPT_USE_DORA:-1}"
USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-0}"
USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-2}"
DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"

TRAIN_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py"
SMOKE_CHECK_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py"
DISTRIBUTED_SMOKE_CHECK_SCRIPT="job/check_torch_distributed_nccl.py"
RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-1}"
SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
TRAIN_MAX_PATIENTS="${DTGPT_TRAIN_MAX_PATIENTS:-}"
VALIDATION_MAX_PATIENTS="${DTGPT_VALIDATION_MAX_PATIENTS:-}"
TEST_MAX_PATIENTS="${DTGPT_TEST_MAX_PATIENTS:-}"
TRAIN_MAX_SAMPLES="${DTGPT_TRAIN_MAX_SAMPLES:-}"
VALIDATION_MAX_SAMPLES="${DTGPT_VALIDATION_MAX_SAMPLES:-}"
DATASET_CACHE_MODE="${DTGPT_DATASET_CACHE_MODE:-auto}"
DATASET_CACHE_NAME="${DTGPT_DATASET_CACHE_NAME:-}"
DATASET_CACHE_BUILD_CHUNK_SIZE="${DTGPT_DATASET_CACHE_BUILD_CHUNK_SIZE:-256}"
SKIP_EVAL="${DTGPT_SKIP_EVAL:-0}"
DEBUG_MODE="${DTGPT_DEBUG:-0}"
PRESERVE_EPOCH_CHECKPOINTS="${DTGPT_PRESERVE_EPOCH_CHECKPOINTS:-}"

DEFAULT_SWEEP_CONFIGS=(
    # Format: lora_r,lora_alpha,gradient_accumulation,num_train_epochs,learning_rate
    "16,32,32,10,8e-6"
)

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

print_header() {
    local message="$1"
    printf '\n[%s] %s\n' "$(timestamp)" "${message}"
}

if [ -n "${DTGPT_SWEEP_CONFIGS:-}" ]; then
    IFS=';' read -r -a SWEEP_CONFIGS <<< "${DTGPT_SWEEP_CONFIGS}"
else
    SWEEP_CONFIGS=("${DEFAULT_SWEEP_CONFIGS[@]}")
fi

if [ "${#SWEEP_CONFIGS[@]}" -eq 0 ]; then
    echo "No sweep configurations were provided."
    exit 1
fi

echo "MIMIC data root: ${DTGPT_MIMIC_DATA_ROOT}"
echo "MIMIC raw events dir: ${DTGPT_MIMIC_RAW_EVENTS_DIR}"
echo "MIMIC raw stats path: ${DTGPT_MIMIC_RAW_STATS_PATH}"
echo "Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}"
echo "Python binary: ${PYTHON_BIN}"
echo "Conda env name: ${CONDA_ENV_NAME}"
echo "Working directory: $(pwd)"
echo "HF home: ${HF_HOME}"
echo "Dataset cache root: ${DTGPT_DATASET_CACHE_ROOT}"
echo "Sequence max length: ${SEQ_MAX_LEN}"
echo "Run timestamp: ${DTGPT_RUN_TIMESTAMP}"
echo "SFT dataset num proc: ${SFT_DATASET_NUM_PROC}"
echo "DF conversion joblib workers: ${DF_CONVERSION_N_JOBS}"
echo "Train batch size per process: ${TRAIN_BATCH_SIZE}"
echo "Validation batch size: ${VALIDATION_BATCH_SIZE}"
echo "Generation samples per patient: ${NUM_SAMPLES_TO_GENERATE}"
echo "Generation max new tokens: ${MAX_NEW_TOKENS}"
echo "LoRA dropout: ${LORA_DROPOUT}"
echo "Max training steps: ${MAX_STEPS}"
echo "Train max patients: ${TRAIN_MAX_PATIENTS:-none}"
echo "Validation max patients: ${VALIDATION_MAX_PATIENTS:-none}"
echo "Test max patients: ${TEST_MAX_PATIENTS:-none}"
echo "Train max samples: ${TRAIN_MAX_SAMPLES:-none}"
echo "Validation max samples: ${VALIDATION_MAX_SAMPLES:-none}"
echo "Dataset cache mode: ${DATASET_CACHE_MODE}"
echo "Dataset cache name: ${DATASET_CACHE_NAME:-manifest default}"
echo "Dataset cache build chunk size: ${DATASET_CACHE_BUILD_CHUNK_SIZE}"
echo "Skip eval after training: ${SKIP_EVAL}"
echo "Preserve epoch checkpoints: ${PRESERVE_EPOCH_CHECKPOINTS:-none}"
echo "Debug/WandB disabled: ${DEBUG_MODE}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi
echo "Distributed smoke check: ${RUN_DISTRIBUTED_SMOKE_CHECK}"
echo "Split smoke check: ${RUN_SPLIT_SMOKE_CHECK}"
if [ "${USE_DISTRIBUTED}" = "1" ]; then
    echo "Distributed launcher: torch.distributed.run (${NPROC_PER_NODE} processes)"
else
    echo "Distributed launcher: disabled"
fi
if [ "${USE_UNSLOTH}" = "1" ] && [ "${USE_DISTRIBUTED}" = "1" ]; then
    echo "Unsloth does not support multi-GPU distributed training in this setup."
    echo "Run with DTGPT_USE_DISTRIBUTED=0, or set DTGPT_USE_UNSLOTH=0 for distributed standard PEFT."
    exit 1
fi
if [ "${USE_DEEPSPEED}" = "1" ]; then
    echo "Distributed training: DeepSpeed ZeRO-3 (${NPROC_PER_NODE} processes)"
    echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
    if ! "${PYTHON_BIN}" -c "import deepspeed"; then
        echo "DeepSpeed requested, but it is not installed for ${PYTHON_BIN}."
        echo "Use DTGPT_USE_DEEPSPEED=0 for single-process training, or install deepspeed in the selected conda env."
        exit 1
    fi
else
    echo "Distributed training: disabled"
fi
if [ "${USE_UNSLOTH}" = "1" ]; then
    echo "Training mode: DoRA + Unsloth (4-bit QLoRA)"
    if ! "${PYTHON_BIN}" -c "import unsloth"; then
        echo "Unsloth requested, but it cannot be imported by ${PYTHON_BIN}."
        echo "Use DTGPT_USE_UNSLOTH=0 if you want to run without unsloth."
        exit 1
    fi
elif [ "${USE_DORA}" = "1" ]; then
    echo "Training mode: DoRA (standard PEFT)"
else
    echo "Training mode: LoRA (standard PEFT)"
fi
if [ "${USE_DORA}" = "1" ] || [ "${USE_UNSLOTH}" = "1" ]; then
    if ! "${PYTHON_BIN}" -c "import inspect; from peft import LoraConfig; raise SystemExit(0 if 'use_dora' in inspect.signature(LoraConfig).parameters else 1)"; then
        echo "DoRA requested, but this PEFT install does not support LoraConfig(use_dora=...)."
        echo "Use DTGPT_USE_DORA=0 DTGPT_USE_UNSLOTH=0 for the dtgpt env, or upgrade PEFT in the selected conda env."
        exit 1
    fi
fi
if [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
    echo "Gradient checkpointing: enabled"
else
    echo "Gradient checkpointing: disabled"
fi
echo "Total sweep configurations: ${#SWEEP_CONFIGS[@]}"

if [ "${USE_DISTRIBUTED}" = "1" ] && [ "${RUN_DISTRIBUTED_SMOKE_CHECK}" = "1" ]; then
    print_header "Running NCCL distributed smoke check"
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${DISTRIBUTED_SMOKE_CHECK_SCRIPT}"
fi

if [ "${RUN_SPLIT_SMOKE_CHECK}" = "1" ]; then
    print_header "Running MIMIC split smoke check"
    "${PYTHON_BIN}" -c "from pipeline.EvaluationManager import EvaluationManager; m = EvaluationManager('2024_03_15_mimic_iv', load_statistics_file=False); [print(f'{split}: {len(m.get_paths_to_events_in_split(split)[1])} patient IDs') for split in ['TRAIN', 'VALIDATION', 'TEST']]"
fi

"${PYTHON_BIN}" "${SMOKE_CHECK_SCRIPT}"

run_index=0
for raw_config in "${SWEEP_CONFIGS[@]}"; do
    run_index=$((run_index + 1))

    IFS=',' read -r lora_r lora_alpha gradient_accumulation num_train_epochs learning_rate extra <<< "${raw_config}"
    unset IFS

    if [ -n "${extra:-}" ] || [ -z "${lora_r:-}" ] || [ -z "${lora_alpha:-}" ] || [ -z "${gradient_accumulation:-}" ] || [ -z "${num_train_epochs:-}" ] || [ -z "${learning_rate:-}" ]; then
        echo "Invalid sweep config #${run_index}: '${raw_config}'"
        echo "Expected format: r,alpha,grad_acc,epochs,lr"
        exit 1
    fi

    run_label="run ${run_index}/${#SWEEP_CONFIGS[@]} | r=${lora_r} alpha=${lora_alpha} grad_acc=${gradient_accumulation} epochs=${num_train_epochs} lr=${learning_rate}"
    print_header "Starting ${run_label}"

    DEEPSPEED_FLAG=()
    DORA_FLAG=()
    UNSLOTH_FLAG=()
    GRADIENT_CHECKPOINTING_FLAG=()
    RESUME_FLAG=()
    DATA_LIMIT_FLAGS=()
    SKIP_EVAL_FLAG=()
    DEBUG_FLAG=()
    PRESERVE_EPOCH_CHECKPOINTS_FLAG=()
    DATASET_CACHE_FLAGS=(--dataset-cache-mode "${DATASET_CACHE_MODE}" --dataset-cache-build-chunk-size "${DATASET_CACHE_BUILD_CHUNK_SIZE}")
    RUNNER=("${PYTHON_BIN}")
    if [ "${USE_DORA}" = "1" ]; then
        DORA_FLAG=(--use-dora)
    fi
    if [ "${USE_UNSLOTH}" = "1" ]; then
        UNSLOTH_FLAG=(--use-unsloth)
    fi
    if [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
        GRADIENT_CHECKPOINTING_FLAG=(--gradient-checkpointing)
    fi
    if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
        RESUME_FLAG=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
    fi
    if [ -n "${TRAIN_MAX_PATIENTS}" ]; then
        DATA_LIMIT_FLAGS+=(--train-max-patients "${TRAIN_MAX_PATIENTS}")
    fi
    if [ -n "${VALIDATION_MAX_PATIENTS}" ]; then
        DATA_LIMIT_FLAGS+=(--validation-max-patients "${VALIDATION_MAX_PATIENTS}")
    fi
    if [ -n "${TEST_MAX_PATIENTS}" ]; then
        DATA_LIMIT_FLAGS+=(--test-max-patients "${TEST_MAX_PATIENTS}")
    fi
    if [ -n "${TRAIN_MAX_SAMPLES}" ]; then
        DATA_LIMIT_FLAGS+=(--train-max-samples "${TRAIN_MAX_SAMPLES}")
    fi
    if [ -n "${VALIDATION_MAX_SAMPLES}" ]; then
        DATA_LIMIT_FLAGS+=(--validation-max-samples "${VALIDATION_MAX_SAMPLES}")
    fi
    if [ "${SKIP_EVAL}" = "1" ]; then
        SKIP_EVAL_FLAG=(--skip-eval)
    fi
    if [ "${DEBUG_MODE}" = "1" ]; then
        DEBUG_FLAG=(--debug)
    fi
    if [ -n "${PRESERVE_EPOCH_CHECKPOINTS}" ]; then
        PRESERVE_EPOCH_CHECKPOINTS_FLAG=(--preserve-epoch-checkpoints "${PRESERVE_EPOCH_CHECKPOINTS}")
    fi
    if [ -n "${DATASET_CACHE_NAME}" ]; then
        DATASET_CACHE_FLAGS+=(--dataset-cache-name "${DATASET_CACHE_NAME}")
    fi
    if [ "${USE_DEEPSPEED}" = "1" ]; then
        DEEPSPEED_FLAG=(--deepspeed-config "${DEEPSPEED_CONFIG}")
    fi
    if [ "${USE_DISTRIBUTED}" = "1" ]; then
        RUNNER=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}")
    fi

    if ! "${RUNNER[@]}" "${TRAIN_SCRIPT}" \
        "${DEBUG_FLAG[@]}" \
        --use-lora \
        "${DORA_FLAG[@]}" \
        "${UNSLOTH_FLAG[@]}" \
        "${DEEPSPEED_FLAG[@]}" \
        --lora-r "${lora_r}" \
        --lora-alpha "${lora_alpha}" \
        --lora-dropout "${LORA_DROPOUT}" \
        "${GRADIENT_CHECKPOINTING_FLAG[@]}" \
        --learning-rate "${learning_rate}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --validation-batch-size "${VALIDATION_BATCH_SIZE}" \
        --gradient-accumulation "${gradient_accumulation}" \
        --num-train-epochs "${num_train_epochs}" \
        --max-steps "${MAX_STEPS}" \
        "${RESUME_FLAG[@]}" \
        "${PRESERVE_EPOCH_CHECKPOINTS_FLAG[@]}" \
        "${DATA_LIMIT_FLAGS[@]}" \
        "${SKIP_EVAL_FLAG[@]}" \
        --seq-max-len "${SEQ_MAX_LEN}" \
        --decimal-precision "${DECIMAL_PRECISION}" \
        --num-samples-to-generate "${NUM_SAMPLES_TO_GENERATE}" \
        --sample-merging-strategy "${SAMPLE_MERGING_STRATEGY}" \
        --max-new-tokens-to-generate "${MAX_NEW_TOKENS}" \
        --logging-steps "${LOGGING_STEPS}" \
        --sft-dataset-num-proc "${SFT_DATASET_NUM_PROC}" \
        --df-conversion-n-jobs "${DF_CONVERSION_N_JOBS}" \
        "${DATASET_CACHE_FLAGS[@]}"; then
        print_header "Failed ${run_label}"
        exit 1
    fi

    print_header "Finished ${run_label}"
done

print_header "Completed all ${#SWEEP_CONFIGS[@]} LoRA sweep runs"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
