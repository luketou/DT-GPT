#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-sweep"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:2
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_sweep_%j.out
#SBATCH --error=logs/mimic_dora_sweep_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT


set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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
export DTGPT_RUN_TIMESTAMP="${DTGPT_RUN_TIMESTAMP:-$(date '+%Y_%m_%d___%H_%M_%S_%6N')}"
export DTGPT_MIMIC_DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/1_experiments/2024_02_08_mimic_iv/1_data}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
export DTGPT_MIMIC_RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
unset TRANSFORMERS_CACHE
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export PYTHONPATH="${PYTHONPATH:-.}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if [ -z "${DTGPT_PYTHON_BIN:-}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # Load conda into this non-interactive shell so batch jobs use the expected env.
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${DTGPT_CONDA_ENV:-dtgpt}"
    PYTHON_BIN="python"
else
    PYTHON_BIN="${DTGPT_PYTHON_BIN:-python3}"
fi
unset TRANSFORMERS_CACHE

# Keep torch distributed on the NCCL runtime bundled with the active Python env.
# A mismatched system libnccl can fail at torch.distributed.barrier() during
# SFTTrainer setup with "Invalid config blocking attribute value".
TORCH_NCCL_LIB_DIR="$("${PYTHON_BIN}" -c "import pathlib, sys; p = pathlib.Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'nvidia' / 'nccl' / 'lib'; print(p if p.is_dir() else '')")"
if [ -n "${TORCH_NCCL_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${TORCH_NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
unset NCCL_COMM_BLOCKING
unset NCCL_BLOCKING_WAIT
unset TORCH_NCCL_BLOCKING_WAIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

"${PYTHON_BIN}" -c "import os, torch; print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda); print('Torch NCCL:', torch.cuda.nccl.version()); print('CUDA available:', torch.cuda.is_available()); print('NCCL lib override:', os.environ.get('LD_LIBRARY_PATH', '').split(':')[0])"

# These defaults target 2x L40S with Unsloth 4-bit QLoRA + DoRA at r=16.
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-6000}"
VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-256}"
TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.05}"
DECIMAL_PRECISION="${DTGPT_DECIMAL_PRECISION:-1}"
LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-10}"
SAMPLE_MERGING_STRATEGY="${DTGPT_SAMPLE_MERGING_STRATEGY:-mean}"
GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
USE_DORA="${DTGPT_USE_DORA:-1}"
USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-1}"
USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-2}"
DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"

TRAIN_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py"
SMOKE_CHECK_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py"
DISTRIBUTED_SMOKE_CHECK_SCRIPT="job/check_torch_distributed_nccl.py"
RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-1}"
SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"

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
echo "HF home: ${HF_HOME}"
echo "Sequence max length: ${SEQ_MAX_LEN}"
echo "Run timestamp: ${DTGPT_RUN_TIMESTAMP}"
echo "SFT dataset num proc: ${SFT_DATASET_NUM_PROC}"
echo "Train batch size per process: ${TRAIN_BATCH_SIZE}"
echo "Validation batch size: ${VALIDATION_BATCH_SIZE}"
echo "Generation samples per patient: ${NUM_SAMPLES_TO_GENERATE}"
echo "Generation max new tokens: ${MAX_NEW_TOKENS}"
echo "LoRA dropout: ${LORA_DROPOUT}"
echo "Distributed smoke check: ${RUN_DISTRIBUTED_SMOKE_CHECK}"
echo "Split smoke check: ${RUN_SPLIT_SMOKE_CHECK}"
if [ "${USE_DISTRIBUTED}" = "1" ]; then
    echo "Distributed launcher: torch.distributed.run (${NPROC_PER_NODE} processes)"
else
    echo "Distributed launcher: disabled"
fi
if [ "${USE_DEEPSPEED}" = "1" ]; then
    echo "Distributed training: DeepSpeed ZeRO-3 (${NPROC_PER_NODE} processes)"
    echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
    if ! "${PYTHON_BIN}" -c "import deepspeed" >/dev/null 2>&1; then
        echo "DeepSpeed requested, but it is not installed for ${PYTHON_BIN}."
        echo "Use DTGPT_USE_DEEPSPEED=0 for single-process training, or install deepspeed in the selected conda env."
        exit 1
    fi
else
    echo "Distributed training: disabled"
fi
if [ "${USE_UNSLOTH}" = "1" ]; then
    echo "Training mode: DoRA + Unsloth (4-bit QLoRA)"
    if ! "${PYTHON_BIN}" -c "import unsloth" >/dev/null 2>&1; then
        echo "Unsloth requested, but it is not installed for ${PYTHON_BIN}."
        echo "Use DTGPT_USE_UNSLOTH=0 for the dtgpt env, or set DTGPT_CONDA_ENV to an env with unsloth installed."
        exit 1
    fi
elif [ "${USE_DORA}" = "1" ]; then
    echo "Training mode: DoRA (standard PEFT)"
else
    echo "Training mode: LoRA (standard PEFT)"
fi
if [ "${USE_DORA}" = "1" ] || [ "${USE_UNSLOTH}" = "1" ]; then
    if ! "${PYTHON_BIN}" -c "import inspect; from peft import LoraConfig; raise SystemExit(0 if 'use_dora' in inspect.signature(LoraConfig).parameters else 1)" >/dev/null 2>&1; then
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
    if [ "${USE_DEEPSPEED}" = "1" ]; then
        DEEPSPEED_FLAG=(--deepspeed-config "${DEEPSPEED_CONFIG}")
    fi
    if [ "${USE_DISTRIBUTED}" = "1" ]; then
        RUNNER=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}")
    fi

    if ! "${RUNNER[@]}" "${TRAIN_SCRIPT}" \
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
        --seq-max-len "${SEQ_MAX_LEN}" \
        --decimal-precision "${DECIMAL_PRECISION}" \
        --num-samples-to-generate "${NUM_SAMPLES_TO_GENERATE}" \
        --sample-merging-strategy "${SAMPLE_MERGING_STRATEGY}" \
        --max-new-tokens-to-generate "${MAX_NEW_TOKENS}" \
        --logging-steps "${LOGGING_STEPS}" \
        --sft-dataset-num-proc "${SFT_DATASET_NUM_PROC}"; then
        print_header "Failed ${run_label}"
        exit 1
    fi

    print_header "Finished ${run_label}"
done

print_header "Completed all ${#SWEEP_CONFIGS[@]} LoRA sweep runs"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
