#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-sweep"
#SBATCH --partition=v100-32g
#SBATCH --account=v100-32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
#SBATCH --time=3-0:0
#SBATCH --output=logs/mimic_dora_sweep_%j.out
#SBATCH --error=logs/mimic_dora_sweep_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT
###SBATCH --test-only

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
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export PYTHONPATH="${PYTHONPATH:-.}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if [ -z "${DTGPT_PYTHON_BIN:-}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # Load conda into this non-interactive shell so batch jobs use the expected env.
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${DTGPT_CONDA_ENV:-dtgpt}"
    PYTHON_BIN="python"
else
    PYTHON_BIN="${DTGPT_PYTHON_BIN:-python3}"
fi

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

# These defaults remain conservative for V100-32GB hardware.
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2048}"
VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-10}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-512}"
TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.05}"
DECIMAL_PRECISION="${DTGPT_DECIMAL_PRECISION:-1}"
LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-10}"
SAMPLE_MERGING_STRATEGY="${DTGPT_SAMPLE_MERGING_STRATEGY:-mean}"
USE_DORA="${DTGPT_USE_DORA:-0}"
USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-1}"
NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-3}"
DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"

TRAIN_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py"
SMOKE_CHECK_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py"
DISTRIBUTED_SMOKE_CHECK_SCRIPT="job/check_torch_distributed_nccl.py"
RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-1}"
SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"

DEFAULT_SWEEP_CONFIGS=(
    "32,64,16,10,8e-6"
    "32,64,16,10,9e-6"
    "32,64,16,9,1e-5"
    "32,64,16,11,1e-5"
    "32,64,16,9,8e-6"
    "32,64,16,10,1e-5"
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
echo "Python binary: ${PYTHON_BIN}"
if [ "${USE_DORA}" = "1" ]; then
    echo "Training mode: DoRA"
else
    echo "Training mode: LoRA"
fi
echo "Run timestamp: ${DTGPT_RUN_TIMESTAMP}"
echo "SFT dataset num proc: ${SFT_DATASET_NUM_PROC}"
echo "Distributed smoke check: ${RUN_DISTRIBUTED_SMOKE_CHECK}"
if [ "${USE_DEEPSPEED}" = "1" ]; then
    echo "Distributed training: DeepSpeed ZeRO-3 (${NPROC_PER_NODE} processes)"
    echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
else
    echo "Distributed training: disabled"
fi
echo "Total sweep configurations: ${#SWEEP_CONFIGS[@]}"

if [ "${USE_DEEPSPEED}" = "1" ] && [ "${RUN_DISTRIBUTED_SMOKE_CHECK}" = "1" ]; then
    print_header "Running NCCL distributed smoke check"
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${DISTRIBUTED_SMOKE_CHECK_SCRIPT}"
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
    RUNNER=("${PYTHON_BIN}")
    if [ "${USE_DORA}" = "1" ]; then
        DORA_FLAG=(--use-dora)
    fi
    if [ "${USE_DEEPSPEED}" = "1" ]; then
        DEEPSPEED_FLAG=(--deepspeed-config "${DEEPSPEED_CONFIG}")
        RUNNER=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}")
    fi

    if ! "${RUNNER[@]}" "${TRAIN_SCRIPT}" \
        --use-lora \
        "${DORA_FLAG[@]}" \
        "${DEEPSPEED_FLAG[@]}" \
        --lora-r "${lora_r}" \
        --lora-alpha "${lora_alpha}" \
        --lora-dropout "${LORA_DROPOUT}" \
        --gradient-checkpointing \
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

print_header "Completed all ${#SWEEP_CONFIGS[@]} DoRA sweep runs"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
