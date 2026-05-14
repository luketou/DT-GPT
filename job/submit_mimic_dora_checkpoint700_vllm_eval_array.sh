#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-vllm700"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:0
#SBATCH --array=0-7%4
#SBATCH --output=logs/mimic_dora_vllm700_shard_%A_%a.out
#SBATCH --error=logs/mimic_dora_vllm700_shard_%A_%a.err
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
export DTGPT_MIMIC_DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT/1_experiments/2024_02_08_mimic_iv/1_data}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
export DTGPT_MIMIC_RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DTGPT_MIMIC_DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
export DTGPT_RUNTIME_CACHE_ROOT="${DTGPT_RUNTIME_CACHE_ROOT:-/tmp/dtgpt_runtime_cache}"
export HF_HOME="${HF_HOME:-${DTGPT_RUNTIME_CACHE_ROOT}/hf_home}"
unset TRANSFORMERS_CACHE
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DTGPT_RUNTIME_CACHE_ROOT}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DTGPT_RUNTIME_CACHE_ROOT}/matplotlib}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${DTGPT_VLLM_CONDA_ENV:-LLMSelection-vllm}"
    PYTHON_BIN="$(command -v python)"
else
    echo "conda is not available in this job environment."
    exit 1
fi
unset TRANSFORMERS_CACHE

"${PYTHON_BIN}" -c "import vllm, openai, pandas, torch; print('vLLM env OK', getattr(vllm, '__version__', 'unknown'))"

EVAL_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py"
CHECKPOINT_PATH="${DTGPT_EVAL_MODEL_PATH:-${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700}"
FULL_MODEL_PATH="${DTGPT_VLLM_FULL_MODEL_PATH:-${REPO_ROOT}/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700-merged-vllm}"
export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-8}"
export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"
PORT="${DTGPT_VLLM_PORT:-$((18100 + DTGPT_EVAL_SHARD_INDEX))}"
SERVED_MODEL="${DTGPT_VLLM_MODEL_NAME:-dtgpt_mimic_dora_checkpoint700}"
SERVER_LOG="logs/mimic_dora_vllm_server_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${DTGPT_EVAL_SHARD_INDEX}.log"
VLLM_MAX_MODEL_LEN="${DTGPT_VLLM_MAX_MODEL_LEN:-4096}"
CLIENT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2900}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-768}"
MAX_CONCURRENT_REQUESTS="${DTGPT_MAX_CONCURRENT_REQUESTS:-8}"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "Python binary: ${PYTHON_BIN}"
echo "Working directory: $(pwd)"
echo "Base model: ${DTGPT_BIOMISTRAL_MODEL_PATH}"
echo "LoRA/DoRA checkpoint: ${CHECKPOINT_PATH}"
echo "Merged full model path: ${FULL_MODEL_PATH}"
echo "Served model: ${SERVED_MODEL}"
echo "Eval shard: ${DTGPT_EVAL_SHARD_INDEX} / ${DTGPT_EVAL_NUM_SHARDS}"
echo "vLLM port: ${PORT}"
echo "vLLM max model length: ${VLLM_MAX_MODEL_LEN}"
echo "Client sequence max length: ${CLIENT_SEQ_MAX_LEN}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Max concurrent requests: ${MAX_CONCURRENT_REQUESTS}"
echo "Eval max samples: ${DTGPT_EVAL_MAX_SAMPLES:-FULL_SHARD}"
echo "VLLM_USE_FLASHINFER_SAMPLER: ${VLLM_USE_FLASHINFER_SAMPLER}"

if [[ -d "${FULL_MODEL_PATH}" ]]; then
    echo "Serving merged full model for vLLM."
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --model "${FULL_MODEL_PATH}" \
        --tokenizer "${FULL_MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL}" \
        --enable-prefix-caching \
        --disable-sliding-window \
        --gpu-memory-utilization "${DTGPT_VLLM_GPU_MEMORY_UTILIZATION:-0.92}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        > "${SERVER_LOG}" 2>&1 &
else
    echo "Merged full model not found; serving adapter directly."
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --model "${DTGPT_BIOMISTRAL_MODEL_PATH}" \
        --tokenizer "${DTGPT_TOKENIZER_MODEL_PATH}" \
        --enable-lora \
        --lora-modules "${SERVED_MODEL}=${CHECKPOINT_PATH}" \
        --enable-prefix-caching \
        --disable-sliding-window \
        --gpu-memory-utilization "${DTGPT_VLLM_GPU_MEMORY_UTILIZATION:-0.92}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        > "${SERVER_LOG}" 2>&1 &
fi
VLLM_PID=$!

for attempt in $(seq 1 180); do
    if "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=2).read()" >/dev/null 2>&1; then
        echo "vLLM server is ready after ${attempt} checks."
        break
    fi
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
        echo "vLLM server exited before becoming ready. Last server log lines:"
        tail -n 80 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 5
done

if ! "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=2).read()" >/dev/null 2>&1; then
    echo "vLLM server did not become ready. Last server log lines:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
fi

conda activate "${DTGPT_CLIENT_CONDA_ENV:-dtgpt-unsloth}"
CLIENT_PYTHON_BIN="$(command -v python)"
echo "Client Python binary: ${CLIENT_PYTHON_BIN}"

"${CLIENT_PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --eval-model-path "${CHECKPOINT_PATH}" \
    --validation-batch-size "${DTGPT_VALIDATION_BATCH_SIZE:-1}" \
    --seq-max-len "${CLIENT_SEQ_MAX_LEN}" \
    --num-samples-to-generate "${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}" \
    --max-new-tokens-to-generate "${MAX_NEW_TOKENS}" \
    --eval-backend vllm \
    --eval-shard-index "${DTGPT_EVAL_SHARD_INDEX}" \
    --eval-num-shards "${DTGPT_EVAL_NUM_SHARDS}" \
    ${DTGPT_EVAL_MAX_SAMPLES:+--eval-max-samples "${DTGPT_EVAL_MAX_SAMPLES}"} \
    --prediction-url "http://127.0.0.1:${PORT}/v1/" \
    --vllm-model-name "${SERVED_MODEL}" \
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}" \
    --vllm-temperature "${DTGPT_VLLM_TEMPERATURE:-1.0}" \
    --vllm-top-p "${DTGPT_VLLM_TOP_P:-0.9}"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
