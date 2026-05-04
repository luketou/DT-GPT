#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-final-data"
#SBATCH --partition=cpu-2g
#SBATCH --account=cpu-2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/mimic_build_final_data_%j.out
#SBATCH --error=logs/mimic_build_final_data_%j.err
#SBATCH --chdir=/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="/home/r15543056/trajectory_forecast/DT-GPT"
DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-${REPO_ROOT}/1_experiments/2024_02_08_mimic_iv/1_data}"
RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
FINAL_DATA_DIR="${DATA_ROOT}/0_final_data"
FINAL_EVENTS_DIR="${FINAL_DATA_DIR}/events"
RUNNER_PATH="${DATA_ROOT}/1_preprocessing/2024_03_15_runner.py"
BACKUP_EXISTING_FINAL_DATA="${BACKUP_EXISTING_FINAL_DATA:-1}"

mkdir -p "${REPO_ROOT}/logs" "${FINAL_DATA_DIR}"
cd "${REPO_ROOT}"

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

if [ -z "${DTGPT_PYTHON_BIN:-}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${DTGPT_CONDA_ENV:-dtgpt}"
    PYTHON_BIN="python"
else
    PYTHON_BIN="${DTGPT_PYTHON_BIN:-python3}"
fi

export DTGPT_MIMIC_DATA_ROOT="${DATA_ROOT}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${RAW_EVENTS_DIR}"
export DTGPT_MIMIC_RAW_STATS_PATH="${RAW_STATS_PATH}"
export DTGPT_MIMIC_NUM_WORKERS="${DTGPT_MIMIC_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

START_TS="$(date '+%Y%m%d_%H%M%S')"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${DTGPT_CONDA_ENV:-dtgpt}"
echo "Python: ${PYTHON_BIN}"
echo "Raw events input: ${DTGPT_MIMIC_RAW_EVENTS_DIR}"
echo "Raw stats path: ${DTGPT_MIMIC_RAW_STATS_PATH}"
echo "Final data dir: ${FINAL_DATA_DIR}"
echo "Runner: ${RUNNER_PATH}"
echo "Worker count: ${DTGPT_MIMIC_NUM_WORKERS}"
echo "Chunking mode: CPU parallel per-stay CSV streaming in 2024_02_01_overall_stats_generation.py and 2024_03_13_filter_columns.py"
echo "GPU mode: CPU-only pandas/CSV workload; scheduled on cpu-2g with no --gres GPU requested"

if [ ! -d "${RAW_EVENTS_DIR}" ]; then
    echo "Missing raw events directory: ${RAW_EVENTS_DIR}" >&2
    exit 1
fi

RAW_STAY_COUNT="$(find "${RAW_EVENTS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "Raw stay folders: ${RAW_STAY_COUNT}"
if [ "${RAW_STAY_COUNT}" -eq 0 ]; then
    echo "No raw stay folders found in ${RAW_EVENTS_DIR}" >&2
    exit 1
fi

if [ "${BACKUP_EXISTING_FINAL_DATA}" = "1" ]; then
    if [ -d "${FINAL_EVENTS_DIR}" ]; then
        EVENTS_BACKUP="${FINAL_EVENTS_DIR}.backup_${START_TS}"
        echo "Backing up existing events directory to: ${EVENTS_BACKUP}"
        mv "${FINAL_EVENTS_DIR}" "${EVENTS_BACKUP}"
    fi
    if [ -f "${FINAL_DATA_DIR}/constants.csv" ]; then
        CONSTANTS_BACKUP="${FINAL_DATA_DIR}/constants.csv.backup_${START_TS}"
        echo "Backing up existing constants.csv to: ${CONSTANTS_BACKUP}"
        cp "${FINAL_DATA_DIR}/constants.csv" "${CONSTANTS_BACKUP}"
    fi
    if [ -f "${RAW_STATS_PATH}" ]; then
        STATS_BACKUP="${RAW_STATS_PATH}.backup_${START_TS}"
        echo "Backing up existing raw stats to: ${STATS_BACKUP}"
        cp "${RAW_STATS_PATH}" "${STATS_BACKUP}"
    fi
fi

mkdir -p "${FINAL_EVENTS_DIR}"

"${PYTHON_BIN}" -u "${RUNNER_PATH}"

FINAL_EVENT_COUNT="$(find "${FINAL_EVENTS_DIR}" -maxdepth 1 -name '*_events.csv' -type f | wc -l)"
echo "Final event files: ${FINAL_EVENT_COUNT}"
ls -lh "${FINAL_DATA_DIR}/constants.csv" "${RAW_STATS_PATH}"

if [ "${FINAL_EVENT_COUNT}" -eq 0 ]; then
    echo "No final event files were generated." >&2
    exit 1
fi

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
