#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-raw-events"
#SBATCH --partition=v100-32g
#SBATCH --account=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/mimic_build_raw_events_%j.out
#SBATCH --error=logs/mimic_build_raw_events_%j.err
#SBATCH --chdir=/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="/home/r15543056/trajectory_forecast/DT-GPT"
DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-${REPO_ROOT}/1_experiments/2024_02_08_mimic_iv/1_data}"
RAW_SOURCE="${DTGPT_MIMIC_DEMO_DATA_DIR:-${REPO_ROOT}/MIMICIV_for方遠_20260429}"
RAW_EVENTS_DIR="${DTGPT_MIMIC_RAW_EVENTS_DIR:-${DATA_ROOT}/1_preprocessing/1_raw_events/csv}"
SCRIPT_PATH="${DATA_ROOT}/1_preprocessing/build_demo_raw_events.py"
BACKUP_EXISTING_RAW_EVENTS="${BACKUP_EXISTING_RAW_EVENTS:-1}"

mkdir -p "${REPO_ROOT}/logs" "$(dirname "${RAW_EVENTS_DIR}")"
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
export DTGPT_MIMIC_DEMO_DATA_DIR="${RAW_SOURCE}"
export DTGPT_MIMIC_RAW_EVENTS_DIR="${RAW_EVENTS_DIR}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${DTGPT_CONDA_ENV:-dtgpt}"
echo "Python: ${PYTHON_BIN}"
echo "Raw MIMIC source: ${DTGPT_MIMIC_DEMO_DATA_DIR}"
echo "Raw events output: ${DTGPT_MIMIC_RAW_EVENTS_DIR}"
echo "Script: ${SCRIPT_PATH}"

for required in \
    "${RAW_SOURCE}/icu/icustays.csv" \
    "${RAW_SOURCE}/icu/chartevents.csv" \
    "${RAW_SOURCE}/icu/inputevents.csv" \
    "${RAW_SOURCE}/icu/outputevents.csv" \
    "${RAW_SOURCE}/icu/procedureevents.csv" \
    "${RAW_SOURCE}/hosp/patients.csv" \
    "${RAW_SOURCE}/hosp/admissions.csv" \
    "${RAW_SOURCE}/hosp/diagnoses_icd.csv"; do
    if [ ! -f "${required}" ]; then
        echo "Missing required input: ${required}" >&2
        exit 1
    fi
done

if [ -d "${RAW_EVENTS_DIR}" ] && [ "${BACKUP_EXISTING_RAW_EVENTS}" = "1" ]; then
    BACKUP_DIR="${RAW_EVENTS_DIR}.backup_$(date '+%Y%m%d_%H%M%S')"
    echo "Backing up existing raw events directory to: ${BACKUP_DIR}"
    mv "${RAW_EVENTS_DIR}" "${BACKUP_DIR}"
elif [ -d "${RAW_EVENTS_DIR}" ]; then
    echo "Reusing existing raw events directory without backup: ${RAW_EVENTS_DIR}"
fi

mkdir -p "${RAW_EVENTS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_PATH}"

echo "Generated stay folders: $(find "${RAW_EVENTS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
