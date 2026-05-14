#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-metadata"
#SBATCH --partition=cpu-2g
#SBATCH --account=cpu-2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mimic_rebuild_metadata_%j.out
#SBATCH --error=logs/mimic_rebuild_metadata_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="/share/home/r15543056/trajectory_forecast/DT-GPT"
DATA_ROOT="${DTGPT_MIMIC_DATA_ROOT:-${REPO_ROOT}/1_experiments/2024_02_08_mimic_iv/1_data}"
RAW_STATS_PATH="${DTGPT_MIMIC_RAW_STATS_PATH:-${DATA_ROOT}/1_preprocessing/2024_02_01_raw_data_stats.json}"
FINAL_DATA_DIR="${DATA_ROOT}/0_final_data"
FINAL_EVENTS_DIR="${FINAL_DATA_DIR}/events"
METADATA_SCRIPT="${DATA_ROOT}/2_data_setup/2024_03_15_post_process_for_meta_data.py"
BACKUP_EXISTING_METADATA="${BACKUP_EXISTING_METADATA:-1}"

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
export DTGPT_MIMIC_RAW_STATS_PATH="${RAW_STATS_PATH}"
export DTGPT_RUNTIME_CACHE_ROOT="${DTGPT_RUNTIME_CACHE_ROOT:-/tmp/dtgpt_runtime_cache}"
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
echo "MIMIC data root: ${DTGPT_MIMIC_DATA_ROOT}"
echo "Raw stats path: ${DTGPT_MIMIC_RAW_STATS_PATH}"
echo "Final data dir: ${FINAL_DATA_DIR}"
echo "Final events dir: ${FINAL_EVENTS_DIR}"
echo "Metadata script: ${METADATA_SCRIPT}"
echo "Resource decision: CPU-only metadata rebuild; no GPU requested."
echo "Reason: workload is large JSON/CSV parsing plus pandas/tokenizer metadata, not model training."
echo "SLURM partition/account: cpu-2g"
echo "Memory request: 160G because raw stats can be several GB and is loaded by mapping_file_generation()."

for required in \
    "${RAW_STATS_PATH}" \
    "${FINAL_EVENTS_DIR}/1_events.csv" \
    "${FINAL_DATA_DIR}/constants.csv" \
    "${METADATA_SCRIPT}"; do
    if [ ! -e "${required}" ]; then
        echo "Missing required input: ${required}" >&2
        exit 1
    fi
done

if [ "${BACKUP_EXISTING_METADATA}" = "1" ]; then
    for metadata_file in \
        "${FINAL_DATA_DIR}/column_mapping.json" \
        "${FINAL_DATA_DIR}/dataset_statistics.json" \
        "${FINAL_DATA_DIR}/column_descriptive_name_mapping.csv"; do
        if [ -f "${metadata_file}" ]; then
            backup_path="${metadata_file}.backup_${START_TS}"
            echo "Backing up ${metadata_file} to ${backup_path}"
            cp "${metadata_file}" "${backup_path}"
        fi
    done
fi

"${PYTHON_BIN}" -u - <<'PY'
import runpy

script = "1_experiments/2024_02_08_mimic_iv/1_data/2_data_setup/2024_03_15_post_process_for_meta_data.py"
namespace = runpy.run_path(script)

print("Regenerating column_descriptive_name_mapping.csv from raw stats")
namespace["mapping_file_generation"]()

print("Regenerating column_mapping.json from current final events/1_events.csv")
namespace["generate_column_mapping"]()

print("Regenerating dataset_statistics.json from current train split events")
namespace["dataset_statistics_loader"]()

print("Disambiguating duplicate descriptive names")
namespace["duplicate_naming_increment"]()

print("Estimating descriptive-name token counts")
namespace["mapping_file_generator_nr_tokens_estimated"]()
PY

"${PYTHON_BIN}" -u - <<'PY'
import csv
import json
from pathlib import Path

base = Path("1_experiments/2024_02_08_mimic_iv/1_data/0_final_data")
expected_targets = {"220635", "220210", "220277"}

with open(base / "events" / "1_events.csv", newline="") as handle:
    event_cols = set(next(csv.reader(handle))[1:])

with open(base / "column_mapping.json") as handle:
    mapping = json.load(handle)

mapping_cols = set(mapping)
targets = {col for col, meta in mapping.items() if meta.get("target")}

with open(base / "dataset_statistics.json") as handle:
    statistics = json.load(handle)

with open(base / "column_descriptive_name_mapping.csv", newline="") as handle:
    descriptive_cols = {
        row["original_column_names"]
        for row in csv.DictReader(handle)
        if row.get("original_column_names")
    }

mapping_not_in_events = sorted(mapping_cols - event_cols)
events_not_in_mapping = sorted(event_cols - mapping_cols)
missing_targets = sorted(expected_targets - targets)
missing_target_stats = sorted(expected_targets - set(statistics))
missing_descriptions = sorted((mapping_cols - {"date", "patientid"}) - descriptive_cols)

print("Verification summary:")
print(f"  event cols: {len(event_cols)}")
print(f"  mapping cols: {len(mapping_cols)}")
print(f"  targets: {sorted(targets)}")
print(f"  mapping_not_in_events: {mapping_not_in_events}")
print(f"  events_not_in_mapping: {events_not_in_mapping}")
print(f"  missing_target_stats: {missing_target_stats}")
print(f"  missing_descriptions_count: {len(missing_descriptions)}")

if mapping_not_in_events:
    raise SystemExit("column_mapping.json still contains columns absent from events/1_events.csv")
if events_not_in_mapping:
    raise SystemExit("events/1_events.csv still contains columns absent from column_mapping.json")
if missing_targets:
    raise SystemExit(f"Missing expected target columns in mapping: {missing_targets}")
if missing_target_stats:
    raise SystemExit(f"Missing expected target columns in dataset_statistics.json: {missing_target_stats}")
if missing_descriptions:
    raise SystemExit(f"Missing descriptive names for mapping columns: {missing_descriptions[:40]}")

print("Metadata rebuild verification passed.")
PY

ls -lh \
    "${FINAL_DATA_DIR}/column_mapping.json" \
    "${FINAL_DATA_DIR}/dataset_statistics.json" \
    "${FINAL_DATA_DIR}/column_descriptive_name_mapping.csv"

if command -v sbatch_post.sh >/dev/null 2>&1; then
    sbatch_post.sh
fi
