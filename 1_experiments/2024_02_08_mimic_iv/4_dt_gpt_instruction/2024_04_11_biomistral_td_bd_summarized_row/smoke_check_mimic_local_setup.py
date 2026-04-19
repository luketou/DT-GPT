import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.local_paths import (
    ensure_runtime_cache_env,
    get_biomistral_model_path,
    get_mimic_column_descriptive_mapping_path,
    get_mimic_column_mapping_json_path,
    get_mimic_constants_path,
    get_mimic_dataset_statistics_path,
    get_mimic_final_data_dir,
    get_mimic_final_events_dir,
    get_mimic_helper_diagnosis_path,
    get_mimic_helper_items_path,
    get_precision_config,
)

ensure_runtime_cache_env()

from dt_gpt_fft_2024_04_11_biomistral_td_bd_sr import DTGPT_mimic_biomistral_fft_ti_bd_sr


def require_path(path, description):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_within(path, root, description):
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise ValueError(
            f"{description} resolved outside expected root {resolved_root}: {resolved_path}"
        )


def main():
    model_path = get_biomistral_model_path()
    if "/" in model_path and Path(model_path).exists():
        print(f"Resolved local BioMistral path: {model_path}")
    else:
        print(f"Resolved BioMistral identifier: {model_path}")

    mimic_data_root = get_mimic_final_data_dir().parent
    print(f"Resolved MIMIC data root: {mimic_data_root}")

    require_path(get_mimic_constants_path(), "MIMIC constants.csv")
    require_path(get_mimic_dataset_statistics_path(), "MIMIC dataset_statistics.json")
    require_path(get_mimic_column_mapping_json_path(), "MIMIC column_mapping.json")
    require_path(get_mimic_column_descriptive_mapping_path(), "MIMIC column_descriptive_name_mapping.csv")
    require_within(get_mimic_constants_path(), mimic_data_root, "MIMIC constants.csv")
    require_within(get_mimic_dataset_statistics_path(), mimic_data_root, "MIMIC dataset_statistics.json")
    require_within(get_mimic_column_mapping_json_path(), mimic_data_root, "MIMIC column_mapping.json")
    require_within(
        get_mimic_column_descriptive_mapping_path(),
        mimic_data_root,
        "MIMIC column_descriptive_name_mapping.csv",
    )

    events_dir = get_mimic_final_events_dir()
    require_path(events_dir, "MIMIC events directory")
    require_within(events_dir, mimic_data_root, "MIMIC events directory")
    event_files = sorted(events_dir.glob("*_events.csv"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {events_dir}")

    helper_items = get_mimic_helper_items_path()
    helper_diagnosis = get_mimic_helper_diagnosis_path()
    require_path(helper_items, "MIMIC helper d_items.csv")
    require_path(helper_diagnosis, "MIMIC helper d_icd_diagnoses.csv")
    require_within(helper_items, mimic_data_root, "MIMIC helper d_items.csv")
    require_within(
        helper_diagnosis,
        mimic_data_root,
        "MIMIC helper d_icd_diagnoses.csv",
    )
    print(f"Resolved MIMIC helper items path: {helper_items}")
    print(f"Resolved MIMIC helper diagnosis path: {helper_diagnosis}")

    with open(get_mimic_dataset_statistics_path()) as handle:
        stats = json.load(handle)
    if not stats:
        raise ValueError("dataset_statistics.json is empty")

    precision = get_precision_config()
    print(f"Precision config: {precision}")

    # Import and instantiate to catch module-level path issues.
    DTGPT_mimic_biomistral_fft_ti_bd_sr()
    print(f"Found {len(event_files)} event files in {events_dir}")
    print("MIMIC local setup smoke check passed.")


if __name__ == "__main__":
    main()
