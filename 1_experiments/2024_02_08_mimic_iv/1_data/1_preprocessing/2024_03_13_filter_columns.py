import pandas as pd
import numpy as np
import os
import json
import time
import sys
from multiprocessing import Pool
from pathlib import Path
import wandb
from pandas.errors import EmptyDataError, ParserError

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.local_paths import (
    ensure_directory,
    get_mimic_constants_path,
    get_mimic_final_data_dir,
    get_mimic_final_events_dir,
    get_mimic_raw_events_dir,
    get_mimic_raw_stats_path,
)


def read_raw_stay_csv(stay_folder, filename, skiprows):
    path = Path(stay_folder) / filename
    try:
        return pd.read_csv(path, skiprows=skiprows)
    except (EmptyDataError, FileNotFoundError, ParserError) as exc:
        print(f"Skipping raw stay {Path(stay_folder).name}: cannot read {path} ({type(exc).__name__}: {exc})")
        return None


def resolve_worker_count(env=None):
    env = os.environ if env is None else env

    for key in ("DTGPT_MIMIC_NUM_WORKERS", "SLURM_CPUS_PER_TASK"):
        value = env.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                print(f"Ignoring invalid {key}={value!r}; expected an integer")

    return max(1, os.cpu_count() or 1)


def list_stay_folders(folder_path):
    return sorted(
        file
        for file in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, file))
    )


def process_stay_for_final_data(args):
    (
        file,
        load_folder_path,
        all_cols_to_keep_dynamic,
        all_cols_to_keep_static,
        all_cols_to_convert_zeros_to_na,
    ) = args
    stay_folder = os.path.join(load_folder_path, file)

    df = read_raw_stay_csv(stay_folder, "dynamic.csv", skiprows=1)
    df_demographics = read_raw_stay_csv(stay_folder, "demo.csv", skiprows=0)
    df_static = read_raw_stay_csv(stay_folder, "static.csv", skiprows=1)

    if df is None or df_demographics is None or df_static is None:
        return None

    df = df[all_cols_to_keep_dynamic].copy()

    zero_columns = set(all_cols_to_convert_zeros_to_na)
    for column in df.columns:
        if column in zero_columns:
            df[column] = df[column].replace(0, np.nan)

    df_static = df_static[all_cols_to_keep_static]
    df_static = df_static.replace(0, np.nan)
    df_static = df_static.replace(1.0, "diagnosed")
    df_static = pd.concat([df_demographics, df_static], axis=1)

    return file, df, df_static


def iter_processed_stays(
    files,
    load_folder_path,
    all_cols_to_keep_dynamic,
    all_cols_to_keep_static,
    all_cols_to_convert_zeros_to_na,
    worker_count,
):
    tasks = [
        (
            file,
            str(load_folder_path),
            all_cols_to_keep_dynamic,
            all_cols_to_keep_static,
            all_cols_to_convert_zeros_to_na,
        )
        for file in files
    ]

    if worker_count == 1:
        for idx, task in enumerate(tasks):
            if idx % 10 == 0:
                print(f"Processing file {idx} out of {len(files)}")
            yield process_stay_for_final_data(task)
        return

    chunksize = max(1, len(tasks) // (worker_count * 8)) if tasks else 1
    with Pool(processes=worker_count) as pool:
        for idx, result in enumerate(
            pool.imap(process_stay_for_final_data, tasks, chunksize=chunksize)
        ):
            if idx % 100 == 0:
                print(f"Processing file {idx} out of {len(files)}")
            yield result



def main():

    print("\n\n\n\n\n")
    print("==========================================================================================================")
    print("Starting main filtering and processing")

    #: go through every csv in the folder, open as csv, then note down stats for non-zero/non-na values
    load_folder_path = get_mimic_raw_events_dir()
    save_folder_path = ensure_directory(get_mimic_final_events_dir())
    save_folder_path_constant = ensure_directory(get_mimic_final_data_dir())

    files = list_stay_folders(load_folder_path)
    worker_count = resolve_worker_count()
    print(f"Filtering workers: {worker_count}")

    #: open stats dic, and select which variables to keep
    with open(get_mimic_raw_stats_path()) as f:
        stats = json.load(f)
    
    # convert everything to numpy arrays if possible
    for key in stats:
        for subkey in stats[key]:
            if isinstance(stats[key][subkey], list):
                stats[key][subkey] = np.array(stats[key][subkey]).astype(float)

    top_k_labs = 50
    top_k_inputevents = 50
    top_k_diagnoses = 100
    top_k_procedures = 50
    top_k_output_events = 50
    paper_target_columns = ["220635", "220210", "220277"]

    key_labs = "chartevents"
    key_outputevents = "outputevents"
    key_inputevents = "inputevents"
    key_procedureevents = "procedureevents"
    key_diagnoses = "COND"

    mapping_dic = {
        key_labs: top_k_labs,
        key_outputevents: top_k_output_events,
        key_inputevents: top_k_inputevents,
        key_procedureevents: top_k_procedures,
        key_diagnoses: top_k_diagnoses,
    }

    zero_to_na_mapping = [key_procedureevents, key_inputevents, key_outputevents, key_diagnoses, key_labs]

    all_cols_to_keep_dynamic = []
    all_cols_to_keep_static = []
    all_cols_to_convert_zeros_to_na = []

    for key in mapping_dic:

        all_cols_in_key = []

        for col in stats:
            if stats[col]["linksto"] == key:
                all_cols_in_key.append((col, stats[col]["non_na_or_zero"]))
        
        #: sort by second key
        all_cols_in_key = sorted(all_cols_in_key, key=lambda x: x[1], reverse=True)

        #: keep top k
        all_cols_in_key = all_cols_in_key[:mapping_dic[key]]

        #: append col names to all_cols_to_keep
        col_names = [x[0] for x in all_cols_in_key]

        if key != "COND":
            all_cols_to_keep_dynamic.extend(col_names)
        else:
            all_cols_to_keep_static.extend(col_names)

        if key in zero_to_na_mapping:
            all_cols_to_convert_zeros_to_na.extend(col_names)

    # Always keep the paper target variables even if they fall outside the
    # frequency-based top-k cutoff, e.g. magnesium (220635) ranks below top 50.
    for col in paper_target_columns:
        if col not in stats:
            raise KeyError(f"Target column {col} was not found in raw MIMIC statistics.")

        if stats[col]["linksto"] == key_diagnoses:
            if col not in all_cols_to_keep_static:
                all_cols_to_keep_static.append(col)
        else:
            if col not in all_cols_to_keep_dynamic:
                all_cols_to_keep_dynamic.append(col)

        if stats[col]["linksto"] in zero_to_na_mapping and col not in all_cols_to_convert_zeros_to_na:
            all_cols_to_convert_zeros_to_na.append(col)

    #: make empty list for constant df 
    constant_list = []

    start_time = time.time()
    processed_stays = iter_processed_stays(
        files,
        load_folder_path,
        all_cols_to_keep_dynamic,
        all_cols_to_keep_static,
        all_cols_to_convert_zeros_to_na,
        worker_count,
    )
    for result in processed_stays:
        if result is None:
            continue

        _file, df, df_static = result

        #: add ids to df
        patientid = len(constant_list)
        df.insert(0, "patientid", patientid)

        #: add 48 hours to date as range of datetime objects
        df.insert(0, "date", list(pd.date_range(start='2024-01-01', periods=len(df), freq='H')))

        #: save df in folder
        df.to_csv(os.path.join(save_folder_path, f"{patientid}_events.csv"))

        df_static.insert(0, "patientid", patientid)

        #: save in constant list
        constant_list.append(df_static)

    print(f"Filtering took {time.time() - start_time:.1f} seconds")

    #: save final constants df
    print("Saving constants df")
    df_constants = pd.concat(constant_list, axis=0, ignore_index=True)
    df_constants.to_csv(get_mimic_constants_path(), index=False)

    
    


if __name__ == "__main__":

    debug = os.getenv("WANDB_MODE", "").lower() == "disabled"

    #: setup wandb
    if debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(project='UC - MIMIC-IV', group="Data Processing")

    main()
