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
    get_mimic_helper_diagnosis_path,
    get_mimic_helper_items_path,
    get_mimic_preprocessing_dir,
    get_mimic_raw_events_dir,
    get_mimic_raw_stats_path,
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


def initialize_dynamic_stats(column, df):
    non_na = df[column].count()
    non_zero = df[column].astype(bool).sum(axis=0)
    num_non_zero_in_second_half = df[column][len(df)//2:].astype(bool).sum(axis=0)

    df[column] = df[column].replace(0, np.nan)
    non_na_or_zero = df[column].count()

    return {
        "non_na": non_na,
        "non_zero": non_zero,
        "non_na_or_zero": non_na_or_zero,
        "std": [str(df[column].std())],
        "std_second_half": [str(df[column][len(df)//2:].std())],
        "num_non_zero_in_second_half": num_non_zero_in_second_half,
        "std_div_mean": [str(df[column].std()/df[column].mean())],
        "std_second_half_div_mean": [
            str(df[column][len(df)//2:].std() / df[column][len(df)//2:].mean())
        ],
        "_source": "dynamic",
    }


def initialize_static_stats(column, df_static):
    non_na = df_static[column].count()
    non_zero = df_static[column].astype(bool).sum(axis=0)
    non_na_or_zero = df_static[column].count()

    return {
        "non_na": non_na,
        "non_zero": non_zero,
        "non_na_or_zero": non_na_or_zero,
        "std": [],
        "std_second_half": [],
        "num_non_zero_in_second_half": 0,
        "std_div_mean": [],
        "std_second_half_div_mean": [],
        "_source": "static",
    }


def merge_column_stats(existing, incoming):
    existing["non_na"] += incoming["non_na"]
    existing["non_zero"] += incoming["non_zero"]
    existing["non_na_or_zero"] += incoming["non_na_or_zero"]
    existing["std"].extend(incoming["std"])
    existing["std_second_half"].extend(incoming["std_second_half"])
    existing["std_div_mean"].extend(incoming["std_div_mean"])
    existing["std_second_half_div_mean"].extend(incoming["std_second_half_div_mean"])
    existing["num_non_zero_in_second_half"] += incoming["num_non_zero_in_second_half"]
    if existing.get("_source") != "dynamic" and incoming.get("_source") == "dynamic":
        existing["_source"] = "dynamic"


def collect_stay_stats(args):
    file, folder_path = args
    stay_folder = os.path.join(folder_path, file)

    df = read_raw_stay_csv(stay_folder, "dynamic.csv", skiprows=1)
    df_static = read_raw_stay_csv(stay_folder, "static.csv", skiprows=1)

    if df is None or df_static is None:
        return None

    stay_stats = {}

    for column in df.columns:
        stay_stats[column] = initialize_dynamic_stats(column, df)

    for column in df_static.columns:
        incoming = initialize_static_stats(column, df_static)
        if column not in stay_stats:
            stay_stats[column] = incoming
        else:
            merge_column_stats(stay_stats[column], incoming)

    return stay_stats


def add_column_metadata(stats, help_df, diagnosis_df):
    help_df = help_df.copy()
    diagnosis_df = diagnosis_df.copy()
    help_df["itemid"] = help_df["itemid"].astype(str)
    diagnosis_df["icd_code"] = diagnosis_df["icd_code"].astype(str)

    helper_lookup = help_df.set_index("itemid").to_dict("index")
    diagnosis_lookup = diagnosis_df.set_index("icd_code")["long_title"].to_dict()

    for column, column_stats in stats.items():
        source = column_stats.pop("_source", "dynamic")

        if source == "static":
            column_stats["label"] = diagnosis_lookup.get(column, "unknown")
            column_stats["linksto"] = "COND"
            column_stats["category"] = "Diagnosis"
            continue

        helper_row = helper_lookup.get(column)
        if helper_row is not None:
            column_stats["label"] = helper_row["label"]
            column_stats["linksto"] = helper_row["linksto"]
            column_stats["category"] = helper_row["category"]
        else:
            column_stats["label"] = "unknown"
            column_stats["linksto"] = "unknown"
            column_stats["category"] = "unknown"


def merge_stats(stay_stats_list, help_df, diagnosis_df):
    stats = {}

    for stay_stats in stay_stats_list:
        if stay_stats is None:
            continue

        for column, incoming in stay_stats.items():
            if column not in stats:
                stats[column] = incoming
            else:
                merge_column_stats(stats[column], incoming)

    add_column_metadata(stats, help_df, diagnosis_df)
    return stats


def collect_all_stay_stats(files, folder_path, worker_count):
    tasks = [(file, str(folder_path)) for file in files]

    if worker_count == 1:
        for idx, task in enumerate(tasks):
            if idx % 10 == 0:
                print(f"Processing file {idx} out of {len(files)}")
            yield collect_stay_stats(task)
        return

    chunksize = max(1, len(tasks) // (worker_count * 8)) if tasks else 1
    with Pool(processes=worker_count) as pool:
        for idx, stay_stats in enumerate(pool.imap(collect_stay_stats, tasks, chunksize=chunksize)):
            if idx % 100 == 0:
                print(f"Processing file {idx} out of {len(files)}")
            yield stay_stats



def main():

    #: go through every csv in the folder, open as csv, then note down stats for non-zero/non-na values

    folder_path = get_mimic_raw_events_dir()
    ensure_directory(get_mimic_preprocessing_dir())

    files = list_stay_folders(folder_path)
    worker_count = resolve_worker_count()
    print(f"Stats generation workers: {worker_count}")

    # Load in helper csv
    help_csv_path = get_mimic_helper_items_path()
    help_df = pd.read_csv(help_csv_path)
    help_df["itemid"] = help_df["itemid"].astype(str)

    # load in diagnosis csv
    diagnosis_csv_path = get_mimic_helper_diagnosis_path()
    diagnosis_df = pd.read_csv(diagnosis_csv_path)
    diagnosis_df["icd_code"] = diagnosis_df["icd_code"].astype(str)

    start_time = time.time()
    stay_stats_list = collect_all_stay_stats(files, folder_path, worker_count)
    stats = merge_stats(stay_stats_list, help_df, diagnosis_df)
    print(f"Stats generation took {time.time() - start_time:.1f} seconds")

    
    # Save dictionary
    with open(get_mimic_raw_stats_path(), 'w') as f:
        json.dump(stats, f, cls=NpEncoder, indent=2)
    


if __name__ == "__main__":

    debug = os.getenv("WANDB_MODE", "").lower() == "disabled"

    #: setup wandb
    if debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(project='UC - MIMIC-IV', group="Data Processing")

    main()
