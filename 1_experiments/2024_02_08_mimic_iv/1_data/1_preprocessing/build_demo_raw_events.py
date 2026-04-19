import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pipeline.local_paths import (
    ensure_directory,
    get_mimic_demo_data_dir,
    get_mimic_raw_events_dir,
)


TIME_BINS = 48


def _load_demo_tables(base_dir):
    icu_dir = base_dir / "icu"
    hosp_dir = base_dir / "hosp"

    tables = {
        "icustays": pd.read_csv(icu_dir / "icustays.csv.gz"),
        "patients": pd.read_csv(hosp_dir / "patients.csv.gz"),
        "admissions": pd.read_csv(hosp_dir / "admissions.csv.gz"),
        "diagnoses": pd.read_csv(hosp_dir / "diagnoses_icd.csv.gz"),
        "chartevents": pd.read_csv(icu_dir / "chartevents.csv.gz", usecols=["stay_id", "charttime", "itemid", "valuenum"]),
        "inputevents": pd.read_csv(icu_dir / "inputevents.csv.gz", usecols=["stay_id", "starttime", "itemid", "amount"]),
        "outputevents": pd.read_csv(icu_dir / "outputevents.csv.gz", usecols=["stay_id", "charttime", "itemid", "value"]),
        "procedureevents": pd.read_csv(icu_dir / "procedureevents.csv.gz", usecols=["stay_id", "starttime", "itemid", "value"]),
    }

    tables["icustays"]["intime"] = pd.to_datetime(tables["icustays"]["intime"])
    tables["icustays"]["outtime"] = pd.to_datetime(tables["icustays"]["outtime"])

    return tables


def _build_hourly_feature_table(events, icustays, time_col, value_col, agg_name):
    curr = events.copy()
    curr[time_col] = pd.to_datetime(curr[time_col], errors="coerce")
    curr[value_col] = pd.to_numeric(curr[value_col], errors="coerce")
    curr = curr.dropna(subset=["stay_id", time_col, value_col])
    curr = curr.merge(icustays[["stay_id", "intime"]], on="stay_id", how="inner")
    curr["hour"] = ((curr[time_col] - curr["intime"]).dt.total_seconds() // 3600).astype("Int64")
    curr = curr[curr["hour"].between(0, TIME_BINS - 1)]
    if curr.empty:
        return pd.DataFrame(columns=["stay_id", "hour"])

    grouped = (
        curr.groupby(["stay_id", "hour", "itemid"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )
    pivot = grouped.pivot_table(
        index=["stay_id", "hour"],
        columns="itemid",
        values="value",
        aggfunc="mean",
    )
    pivot.columns = [str(col) for col in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def _merge_hourly_tables(icustays, hourly_tables):
    base_index = pd.MultiIndex.from_product(
        [icustays["stay_id"].tolist(), list(range(TIME_BINS))],
        names=["stay_id", "hour"],
    )
    merged = pd.DataFrame(index=base_index).reset_index()

    for curr in hourly_tables:
        merged = merged.merge(curr, on=["stay_id", "hour"], how="left")

    return merged


def _build_static_diagnosis_table(icustays, diagnoses):
    merged = icustays[["stay_id", "hadm_id"]].merge(
        diagnoses[["hadm_id", "icd_code"]],
        on="hadm_id",
        how="left",
    )
    merged = merged.dropna(subset=["icd_code"]).copy()
    merged["value"] = 1.0
    static = merged.pivot_table(
        index="stay_id",
        columns="icd_code",
        values="value",
        aggfunc="max",
    )
    if static.empty:
        return pd.DataFrame(index=icustays["stay_id"].tolist())
    static.columns = [str(col) for col in static.columns]
    return static


def _build_demo_table(icustays, admissions, patients):
    demo = icustays[["stay_id", "subject_id", "hadm_id"]].merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    ).merge(
        admissions[["hadm_id", "race", "insurance"]],
        on="hadm_id",
        how="left",
    )
    demo = demo.rename(
        columns={
            "anchor_age": "Age",
            "race": "ethnicity",
        }
    )
    demo = demo[["stay_id", "Age", "gender", "ethnicity", "insurance"]]
    return demo.set_index("stay_id")


def _write_skiprows_csv(df, path):
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("metadata_row\n")
        df.to_csv(handle, index=False)


def build_demo_raw_events():
    base_dir = get_mimic_demo_data_dir()
    output_root = ensure_directory(get_mimic_raw_events_dir())

    tables = _load_demo_tables(base_dir)

    hourly_tables = [
        _build_hourly_feature_table(tables["chartevents"], tables["icustays"], "charttime", "valuenum", "chartevents"),
        _build_hourly_feature_table(tables["inputevents"], tables["icustays"], "starttime", "amount", "inputevents"),
        _build_hourly_feature_table(tables["outputevents"], tables["icustays"], "charttime", "value", "outputevents"),
        _build_hourly_feature_table(tables["procedureevents"], tables["icustays"], "starttime", "value", "procedureevents"),
    ]

    dynamic = _merge_hourly_tables(tables["icustays"], hourly_tables)
    static = _build_static_diagnosis_table(tables["icustays"], tables["diagnoses"])
    demo = _build_demo_table(tables["icustays"], tables["admissions"], tables["patients"])

    for stay_id in tables["icustays"]["stay_id"].tolist():
        stay_folder = ensure_directory(output_root / str(stay_id))

        curr_dynamic = (
            dynamic[dynamic["stay_id"] == stay_id]
            .drop(columns=["stay_id"])
            .sort_values("hour")
            .reset_index(drop=True)
        )
        curr_dynamic = curr_dynamic.drop(columns=["hour"])
        _write_skiprows_csv(curr_dynamic, stay_folder / "dynamic.csv")

        if stay_id in static.index:
            curr_static = static.loc[[stay_id]].reset_index(drop=True)
        else:
            curr_static = pd.DataFrame(index=[0])
        _write_skiprows_csv(curr_static, stay_folder / "static.csv")

        curr_demo = demo.loc[[stay_id]].reset_index(drop=True)
        curr_demo.to_csv(stay_folder / "demo.csv", index=False)

    return {
        "num_stays": int(tables["icustays"]["stay_id"].nunique()),
        "output_root": str(output_root),
    }


if __name__ == "__main__":
    result = build_demo_raw_events()
    print(result)
