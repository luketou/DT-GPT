#!/usr/bin/env python3
"""Compute MIMIC DoRA r64 2-epoch metrics and plots from vLLM shard outputs."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_GLOB = "mimic_dora_paper_r2_vllm_shard_40007_*.out"
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "plot" / "64_2epoch"
OUTLIER_ABS_TARGET_LIMIT = 1000.0

OUT_DIR.mkdir(parents=True, exist_ok=True)
matplotlib_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "dtgpt_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib.pyplot as plt

VARIABLES = {
    "220210": "Respiratory Rate",
    "220277": "SpO2",
    "220635": "Magnesium",
}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    if ss_tot == 0:
        return math.nan
    return float(1.0 - ss_res / ss_tot)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def extract_dataframe_pairs() -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for log_path in sorted(LOG_DIR.glob(LOG_GLOB)):
        text = log_path.read_text(errors="replace")
        target_match = re.search(r"Saved target dataframe: (.*?target_dataframe\.csv)", text)
        pred_match = re.search(r"Saved prediction dataframe: (.*?prediction_dataframe\.csv)", text)
        if not target_match or not pred_match:
            raise RuntimeError(f"Missing saved dataframe paths in {log_path}")
        target_path = Path(target_match.group(1))
        pred_path = Path(pred_match.group(1))
        if not target_path.exists() or not pred_path.exists():
            raise FileNotFoundError(f"Missing dataframe for {log_path}: {target_path}, {pred_path}")
        pairs.append((log_path, target_path, pred_path))
    if len(pairs) != 8:
        raise RuntimeError(f"Expected 8 shard logs, found {len(pairs)}")
    return pairs


def load_combined_frames(pairs: list[tuple[Path, Path, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frames = []
    pred_frames = []
    usecols = ["patientid", "patient_sample_index", "date", *VARIABLES]
    for log_path, target_path, pred_path in pairs:
        shard = re.search(r"shard_(\d+)_of_008", target_path.name)
        shard_id = int(shard.group(1)) if shard else -1
        target = pd.read_csv(target_path, usecols=lambda c: c in usecols)
        pred = pd.read_csv(pred_path, usecols=lambda c: c in usecols)
        if len(target) != len(pred):
            raise RuntimeError(f"Row count mismatch for {log_path}: {len(target)} != {len(pred)}")
        target = target.copy()
        pred = pred.copy()
        target["source_shard"] = shard_id
        pred["source_shard"] = shard_id
        target["source_row"] = np.arange(len(target))
        pred["source_row"] = np.arange(len(pred))
        target_frames.append(target)
        pred_frames.append(pred)
    return pd.concat(target_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def compute_step_metrics(target: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_abs_error = 0.0
    total_count = 0
    for code, name in VARIABLES.items():
        y_true = numeric_series(target, code)
        y_pred = numeric_series(pred, code)
        mask = y_true.notna() & y_pred.notna() & (y_true.abs() <= OUTLIER_ABS_TARGET_LIMIT)
        yt = y_true[mask].to_numpy(dtype=float)
        yp = y_pred[mask].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(yt - yp)))
        rows.append(
            {
                "metric_scope": "step",
                "variable_code": code,
                "variable": name,
                "rows": int(mask.sum()),
                "r2": r2_score(yt, yp),
                "mae_scaled": mae,
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "smae": mae,
            }
        )
        total_abs_error += float(np.sum(np.abs(yt - yp)))
        total_count += int(mask.sum())
    step_df = pd.DataFrame(rows)
    step_df.loc[len(step_df)] = {
        "metric_scope": "step",
        "variable_code": "mean_unweighted",
        "variable": "Unweighted mean",
        "rows": int(step_df["rows"].sum()),
        "r2": math.nan,
        "mae_scaled": float(step_df["mae_scaled"].mean()),
        "rmse": math.nan,
        "smae": float(step_df["smae"].mean()),
    }
    step_df.loc[len(step_df)] = {
        "metric_scope": "step",
        "variable_code": "mean_weighted",
        "variable": "Weighted mean",
        "rows": total_count,
        "r2": math.nan,
        "mae_scaled": total_abs_error / total_count,
        "rmse": math.nan,
        "smae": total_abs_error / total_count,
    }
    return step_df


def compute_patient_metrics(target: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    patient_frames: dict[str, pd.DataFrame] = {}
    patient_ids = target["patientid"].astype(str)
    for code, name in VARIABLES.items():
        y_true = numeric_series(target, code)
        y_pred = numeric_series(pred, code)
        mask = y_true.notna() & y_pred.notna() & (y_true.abs() <= OUTLIER_ABS_TARGET_LIMIT)
        per_step = pd.DataFrame(
            {
                "patientid": patient_ids[mask],
                "target": y_true[mask].astype(float),
                "prediction": y_pred[mask].astype(float),
            }
        )
        patient_avg = per_step.groupby("patientid", as_index=False).mean(numeric_only=True)
        yt = patient_avg["target"].to_numpy(dtype=float)
        yp = patient_avg["prediction"].to_numpy(dtype=float)
        true_std = float(np.std(yt, ddof=0))
        raw_mae = float(np.mean(np.abs(yt - yp)))
        smae = raw_mae / true_std if true_std else math.nan
        rows.append(
            {
                "metric_scope": "patient_avg",
                "variable_code": code,
                "variable": name,
                "patients": int(len(patient_avg)),
                "r2_patient": r2_score(yt, yp),
                "raw_mae": raw_mae,
                "true_patient_avg_std": true_std,
                "smae": smae,
            }
        )
        patient_frames[code] = patient_avg
    return pd.DataFrame(rows), patient_frames


def compute_correlation_metrics(target: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    codes = list(VARIABLES)
    for left_i, left_code in enumerate(codes):
        for right_code in codes[left_i + 1 :]:
            left_true = numeric_series(target, left_code)
            right_true = numeric_series(target, right_code)
            left_pred = numeric_series(pred, left_code)
            right_pred = numeric_series(pred, right_code)
            mask = (
                left_true.notna()
                & right_true.notna()
                & left_pred.notna()
                & right_pred.notna()
                & (left_true.abs() <= OUTLIER_ABS_TARGET_LIMIT)
                & (right_true.abs() <= OUTLIER_ABS_TARGET_LIMIT)
            )
            true_corr = corr(
                left_true[mask].to_numpy(dtype=float),
                right_true[mask].to_numpy(dtype=float),
            )
            pred_corr = corr(
                left_pred[mask].to_numpy(dtype=float),
                right_pred[mask].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "pair": f"{VARIABLES[left_code]} vs {VARIABLES[right_code]}",
                    "left_code": left_code,
                    "right_code": right_code,
                    "rows": int(mask.sum()),
                    "true_corr": true_corr,
                    "pred_corr": pred_corr,
                    "diff": pred_corr - true_corr,
                }
            )
    corr_df = pd.DataFrame(rows)
    corr_r2 = r2_score(corr_df["true_corr"].to_numpy(), corr_df["pred_corr"].to_numpy())
    return corr_df, corr_r2


def plot_patient_avg(patient_frames: dict[str, pd.DataFrame], patient_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#2f6f9f", "#2a9d55", "#7b4bb3"]
    for ax, (code, name), color in zip(axes, VARIABLES.items(), colors):
        frame = patient_frames[code]
        metrics = patient_df[patient_df["variable_code"] == code].iloc[0]
        true = frame["target"].to_numpy(dtype=float)
        pred = frame["prediction"].to_numpy(dtype=float)
        true_mean = np.mean(true)
        true_std = np.std(true, ddof=0)
        true_z = (true - true_mean) / true_std
        pred_z = (pred - true_mean) / true_std
        ax.scatter(true_z, pred_z, s=12, alpha=0.45, color=color, edgecolors="none")
        xy_min = float(min(true_z.min(), pred_z.min()))
        xy_max = float(max(true_z.max(), pred_z.max()))
        pad = (xy_max - xy_min) * 0.08
        ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], color="#333333", linewidth=1)
        ax.set_xlim(xy_min - pad, xy_max + pad)
        ax.set_ylim(xy_min - pad, xy_max + pad)
        ax.set_title(f"{name}\nR2={metrics['r2_patient']:.4f}, sMAE={metrics['smae']:.4f}", fontsize=11)
        ax.set_xlabel("Patient avg true (z)")
        ax.set_ylabel("Patient avg predicted (z)")
        ax.grid(alpha=0.25)
    fig.suptitle("MIMIC DoRA r64 2-epoch patient-averaged true vs predicted", fontsize=14)
    path = OUT_DIR / "patient_avg_true_vs_pred_64_2epoch.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_step_smae(step_df: pd.DataFrame) -> Path:
    rows = step_df[step_df["variable_code"].isin(VARIABLES.keys())]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(rows["variable"], rows["smae"], color=["#2f6f9f", "#2a9d55", "#7b4bb3"])
    ax.set_ylabel("Step-level scaled MAE")
    ax.set_title("MIMIC DoRA r64 2-epoch step-level sMAE")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rows["smae"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    path = OUT_DIR / "step_scaled_mae_64_2epoch.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_correlation(corr_df: pd.DataFrame, corr_r2: float) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.scatter(corr_df["true_corr"], corr_df["pred_corr"], s=70, color="#555555")
    for _, row in corr_df.iterrows():
        ax.annotate(row["pair"].replace(" vs ", "\nvs "), (row["true_corr"], row["pred_corr"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    low = float(min(corr_df["true_corr"].min(), corr_df["pred_corr"].min()))
    high = float(max(corr_df["true_corr"].max(), corr_df["pred_corr"].max()))
    pad = (high - low) * 0.25 if high > low else 0.05
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#333333", linewidth=1)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("True pairwise correlation")
    ax.set_ylabel("Predicted pairwise correlation")
    ax.set_title(f"Correlation preservation R2={corr_r2:.4f}")
    ax.grid(alpha=0.25)
    path = OUT_DIR / "correlation_preservation_64_2epoch.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str]) -> str:
    table = df[columns].copy()
    for column in float_cols:
        if column in table:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    return table.to_markdown(index=False)


def write_summary(
    pairs: list[tuple[Path, Path, Path]],
    step_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    corr_r2: float,
    plot_paths: list[Path],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step_df.to_csv(OUT_DIR / "step_metrics_64_2epoch.csv", index=False)
    patient_df.to_csv(OUT_DIR / "patient_avg_metrics_64_2epoch.csv", index=False)
    corr_df.to_csv(OUT_DIR / "correlation_metrics_64_2epoch.csv", index=False)
    summary = {
        "logs": [str(p[0].relative_to(ROOT)) for p in pairs],
        "target_dataframes": [str(p[1]) for p in pairs],
        "prediction_dataframes": [str(p[2]) for p in pairs],
        "outlier_abs_target_limit": OUTLIER_ABS_TARGET_LIMIT,
        "paper_style_correlation_r2": corr_r2,
        "step_metrics": step_df.to_dict(orient="records"),
        "patient_avg_metrics": patient_df.to_dict(orient="records"),
        "correlation_metrics": corr_df.to_dict(orient="records"),
        "plots": [str(path.relative_to(ROOT)) for path in plot_paths],
    }
    (OUT_DIR / "metrics_summary_64_2epoch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# MIMIC DoRA r64 2-Epoch Metrics

Source logs: `logs/{LOG_GLOB}`

Outlier rule: rows with `abs(target) > {OUTLIER_ABS_TARGET_LIMIT:.1f}` are excluded before metric calculation.

## Paper-Style Correlation Preservation R2

`R2_corr = {corr_r2:.6f}`

{markdown_table(corr_df, ["pair", "rows", "true_corr", "pred_corr", "diff"], {"true_corr", "pred_corr", "diff"})}

## Step-Level sMAE

{markdown_table(step_df, ["variable", "rows", "mae_scaled", "smae", "r2"], {"mae_scaled", "smae", "r2"})}

## Patient-Averaged Metrics

{markdown_table(patient_df, ["variable", "patients", "r2_patient", "raw_mae", "true_patient_avg_std", "smae"], {"r2_patient", "raw_mae", "true_patient_avg_std", "smae"})}

## Figures

{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in plot_paths)}
"""
    out_path = OUT_DIR / "patient_avg_metrics_summary_64_2epoch.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = extract_dataframe_pairs()
    target, pred = load_combined_frames(pairs)
    step_df = compute_step_metrics(target, pred)
    patient_df, patient_frames = compute_patient_metrics(target, pred)
    corr_df, corr_r2 = compute_correlation_metrics(target, pred)
    plot_paths = [
        plot_patient_avg(patient_frames, patient_df),
        plot_step_smae(step_df),
        plot_correlation(corr_df, corr_r2),
    ]
    summary_path = write_summary(pairs, step_df, patient_df, corr_df, corr_r2, plot_paths)
    print(f"Wrote {summary_path}")
    print(f"Paper-style correlation R2: {corr_r2:.6f}")
    print(step_df.to_string(index=False))
    print(patient_df.to_string(index=False))


if __name__ == "__main__":
    main()
