#!/usr/bin/env python3
"""Visualize which rows each outlier rule removes for MIMIC job 40131."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_GLOB = "mimic_dora_paper_r2_vllm_shard_40131_*.out"
OUT_DIR = ROOT / "plot" / "64_2epoch_30sample_40131"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = OUT_DIR / "outlier_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
NATURE_STYLE = ROOT / ".agents" / "skills" / "scientific-visualization" / "assets" / "nature.mplstyle"

matplotlib_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "dtgpt_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib.pyplot as plt


if NATURE_STYLE.exists():
    plt.style.use(str(NATURE_STYLE))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    }
)


VARIABLES = {
    "220210": "Respiratory Rate",
    "220277": "SpO2",
    "220635": "Magnesium",
}

RULE_ORDER = [
    "target_abs_le_1000",
    "target_3sd_clip_after_3sd_removal",
    "target_and_pred_abs_le_1000",
    "target_iqr3",
    "target_and_pred_iqr3",
    "target_p0.1_99.9",
    "target_and_pred_p0.1_99.9",
    "target_p1_99",
    "target_and_pred_p1_99",
]

RULE_LABELS = {
    "target_abs_le_1000": "target\nabs<=1000",
    "target_3sd_clip_after_3sd_removal": "target\n3SD+clip",
    "target_and_pred_abs_le_1000": "target+pred\nabs<=1000",
    "target_iqr3": "target\nIQR3",
    "target_and_pred_iqr3": "target+pred\nIQR3",
    "target_p0.1_99.9": "target\np0.1-99.9",
    "target_and_pred_p0.1_99.9": "target+pred\np0.1-99.9",
    "target_p1_99": "target\np1-99",
    "target_and_pred_p1_99": "target+pred\np1-99",
}


def extract_dataframe_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
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
        pairs.append((target_path, pred_path))
    if len(pairs) != 8:
        raise RuntimeError(f"Expected 8 shard logs, found {len(pairs)}")
    return pairs


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frames = []
    pred_frames = []
    usecols = ["patientid", "patient_sample_index", "date", *VARIABLES]
    for shard_id, (target_path, pred_path) in enumerate(extract_dataframe_pairs()):
        target = pd.read_csv(target_path, usecols=lambda c: c in usecols)
        pred = pd.read_csv(pred_path, usecols=lambda c: c in usecols)
        if len(target) != len(pred):
            raise RuntimeError(f"Row count mismatch for {target_path}: {len(target)} != {len(pred)}")
        target = target.copy()
        pred = pred.copy()
        target["source_shard"] = shard_id
        pred["source_shard"] = shard_id
        target["source_row"] = np.arange(len(target))
        pred["source_row"] = np.arange(len(pred))
        target_frames.append(target)
        pred_frames.append(pred)
    return pd.concat(target_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def iqr_bounds(series: pd.Series, k: float = 3.0) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def percentile_bounds(series: pd.Series, low: float, high: float) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    return tuple(float(v) for v in np.percentile(values, [low, high]))


def mean_sd_bounds(series: pd.Series, k: float = 3.0) -> tuple[float, float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    return mean, sd, mean - k * sd, mean + k * sd


def between(series: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    low, high = bounds
    return series.between(low, high)


def build_rule_masks(
    target: pd.DataFrame,
    pred: pd.DataFrame,
) -> tuple[dict[str, dict[str, pd.Series]], list[dict[str, object]], pd.DataFrame]:
    masks: dict[str, dict[str, pd.Series]] = {rule: {} for rule in RULE_ORDER}
    bound_rows: list[dict[str, object]] = []
    two_step_rows: list[dict[str, object]] = []
    for code, variable in VARIABLES.items():
        y_true = numeric(target, code)
        y_pred = numeric(pred, code)
        valid = y_true.notna() & y_pred.notna()
        target_available = y_true.notna()

        first_mean, first_sd, first_low, first_high = mean_sd_bounds(y_true[target_available])
        stage1_target_inlier = target_available & y_true.between(first_low, first_high)
        second_mean, second_sd, clip_low, clip_high = mean_sd_bounds(y_true[stage1_target_inlier])
        stage1_valid_inlier = valid & y_true.between(first_low, first_high)
        clipped_valid = stage1_valid_inlier & ~y_true.between(clip_low, clip_high)
        target_iqr3 = iqr_bounds(y_true)
        pred_iqr3 = iqr_bounds(y_pred)
        target_p001 = percentile_bounds(y_true, 0.1, 99.9)
        pred_p001 = percentile_bounds(y_pred, 0.1, 99.9)
        target_p1 = percentile_bounds(y_true, 1.0, 99.0)
        pred_p1 = percentile_bounds(y_pred, 1.0, 99.0)

        for name, bounds in [
            ("target_3sd_initial_remove", (first_low, first_high)),
            ("target_3sd_recomputed_clip", (clip_low, clip_high)),
            ("target_iqr3", target_iqr3),
            ("pred_iqr3", pred_iqr3),
            ("target_p0.1_99.9", target_p001),
            ("pred_p0.1_99.9", pred_p001),
            ("target_p1_99", target_p1),
            ("pred_p1_99", pred_p1),
        ]:
            bound_rows.append(
                {
                    "variable_code": code,
                    "variable": variable,
                    "bound": name,
                    "low": bounds[0],
                    "high": bounds[1],
                }
            )

        two_step_rows.append(
            {
                "variable_code": code,
                "variable": variable,
                "target_rows_available": int(target_available.sum()),
                "valid_rows_before_outlier_rule": int(valid.sum()),
                "first_pass_mean": first_mean,
                "first_pass_sd": first_sd,
                "first_pass_low": first_low,
                "first_pass_high": first_high,
                "removed_stage1_target_rows": int((target_available & ~stage1_target_inlier).sum()),
                "removed_stage1_valid_pairs": int((valid & ~stage1_valid_inlier).sum()),
                "second_pass_mean_after_stage1": second_mean,
                "second_pass_sd_after_stage1": second_sd,
                "clip_low": clip_low,
                "clip_high": clip_high,
                "kept_valid_pairs_after_stage1": int(stage1_valid_inlier.sum()),
                "clipped_valid_pairs_after_stage1": int(clipped_valid.sum()),
                "removed_stage1_pct_of_valid": 100.0 * (valid & ~stage1_valid_inlier).sum() / valid.sum() if valid.sum() else np.nan,
                "clipped_pct_of_stage1_kept": 100.0 * clipped_valid.sum() / stage1_valid_inlier.sum() if stage1_valid_inlier.sum() else np.nan,
            }
        )

        masks["target_abs_le_1000"][code] = valid & (y_true.abs() <= 1000)
        masks["target_3sd_clip_after_3sd_removal"][code] = stage1_valid_inlier
        masks["target_and_pred_abs_le_1000"][code] = valid & (y_true.abs() <= 1000) & (y_pred.abs() <= 1000)
        masks["target_iqr3"][code] = valid & between(y_true, target_iqr3)
        masks["target_and_pred_iqr3"][code] = valid & between(y_true, target_iqr3) & between(y_pred, pred_iqr3)
        masks["target_p0.1_99.9"][code] = valid & between(y_true, target_p001)
        masks["target_and_pred_p0.1_99.9"][code] = valid & between(y_true, target_p001) & between(y_pred, pred_p001)
        masks["target_p1_99"][code] = valid & between(y_true, target_p1)
        masks["target_and_pred_p1_99"][code] = valid & between(y_true, target_p1) & between(y_pred, pred_p1)
    return masks, bound_rows, pd.DataFrame(two_step_rows)


def build_accounting(target: pd.DataFrame, pred: pd.DataFrame, masks: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    total_rows = len(target)
    for rule in RULE_ORDER:
        for code, variable in VARIABLES.items():
            y_true = numeric(target, code)
            y_pred = numeric(pred, code)
            valid = y_true.notna() & y_pred.notna()
            kept = masks[rule][code]
            valid_rows = int(valid.sum())
            kept_rows = int(kept.sum())
            missing_or_invalid = int(total_rows - valid_rows)
            removed_by_rule = int(valid_rows - kept_rows)
            rows.append(
                {
                    "rule": rule,
                    "rule_label": RULE_LABELS[rule].replace("\n", " "),
                    "variable_code": code,
                    "variable": variable,
                    "total_rows": total_rows,
                    "valid_rows_before_outlier_rule": valid_rows,
                    "missing_or_invalid_rows": missing_or_invalid,
                    "kept_rows": kept_rows,
                    "removed_by_outlier_rule": removed_by_rule,
                    "removed_pct_of_valid": 100.0 * removed_by_rule / valid_rows if valid_rows else np.nan,
                    "kept_pct_of_valid": 100.0 * kept_rows / valid_rows if valid_rows else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_removed_examples(
    target: pd.DataFrame,
    pred: pd.DataFrame,
    masks: dict[str, dict[str, pd.Series]],
    max_rows_per_group: int = 30,
) -> pd.DataFrame:
    rows = []
    for rule in RULE_ORDER:
        for code, variable in VARIABLES.items():
            y_true = numeric(target, code)
            y_pred = numeric(pred, code)
            valid = y_true.notna() & y_pred.notna()
            removed = valid & ~masks[rule][code]
            frame = pd.DataFrame(
                {
                    "rule": rule,
                    "variable_code": code,
                    "variable": variable,
                    "patientid": target.loc[removed, "patientid"].astype(str),
                    "date": target.loc[removed, "date"].astype(str),
                    "source_shard": target.loc[removed, "source_shard"].astype(int),
                    "source_row": target.loc[removed, "source_row"].astype(int),
                    "target": y_true[removed].astype(float),
                    "prediction": y_pred[removed].astype(float),
                }
            )
            if frame.empty:
                continue
            frame["abs_target"] = frame["target"].abs()
            frame["abs_prediction"] = frame["prediction"].abs()
            frame["abs_error"] = (frame["target"] - frame["prediction"]).abs()
            frame = frame.sort_values(["abs_target", "abs_error"], ascending=False).head(max_rows_per_group)
            rows.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_two_step_examples(target: pd.DataFrame, pred: pd.DataFrame, two_step: pd.DataFrame, max_rows_per_group: int = 30) -> pd.DataFrame:
    rows = []
    for code, variable in VARIABLES.items():
        y_true = numeric(target, code)
        y_pred = numeric(pred, code)
        valid = y_true.notna() & y_pred.notna()
        rule = two_step[two_step["variable_code"] == code].iloc[0]
        stage1_removed = valid & ~y_true.between(rule["first_pass_low"], rule["first_pass_high"])
        stage1_kept = valid & y_true.between(rule["first_pass_low"], rule["first_pass_high"])
        clipped = stage1_kept & ~y_true.between(rule["clip_low"], rule["clip_high"])
        for action, mask in [("removed_stage1_3sd", stage1_removed), ("clipped_stage2_recomputed_3sd", clipped)]:
            frame = pd.DataFrame(
                {
                    "action": action,
                    "variable_code": code,
                    "variable": variable,
                    "patientid": target.loc[mask, "patientid"].astype(str),
                    "date": target.loc[mask, "date"].astype(str),
                    "source_shard": target.loc[mask, "source_shard"].astype(int),
                    "source_row": target.loc[mask, "source_row"].astype(int),
                    "target_original": y_true[mask].astype(float),
                    "target_after_clip": y_true[mask].clip(rule["clip_low"], rule["clip_high"]).astype(float),
                    "prediction": y_pred[mask].astype(float),
                }
            )
            if frame.empty:
                continue
            frame["abs_target_original"] = frame["target_original"].abs()
            frame["abs_clip_delta"] = (frame["target_original"] - frame["target_after_clip"]).abs()
            frame = frame.sort_values(["abs_target_original", "abs_clip_delta"], ascending=False).head(max_rows_per_group)
            rows.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def plot_accounting(accounting: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    colors = {
        "kept": "#4C78A8",
        "removed": "#F58518",
        "missing": "#BAB0AC",
    }
    labels = [RULE_LABELS[rule] for rule in RULE_ORDER]
    x = np.arange(len(RULE_ORDER))
    for ax, (code, variable) in zip(axes, VARIABLES.items()):
        subset = accounting[(accounting["variable_code"] == code)].set_index("rule").loc[RULE_ORDER]
        total = subset["total_rows"].to_numpy(dtype=float)
        kept_pct = 100.0 * subset["kept_rows"].to_numpy(dtype=float) / total
        removed_pct = 100.0 * subset["removed_by_outlier_rule"].to_numpy(dtype=float) / total
        missing_pct = 100.0 * subset["missing_or_invalid_rows"].to_numpy(dtype=float) / total
        ax.bar(x, kept_pct, label="Kept valid rows", color=colors["kept"])
        ax.bar(x, removed_pct, bottom=kept_pct, label="Removed by outlier rule", color=colors["removed"])
        ax.bar(x, missing_pct, bottom=kept_pct + removed_pct, label="Missing / invalid before rule", color=colors["missing"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of all rows")
        ax.set_title(variable)
        for idx, row in enumerate(subset.itertuples()):
            if row.removed_by_outlier_rule:
                ax.text(idx, kept_pct[idx] + removed_pct[idx] / 2, f"{row.removed_by_outlier_rule}", ha="center", va="center", fontsize=7)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=8)
    fig.suptitle("Deletion accounting: kept vs outlier-removed vs missing rows", fontsize=15)
    path = ANALYSIS_DIR / "outlier_removal_accounting_40131_30sample.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_removed_counts(accounting: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(RULE_ORDER))
    width = 0.24
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    for offset, (code, variable), color in zip([-width, 0, width], VARIABLES.items(), palette):
        subset = accounting[accounting["variable_code"] == code].set_index("rule").loc[RULE_ORDER]
        values = subset["removed_by_outlier_rule"].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=variable, color=color)
        ax.bar_label(bars, labels=[f"{int(v)}" if v else "" for v in values], fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_LABELS[rule] for rule in RULE_ORDER], fontsize=8)
    ax.set_ylabel("Rows removed among valid rows")
    ax.set_title("Actual outlier-rule removals only (missing rows excluded)")
    ax.legend()
    path = ANALYSIS_DIR / "outlier_removed_counts_only_40131_30sample.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_cutoff_line(
    ax,
    value: float,
    low_clip: float,
    high_clip: float,
    color: str,
    style: str,
    label: str,
    offscreen_slot: int,
) -> None:
    """Draw visible cutoff lines; pin off-view cutoffs to the display edge."""
    if low_clip <= value <= high_clip:
        ax.axvline(value, color=color, linestyle=style, linewidth=1.4, label=label)
        return

    span = high_clip - low_clip
    inset = span * 0.006 * offscreen_slot
    edge = low_clip + inset if value < low_clip else high_clip - inset
    ax.axvline(edge, color=color, linestyle=style, linewidth=1.4, alpha=0.8, label=label)


def plot_target_distributions(target: pd.DataFrame, pred: pd.DataFrame, bounds_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    line_styles = {
        "target_iqr3": ("#F58518", "--", "IQR3"),
        "target_p0.1_99.9": ("#54A24B", "-.", "p0.1-99.9"),
        "target_p1_99": ("#B279A2", ":", "p1-99"),
    }
    for ax, (code, variable) in zip(axes, VARIABLES.items()):
        y_true = numeric(target, code)
        y_pred = numeric(pred, code)
        valid_values = y_true[y_true.notna() & y_pred.notna()].astype(float)
        low_clip, high_clip = np.percentile(valid_values, [0.2, 99.8])
        clipped = valid_values.clip(low_clip, high_clip)
        ax.hist(clipped, bins=80, color="#4C78A8", alpha=0.75)
        ax.set_title(f"{variable}: target distribution among valid rows")
        ax.set_ylabel("Rows")
        ax.set_xlabel("Target value (standardized scale, clipped for display)")
        for offscreen_slot, (bound_name, (color, style, label)) in enumerate(line_styles.items()):
            row = bounds_df[(bounds_df["variable_code"] == code) & (bounds_df["bound"] == bound_name)].iloc[0]
            for value in [row["low"], row["high"]]:
                draw_cutoff_line(ax, value, low_clip, high_clip, color, style, label, offscreen_slot)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        outside = int(((valid_values < low_clip) | (valid_values > high_clip)).sum())
        ax.text(0.99, 0.92, f"display clipped: {outside} rows outside", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.text(0.99, 0.85, "edge lines mark off-view cutoffs", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#4D4D4D")
    fig.suptitle("Where target-based rules place their cutoffs", fontsize=15)
    path = ANALYSIS_DIR / "outlier_target_distribution_thresholds_40131_30sample.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_two_step_filter(target: pd.DataFrame, pred: pd.DataFrame, two_step: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for panel_label, ax, (code, variable) in zip(["a", "b", "c"], axes, VARIABLES.items()):
        y_true = numeric(target, code)
        y_pred = numeric(pred, code)
        valid_values = y_true[y_true.notna() & y_pred.notna()].astype(float)
        rule = two_step[two_step["variable_code"] == code].iloc[0]
        low_display, high_display = np.percentile(valid_values, [0.2, 99.8])
        displayed = valid_values.clip(low_display, high_display)
        ax.hist(displayed, bins=60, color="#56B4E9", alpha=0.85, linewidth=0)
        for value, color, style, label in [
            (rule["first_pass_low"], "#D55E00", "--", "initial 3SD removal"),
            (rule["first_pass_high"], "#D55E00", "--", "initial 3SD removal"),
            (rule["clip_low"], "#0072B2", "-", "recomputed 3SD clip"),
            (rule["clip_high"], "#0072B2", "-", "recomputed 3SD clip"),
        ]:
            draw_cutoff_line(ax, float(value), low_display, high_display, color, style, label, 1)
        ax.set_title(variable)
        ax.set_xlabel("Standardized target")
        if ax is axes[0]:
            ax.set_ylabel("Rows")
        ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        annotation = (
            f"valid n={int(rule['valid_rows_before_outlier_rule']):,}\n"
            f"removed={int(rule['removed_stage1_valid_pairs']):,}\n"
            f"clipped={int(rule['clipped_valid_pairs_after_stage1']):,}"
        )
        ax.text(
            0.96,
            0.92,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Target-only two-step outlier filtering", fontsize=9)
    path = ANALYSIS_DIR / "outlier_target_two_step_filter_40131_30sample.png"
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def update_outlier_analysis_markdown(two_step: pd.DataFrame) -> None:
    analysis_md = ANALYSIS_DIR / "outlier_analysis.md"
    two_step_display = two_step.copy()
    summary_text = f"""
# Target-only two-step outlier filtering

這是目前主檢查要看的 target-based 規則，不使用 prediction value 來決定 outlier boundary。

流程：

- Step 1: 對 target values 計算 mean ± 3 SD，先移除超過這個範圍的 target rows。
- Step 2: 在 Step 1 保留下來的 target values 上重新計算 mean 和 SD。
- Step 3: 對 Step 1 保留下來但超過重新計算後 mean ± 3 SD 的 target values 做 clipping；這一步不刪 row。

| variable | valid rows | removed in step 1 | removed % | clipped in step 2 | clipped % among kept |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(
    "| {variable} | {valid_rows_before_outlier_rule:,} | {removed_stage1_valid_pairs:,} | {removed_stage1_pct_of_valid:.3f} | {clipped_valid_pairs_after_stage1:,} | {clipped_pct_of_stage1_kept:.3f} |".format(**row)
    for row in two_step_display.to_dict(orient="records")
)}

![Target-only two-step filter](outlier_target_two_step_filter_40131_30sample.png)

重點解讀：

- Respiratory Rate 的第一步 SD 被兩個巨大 target outliers 拉大；移除後重新計算 SD，才會有 730 筆進入 clipping。
- SpO2 第一階段只移除 1 筆，但重算 SD 後有 843 筆被 clipping。
- Magnesium 第一階段移除 54 筆，第二階段 clipping 70 筆；Magnesium 少很多 row 的主要原因仍然是原本 valid target/prediction pair 少，不是 prediction-based deletion。

"""
    start = "<!-- TARGET_TWO_STEP_FILTER:START -->"
    end = "<!-- TARGET_TWO_STEP_FILTER:END -->"
    block = f"{start}\n{summary_text.strip()}\n{end}\n"
    text = analysis_md.read_text(encoding="utf-8")
    if start in text and end in text:
        text = text.split(start)[0].rstrip() + "\n\n" + block + "\n" + text.split(end, 1)[1].lstrip()
    else:
        text = block + "\n" + text
    analysis_md.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str]) -> str:
    table = df[columns].copy()
    for column in float_cols:
        if column in table:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    table = table.fillna("")
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def update_markdown(accounting: pd.DataFrame, two_step: pd.DataFrame, plot_paths: list[Path]) -> None:
    outlier_md = OUT_DIR / "outlier_sensitivity_analysis_40131_30sample.md"
    standard = accounting[
        accounting["rule"].isin(["target_abs_le_1000", "target_3sd_clip_after_3sd_removal", "target_iqr3", "target_and_pred_iqr3"])
    ]
    two_step_display = two_step.copy()
    for column in [
        "first_pass_mean",
        "first_pass_sd",
        "first_pass_low",
        "first_pass_high",
        "second_pass_mean_after_stage1",
        "second_pass_sd_after_stage1",
        "clip_low",
        "clip_high",
        "removed_stage1_pct_of_valid",
        "clipped_pct_of_stage1_kept",
    ]:
        two_step_display[column] = two_step_display[column].astype(float)
    summary_text = f"""

---

## Deleted-Row Visualization

These plots separate three different concepts that were previously easy to confuse:

- `missing / invalid before rule`: rows that did not have both target and prediction, so they were not evaluated by that variable.
- `removed by outlier rule`: rows that had both target and prediction but failed the outlier rule.
- `kept valid rows`: rows used for metric calculation.

### Key table: standard rules

{markdown_table(standard, ["rule", "variable", "valid_rows_before_outlier_rule", "kept_rows", "removed_by_outlier_rule", "removed_pct_of_valid"], {"removed_pct_of_valid"})}

### Target-only two-step 3SD + clipping

This rule is target-based only. The prediction value is used only to decide whether a row is evaluable, not to set the outlier boundary.

Step 1 removes target values outside mean ± 3 SD. Step 2 recomputes mean and SD after Step 1, then clips remaining target values to the recomputed mean ± 3 SD range.

{markdown_table(two_step_display, ["variable", "valid_rows_before_outlier_rule", "removed_stage1_valid_pairs", "removed_stage1_pct_of_valid", "kept_valid_pairs_after_stage1", "clipped_valid_pairs_after_stage1", "clipped_pct_of_stage1_kept"], {"removed_stage1_pct_of_valid", "clipped_pct_of_stage1_kept"})}

![Target-only two-step filter](outlier_analysis/outlier_target_two_step_filter_40131_30sample.png)

**Figure 0.** Target-only two-step rule requested for the main outlier check. Dashed orange lines are the first mean ± 3 SD removal boundary. Solid blue lines are the recomputed mean ± 3 SD clipping boundary. Clipping changes target values; it does not delete additional rows.

### Figures

![Deletion accounting](outlier_analysis/outlier_removal_accounting_40131_30sample.png)

**Figure A.** Full accounting for each variable and rule. Gray means missing/invalid before any outlier rule. Orange is the actual rule deletion. This shows Magnesium has many missing rows, not many IQR-deleted rows.

![Actual removed counts](outlier_analysis/outlier_removed_counts_only_40131_30sample.png)

**Figure B.** Actual outlier-rule deletion counts only, after excluding missing rows. This is the cleanest plot for answering which method deletes how much valid data.

![Target thresholds](outlier_analysis/outlier_target_distribution_thresholds_40131_30sample.png)

**Figure C.** Target value distributions and target-based cutoff lines. This shows where IQR3 and percentile rules draw their boundaries on the standardized scale.

Generated files:

{chr(10).join(f"- `{path.name}`" for path in plot_paths)}
- `outlier_removal_accounting_40131_30sample.csv`
- `outlier_removed_data_examples_40131_30sample.csv`
- `outlier_target_two_step_filter_40131_30sample.csv`
- `outlier_target_two_step_examples_40131_30sample.csv`
"""
    start = "<!-- DELETED_ROW_VISUALIZATION:START -->"
    end = "<!-- DELETED_ROW_VISUALIZATION:END -->"
    block = f"{start}\n{summary_text.strip()}\n{end}\n"
    text = outlier_md.read_text(encoding="utf-8")
    if start in text and end in text:
        text = text.split(start)[0].rstrip() + "\n\n" + block + "\n" + text.split(end, 1)[1].lstrip()
    else:
        text = text.rstrip() + "\n\n" + block
    outlier_md.write_text(text, encoding="utf-8")


def main() -> None:
    target, pred = load_frames()
    masks, bound_rows, two_step = build_rule_masks(target, pred)
    bounds_df = pd.DataFrame(bound_rows)
    accounting = build_accounting(target, pred, masks)
    examples = build_removed_examples(target, pred, masks)
    two_step_examples = build_two_step_examples(target, pred, two_step)

    accounting.to_csv(OUT_DIR / "outlier_removal_accounting_40131_30sample.csv", index=False)
    bounds_df.to_csv(OUT_DIR / "outlier_removal_rule_bounds_40131_30sample.csv", index=False)
    examples.to_csv(OUT_DIR / "outlier_removed_data_examples_40131_30sample.csv", index=False)
    two_step.to_csv(OUT_DIR / "outlier_target_two_step_filter_40131_30sample.csv", index=False)
    two_step_examples.to_csv(OUT_DIR / "outlier_target_two_step_examples_40131_30sample.csv", index=False)

    plot_paths = [
        plot_two_step_filter(target, pred, two_step),
        plot_accounting(accounting),
        plot_removed_counts(accounting),
        plot_target_distributions(target, pred, bounds_df),
    ]
    update_markdown(accounting, two_step, plot_paths)
    update_outlier_analysis_markdown(two_step)

    print(accounting.to_string(index=False))
    print(two_step.to_string(index=False))
    print("Wrote:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
