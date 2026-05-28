import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Set professional layout parameters directly in matplotlib
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 18,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--"
})

# Paths to Shard 003 evaluation data
t_path = "/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPT/setup/setup/2026_05_27___20_39_19_161618/eval_meta_data/TEST_shard_003_of_016_target_dataframe.csv"
p_path = "/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPT/setup/setup/2026_05_27___20_39_19_161618/eval_meta_data/TEST_shard_003_of_016_prediction_dataframe.csv"

# Load dataframes
t_df = pd.read_csv(t_path)
p_df = pd.read_csv(p_path)

variables = {
    "220635": "Magnesium (220635)",
    "220277": "SpO2 (220277)",
    "220210": "Respiratory Rate (220210)"
}

# -------------------------------------------------------------
# CHART A: True vs. Predicted Scatter Plots
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (col, var_name) in enumerate(variables.items()):
    t_vals = t_df[col].dropna()
    p_vals = p_df.loc[t_vals.index, col]
    
    ax = axes[i]
    ax.grid(True)
    
    # Calculate R2 for the subtitle
    r2 = r2_score(t_vals, p_vals)
    
    # Highlight Patient 6704 in Magnesium
    if col == "220635":
        pat_ids = t_df.loc[t_vals.index, "patientid"]
        is_outlier = pat_ids == 6704
        
        # Plot normal points
        ax.scatter(t_vals[~is_outlier], p_vals[~is_outlier], alpha=0.6, color=colors[i], label="Normal patients")
        # Plot outlier points in red
        ax.scatter(t_vals[is_outlier], p_vals[is_outlier], alpha=0.9, color="#d62728", edgecolor="black", s=100, zorder=5, label="Patient 6704 (Outlier)")
        ax.legend(loc="upper left")
    else:
        ax.scatter(t_vals, p_vals, alpha=0.4, color=colors[i])
        
    # Draw y = x reference line
    min_val = min(t_vals.min(), p_vals.min())
    max_val = max(t_vals.max(), p_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="darkgray", linestyle="--", linewidth=1.5, label="y = x")
    
    ax.set_title(f"{var_name}\n$R^2 = {r2:.4f}$")
    ax.set_xlabel("True Standardized Value")
    if i == 0:
        ax.set_ylabel("Predicted Standardized Value")
    ax.set_xlim(min_val - 0.5, max_val + 0.5)
    ax.set_ylim(min_val - 0.5, max_val + 0.5)

plt.suptitle("True vs. Predicted Values for Shard 003 (Standardized)", y=1.02)
plt.tight_layout()
plt.savefig("/home/r15543056/trajectory_forecast/DT-GPT/scatter_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved scatter_comparison.png")

# -------------------------------------------------------------
# CHART B: Autocorrelation Function (ACF) Plot
# -------------------------------------------------------------
# We want to calculate how values correlate with their past values.
# Since SpO2/RR are hourly and Magnesium is daily, let's compute correlation between t and t-k.
# We sort by patient and date, then shift to compute autocorrelation.

plt.figure(figsize=(10, 6))
plt.grid(True)

for i, (col, var_name) in enumerate(variables.items()):
    autocorrs = []
    lags = range(1, 13) # Lags from 1 to 12 periods
    
    # Sort by patient and date
    df_sorted = t_df[["patientid", "date", col]].dropna().copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"], format="mixed", errors="coerce")
    df_sorted = df_sorted.dropna(subset=["date"])
    df_sorted = df_sorted.sort_values(by=["patientid", "date"])
    
    # We will compute autocorrelation for each lag
    lag_corrs = []
    for lag in lags:
        # Shift within each patient group to avoid cross-patient correlation
        df_sorted["lagged"] = df_sorted.groupby("patientid")[col].shift(lag)
        valid_rows = df_sorted[[col, "lagged"]].dropna()
        if len(valid_rows) > 5:
            corr = np.corrcoef(valid_rows[col], valid_rows["lagged"])[0, 1]
        else:
            corr = np.nan
        lag_corrs.append(corr)
        
    plt.plot(list(lags), lag_corrs, marker="o", linewidth=2.5, color=colors[i], label=var_name)

plt.title("Autocorrelation (ACF) of Patient Sequences")
plt.xlabel("Time Lag (Observation Steps)")
plt.ylabel("Correlation Coefficient")
plt.xticks(list(lags))
plt.ylim(-0.2, 1.1)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("/home/r15543056/trajectory_forecast/DT-GPT/acf_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved acf_comparison.png")
