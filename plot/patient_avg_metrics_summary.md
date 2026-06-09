# Patient-Averaged Evaluation Metrics & Plotting Summary

This document summarizes the math and processing workflow used to calculate the patient-averaged evaluation metrics ($R^2$ and sMAE) and the implementation details of the plotting scripts for the 1-epoch DT-GPT DoRA model evaluation.

---

## 1. Metric Calculation Methodology

The metrics shown in the plots represent **patient-level averages** across the clinical sequences, rather than step-level predictions. This assesses how well the model predicts the overall physiological baseline of individual patients.

> **Important distinction:** these plot metrics are **not** the same as the paper's main step-level scaled MAE. The plots first average each patient's trajectory and then compute patient-level metrics. The paper's reported ICU scaled MAE is computed from step-level forecast errors and then normalized by the clinical variable's standard deviation after outlier filtering.

### Step A: Patient-Level Averaging
For each patient $p$ in the dataset, we aggregate all sequence time steps to compute a single average true value $\overline{y}_p$ and average predicted value $\overline{\hat{y}}_p$:

$$\overline{y}_p = \frac{1}{T_p} \sum_{t=1}^{T_p} y_{p,t}, \quad \overline{\hat{y}}_p = \frac{1}{T_p} \sum_{t=1}^{T_p} \hat{y}_{p,t}$$

*   $T_p$: Total number of valid time steps recorded for patient $p$.
*   $y_{p,t}$: True value of the clinical variable for patient $p$ at time step $t$.
*   $\hat{y}_{p,t}$: Model prediction of the clinical variable for patient $p$ at time step $t$.

### Step B: Z-score Standardization
To make metrics comparable across variables with different baseline values and physical units (e.g., Respiratory Rate vs. Magnesium), patient-level averages are standardized using the mean $\mu_{\overline{y}}$ and standard deviation $\sigma_{\overline{y}}$ of the true patient-averaged values:

$$mt\_z_p = \frac{\overline{y}_p - \mu_{\overline{y}}}{\sigma_{\overline{y}}}$$

$$mp\_z_p = \frac{\overline{\hat{y}}_p - \mu_{\overline{y}}}{\sigma_{\overline{y}}}$$

### Step C: Computing $R^2$ and sMAE
Using the standardized arrays $mt\_z$ and $mp\_z$:

1.  **sMAE (Scaled Mean Absolute Error / 標準化平均絕對誤差)**:
    $$\text{sMAE} = \frac{1}{N} \sum_{p=1}^N |mt\_z_p - mp\_z_p|$$
    *This is mathematically equivalent to scaling the raw patient-level MAE by the true standard deviation:* $\text{sMAE} = \frac{\text{MAE}}{\sigma_{\overline{y}}}$.

2.  **Patient-level $R^2$ (Coefficient of Determination / 決定係數)**:
    $$R^2 = 1 - \frac{\sum_{p=1}^N (mt\_z_p - mp\_z_p)^2}{\sum_{p=1}^N (mt\_z_p - \overline{mt\_z})^2}$$
    *Since $mt\_z$ is standardized, the denominator $\sum (mt\_z_p - \overline{mt\_z})^2$ simplifies to the total number of patients $N$.*

### Step D: Interpreting the Plot MAE

The plot title labels use **patient-averaged MAE on the dataframe scale**. This is not identical to the patient-level sMAE defined above unless the standard deviation of the true patient averages is exactly 1.

For example, Respiratory Rate without the anomalous Patient 13793 outlier has:

*   patient-level raw MAE on the normalized dataframe scale: approximately **0.364**
*   patient-level sMAE after dividing by the standard deviation of true patient averages: approximately **0.490**

Therefore, when comparing to the paper, use the **step-level scaled MAE** section below rather than the plot MAE.

---

## 2. Paper-Style Step-Level Scaled MAE

The DT-GPT paper reports scaled MAE as:

$$\text{scaled MAE}_j = \frac{\frac{1}{N_j}\sum_i |y_{i,j} - \hat{y}_{i,j}|}{\sigma_j}$$

where:

*   $j$ is a clinical variable.
*   $i$ indexes valid step-level forecast rows for that variable.
*   $\sigma_j$ is the standard deviation of the clinical variable after outlier filtering.

In this repository's saved evaluation dataframes, numeric targets and predictions are already on the normalized dataframe scale used by `MetricManager.mae`. Therefore:

*   `MetricManager.mae` is a step-level MAE on the normalized/scaled dataframe values.
*   With the same outlier filtering, this value is the practical paper-style scaled MAE for each variable.
*   Do **not** first average by patient if the goal is to reproduce the paper's main ICU scaled MAE.

### Recommended Calculation

1.  Load all saved shard target and prediction dataframes.
2.  For each variable independently, keep rows where the target is non-missing.
3.  Apply outlier filtering on the target value, e.g. `abs(target) > 1000.0`.
4.  Compute step-level MAE:

    $$\text{MAE}_{scaled,j} = \frac{1}{N_j}\sum_i |target_{i,j} - prediction_{i,j}|$$

5.  Aggregate across variables either:
    *   **unweighted mean** across variables, matching `MetricManager`'s all-numeric-column averaging style; or
    *   **weighted mean** by number of observed values, if the goal is an observation-weighted error.

### Example Result from Checkpoint 5602 8-Shard Evaluation

Using the 8 `mimic_dora_vllm5602_shard_39082_*` target/prediction CSVs and filtering `abs(target) > 1000.0`:

| Variable | Step Rows | Step-Level Scaled MAE |
|---|---:|---:|
| Respiratory Rate | 51,169 | 0.6448 |
| SpO2 | 50,578 | 0.5766 |
| Magnesium | 3,461 | 0.4578 |
| **Unweighted mean across variables** | — | **0.5598** |
| **Weighted mean by observed values** | 105,208 | **0.6059** |

The unweighted mean (**0.5598**) is the closest analogue to the repository's `all_numeric_columns mae overall` convention, while the weighted mean (**0.6059**) reflects the fact that Magnesium is much sparser than Respiratory Rate and SpO2.

Without the outlier filter, Patient 13793's anomalous Respiratory Rate target dominates Respiratory Rate MAE and inflates the overall result.

---

## 3. Visualization Script Details

*   **Script Path**: [generate_final_plots.py](file:///home/r15543056/.gemini/antigravity-cli/brain/77e18194-0edc-4ad4-b1cc-bcdf9c1f7b62/scratch/generate_final_plots.py)
*   **Data Source**: 8 evaluation shards from directory `/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPT/setup/setup/` (1-epoch setup).

### Features Implemented in the Script:
1.  **Outlier Filtering**: Includes a togglable outlier exclusion logic (`abs(target) > 1000.0`) which isolates and removes anomalous database recordings (such as Patient 13793 in Respiratory Rate) to show normal patient distributions.
2.  **Gaussian KDE Density Coloring**: Computes point density using `scipy.stats.gaussian_kde` and colors points via colorbars (Blues for Respiratory Rate, Greens for SpO₂, Purples for Magnesium) to reveal overlapping core clusters.
3.  **Outlier Annotation & Layout Padding**: Annotates the outlier Patient 13793 on the Respiratory Rate plot using standard offset vectors. The text box position is offset vertically to prevent overlap with the legend.
4.  **Tighter Spacing Layout**: Employs `figsize=(18, 5.6)` and `plt.tight_layout(rect=[0, 0, 1, 0.92])` with adjusted title y-positions to minimize whitespace above the subplots.

---

## 4. Output Figures

The script outputs the following two high-resolution (150 DPI) plots directly in the plot directory:

1.  **Without Outliers**: [patient_avg_true_vs_pred_merged_1epoch.png](file:///home/r15543056/trajectory_forecast/DT-GPT/plot/patient_avg_true_vs_pred_merged_1epoch.png)
    *   *Titles*: "Patient-Averaged True vs Predicted (Standardized)", "DT-GPT with DoRA Finetune 1 epoch"
    *   *Respiratory Rate $R^2$*: **0.5717** | *MAE*: **0.3637**
2.  **With Outlier Included**: [patient_avg_true_vs_pred_merged_with_outlier_1epoch.png](file:///home/r15543056/trajectory_forecast/DT-GPT/plot/patient_avg_true_vs_pred_merged_with_outlier_1epoch.png)
    *   *Titles*: "Patient-Averaged True vs Predicted (Standardized)", "DT-GPT with DoRA Finetune 1 epoch  |  outlier included"
    *   *Respiratory Rate $R^2$*: **0.0004** | *MAE*: **high scale** (outlier labeled at top-right)

---

## 5. Paper-Style Inter-Variable Correlation Preservation $R^2$

The $R^2$ reported in the DT-GPT paper for preserved inter-variable relationships is **not** the same as the patient-level true-vs-predicted $R^2$ above. The paper-style value compares whether the predicted trajectories preserve the **correlation structure between clinical variables**.

Reference interpretation:

*   **Patient-level prediction $R^2$**: compares $\overline{y}_p$ against $\overline{\hat{y}}_p$ for one variable at a time.
*   **Paper-style correlation $R^2$**: compares the pairwise correlation coefficients from the true dataset against the pairwise correlation coefficients from the predicted dataset.

### Step A: Build True and Predicted Correlation Vectors

For the ICU variables:

*   Respiratory Rate (`220210`)
*   O2 saturation pulseoxymetry / SpO2 (`220277`)
*   Magnesium (`220635`)

compute the true and predicted pairwise Pearson correlations:

$$C^{true}_{ij} = corr(y_i, y_j), \quad C^{pred}_{ij} = corr(\hat{y}_i, \hat{y}_j)$$

using the upper-triangle variable pairs:

$$[(RR, SpO2), (RR, Mg), (SpO2, Mg)]$$

This produces two vectors:

$$c^{true} = [C^{true}_{RR,SpO2}, C^{true}_{RR,Mg}, C^{true}_{SpO2,Mg}]$$

$$c^{pred} = [C^{pred}_{RR,SpO2}, C^{pred}_{RR,Mg}, C^{pred}_{SpO2,Mg}]$$

### Step B: Compute Correlation Preservation $R^2$

Compute the coefficient of determination between these two correlation vectors:

$$R^2_{corr} = 1 - \frac{\sum_k (c^{true}_k - c^{pred}_k)^2}{\sum_k (c^{true}_k - \overline{c^{true}})^2}$$

This measures how closely the predicted data preserves the true inter-variable correlation structure.

### Recommended Processing Details

*   Use the same target and prediction dataframe rows saved during shard evaluation.
*   Use **pairwise complete observations** for each variable pair, because Magnesium is much sparser than Respiratory Rate and SpO2.
*   Apply the same outlier exclusion rule used in the plots (`abs(target) > 1000.0`) before computing correlations. This removes anomalous database recordings such as Patient 13793's Respiratory Rate outlier.
*   Report the individual true/predicted pairwise correlations together with $R^2_{corr}$, because there are only three variables and therefore only three pairwise correlation coefficients.

### Example Result from Checkpoint 5602 8-Shard Evaluation

Using the 8 `mimic_dora_vllm5602_shard_39082_*` evaluation logs and their saved CSVs, with pairwise complete observations and `abs(target) > 1000.0` filtering:

| Variable Pair | True Corr. | Pred Corr. | Difference |
|---|---:|---:|---:|
| Respiratory Rate vs SpO2 | -0.1234 | -0.1325 | -0.0091 |
| Respiratory Rate vs Magnesium | -0.0318 | -0.0043 | +0.0275 |
| SpO2 vs Magnesium | -0.0565 | -0.0250 | +0.0315 |

$$R^2_{corr} \approx 0.5934$$

Without the outlier filter, the Respiratory Rate outlier can dominate the Respiratory Rate--SpO2 correlation and make this paper-style $R^2$ unstable or negative.

### Minimal Recalculation Script

```python
import csv
import math
import pathlib
import re

log_glob = "logs/mimic_dora_vllm5602_shard_39082_*.out"
vars_map = {
    "220210": "Respiratory Rate",
    "220277": "SpO2",
    "220635": "Magnesium",
}

def corr(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs)
        * sum((y - my) ** 2 for y in ys)
    )
    return num / den

def r2_score(y_true, y_pred):
    mean = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    return 1 - ss_res / ss_tot

pairs = []
for log in sorted(pathlib.Path(".").glob(log_glob)):
    text = log.read_text(errors="replace")
    target = re.search(r"Saved target dataframe: (.*?target_dataframe\.csv)", text)
    pred = re.search(r"Saved prediction dataframe: (.*?prediction_dataframe\.csv)", text)
    if target and pred:
        pairs.append((pathlib.Path(target.group(1)), pathlib.Path(pred.group(1))))

true_corrs = []
pred_corrs = []
labels = list(vars_map.items())

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        c1, n1 = labels[i]
        c2, n2 = labels[j]
        t1, t2, p1, p2 = [], [], [], []

        for target_path, pred_path in pairs:
            with target_path.open(newline="") as ft, pred_path.open(newline="") as fp:
                rt = csv.DictReader(ft)
                rp = csv.DictReader(fp)
                for tr, pr in zip(rt, rp):
                    vals = [
                        (tr.get(c1) or "").strip(),
                        (tr.get(c2) or "").strip(),
                        (pr.get(c1) or "").strip(),
                        (pr.get(c2) or "").strip(),
                    ]
                    if any(v == "" for v in vals):
                        continue
                    a, b, pa, pb = map(float, vals)
                    if abs(a) > 1000.0 or abs(b) > 1000.0:
                        continue
                    t1.append(a)
                    t2.append(b)
                    p1.append(pa)
                    p2.append(pb)

        true_corr = corr(t1, t2)
        pred_corr = corr(p1, p2)
        true_corrs.append(true_corr)
        pred_corrs.append(pred_corr)
        print(f"{n1} vs {n2}: true={true_corr:.6f}, pred={pred_corr:.6f}")

print("paper-style correlation R2:", r2_score(true_corrs, pred_corrs))
```
