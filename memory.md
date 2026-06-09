# 2026-06-09 Session Notes

## Scope

This session focused on the MIMIC DoRA evaluation outputs under `logs/mimic_dora_vllm5602_shard_39082_*.out`, the metric definitions used in the local plotting workflow, and how those metrics differ from the Nature DT-GPT paper.

## Metric Definition Clarifications

- The current local plot and `patient_avg_metrics_summary.md` use **patient-level prediction R2**:
  - Group by patient.
  - Average all time steps for each patient.
  - Compute `R2(y_true_patient_avg, y_pred_patient_avg)`.
- The paper's reported ICU `R2 ~= 0.99` is **not** the same metric.
- The paper-style `R2` is **inter-variable correlation preservation R2**:
  - Compute pairwise correlations between the true target variables.
  - Compute pairwise correlations between the predicted target variables.
  - Compare those two correlation vectors with an R2 formula.

Formulas discussed in this session:

```text
Patient-level prediction R2
= 1 - sum_p((ybar_p - yhatbar_p)^2) / sum_p((ybar_p - mean(ybar))^2)
```

```text
Paper-style correlation-preservation R2
= 1 - sum_k((c_true_k - c_pred_k)^2) / sum_k((c_true_k - mean(c_true))^2)
```

```text
Step-level scaled MAE
= MAE / sigma
= mean_i(|y_i - yhat_i|) / sigma
```

## What The Current Code Is Measuring

- `pipeline/MetricManager.py` uses `sklearn.metrics.r2_score` for `r2()`.
- `pipeline/MetricManager.py` uses `sklearn.metrics.mean_absolute_error` for `mae()`.
- The built-in log summary at evaluation time is **step-level**, based on saved target/prediction rows, not patient-averaged unless aggregated manually later.

## Key Findings From The Logs

- All 8 shards completed successfully.
- A major Respiratory Rate outlier was identified:
  - `patientid = 13793`
  - `date = 2024-01-02 08:00:00`
  - `Respiratory Rate target = 32977.80967121861`
  - `prediction = 0.10540834`
- This outlier is in shard 2 and heavily distorts Respiratory Rate metrics if not filtered.

## Recomputed Results From Current Files

### Patient-level metrics

- Respiratory Rate with outlier:
  - `R2 ~= 0.0003`
  - `raw MAE ~= 0.8810`
- Respiratory Rate after removing patients with `abs(patient_avg_target) > 1000`:
  - `R2 ~= 0.5689`
  - `raw MAE ~= 0.3640`
  - `sMAE ~= 0.4903`
- SpO2:
  - `R2 ~= 0.3395`
  - `raw MAE ~= 0.3638`
  - `sMAE ~= 0.4278`
- Magnesium:
  - `R2 ~= 0.2622`
  - `raw MAE ~= 0.4098`
  - `sMAE ~= 0.3589`

### Step-level scaled MAE aligned to the paper

Using saved shard CSVs and outlier filtering `abs(target) > 1000`:

- Respiratory Rate: `0.6448`
- SpO2: `0.5766`
- Magnesium: `0.4578`
- Unweighted mean across the 3 variables: `0.5598`
- Weighted mean across all observed rows: `0.6059`

Interpretation from this session:

- On **step-level scaled MAE**, the current DoRA checkpoint is already close to the paper's reported ICU range around `0.59`.

### Paper-style correlation-preservation R2

Using the same three MIMIC target variables:

- `220210` Respiratory Rate
- `220277` SpO2
- `220635` Magnesium

Recommended reporting variant:

- Pairwise complete observations
- Outlier filter: `abs(target) > 1000`

Computed result:

- `paper-style R2 ~= 0.593378`

Pairwise correlations used:

- RR vs SpO2:
  - true `-0.123431`
  - pred `-0.132482`
  - diff `-0.009051`
- RR vs Magnesium:
  - true `-0.031829`
  - pred `-0.004341`
  - diff `+0.027488`
- SpO2 vs Magnesium:
  - true `-0.056459`
  - pred `-0.024995`
  - diff `+0.031463`

Alternative variant also computed:

- Global complete-case `paper-style R2 ~= 0.662922`

Session conclusion:

- The paper-aligned R2 from the current files is **about 0.59**, not `0.99`.
- The local patient-level R2 and the paper R2 are different metrics and cannot be compared directly.

## Training Decisions Discussed

- Current training setup discussed by the user:
  - DoRA / LoRA `r = 64`
  - `alpha = 128`
- The plateau from epoch 1 to epoch 2 does **not** justify immediately increasing `lora_r`.
- Recommended next ablations discussed:
  - `r64 alpha128 lr2e-5 dropout0.05 split0.5`
  - `r64 alpha128 lr5e-5 dropout0.10 split0.5`
  - `r64 alpha128 lr2e-5 dropout0.10 split0.5`
- For fair comparison:
  - Prefer training from scratch if resources allow.
  - Otherwise resume from the **1-epoch** checkpoint, not the 4-epoch checkpoint.

## Documentation Added During This Session

- `plot/patient_avg_metrics_summary.md`
  - expanded with metric interpretation and paper-style calculations
- `plot/step_level_scaled_mae_notes.md`
  - step-level scaled MAE definition and comparison
- `plot/r2_metric_definitions.md`
  - local R2 vs paper R2 definition and formulas

## Reuse Notes For Future Sessions

- When comparing against the DT-GPT paper on MIMIC, use:
  - **step-level scaled MAE**
  - **paper-style correlation-preservation R2**
- Do not compare the paper's `R2 ~= 0.99` against local patient-level prediction R2.
- Always check for the RR outlier patient `13793` before interpreting Respiratory Rate results.
