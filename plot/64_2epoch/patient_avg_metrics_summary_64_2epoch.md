# MIMIC DoRA r64 2-Epoch Metrics

Source logs: `logs/mimic_dora_paper_r2_vllm_shard_40007_*.out`

Outlier rule: rows with `abs(target) > 1000.0` are excluded before metric calculation.

## Paper-Style Correlation Preservation R2

`R2_corr = -1.127794`

| pair                          |   rows |   true_corr |   pred_corr |      diff |
|:------------------------------|-------:|------------:|------------:|----------:|
| Respiratory Rate vs SpO2      |  98238 |   -0.066436 |   -0.115799 | -0.049362 |
| Respiratory Rate vs Magnesium |   6728 |   -0.010806 |   -0.007499 |  0.003307 |
| SpO2 vs Magnesium             |   6665 |   -0.039346 |   -0.010265 |  0.029081 |

## Step-Level sMAE

| variable         |   rows |   mae_scaled |     smae |        r2 |
|:-----------------|-------:|-------------:|---------:|----------:|
| Respiratory Rate | 101083 |     0.785076 | 0.785076 |  0.005669 |
| SpO2             |  99598 |     0.691073 | 0.691073 | -0.000264 |
| Magnesium        |   6929 |     0.574592 | 0.574592 |  0.154716 |
| Unweighted mean  | 207610 |     0.68358  | 0.68358  |           |
| Weighted mean    | 207610 |     0.732954 | 0.732954 |           |

## Patient-Averaged Metrics

| variable         |   patients |   r2_patient |   raw_mae |   true_patient_avg_std |     smae |
|:-----------------|-----------:|-------------:|----------:|-----------------------:|---------:|
| Respiratory Rate |       5520 |     0.283142 |  0.466342 |               0.79129  | 0.589344 |
| SpO2             |       5507 |     0.293807 |  0.444862 |               0.858514 | 0.518177 |
| Magnesium        |       4322 |     0.13895  |  0.508093 |               0.963151 | 0.527532 |

## Figures

- `plot/64_2epoch/patient_avg_true_vs_pred_64_2epoch.png`
- `plot/64_2epoch/step_scaled_mae_64_2epoch.png`
- `plot/64_2epoch/correlation_preservation_64_2epoch.png`
