# Outlier Sensitivity Analysis — MIMIC DoRA job 40131, 30 samples

Source logs: `logs/mimic_dora_paper_r2_vllm_shard_40131_*.out`. All 8 shards completed and saved target/prediction CSVs.

This file compares R2, R2-corr, and sMAE under several outlier-removal rules. The previously reported local rule is `target_abs_le_1000`.

## Rule summary

| rule                        |   r2_corr_definition_diagonal |   r2_corr_scatter_fit_corr_squared |   step_smae_unweighted |   step_smae_weighted |   mean_patient_r2 |   mean_patient_smae |
|:----------------------------|------------------------------:|-----------------------------------:|-----------------------:|---------------------:|------------------:|--------------------:|
| none                        |                    -30.6366   |                           0.257812 |               0.679367 |             0.783041 |          0.13677  |            0.206793 |
| target_abs_le_1000          |                     -5.21711  |                           0.924896 |               0.559732 |             0.608527 |          0.439173 |            0.431049 |
| target_and_pred_abs_le_1000 |                     -5.21711  |                           0.924896 |               0.559732 |             0.608527 |          0.439173 |            0.431049 |
| target_iqr3                 |                      0.691803 |                           0.909459 |               0.545888 |             0.59581  |          0.492479 |            0.517813 |
| target_and_pred_iqr3        |                      0.603291 |                           0.891143 |               0.544054 |             0.594497 |          0.509241 |            0.52794  |
| target_p0.1_99.9            |                      0.681706 |                           0.921904 |               0.549832 |             0.59944  |          0.514898 |            0.506922 |
| target_and_pred_p0.1_99.9   |                      0.561066 |                           0.887249 |               0.549399 |             0.59899  |          0.531903 |            0.509617 |
| target_p1_99                |                      0.66162  |                           0.92324  |               0.529568 |             0.57692  |          0.467907 |            0.532013 |
| target_and_pred_p1_99       |                      0.407001 |                           0.874081 |               0.529326 |             0.577096 |          0.500805 |            0.538283 |

## Interpretation

- `target_abs_le_1000` removes the known impossible target outlier(s), but does **not** remove extreme prediction values.
- `target_and_pred_abs_le_1000` is almost identical here, so there are no prediction values beyond ±1000 driving the result.
- IQR / percentile trimming can make patient R2 and sMAE look better, but it does **not** recover paper-level `R2_corr ~= 0.99` under the strict definition.
- The highest scatter-style correlation among these rules is shown in `r2_corr_scatter_fit_corr_squared`; this is not the same formula as `plot/r2_metric_definitions.md`'s strict `R2_corr`.

## Correlation-pair details for the standard outlier rule (`target_abs_le_1000`)

| pair                          |   rows |   true_corr |   pred_corr |      diff |
|:------------------------------|-------:|------------:|------------:|----------:|
| Respiratory Rate vs SpO2      |  98238 |   -0.066436 |   -0.157892 | -0.091455 |
| Respiratory Rate vs Magnesium |   6728 |   -0.010806 |    0.022072 |  0.032877 |
| SpO2 vs Magnesium             |   6665 |   -0.039346 |   -0.026024 |  0.013322 |

## Patient metrics for the standard outlier rule (`target_abs_le_1000`)

| variable         |   patients |   r2_patient |   raw_mae |   true_patient_avg_std |     smae |
|:-----------------|-----------:|-------------:|----------:|-----------------------:|---------:|
| Respiratory Rate |       5520 |     0.510057 |  0.368719 |               0.79129  | 0.465972 |
| SpO2             |       5507 |     0.477341 |  0.357433 |               0.858514 | 0.416339 |
| Magnesium        |       4322 |     0.33012  |  0.395696 |               0.963151 | 0.410835 |

## Step metrics for the standard outlier rule (`target_abs_le_1000`)

| variable         |   rows |   removed_rows |   r2_step |     smae |
|:-----------------|-------:|---------------:|----------:|---------:|
| Respiratory Rate | 101083 |           1486 |  0.103775 | 0.647648 |
| SpO2             |  99598 |           2971 |  0.286384 | 0.579725 |
| Magnesium        |   6929 |          95640 |  0.381289 | 0.451821 |
| Unweighted mean  | 207610 |                |           | 0.559732 |
| Weighted mean    | 207610 |                |           | 0.608527 |

## Files

- `outlier_sensitivity_summary_40131_30sample.csv`
- `outlier_sensitivity_step_metrics_40131_30sample.csv`
- `outlier_sensitivity_patient_metrics_40131_30sample.csv`
- `outlier_sensitivity_correlation_metrics_40131_30sample.csv`
- `outlier_sensitivity_bounds_40131_30sample.csv`
- `outlier_sensitivity_extreme_values_40131_30sample.csv`
