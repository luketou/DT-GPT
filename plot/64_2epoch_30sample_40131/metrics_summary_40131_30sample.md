# MIMIC DoRA Paper-R2 vLLM Evaluation Metrics — job 40131 (30 samples)

Source logs: `logs/mimic_dora_paper_r2_vllm_shard_40131_*.out`  
All 8 shards report `Num samples to generate: 30` and saved target/prediction CSVs.

Outlier rule from `memory.md`: exclude rows with `abs(target) > 1000.0` before metric calculation.

## R2 correlation / paper-style correlation preservation

- Definition in `plot/r2_metric_definitions.md`: `R2_corr = -5.217115`
- Scatter-fit alternative (`corr(c_true, c_pred)^2`, useful because paper figure is a scatter with fit): `0.924896`
- Correlation between the two 3-point correlation vectors: `0.961715`

| pair                          |   rows |   true_corr |   pred_corr |      diff |
|:------------------------------|-------:|------------:|------------:|----------:|
| Respiratory Rate vs SpO2      |  98238 |   -0.066436 |   -0.157892 | -0.091455 |
| Respiratory Rate vs Magnesium |   6728 |   -0.010806 |    0.022072 |  0.032877 |
| SpO2 vs Magnesium             |   6665 |   -0.039346 |   -0.026024 |  0.013322 |

## Step-level sMAE

| variable         |   rows |   mae_scaled |     smae |   r2_step |
|:-----------------|-------:|-------------:|---------:|----------:|
| Respiratory Rate | 101083 |     0.647648 | 0.647648 |  0.103775 |
| SpO2             |  99598 |     0.579725 | 0.579725 |  0.286384 |
| Magnesium        |   6929 |     0.451821 | 0.451821 |  0.381289 |
| Unweighted mean  | 207610 |     0.559732 | 0.559732 |           |
| Weighted mean    | 207610 |     0.608527 | 0.608527 |           |

## Patient-averaged R2 / sMAE

| variable         |   patients |   r2_patient |   raw_mae |   true_patient_avg_std |     smae |
|:-----------------|-----------:|-------------:|----------:|-----------------------:|---------:|
| Respiratory Rate |       5520 |     0.510057 |  0.368719 |               0.79129  | 0.465972 |
| SpO2             |       5507 |     0.477341 |  0.357433 |               0.858514 | 0.416339 |
| Magnesium        |       4322 |     0.33012  |  0.395696 |               0.963151 | 0.410835 |
