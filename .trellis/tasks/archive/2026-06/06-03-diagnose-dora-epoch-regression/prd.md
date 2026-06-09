# brainstorm: diagnose dora epoch regression

## Goal

Diagnose why a MIMIC DoRA fine-tuning run appears to regress from epoch 1 to epoch 2, and decide whether continued fine-tuning is likely to help reach R2 > 0.90.

## What I already know

* User is training a DoRA model and currently has results after epoch 2.
* Epoch 1 reported patient-averaged metrics: Magnesium R2 0.2387 MAE 0.4274; SpO2 R2 0.3443 MAE 0.3602; Respiratory Rate R2 0.5717 MAE 0.3637.
* Earlier epoch 2 shard-level analysis found one Respiratory Rate target outlier in shard 002: target 32977.809671.
* Removing that outlier improves shard-averaged Respiratory Rate MAE substantially.

## Assumptions (temporary)

* Epoch 1 and epoch 2 metrics may not be using identical aggregation methods.
* The first diagnosis should separate metric mismatch, data outliers, and true overfitting/continued-training degradation.

## Open Questions

* Confirm whether epoch 1 and epoch 2 should be compared with patient-averaged metrics using the same evaluation script.

## Requirements (evolving)

* Compare epoch 1 vs epoch 2 using consistent metric definitions before deciding whether DoRA training is degrading.
* Identify likely causes: metric mismatch, data preprocessing/outlier, overfitting, learning-rate instability, or checkpoint/eval inconsistency.

## Acceptance Criteria (evolving)

* [ ] Produce a clear recommendation on whether to continue finetuning.
* [ ] Identify the most likely root cause(s) and next diagnostic run.

## Definition of Done (team quality bar)

* No code changes required unless user asks for scripts.
* Use evidence from logs/results where available.
* Document assumptions and metric definitions.

## Out of Scope (explicit)

* Modifying training code or job scripts in this brainstorm.

## Technical Notes

* Epoch 2 logs previously inspected: `logs/mimic_dora_vllm2801_0531_shard_38685_*.out`.
* Need compare patient-averaged metrics vs row/shard macro metrics.

## Auto-Context Findings

* Recomputed epoch 2 patient-level MAE from shard CSVs:
  * Magnesium MAE 0.450516, N=2194
  * SpO2 MAE 0.579240, N=2765
  * Respiratory Rate MAE 1.144700 before outlier removal, 0.627648 after removing RR target abs > 100, N=2773
* Patient-averaged R2 recomputation produced extreme negative values because some per-patient target variance is near zero; this suggests the exact epoch 1 R2 script/definition must be matched before comparing R2.
* The epoch 2 MAE degradation is real for SpO2 and Respiratory Rate under patient-level MAE, even after removing the single RR outlier.

## Current Diagnostic Hypotheses

1. Metric/evaluation mismatch: epoch 1 and epoch 2 R2 may not be calculated with identical filtering/aggregation.
2. Training degradation after epoch 1: likely overfitting or learning-rate instability if metric definitions are confirmed identical.
3. Data/eval issue: at least one impossible target exists; additional preprocessing anomalies may exist.
4. DoRA configuration may be too aggressive for continued epochs if LR/rank/dropout are high.

## Method Comparison Update

* `plot/patient_avg_metrics_summary.md` defines the intended 1-epoch method as patient-level averaging first, then metric calculation on patient-average true/pred pairs.
* Recomputing epoch 2 with this patient-average method gives values that closely match the user's 1-epoch table, so the earlier row/shard-level epoch 2 table was not comparable.
* Important discrepancy: the markdown describes z-score sMAE, but the reported 1-epoch MAE numbers align more closely with raw patient-average MAE from the already-normalized CSV columns, not an extra z-score by patient-average standard deviation.
* With the single RR outlier removed (`abs(target) > 1000`), epoch 2 appears similar or slightly improved vs epoch 1 under the comparable patient-average method.

## Learning Rate / Continuation Findings

* Current r=64 resume wrapper defaults to `DTGPT_SWEEP_CONFIGS=64,128,8,2,1e-4`.
* Training log `logs/mimic_dora_r64_resume_38625.out` shows the checkpoint evaluated at global step 2801 with `epoch: 1` and final LR near zero (`3.885e-11`), despite the wrapper label `epochs=2`.
* This indicates the evaluated `checkpoint-2801`/vLLM run is effectively an epoch-sized checkpoint, not clear evidence of a completed second epoch with a fresh LR schedule.
* To continue from this checkpoint, scheduler behavior must be checked/reset; otherwise resuming from a checkpoint whose scheduler already decayed to ~0 can make further training ineffective.
