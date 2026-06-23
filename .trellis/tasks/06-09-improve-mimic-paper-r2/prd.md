# Improve MIMIC Paper-Style R2 To 0.8

## Goal

Find a practical experiment path to raise the MIMIC DoRA paper-style inter-variable correlation preservation R2 from the current baseline around 0.59 toward 0.8, while staying within the single-L40S / 48GB CPU RAM constraint.

## What I Already Know

* The target metric is paper-style correlation-preservation R2, not patient-level prediction R2.
* Current baseline from `memory.md`: pairwise-complete paper-style R2 is approximately `0.593378`.
* Current step-level scaled MAE is already close to the paper range:
  * Respiratory Rate: `0.6448`
  * SpO2: `0.5766`
  * Magnesium: `0.4578`
  * Unweighted mean: `0.5598`
  * Weighted mean: `0.6059`
* Current pairwise correlation diffs:
  * RR vs SpO2 diff: `-0.009051`
  * RR vs Magnesium diff: `+0.027488`
  * SpO2 vs Magnesium diff: `+0.031463`
* The largest useful improvement target is likely the Magnesium-related correlation pairs.
* `split=1.0` with the new paper-R2 job exceeded CPU RAM at `48GB`.
* Prior `split=0.5` r64 DoRA runs completed.
* New job wrapper supports overriding `DTGPT_PATIENT_SPLIT_FRACTION`, `DTGPT_SWEEP_CONFIGS`, and `DTGPT_LORA_DROPOUT`.
* New checkpoint preservation support can keep epoch-1 checkpoints outside Hugging Face rotation.

## Assumptions

* The goal R2=0.8 refers to the paper-style correlation-preservation R2.
* The target should be reached without full finetuning unless adapter experiments fail.
* CPU RAM is currently the limiting resource, not GPU memory.
* The data fraction should stay in the approximate `0.5` to `0.75` range unless memory settings are also changed.

## Open Questions

* Should the first experiment prioritize staying within known-stable RAM (`split=0.5`) or pushing more data with moderate RAM risk (`split=0.65` or `0.75`)?
* For full-data training, should we solve CPU preprocessing/RAM peak by adding a durable pre-tokenized dataset artifact pipeline, a streaming iterable dataset, or a smaller patch to the current cache builder?

## Requirements

* Use paper-style correlation-preservation R2 as the primary success metric.
* Keep step-level scaled MAE from regressing substantially from the current `0.56` to `0.61` range.
* Use the same outlier filtering convention: remove values with `abs(target) > 1000`.
* Preserve epoch-1 checkpoint for epoch comparison.
* Avoid `split=1.0` until CPU RAM behavior is improved.
* Run the next experiment with aggressive data fraction `split=0.75`.
* If `96G` is not schedulable on the L40S partition, fall back to a schedulable intermediate setting.
* Future full-data training should avoid holding raw patient events, converted text records, and tokenized Hugging Face datasets in memory at the same time.
* Future full-data training should make GPU training jobs load a reusable on-disk dataset artifact instead of rebuilding dataframe/text/tokenizer preprocessing every run.

## Acceptance Criteria

* [ ] One or more runnable experiment configurations are selected.
* [ ] Each selected configuration fits within the known cluster constraints or has an explicit memory-risk rationale.
* [ ] Evaluation plan reports paper-style R2, pairwise correlation diffs, and step-level scaled MAE.
* [ ] Success threshold is defined as paper-style R2 `>= 0.8`.
* [ ] Secondary threshold prevents unacceptable degradation in step-level scaled MAE.

## Definition of Done

* Experiment commands or job settings are documented.
* Expected runtime and memory risk are stated.
* Evaluation procedure is tied back to `memory.md` baseline.
* Any job/script change is verified with shell syntax checks and relevant Python compile checks.

## Out of Scope

* Full finetuning 7B unless adapter-based strategies fail.
* Changing the MIMIC target variables beyond the original three variables.
* Claiming paper equivalence based on patient-level prediction R2.

## Technical Notes

* Baseline notes: `memory.md`.
* Paper-style R2 definition: `plot/r2_metric_definitions.md`.
* Current experiment wrapper: `job/submit_mimic_dora_r64_paper_r2_l40s.sh`.
* The current wrapper default `split=1.0` is too memory-heavy for the current 48GB CPU RAM request.
* Candidate first-pass experiment families:
  * Conservative: `split=0.5`, `r64 alpha128`, `lr2e-5`, `dropout0.10`, `2 epochs`.
  * Balanced: `split=0.65`, `r64 alpha128`, `lr2e-5`, `dropout0.10`, `2 epochs`.
  * Higher data / higher risk: `split=0.75`, `r64 alpha128`, `lr2e-5`, `dropout0.10`, `2 epochs`.
* Full-data RAM root cause from code inspection:
  * `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py` currently builds `training_records = list(iter_converted_tuples(...))` and `validation_records = list(...)`, so converted prompt/target text is fully materialized before tokenization.
  * `DataProcessorBiomistral.preprocess_converted_records()` then calls `Dataset.from_list(list(converted_records))`, which can materialize another full in-memory copy before `Dataset.map()` tokenization.
  * The script does save a tokenized dataset cache via `save_to_disk()`, but the first cache build still has a large memory peak because raw events, text records, and tokenized dataset construction overlap.
  * `DTGPT_SFT_DATASET_NUM_PROC=1` and `DTGPT_DF_CONVERSION_N_JOBS=1` reduce parallel-memory multiplication but do not remove the single-process peak.

## Full-Data Preprocessing Architecture Options

**Approach A: two-stage pre-tokenized artifact pipeline** (recommended)

* Add a CPU-oriented preprocessing job that writes train/validation tokenized Hugging Face datasets to disk in bounded shards.
* GPU fine-tuning jobs require an existing dataset artifact and only call `load_from_disk()`.
* This removes repeated preprocessing from GPU jobs and makes memory behavior easier to control.
* Trade-off: requires a new preprocessing command/job and artifact versioning by split, tokenizer, sequence length, variables, normalization config, and data fraction.

**Approach B: streaming iterable dataset**

* Convert patient events to examples lazily during training instead of materializing full records/datasets.
* This can minimize RAM but makes shuffling, exact reproducibility, length grouping, and Trainer/SFTTrainer integration harder.
* Trade-off: larger code change and more risk to training behavior.

**Approach C: patch current cache builder to chunk writes**

* Keep the current training script structure, but replace full `list(...)` record construction with chunked conversion/tokenization/save.
* Lower implementation cost than a full artifact pipeline.
* Trade-off: still couples preprocessing to GPU training jobs and is less reusable for repeated sweeps.

## Decision

User selected the aggressive path:

* Set `DTGPT_PATIENT_SPLIT_FRACTION=0.75` in the paper-R2 L40S job.
* Raise Slurm `--mem` from `48G` to `96G` for the same job.

Follow-up result:

* Slurm rejected `--mem=96G` with `Memory specification can not be satisfied`.
* The implementation should fall back to a schedulable configuration before re-submission.
* Selected fallback for the wrapper: `split=0.65`, `--mem=48G`, no fixed `--nodelist`.
