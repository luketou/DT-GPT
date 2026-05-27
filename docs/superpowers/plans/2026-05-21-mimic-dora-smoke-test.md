# MIMIC DoRA Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe MIMIC DoRA resume smoke path that trains on a tiny sample before running the full ~30k-patient job.

**Architecture:** Keep the reusable launcher as the control surface and add explicit sample-limit flags to the MIMIC training entrypoint. Smoke mode should limit patients before expensive DF loading, limit split samples after splitter filtering, run only a few optimizer steps, and skip vLLM evaluation.

**Tech Stack:** Bash Slurm wrappers, Python argparse, existing `EvaluationManager`, existing `DTGPT_mimic_biomistral_fft_ti_bd_sr.run()` training flow.

---

### Task 1: Add exact small-data controls to the MIMIC training script

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Add CLI args**

Add:
```python
parser.add_argument("--train-max-patients", type=int, default=None)
parser.add_argument("--validation-max-patients", type=int, default=None)
parser.add_argument("--test-max-patients", type=int, default=None)
parser.add_argument("--train-max-samples", type=int, default=None)
parser.add_argument("--validation-max-samples", type=int, default=None)
parser.add_argument("--skip-eval", action="store_true")
```

- [ ] **Step 2: Thread args into `experiment.run(...)`**

Pass:
```python
train_max_patients=args.train_max_patients,
validation_max_patients=args.validation_max_patients,
test_max_patients=args.test_max_patients,
train_max_samples=args.train_max_samples,
validation_max_samples=args.validation_max_samples,
skip_eval=args.skip_eval,
```

- [ ] **Step 3: Apply patient limits before loading DFs**

In `DTGPT_mimic_biomistral_fft_ti_bd_sr.run()`, slice `training_full_patientids`, `validation_full_patientids`, and `test_full_patientids` before `load_list_of_patient_dfs_and_constants(...)`.

- [ ] **Step 4: Apply split-sample limits after `After24HSplitter`**

After `splitter.setup_split_indices(...)`, slice `training_events/training_meta_data` and `validation_events/validation_meta` so tokenization and truncation only process the smoke sample.

- [ ] **Step 5: Skip evaluation after successful train smoke**

After saving the LoRA adapter and distributed barrier, if `skip_eval` is true, finish W&B on rank 0 and return before model reload / vLLM evaluation.

### Task 2: Add launcher env flags and a dedicated smoke Slurm wrapper

**Files:**
- Modify: `job/submit_mimic_dora.sh`
- Create: `job/submit_mimic_dora_resume1395_to4185_smoke.sh`

- [ ] **Step 1: Read env vars in launcher**

Add env vars:
```bash
DTGPT_TRAIN_MAX_PATIENTS
DTGPT_VALIDATION_MAX_PATIENTS
DTGPT_TEST_MAX_PATIENTS
DTGPT_TRAIN_MAX_SAMPLES
DTGPT_VALIDATION_MAX_SAMPLES
DTGPT_SKIP_EVAL
DTGPT_DEBUG
```

- [ ] **Step 2: Convert non-empty env vars into CLI flags**

Build arrays and pass them into `${TRAIN_SCRIPT}` invocation.

- [ ] **Step 3: Create smoke wrapper**

Use 2 L40S, same checkpoint, but set:
```bash
DTGPT_MAX_STEPS=1397
DTGPT_SWEEP_CONFIGS=16,32,1,1,8e-6
DTGPT_TRAIN_MAX_PATIENTS=80
DTGPT_VALIDATION_MAX_PATIENTS=20
DTGPT_TEST_MAX_PATIENTS=20
DTGPT_TRAIN_MAX_SAMPLES=32
DTGPT_VALIDATION_MAX_SAMPLES=8
DTGPT_SKIP_EVAL=1
DTGPT_DEBUG=1
DTGPT_LOGGING_STEPS=1
```

### Task 3: Verify safely

**Files:**
- Verify only; no new source files.

- [ ] **Step 1: Syntax check shell scripts**

Run:
```bash
bash -n job/submit_mimic_dora.sh job/submit_mimic_dora_resume1395_to4185.sh job/submit_mimic_dora_resume1395_to4185_smoke.sh
```
Expected: no output, exit code 0.

- [ ] **Step 2: Python compile check**

Run:
```bash
python -m compileall 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
```
Expected: both files compile.

- [ ] **Step 3: Submit smoke job only**

Run:
```bash
sbatch job/submit_mimic_dora_resume1395_to4185_smoke.sh
```
Expected: new job ID, logs under `logs/mimic_dora_resume1395_to4185_smoke_<jobid>.out/.err`.

- [ ] **Step 4: Inspect smoke logs**

Check for:
```text
Train max patients: 80
Train max samples: 32
Skip eval after training: 1
Start training
```
and either successful completion or a concrete traceback to debug next.
