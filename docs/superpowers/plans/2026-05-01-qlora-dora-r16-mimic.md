# QLoRA DoRA R16 MIMIC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MIMIC BioMistral run explicitly execute memory-conscious `QLoRA + DoRA` with `r=16`, 6000-token context support, and bounded evaluation generation.

**Architecture:** Keep the existing experiment flow intact: the submit script controls runtime defaults and sweep values, while the Python training module remains responsible for mapping CLI flags into `DTGPT_mimic_biomistral_fft_ti_bd_sr.run()`. Fix one Python wiring bug so `--max-new-tokens-to-generate` actually limits evaluation generation, then update the L40S DoRA submit script defaults to the chosen `r=16` configuration. Add a compatibility script at the path previously referenced by the workflow if that file is missing.

**Tech Stack:** Bash, SLURM, Python, Hugging Face Transformers, TRL `SFTTrainer`, Unsloth 4-bit loading, PEFT DoRA, conda `dtgpt`.

---

## File Structure

- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Responsibility: Pass the configured `max_new_tokens_to_generate` into HF generation during evaluation so evaluation memory is bounded.
- Modify: `job/submit_mimic_dora.sh`
  - Responsibility: Provide the canonical L40S `QLoRA + DoRA, r=16` sweep defaults.

## Current State Notes

- `job/submit_mimic_dora.sh` contains the L40S DoRA sweep logic and currently defaults to `r=32`.
- `--max-new-tokens-to-generate` is parsed by `2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py` and passed into `run()`, but evaluation currently calls `get_output_for_split_hf_default(... max_new_tokens=None ...)`, which ignores the configured value.

---

### Task 1: Bound Evaluation Generation

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:555-572`

- [ ] **Step 1: Inspect the current generation call**

Run:

```bash
nl -ba 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py | sed -n '552,572p'
```

Expected: output includes this line:

```python
max_new_tokens=None,
```

- [ ] **Step 2: Update `max_new_tokens` wiring**

Replace this argument inside `evaluate_and_record()`:

```python
                                                                                        max_new_tokens=None,
```

with:

```python
                                                                                        max_new_tokens=max_new_tokens_to_generate,
```

The resulting block should be:

```python
            eval_targets, eval_prediction, return_meta_data_list = experiment.get_output_for_split_hf_default(eval_set_events, 
                                                                                        eval_manager, 
                                                                                        preprocessing_function=preprocessing_function, 
                                                                                        tokenizer=dp.tokenizer,
                                                                                        encoding_function=encoding_function, 
                                                                                        decoding_function=decoding_function, 
                                                                                        post_processing_function=post_processing_function,
                                                                                        batch_size=batch_size,
                                                                                        verbose=verbose,
                                                                                        gen_top_p=0.9,
                                                                                        gen_do_sample=True,
                                                                                        max_new_tokens=max_new_tokens_to_generate,
                                                                                        num_samples_to_generate=num_samples_to_generate, 
                                                                                        sample_merging_strategy=sample_merging_strategy,
                                                                                        pad_token_id=dp.tokenizer.eos_token_id,
                                                                                        max_output_length=4000,                     # Setting this here due to mistral long context bug, need to check if causes errors
                                                                                        return_meta_data=True,
                                                                                        note_down_probabilities=True)
```

- [ ] **Step 3: Verify the replacement**

Run:

```bash
rg -n "max_new_tokens=" 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
```

Expected: output includes:

```text
max_new_tokens=max_new_tokens_to_generate,
```

- [ ] **Step 4: Run Python syntax check**

Run:

```bash
python -m py_compile 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 5: Commit**

Run:

```bash
git add 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
git commit -m "Fix MIMIC evaluation max new tokens"
```

Expected: commit succeeds with one modified Python file.

---

### Task 2: Set Canonical QLoRA DoRA R16 Defaults

**Files:**
- Modify: `job/submit_mimic_dora.sh:73-104`

- [ ] **Step 1: Inspect current submit defaults**

Run:

```bash
nl -ba job/submit_mimic_dora.sh | sed -n '73,104p'
```

Expected: output shows:

```bash
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-1024}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-10}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-512}"
DEFAULT_SWEEP_CONFIGS=(
    "32,64,16,10,8e-6"
)
```

- [ ] **Step 2: Update memory-conscious defaults**

Replace the default variable block with:

```bash
# These defaults target 2x L40S with Unsloth 4-bit QLoRA + DoRA at r=16.
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-6000}"
VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-256}"
TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0.05}"
DECIMAL_PRECISION="${DTGPT_DECIMAL_PRECISION:-1}"
LOGGING_STEPS="${DTGPT_LOGGING_STEPS:-10}"
SAMPLE_MERGING_STRATEGY="${DTGPT_SAMPLE_MERGING_STRATEGY:-mean}"
GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
USE_DORA="${DTGPT_USE_DORA:-1}"
USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-1}"
USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-2}"
DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"
```

- [ ] **Step 3: Update the default sweep**

Replace:

```bash
DEFAULT_SWEEP_CONFIGS=(
    "32,64,16,10,8e-6"
    # "32,64,16,10,9e-6"
    # "32,64,16,9,1e-5"
    # "32,64,16,11,1e-5"
    # "32,64,16,9,8e-6"
    # "32,64,16,10,1e-5"
)
```

with:

```bash
DEFAULT_SWEEP_CONFIGS=(
    # Format: lora_r,lora_alpha,gradient_accumulation,num_train_epochs,learning_rate
    "16,32,32,10,8e-6"
)
```

- [ ] **Step 4: Add clear runtime logging for the selected mode**

After this line:

```bash
echo "SFT dataset num proc: ${SFT_DATASET_NUM_PROC}"
```

add:

```bash
echo "Train batch size per process: ${TRAIN_BATCH_SIZE}"
echo "Validation batch size: ${VALIDATION_BATCH_SIZE}"
echo "Generation samples per patient: ${NUM_SAMPLES_TO_GENERATE}"
echo "Generation max new tokens: ${MAX_NEW_TOKENS}"
echo "LoRA dropout: ${LORA_DROPOUT}"
```

- [ ] **Step 5: Verify script syntax**

Run:

```bash
bash -n job/submit_mimic_dora.sh
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 6: Verify defaults by static inspection**

Run:

```bash
rg -n 'SEQ_MAX_LEN|NUM_SAMPLES_TO_GENERATE|MAX_NEW_TOKENS|DEFAULT_SWEEP_CONFIGS|"16,32,32,10,8e-6"|Training mode' job/submit_mimic_dora.sh
```

Expected: output includes:

```text
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-6000}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-256}"
"16,32,32,10,8e-6"
Training mode: DoRA + Unsloth (4-bit QLoRA)
```

- [ ] **Step 7: Commit**

Run:

```bash
git add job/submit_mimic_dora.sh
git commit -m "Tune MIMIC DoRA submit defaults for QLoRA r16"
```

Expected: commit succeeds with one modified shell script.

---

### Task 3: Add Compatibility Submit Script

**Files:**
- Create: `job/submit_mimic_dora.sh`

- [ ] **Step 1: Confirm compatibility script is absent**

Run:

```bash
test ! -e job/submit_mimic_dora.sh
```

Expected: command exits with status 0.

- [ ] **Step 2: Create the compatibility wrapper**

Create `job/submit_mimic_dora.sh` with exactly:

```bash
#!/bin/bash
# Compatibility entry point for older commands. The canonical L40S
# QLoRA + DoRA sweep implementation lives in submit_mimic_dora.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/submit_mimic_dora.sh" "$@"
```

- [ ] **Step 3: Make it executable**

Run:

```bash
chmod +x job/submit_mimic_dora.sh
```

Expected: command exits with status 0.

- [ ] **Step 4: Verify wrapper syntax**

Run:

```bash
bash -n job/submit_mimic_dora.sh
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 5: Verify wrapper target**

Run:

```bash
sed -n '1,12p' job/submit_mimic_dora.sh
```

Expected: output includes:

```bash
exec "${SCRIPT_DIR}/submit_mimic_dora.sh" "$@"
```

- [ ] **Step 6: Commit**

Run:

```bash
git add job/submit_mimic_dora.sh
git commit -m "Add MIMIC DoRA sweep compatibility launcher"
```

Expected: commit succeeds with one new shell script.

---

### Task 4: Final Static Verification

**Files:**
- Verify: `job/submit_mimic_dora.sh`
- Verify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Run shell syntax checks**

Run:

```bash
bash -n job/submit_mimic_dora.sh
```

Expected: both commands exit with status 0 and print no output.

- [ ] **Step 2: Run Python syntax checks**

Run:

```bash
python -m compileall pipeline 1_experiments
```

Expected: command exits with status 0. Existing `__pycache__` output is acceptable.

- [ ] **Step 3: Verify the training path is QLoRA + DoRA**

Run:

```bash
rg -n 'load_in_4bit=True|use_dora=use_dora|--use-unsloth|--use-dora|--lora-r' \
  job/submit_mimic_dora.sh \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
```

Expected: output shows:

```text
job/submit_mimic_dora.sh:...:"${DORA_FLAG[@]}"
job/submit_mimic_dora.sh:...:"${UNSLOTH_FLAG[@]}"
job/submit_mimic_dora.sh:...:--lora-r "${lora_r}"
.../2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py:...:use_dora=args.use_dora or args.use_unsloth
.../dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:...:load_in_4bit=True
.../dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:...:use_dora=use_dora
```

- [ ] **Step 4: Prepare the exact recommended run command**

Use this command for the first production run:

```bash
DTGPT_SEQ_MAX_LEN=6000 \
DTGPT_TRAIN_BATCH_SIZE=1 \
DTGPT_VALIDATION_BATCH_SIZE=1 \
DTGPT_NUM_SAMPLES_TO_GENERATE=1 \
DTGPT_MAX_NEW_TOKENS=256 \
DTGPT_GRADIENT_CHECKPOINTING=1 \
DTGPT_USE_UNSLOTH=1 \
DTGPT_USE_DORA=1 \
DTGPT_USE_DISTRIBUTED=1 \
DTGPT_NPROC_PER_NODE=2 \
DTGPT_SWEEP_CONFIGS="16,32,32,10,8e-6" \
sbatch job/submit_mimic_dora.sh
```

Expected: SLURM returns a submitted batch job id.

- [ ] **Step 5: Monitor logs for the selected configuration**

After the job starts, run:

```bash
tail -n 80 logs/mimic_dora<JOB_ID>.out
```

Expected: output includes:

```text
Sequence max length: 6000
Training mode: DoRA + Unsloth (4-bit QLoRA)
Gradient checkpointing: enabled
run 1/1 | r=16 alpha=32 grad_acc=32 epochs=10 lr=8e-6
```

- [ ] **Step 6: Commit final verification notes**

If verification produced any project-local notes or log excerpts that should be kept, add them to the PR description rather than committing logs. Then confirm no unintended generated files are staged:

```bash
git status --short
```

Expected: output is empty after the previous commits, or only shows intentional untracked files that are not part of this plan.

---

## Self-Review

**Spec coverage:** This plan covers the requested `QLoRA + DoRA, r=16` path, the missing `job/submit_mimic_dora.sh` path, the current submit script defaults, 6000-token memory controls, and the evaluation generation limit.

**Placeholder scan:** The plan contains no TBD/TODO placeholders. Each code change includes exact replacement text or full file content.

**Type consistency:** The plan uses the existing CLI names: `--max-new-tokens-to-generate`, `--use-unsloth`, `--use-dora`, `--lora-r`, `--lora-alpha`, `--gradient-accumulation`, and the existing environment variables used by `job/submit_mimic_dora.sh`.
