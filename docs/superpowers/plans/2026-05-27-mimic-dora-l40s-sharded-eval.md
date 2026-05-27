# MIMIC DoRA L40S Sharded Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a safe L40S Slurm array entrypoint for sharded MIMIC DoRA HF evaluation and remove the Transformers `max_new_tokens`/`max_length` warning while defaulting generation to 1024 new tokens.

**Architecture:** Keep the existing single-node eval script as the reusable execution body because it already forces `DTGPT_ATTN_IMPLEMENTATION=eager`, avoiding the prior L40S FlashAttention2 failure. Add a small L40S array wrapper that sets shard environment variables and delegates to that body, then remove redundant `max_length` arguments from HF generation so `max_new_tokens` is the only generation length cap.

**Tech Stack:** Bash Slurm scripts, Python `transformers.generate`, Python `unittest`, `compileall`.

---

## File Structure

- Modify: `job/submit_mimic_dora_checkpoint700_eval_v100.sh`
  - Responsibility: reusable eval body; update default `DTGPT_MAX_NEW_TOKENS` from `256` to `1024`.
- Create: `job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh`
  - Responsibility: L40S Slurm array wrapper; set `DTGPT_EVAL_NUM_SHARDS=8`, `DTGPT_EVAL_SHARD_INDEX=$SLURM_ARRAY_TASK_ID`, run one shard at a time for a single available L40S, and call the eager-attention eval body.
- Modify: `pipeline/Experiment.py`
  - Responsibility: HF generation implementation; stop passing `max_length` together with `max_new_tokens`.
- Modify: `tests/test_mimic_dora_job_config.py`
  - Responsibility: regression coverage for L40S sharded wrapper and generation warning fix.

### Task 1: Add tests for sharded L40S eval config and generation length arguments

**Files:**
- Modify: `tests/test_mimic_dora_job_config.py`

- [ ] **Step 1: Add regression tests**

Add tests that read the scripts/source and assert the desired static configuration:

```python
    def test_l40s_sharded_eval_wrapper_uses_eager_eval_body_and_1024_tokens(self):
        wrapper = (REPO_ROOT / "job" / "submit_mimic_dora_checkpoint700_eval_l40s_array.sh").read_text()
        body = (REPO_ROOT / "job" / "submit_mimic_dora_checkpoint700_eval_v100.sh").read_text()

        self.assertIn("#SBATCH --partition=l40s", wrapper)
        self.assertIn("#SBATCH --account=l40s", wrapper)
        self.assertIn("#SBATCH --array=0-7%1", wrapper)
        self.assertIn('export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-8}"', wrapper)
        self.assertIn('export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"', wrapper)
        self.assertIn("bash job/submit_mimic_dora_checkpoint700_eval_v100.sh", wrapper)
        self.assertIn('export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-eager}"', body)
        self.assertIn('--max-new-tokens-to-generate "${DTGPT_MAX_NEW_TOKENS:-1024}"', body)

    def test_hf_generation_uses_only_max_new_tokens_length_cap(self):
        experiment_source = (REPO_ROOT / "pipeline" / "Experiment.py").read_text()

        self.assertIn("max_new_tokens=max_new_tokens", experiment_source)
        self.assertNotIn("max_new_tokens=max_new_tokens, max_length=max_output_length + 92", experiment_source)
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run: `python -m unittest tests.test_mimic_dora_job_config -v`

Expected before implementation: FAIL because the new wrapper does not exist and the old eval body still defaults to 256.

### Task 2: Implement the L40S sharded wrapper and 1024-token default

**Files:**
- Create: `job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh`
- Modify: `job/submit_mimic_dora_checkpoint700_eval_v100.sh`

- [ ] **Step 1: Create wrapper script**

Create `job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh`:

```bash
#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-eval-l40s-shard"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:0
#SBATCH --array=0-7%1
#SBATCH --output=logs/mimic_dora_eval_l40s_shard_%A_%a.out
#SBATCH --error=logs/mimic_dora_eval_l40s_shard_%A_%a.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

export DTGPT_EVAL_BACKEND="${DTGPT_EVAL_BACKEND:-hf}"
export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-8}"
export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"
export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-eager}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-1024}"

bash job/submit_mimic_dora_checkpoint700_eval_v100.sh
```

- [ ] **Step 2: Update eval body default**

Change in `job/submit_mimic_dora_checkpoint700_eval_v100.sh`:

```bash
--max-new-tokens-to-generate "${DTGPT_MAX_NEW_TOKENS:-1024}" \
```

- [ ] **Step 3: Run tests**

Run: `python -m unittest tests.test_mimic_dora_job_config -v`

Expected: only generation-source test may still fail until Task 3.

### Task 3: Remove redundant HF `max_length` generation caps

**Files:**
- Modify: `pipeline/Experiment.py`

- [ ] **Step 1: Edit both HF generation calls**

In `pipeline/Experiment.py`, remove `max_length=max_output_length + 92` from both `self.model.generate(...)` calls that already pass `max_new_tokens=max_new_tokens`.

Expected resulting argument shape:

```python
outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                do_sample=gen_do_sample, num_beams=gen_num_beams,
                                penalty_alpha=gen_penalty_alpha, top_k=gen_top_k,
                                top_p=gen_top_p,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=pad_token_id,
                                return_dict_in_generate=True, output_scores=True)
```

and:

```python
predictions = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    do_sample=gen_do_sample, num_beams=gen_num_beams,
                                    penalty_alpha=gen_penalty_alpha, top_k=gen_top_k,
                                    top_p=gen_top_p,
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=pad_token_id)
```

- [ ] **Step 2: Run tests**

Run: `python -m unittest tests.test_mimic_dora_job_config -v`

Expected: PASS.

### Task 4: Validate syntax and provide submit command

**Files:**
- Validate: `job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh`
- Validate: `job/submit_mimic_dora_checkpoint700_eval_v100.sh`
- Validate: `pipeline/Experiment.py`

- [ ] **Step 1: Bash syntax check**

Run:

```bash
bash -n job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh
bash -n job/submit_mimic_dora_checkpoint700_eval_v100.sh
```

Expected: no output and exit code 0.

- [ ] **Step 2: Python syntax check**

Run:

```bash
python -m compileall pipeline/Experiment.py tests/test_mimic_dora_job_config.py
```

Expected: both files compile.

- [ ] **Step 3: Final command for user**

Recommended submission command:

```bash
sbatch job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh
```

Optional override for a different checkpoint:

```bash
sbatch --export=ALL,DTGPT_EVAL_MODEL_PATH=/path/to/checkpoint,DTGPT_MAX_NEW_TOKENS=1024 job/submit_mimic_dora_checkpoint700_eval_l40s_array.sh
```

## Self-Review

- Spec coverage: L40S+shard is covered by Task 2; warning removal is covered by Task 3; token value 1024 is covered by Tasks 1 and 2.
- Placeholder scan: no placeholders or deferred implementation steps remain.
- Type consistency: all paths and environment variable names match existing scripts and parser arguments.
