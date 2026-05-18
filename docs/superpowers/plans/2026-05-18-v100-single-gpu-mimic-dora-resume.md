# V100 Single-GPU MIMIC DoRA Resume Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-GPU V100 fallback Slurm script for resuming the MIMIC DoRA/Unsloth run from checkpoint 1395 to max step 2100 without changing the existing L40S script.

**Architecture:** Create a new wrapper script beside the existing job scripts. The new script requests one `v100-32g` GPU, sets conservative V100-safe defaults, and delegates to the existing `job/submit_mimic_dora.sh` training driver so the model/data/checkpoint behavior stays consistent with the current workflow.

**Tech Stack:** Bash, Slurm, conda `dtgpt-vllm` or caller-provided `DTGPT_CONDA_ENV`, existing Python training entrypoint via `job/submit_mimic_dora.sh`, Unsloth 4-bit QLoRA + DoRA in single-process mode.

---

## File Structure

- Create: `job/submit_mimic_dora_resume1395_to2100_v100.sh`
  - Responsibility: Slurm wrapper for one V100-32G fallback run.
  - Keeps all MIMIC/DoRA training logic delegated to `job/submit_mimic_dora.sh`.
  - Uses single GPU only: `--gres=gpu:1`, `DTGPT_USE_DISTRIBUTED=0`, `DTGPT_USE_DEEPSPEED=0`.
  - Uses V100-safe context length default: `DTGPT_SEQ_MAX_LEN=4096` initially, with comments showing how to lower to 3072/2048 if OOM.
- Read-only reference: `job/submit_mimic_dora_resume1395_to2100.sh`
  - Existing L40S resume wrapper. Do not modify.
- Read-only reference: `job/submit_mimic_dora.sh`
  - Existing training driver. Do not modify in this task.

## Behavioral Constraints

- Do not alter the existing L40S job script.
- Do not enable distributed training because the current driver rejects `DTGPT_USE_UNSLOTH=1` with `DTGPT_USE_DISTRIBUTED=1`.
- Do not enable DeepSpeed because this fallback is meant to keep the current Unsloth 4-bit DoRA path.
- Preserve checkpoint resume path and `DTGPT_MAX_STEPS=2100`.
- Keep `DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK=0` because the job is single-process.

---

### Task 1: Create the V100 single-GPU resume wrapper

**Files:**
- Create: `job/submit_mimic_dora_resume1395_to2100_v100.sh`

- [ ] **Step 1: Create the script with V100 Slurm resources and conservative runtime defaults**

Write this exact file:

```bash
#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-v100-1395to2100"
#SBATCH --partition=v100-32g
#SBATCH --account=v100-32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_v100_resume1395_to2100_%j.out
#SBATCH --error=logs/mimic_dora_v100_resume1395_to2100_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

# Single-V100 fallback for the L40S resume job.
# This intentionally keeps the existing Unsloth 4-bit QLoRA + DoRA path and
# does not try to combine multiple V100 GPUs. Multi-GPU Unsloth is rejected by
# job/submit_mimic_dora.sh in this repository.

export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_05___16_46_44_938674/models/checkpoint-1395}"
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-2100}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-1.0}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-16,32,32,2,8e-6}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0}"

# V100-32G fallback default. If this OOMs, resubmit with:
#   sbatch --export=ALL,DTGPT_SEQ_MAX_LEN=3072 job/submit_mimic_dora_resume1395_to2100_v100.sh
# or, more conservatively:
#   sbatch --export=ALL,DTGPT_SEQ_MAX_LEN=2048 job/submit_mimic_dora_resume1395_to2100_v100.sh
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-4096}"

export DTGPT_NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-128}"
export DTGPT_TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
export DTGPT_GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
export DTGPT_USE_DORA="${DTGPT_USE_DORA:-1}"
export DTGPT_USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"
export DTGPT_USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-0}"
export DTGPT_USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-0}"
export DTGPT_RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
export DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-0}"

if [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

echo "Resume checkpoint: ${DTGPT_RESUME_FROM_CHECKPOINT}"
echo "Conda env: ${DTGPT_CONDA_ENV}"
echo "Target max global step: ${DTGPT_MAX_STEPS}"
echo "GPU fallback: single V100-32G"
echo "Sequence max length: ${DTGPT_SEQ_MAX_LEN}"
echo "Distributed: ${DTGPT_USE_DISTRIBUTED}"
echo "DeepSpeed: ${DTGPT_USE_DEEPSPEED}"
echo "Training mode: Unsloth + DoRA single-process fallback"

bash job/submit_mimic_dora.sh
```

- [ ] **Step 2: Run Bash syntax check**

Run:

```bash
bash -n job/submit_mimic_dora_resume1395_to2100_v100.sh
```

Expected: command exits with status `0` and prints no output.

- [ ] **Step 3: Compare resource lines against the existing L40S wrapper**

Run:

```bash
grep -nE '^#SBATCH --(job-name|partition|account|nodes|cpus-per-task|gres|time|output|error|chdir)' job/submit_mimic_dora_resume1395_to2100.sh job/submit_mimic_dora_resume1395_to2100_v100.sh
```

Expected differences:

```text
job/submit_mimic_dora_resume1395_to2100.sh:3:#SBATCH --partition=l40s
job/submit_mimic_dora_resume1395_to2100.sh:4:#SBATCH --account=l40s
job/submit_mimic_dora_resume1395_to2100_v100.sh:3:#SBATCH --partition=v100-32g
job/submit_mimic_dora_resume1395_to2100_v100.sh:4:#SBATCH --account=v100-32g
job/submit_mimic_dora_resume1395_to2100_v100.sh:7:#SBATCH --gres=gpu:1
```

The exact line numbers may differ if comments are added, but the V100 script must request `v100-32g` and exactly one GPU.

- [ ] **Step 4: Validate that the script keeps single-process Unsloth behavior**

Run:

```bash
grep -nE 'DTGPT_USE_UNSLOTH|DTGPT_USE_DISTRIBUTED|DTGPT_USE_DEEPSPEED|DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK' job/submit_mimic_dora_resume1395_to2100_v100.sh
```

Expected output includes these effective defaults:

```text
DTGPT_USE_UNSLOTH:-1
DTGPT_USE_DISTRIBUTED:-0
DTGPT_USE_DEEPSPEED:-0
DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-0
```

- [ ] **Step 5: Commit the new wrapper if this is part of a tracked branch**

Run:

```bash
git add job/submit_mimic_dora_resume1395_to2100_v100.sh docs/superpowers/plans/2026-05-18-v100-single-gpu-mimic-dora-resume.md
git commit -m "Add V100 fallback for MIMIC DoRA resume

Constraint: L40S queue contention requires a single-GPU V100 fallback without changing the main L40S job.
Rejected: DeepSpeed multi-V100 fallback | The current goal is fast low-risk scheduling fallback, and the existing Unsloth path is single-process.
Confidence: high
Scope-risk: narrow
Directive: Keep this wrapper single-GPU unless the training driver gains supported Unsloth distributed training.
Tested: bash -n job/submit_mimic_dora_resume1395_to2100_v100.sh
Not-tested: Full sbatch execution; requires cluster GPU allocation and checkpoint availability."
```

Expected: commit succeeds if the working tree is ready. If unrelated files are modified, do not include them.

---

## Self-Review

**Spec coverage:**
- Single-card V100 script: covered by Task 1 Step 1.
- Preserve current resume behavior from checkpoint 1395 to step 2100: covered by `DTGPT_RESUME_FROM_CHECKPOINT` and `DTGPT_MAX_STEPS` exports.
- Avoid DeepSpeed/Unsloth incompatibility risk: covered by `DTGPT_USE_DISTRIBUTED=0`, `DTGPT_USE_DEEPSPEED=0`, and comments.
- Support OOM fallback: covered by documented `DTGPT_SEQ_MAX_LEN=3072` and `2048` resubmission commands.

**Placeholder scan:**
- No TBD/TODO/fill-in-later placeholders are present.
- All commands and expected outcomes are explicit.

**Type/name consistency:**
- Environment variable names match the existing `job/submit_mimic_dora.sh` driver variables.
- File paths match repository paths under `/share/home/r15543056/trajectory_forecast/DT-GPT`.
