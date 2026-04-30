# MIMIC LoRA V100 NCCL Barrier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `job/submit_mimic_lora_sweep_v100.sh` fail fast on broken NCCL distributed setup and avoid the observed `SFTTrainer` initialization barrier crash when the runtime can be configured safely.

**Architecture:** Add a small distributed environment probe that verifies the active Python environment, loaded NCCL library candidates, and a real `torch.distributed.barrier()` before the expensive MIMIC training starts. Keep launcher changes in the Slurm script and add one explicit training-code option that lets this sweep request serial dataset preprocessing in `SFTTrainer`, reducing the barrier surface during TRL dataset preparation while preserving DeepSpeed training.

**Tech Stack:** Bash, Slurm, Python 3.8 `dtgpt` conda env, PyTorch distributed/NCCL, TRL `SFTTrainer`, DeepSpeed ZeRO-3.

---

## File Structure

- Create: `job/check_torch_distributed_nccl.py`
  - Responsibility: minimal multi-process NCCL probe run through `torch.distributed.run`; prints version/runtime diagnostics and executes a barrier.
- Modify: `job/submit_mimic_lora_sweep_v100.sh`
  - Responsibility: harden CUDA/NCCL environment setup, run the probe before the MIMIC smoke check, and pass the new SFT dataset-processing option to the training script.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
  - Responsibility: expose a CLI flag for `SFTTrainer` dataset preprocessing worker count.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Responsibility: pass `dataset_num_proc` into `SFTTrainer` when supported by the installed TRL version.

## Root Cause Working Model

The stack trace fails before model training at `accelerate.state.PartialState.local_main_process_first()` inside `trl.trainer.sft_trainer.SFTTrainer.__init__`. That context manager calls `torch.distributed.barrier()`, and NCCL reports:

```text
ncclInvalidArgument: Invalid config blocking attribute value -2147483648
```

The current Slurm script already unsets `NCCL_COMM_BLOCKING`, `NCCL_BLOCKING_WAIT`, and `TORCH_NCCL_BLOCKING_WAIT`, and prepends the conda NCCL library directory. The next implementation should therefore prove whether a clean `torch.distributed.barrier()` works before training, print enough library diagnostics to identify a runtime mismatch, and only then make code-level changes that reduce TRL preprocessing barriers.

### Task 1: Add A NCCL Barrier Probe

**Files:**
- Create: `job/check_torch_distributed_nccl.py`

- [ ] **Step 1: Write the probe script**

Create `job/check_torch_distributed_nccl.py` with this exact content:

```python
import ctypes.util
import os
import socket

import torch


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nccl_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("NCCL") or key.startswith("TORCH_NCCL")
    }

    if rank == 0:
        print(f"Hostname: {socket.gethostname()}", flush=True)
        print(f"Torch: {torch.__version__}", flush=True)
        print(f"Torch CUDA: {torch.version.cuda}", flush=True)
        print(f"Torch NCCL: {torch.cuda.nccl.version()}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
        print(f"CUDA_VISIBLE_DEVICES: {visible_devices}", flush=True)
        print(f"ctypes nccl lookup: {ctypes.util.find_library('nccl')}", flush=True)
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)
        print(f"NCCL env: {nccl_env}", flush=True)

    device_name = torch.cuda.get_device_name(local_rank)
    print(
        f"Rank {rank}/{world_size} local_rank={local_rank} device={device_name}",
        flush=True,
    )
    torch.distributed.barrier()
    if rank == 0:
        print("NCCL barrier probe passed.", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run:

```bash
bash -n job/submit_mimic_lora_sweep_v100.sh
python -m py_compile job/check_torch_distributed_nccl.py
```

Expected: both commands exit with status `0` and print no Python syntax error.

- [ ] **Step 3: Commit**

```bash
git add job/check_torch_distributed_nccl.py
git commit -m "Add NCCL distributed smoke probe"
```

### Task 2: Harden Slurm CUDA/NCCL Setup And Run Probe

**Files:**
- Modify: `job/submit_mimic_lora_sweep_v100.sh`

- [ ] **Step 1: Replace the CUDA module load block**

In `job/submit_mimic_lora_sweep_v100.sh`, replace:

```bash
module load cuda/12.1
```

with:

```bash
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.1
fi
```

- [ ] **Step 2: Replace the existing NCCL environment block**

Replace the block from:

```bash
# Keep torch distributed on the NCCL runtime bundled with the active Python env.
```

through:

```bash
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
```

with:

```bash
# Keep torch distributed on the NCCL runtime bundled with the active Python env.
# A mismatched system libnccl can fail at torch.distributed.barrier() during
# SFTTrainer setup with "Invalid config blocking attribute value".
TORCH_NCCL_LIB_DIR="$("${PYTHON_BIN}" -c "import pathlib, sys; p = pathlib.Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'nvidia' / 'nccl' / 'lib'; print(p if p.is_dir() else '')")"
if [ -n "${TORCH_NCCL_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${TORCH_NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
unset NCCL_COMM_BLOCKING
unset NCCL_BLOCKING_WAIT
unset TORCH_NCCL_BLOCKING_WAIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
```

- [ ] **Step 3: Add the distributed probe variables**

After:

```bash
SMOKE_CHECK_SCRIPT="1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py"
```

add:

```bash
DISTRIBUTED_SMOKE_CHECK_SCRIPT="job/check_torch_distributed_nccl.py"
RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-1}"
SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"
```

- [ ] **Step 4: Print the new defaults**

After:

```bash
echo "Run timestamp: ${DTGPT_RUN_TIMESTAMP}"
```

add:

```bash
echo "SFT dataset num proc: ${SFT_DATASET_NUM_PROC}"
echo "Distributed smoke check: ${RUN_DISTRIBUTED_SMOKE_CHECK}"
```

- [ ] **Step 5: Run the NCCL probe before the local MIMIC smoke check**

Replace:

```bash
"${PYTHON_BIN}" "${SMOKE_CHECK_SCRIPT}"
```

with:

```bash
if [ "${USE_DEEPSPEED}" = "1" ] && [ "${RUN_DISTRIBUTED_SMOKE_CHECK}" = "1" ]; then
    print_header "Running NCCL distributed smoke check"
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${DISTRIBUTED_SMOKE_CHECK_SCRIPT}"
fi

"${PYTHON_BIN}" "${SMOKE_CHECK_SCRIPT}"
```

- [ ] **Step 6: Pass the SFT dataset preprocessing value to training**

In the training command, after:

```bash
--logging-steps "${LOGGING_STEPS}"
```

add a trailing backslash to that line and then add:

```bash
        --sft-dataset-num-proc "${SFT_DATASET_NUM_PROC}"
```

The end of the command should become:

```bash
        --sample-merging-strategy "${SAMPLE_MERGING_STRATEGY}" \
        --max-new-tokens-to-generate "${MAX_NEW_TOKENS}" \
        --logging-steps "${LOGGING_STEPS}" \
        --sft-dataset-num-proc "${SFT_DATASET_NUM_PROC}"; then
```

- [ ] **Step 7: Run shell syntax check**

Run:

```bash
bash -n job/submit_mimic_lora_sweep_v100.sh
```

Expected: exits with status `0`.

- [ ] **Step 8: Commit**

```bash
git add job/submit_mimic_lora_sweep_v100.sh
git commit -m "Run NCCL smoke check before MIMIC LoRA sweep"
```

### Task 3: Expose SFT Dataset Processing Control

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Add CLI argument**

In `2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`, after:

```python
parser.add_argument("--logging-steps", type=int, default=10)
```

add:

```python
parser.add_argument(
    "--sft-dataset-num-proc",
    type=int,
    default=1,
    help="Number of processes TRL should use for SFT dataset preprocessing.",
)
```

- [ ] **Step 2: Pass CLI argument into experiment.run**

In the `experiment.run(...)` call, after:

```python
logging_steps=args.logging_steps,
```

add:

```python
sft_dataset_num_proc=args.sft_dataset_num_proc,
```

- [ ] **Step 3: Add run parameter**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, change the end of the `run` signature from:

```python
use_unsloth=False,
deepspeed_config=None):
```

to:

```python
use_unsloth=False,
deepspeed_config=None,
sft_dataset_num_proc=1):
```

- [ ] **Step 4: Import inspect**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, after:

```python
import json
```

add:

```python
import inspect
```

- [ ] **Step 5: Build version-compatible SFT kwargs**

Immediately before:

```python
trainer = SFTTrainer(
```

add:

```python
            sft_trainer_kwargs = {}
            if "dataset_num_proc" in inspect.signature(SFTTrainer).parameters:
                sft_trainer_kwargs["dataset_num_proc"] = sft_dataset_num_proc
```

- [ ] **Step 6: Pass kwargs into SFTTrainer**

In the `SFTTrainer(...)` call, after:

```python
dataset_text_field="concatenated_text",
```

add:

```python
**sft_trainer_kwargs,
```

The final block should read:

```python
            trainer = SFTTrainer(
                model=model,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                tokenizer=dp.tokenizer,
                data_collator=data_collator,
                max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                args=train_params,
                packing=False,
                dataset_text_field="concatenated_text",
                **sft_trainer_kwargs,
            )
```

- [ ] **Step 7: Run Python syntax checks**

Run:

```bash
python -m py_compile 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py
python -m py_compile 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
```

Expected: both commands exit with status `0`.

- [ ] **Step 8: Commit**

```bash
git add 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
git commit -m "Expose SFT dataset preprocessing control"
```

### Task 4: Validate On The V100 Partition

**Files:**
- Modify: none

- [ ] **Step 1: Run local syntax validation**

Run:

```bash
bash -n job/submit_mimic_lora_sweep_v100.sh
python -m compileall job 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
```

Expected: `bash -n` exits with status `0`; `compileall` prints successful compile output and exits with status `0`.

- [ ] **Step 2: Submit a short distributed smoke-only job**

Run:

```bash
sbatch --time=00:10:00 --export=ALL,DTGPT_SWEEP_CONFIGS=32,64,16,0.01,8e-6,DTGPT_NUM_SAMPLES_TO_GENERATE=1,DTGPT_MAX_NEW_TOKENS=16,DTGPT_SEQ_MAX_LEN=256,DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK=1 job/submit_mimic_lora_sweep_v100.sh
```

Expected: Slurm returns `Submitted batch job <job_id>`.

- [ ] **Step 3: Inspect Slurm output**

Run:

```bash
squeue -u "$USER"
```

After the job finishes, inspect the matching files:

```bash
ls -t logs/mimic_dora_sweep_*.out logs/mimic_dora_sweep_*.err
```

Expected stdout contains:

```text
Running NCCL distributed smoke check
NCCL barrier probe passed.
MIMIC local setup smoke check passed.
```

Expected stderr does not contain:

```text
Invalid config blocking attribute value -2147483648
torch.distributed.DistBackendError
ChildFailedError
```

- [ ] **Step 4: If the probe still fails, capture the runtime mismatch evidence**

If the smoke job fails before training, copy these exact lines from the Slurm stdout/stderr into the PR or issue:

```text
Torch:
Torch CUDA:
Torch NCCL:
ctypes nccl lookup:
LD_LIBRARY_PATH:
NCCL env:
ncclInvalidArgument:
```

Then run a single-process fallback to confirm the training code path is otherwise intact:

```bash
sbatch --time=00:20:00 --gres=gpu:1 --export=ALL,DTGPT_USE_DEEPSPEED=0,DTGPT_NPROC_PER_NODE=1,DTGPT_SWEEP_CONFIGS=32,64,1,0.01,8e-6,DTGPT_NUM_SAMPLES_TO_GENERATE=1,DTGPT_MAX_NEW_TOKENS=16,DTGPT_SEQ_MAX_LEN=256 job/submit_mimic_lora_sweep_v100.sh
```

Expected: single-process run reaches `Start training` or fails later with a model/data memory issue, not an NCCL barrier error.

- [ ] **Step 5: Commit validation notes**

Create `docs/superpowers/plans/2026-04-27-mimic-lora-v100-nccl-barrier-validation.md` with the observed job id, command, and result:

````markdown
# MIMIC LoRA V100 NCCL Barrier Validation

Command:

```bash
sbatch --time=00:10:00 --export=ALL,DTGPT_SWEEP_CONFIGS=32,64,16,0.01,8e-6,DTGPT_NUM_SAMPLES_TO_GENERATE=1,DTGPT_MAX_NEW_TOKENS=16,DTGPT_SEQ_MAX_LEN=256,DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK=1 job/submit_mimic_lora_sweep_v100.sh
```

Result:

```text
Submitted batch job REPLACE_WITH_JOB_ID
NCCL barrier probe passed.
MIMIC local setup smoke check passed.
```
````

Run:

```bash
git add docs/superpowers/plans/2026-04-27-mimic-lora-v100-nccl-barrier-validation.md
git commit -m "Document MIMIC LoRA V100 validation"
```

## Self-Review

- Spec coverage: The plan addresses the reported `job/submit_mimic_lora_sweep_v100.sh` failure, the NCCL `barrier()` crash, and the exact training script path in the traceback. It includes launcher diagnostics, training-code control, and V100 validation commands.
- Placeholder scan: No placeholder markers or vague implementation steps remain.
- Type consistency: The new CLI option is `--sft-dataset-num-proc`, the parsed attribute is `args.sft_dataset_num_proc`, the experiment parameter is `sft_dataset_num_proc`, and the TRL kwarg is `dataset_num_proc`.
