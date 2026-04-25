## Goal

Add a Slurm batch script that runs a small, fixed LoRA hyperparameter sweep for the MIMIC BioMistral training pipeline inside a single submitted job. The sweep should run configurations sequentially within one allocation so all runs share the same node, environment setup, and cache directories.

## Scope

The change is limited to a new job script under `job/`. It does not modify the training Python entrypoint, add distributed training, or change evaluation behavior.

## Proposed File

- `job/submit_mimic_lora_sweep_v100.sh`

## Behavior

The new script will:

- reuse the environment setup from `job/submit_mimic_train_v100.sh`
- run the existing MIMIC training entrypoint
- execute multiple LoRA configurations sequentially in one Slurm job
- print a clear header before each run showing the active hyperparameters
- stop the sweep immediately if any individual run fails
- keep each run's artifacts separate by relying on the experiment code's timestamped output directories

## Default Sweep

The script will ship with a default fixed sweep of four configurations:

1. `lora_r=16`, `lora_alpha=32`, `gradient_accumulation=8`, `epochs=8`, `learning_rate=1e-5`
2. `lora_r=32`, `lora_alpha=64`, `gradient_accumulation=8`, `epochs=8`, `learning_rate=1e-5`
3. `lora_r=64`, `lora_alpha=128`, `gradient_accumulation=8`, `epochs=8`, `learning_rate=1e-5`
4. `lora_r=32`, `lora_alpha=64`, `gradient_accumulation=16`, `epochs=10`, `learning_rate=1e-5`

## Override Mechanism

The script will support an optional `DTGPT_SWEEP_CONFIGS` environment variable for advanced overrides.

Format:

`r,alpha,grad_acc,epochs,lr;r,alpha,grad_acc,epochs,lr`

Example:

`DTGPT_SWEEP_CONFIGS="16,32,8,8,1e-5;32,64,16,10,5e-6"`

If the variable is unset, the script uses the default four-run sweep.

## Logging

The script should emit:

- the total number of sweep configurations
- a numbered progress line for each run
- the exact hyperparameters for each run
- timestamps for run start and finish

The script-level Slurm log remains the top-level record. Per-run detailed metrics continue to be stored by the training code in the normal experiment output folders and WandB.

## Failure Handling

- `set -euo pipefail` remains enabled
- any failed training invocation aborts the full sweep
- the script should print which configuration failed before exiting

## GPU Assumption

The script should keep the current `3 x V100 32GB` allocation from the existing training job.

Reasoning:

- the current training path uses long sequence inputs, LoRA training, reload-for-eval, and `device_map="auto"`
- the repository is not using a formal distributed training stack for this path
- reducing to fewer GPUs would increase OOM risk and lower confidence in sweep stability

This is a stability-oriented choice for the current implementation, not a theoretical lower bound for LoRA.
