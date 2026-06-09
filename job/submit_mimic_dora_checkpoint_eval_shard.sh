#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-eval-shard"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:0
#SBATCH --array=0-7%4
#SBATCH --output=logs/mimic_dora_eval700_shard_%A_%a.out
#SBATCH --error=logs/mimic_dora_eval700_shard_%A_%a.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

export DTGPT_EVAL_BACKEND="${DTGPT_EVAL_BACKEND:-hf}"
export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-8}"
export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"

bash job/submit_mimic_dora_checkpoint700_eval.sh
