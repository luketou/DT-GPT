#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-eval-v100-shard"
#SBATCH --partition=v100-32g
#SBATCH --account=v100-32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:0
#SBATCH --array=0-15%2
#SBATCH --output=logs/mimic_dora_eval_v100_shard_%A_%a.out
#SBATCH --error=logs/mimic_dora_eval_v100_shard_%A_%a.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

export DTGPT_EVAL_BACKEND="${DTGPT_EVAL_BACKEND:-hf}"
export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-16}"
export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"
export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-eager}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-1024}"

bash job/submit_mimic_dora_checkpoint700_eval_v100.sh
