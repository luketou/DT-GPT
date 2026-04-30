#!/bin/bash
#SBATCH --job-name="dtgpt-node181-smoke"
#SBATCH --partition=llm
#SBATCH --account=l40s
#SBATCH --nodelist=node-181
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/node181_%j.out
#SBATCH --error=logs/node181_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

source /home/r15543056/miniconda3/etc/profile.d/conda.sh
conda activate dtgpt

echo "[ENV] hostname=$(hostname)"
echo "[ENV] account=${SLURM_JOB_ACCOUNT:-unset}"
echo "[ENV] partition=${SLURM_JOB_PARTITION:-unset}"
echo "[ENV] node_list=${SLURM_JOB_NODELIST:-unset}"

python - <<'PY'
from pathlib import Path

import torch

model_path = Path("/home/r15543056/llm_model/BioMistral-7B-DARE")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
print(f"Model path exists: {model_path.exists()}")
print(f"Config exists: {(model_path / 'config.json').exists()}")
print("node181 smoke check passed.")
PY
