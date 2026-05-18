#!/bin/bash
#SBATCH --job-name="install-dtgpt-vllm"
#SBATCH --partition=cpu-2g
#SBATCH --account=cpu-2g
#SBATCH --nodes=1
#SBATCH --nodelist=node-11
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/install_dtgpt_vllm_%j.out
#SBATCH --error=logs/install_dtgpt_vllm_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

REPO_ROOT="${DTGPT_REPO_ROOT:-/share/home/r15543056/trajectory_forecast/DT-GPT}"
cd "${REPO_ROOT}"
mkdir -p logs

if command -v sbatch_pre.sh >/dev/null 2>&1; then
    sbatch_pre.sh
fi

ENV_PREFIX="${DTGPT_VLLM_ENV_PREFIX:-/home/r15543056/miniconda3/envs/dtgpt-vllm}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
VLLM_VERSION="${DTGPT_VLLM_VERSION:-0.19.1}"

# Use /share/home instead of /tmp: vLLM pulls multi-GB torch/CUDA wheels.
export TMPDIR="${DTGPT_INSTALL_TMPDIR:-/share/home/r15543056/.cache/tmp-dtgpt-vllm}"
export PIP_CACHE_DIR="${DTGPT_INSTALL_PIP_CACHE_DIR:-/share/home/r15543056/.cache/pip-dtgpt-vllm}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

LOCK_FILE="${DTGPT_INSTALL_LOCK_FILE:-/share/home/r15543056/.cache/dtgpt-vllm-install.lock}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another dtgpt-vllm install job is holding ${LOCK_FILE}; exiting safely."
    exit 75
fi

# Avoid corrupting the env if an interactive pip install is already modifying it.
if pgrep -u "${USER}" -f "${PYTHON_BIN} -m pip install" >/dev/null 2>&1; then
    echo "Detected an existing interactive pip install using ${PYTHON_BIN}; exiting safely."
    echo "Re-submit this job after that install finishes, or kill the old process intentionally."
    exit 75
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found at ${PYTHON_BIN}; creating a clean conda env at ${ENV_PREFIX}."
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda create -p "${ENV_PREFIX}" python=3.11 -y
        PYTHON_BIN="${ENV_PREFIX}/bin/python"
    else
        echo "conda is not available in this job environment."
        exit 1
    fi
fi

"${PYTHON_BIN}" -m pip install --upgrade pip

echo "=== install-dtgpt-vllm ==="
echo "date:        $(date)"
echo "host:        $(hostname)"
echo "repo:        ${REPO_ROOT}"
echo "python:      ${PYTHON_BIN}"
echo "vllm:        ${VLLM_VERSION}"
echo "TMPDIR:      ${TMPDIR}"
echo "PIP_CACHE:   ${PIP_CACHE_DIR}"
echo "df /tmp:"
df -h /tmp || true
echo "df /share/home:"
df -h /share/home/r15543056 || true

echo "=== pip install vLLM server deps ==="
"${PYTHON_BIN}" -m pip install --only-binary=:all: \
    "vllm==${VLLM_VERSION}" \
    openai \
    pandas

if [[ "${DTGPT_INSTALL_UNSLOTH:-0}" == "1" || "${DTGPT_INSTALL_UNSLOTH:-0}" == "true" ]]; then
    echo "=== optional: pip install unsloth ==="
    "${PYTHON_BIN}" -m pip install --only-binary=:all: unsloth
fi

echo "=== verify imports ==="
"${PYTHON_BIN}" - <<'PY'
import sys
print('python', sys.executable)
for name in ['torch', 'vllm', 'openai', 'pandas']:
    mod = __import__(name)
    print(name, getattr(mod, '__version__', 'no __version__'))
try:
    import unsloth
    print('unsloth', getattr(unsloth, '__version__', 'no __version__'))
except Exception as exc:
    print('unsloth not installed/verified:', repr(exc))
PY

echo "=== done $(date) ==="
