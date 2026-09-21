#!/bin/bash -l
#SBATCH --job-name=watermark-llama-capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=108G
#SBATCH --time=40:00:00
#SBATCH --export=ALL
#SBATCH --output=job_outputs/slurm-%x-%j.out
#SBATCH --error=job_outputs/slurm-%x-%j.err

set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <python-module-path> [arguments ...]" >&2
    exit 64
fi

readonly RUN_MODULE="$1"
shift
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly VENV_DIR="${LLAMA_VENV_DIR:-${PROJECT_DIR}/.venv}"
readonly LOG_DIR="${PROJECT_DIR}/logs"

if [[ ! -f "${PROJECT_DIR}/data/watermark_config.json" ]]; then
    echo "Submit this script from the repository root." >&2
    exit 1
fi

module purge
module load release/24.04 GCCcore/13.3.0 Python/3.12.3

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Llama environment not found: ${VENV_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"

mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/.cache/huggingface" "${PROJECT_DIR}/.cache/torch"
export PYTHONPATH="${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor-${SLURM_JOB_ID}"
export UNSLOTH_COMPILE_LOCATION="${PROJECT_DIR}/.cache/unsloth-${SLURM_JOB_ID}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES is empty; Slurm did not expose the requested GPU." >&2
    exit 1
fi

NVIDIA_SMI_PID=""
cleanup() {
    if [[ -n "${NVIDIA_SMI_PID}" ]]; then
        kill "${NVIDIA_SMI_PID}" 2>/dev/null || true
        wait "${NVIDIA_SMI_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Python executable: $(command -v python)"
echo "Python: $(python --version 2>&1)"
echo "Module: ${RUN_MODULE}"
echo "Arguments: $*"

nvidia-smi
nvidia-smi dmon -s u -d 10 > "${LOG_DIR}/gpu_util_${SLURM_JOB_ID}.log" &
NVIDIA_SMI_PID=$!

python -m "${RUN_MODULE}" "$@"
