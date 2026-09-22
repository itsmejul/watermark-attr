#!/usr/bin/env bash
# Submit direct-detection Qwen sampling ablations around temperature=1.0.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <capella|horeka-green>" >&2
    exit 64
fi

case "$1" in
    capella) launcher="scripts/launch_capella_qwen.sh" ;;
    horeka-green) launcher="scripts/launch_horeka_green.sh" ;;
    *) echo "Unknown cluster: $1" >&2; exit 64 ;;
esac

if [[ ! -f data/watermark_config_qwen3_5_9b.json || ! -f data/keys.json ]]; then
    echo "Run this helper from the repository root." >&2
    exit 1
fi

export QWEN_VENV_DIR="${QWEN_VENV_DIR:-${PWD}/.venv-qwen}"
if [[ ! -x "${QWEN_VENV_DIR}/bin/python" ]]; then
    echo "Missing Qwen watermark environment: ${QWEN_VENV_DIR}" >&2
    exit 1
fi

if ! "${QWEN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if [[ "$1" == "capella" ]]; then
        if ! type module >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            source /etc/profile
        fi
        module purge
        module load release/24.04 GCCcore/13.3.0 Python/3.12.3
    fi
fi
if ! "${QWEN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    echo "The Qwen-environment Python cannot start: ${QWEN_VENV_DIR}/bin/python" >&2
    exit 1
fi

mkdir -p job_outputs
module_name="src.experiments.ablations.qwen_kappa_ablation"

# All arms keep kappa=6 and temperature=1.0. They remove top-p, top-k,
# or both while retaining the other baseline truncation setting.
declare -a conditions=(
    "--temperature 1.0 --top-p 1.0"
    "--temperature 1.0 --top-k 0"
    "--temperature 1.0 --top-p 1.0 --top-k 0"
)
declare -a names=(
    "qwen-t1-notopp"
    "qwen-t1-notopk"
    "qwen-t1-unfiltered"
)

for index in "${!conditions[@]}"; do
    # shellcheck disable=SC2206
    args=( ${conditions[$index]} )
    "${QWEN_VENV_DIR}/bin/python" -m "$module_name" 6 "${args[@]}" --dry-run >/dev/null
    sbatch --job-name="${names[$index]}" "$launcher" "$module_name" 6 "${args[@]}"
done
