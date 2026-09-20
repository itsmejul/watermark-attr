#!/usr/bin/env bash
# Submit the three direct-detection Qwen watermark-strength conditions.
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
for kappa in 6 10 14; do
    "${QWEN_VENV_DIR}/bin/python" \
        -m src.experiments.ablations.qwen_kappa_ablation \
        "$kappa" --dry-run >/dev/null
done

for kappa in 6 10 14; do
    sbatch --job-name="qwen-kappa-${kappa}" "$launcher" \
        src.experiments.ablations.qwen_kappa_ablation "$kappa"
done
