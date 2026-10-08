#!/usr/bin/env bash
# Twelve independent direct-text baselines, never modifying the main grids.
set -Eeuo pipefail
if [[ $# -ne 1 || "$1" != capella ]]; then
    echo "Usage (from repository root): bash $0 capella" >&2
    exit 64
fi
export RUN_VENV_DIR="${RUN_VENV_DIR:-${PWD}/.venv-experiment}"
if [[ ! -x "$RUN_VENV_DIR/bin/python" ]]; then
    echo "Missing environment: $RUN_VENV_DIR" >&2
    exit 1
fi
if ! "$RUN_VENV_DIR/bin/python" -c 'pass' >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then
        source /etc/profile
    fi
    module purge
    module load release/24.04 GCCcore/13.3.0 Python/3.12.3
fi
"$RUN_VENV_DIR/bin/python" -c \
    'from importlib.metadata import version; assert version("waterfall") == "0.3.4", "Use waterfall==0.3.4"; assert version("transformers") == "5.5.0", "Use req-experiment.txt"'
# Validate the entire grid before submitting any jobs. No ML imports needed.
for source in llama qwen; do
    for n in 100 500 1000 5000 10000 50000; do
        "$RUN_VENV_DIR/bin/python" -m src.experiments.main.oracle_watermark_eval \
            --source "$source" --n-samples "$n" --preflight
    done
done
mkdir -p job_outputs
for source in llama qwen; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --partition=capella --time=02:00:00 \
            --job-name="oracle-${source}-${n}" \
            scripts/launch_capella.sh src.experiments.main.oracle_watermark_eval \
            --source "$source" --n-samples "$n"
    done
done
