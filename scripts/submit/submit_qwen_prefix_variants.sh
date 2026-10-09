#!/usr/bin/env bash
# Generation only: four independent, resumable same-prefix variant jobs.
set -Eeuo pipefail
if [[ $# -ne 1 || "$1" != capella ]]; then
    echo "Usage: bash $0 capella (from repository root)" >&2
    exit 64
fi
# Legacy HPC generation environment name, NOT .venv-qwen-experiments.
if [[ -z "${RUN_VENV_DIR:-}" ]]; then
    if [[ -x "$PWD/.venv-watermark/bin/python" ]]; then
        export RUN_VENV_DIR="$PWD/.venv-watermark"
    else
        export RUN_VENV_DIR="$PWD/.venv-qwen"
    fi
fi
if [[ ! -x "$RUN_VENV_DIR/bin/python" ]]; then
    echo "Missing generation environment: $RUN_VENV_DIR" >&2
    exit 1
fi
if ! "$RUN_VENV_DIR/bin/python" -c 'pass' >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then source /etc/profile; fi
    module purge
    module load release/24.04 GCCcore/13.3.0 Python/3.12.3
fi
"$RUN_VENV_DIR/bin/python" -c \
  'from importlib.metadata import version; assert version("waterfall") == "0.3.4"; assert version("transformers") == "5.17.0", "Use the watermark-generation environment"'
"$RUN_VENV_DIR/bin/python" -m src.data_creation.create_qwen_prefix_variants --preflight
mkdir -p job_outputs
for v in 2 3 4 5; do
    sbatch --partition=capella --job-name="qwen-prefix-v${v}-1000" \
        scripts/launch_capella.sh src.data_creation.create_qwen_prefix_variants \
        --n-samples 1000 --version "$v"
done
echo 'After completion: python -m src.data_creation.create_qwen_prefix_variants --combine'
