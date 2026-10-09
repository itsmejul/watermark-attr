#!/usr/bin/env bash
# Two 100-epoch recitation runs; reuse existing M=1 without submitting it.
set -Eeuo pipefail
if [[ $# -ne 1 || "$1" != capella ]]; then
    echo "Usage: bash $0 capella (after watermark variants finish)" >&2
    exit 64
fi
if [[ -z "${RUN_VENV_DIR:-}" ]]; then
    if [[ -x "$PWD/.venv-experiment/bin/python" ]]; then
        export RUN_VENV_DIR="$PWD/.venv-experiment"
    else
        export RUN_VENV_DIR="$PWD/.venv-qwen-experiments"
    fi
fi
if [[ ! -x "$RUN_VENV_DIR/bin/python" ]]; then
    echo "Missing training environment: $RUN_VENV_DIR" >&2
    exit 1
fi
if ! "$RUN_VENV_DIR/bin/python" -c 'pass' >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then source /etc/profile; fi
    module purge
    module load release/24.04 GCCcore/13.3.0 Python/3.12.3
fi
"$RUN_VENV_DIR/bin/python" -c \
  'from importlib.metadata import version; assert version("waterfall") == "0.3.4"; assert version("transformers") == "5.5.0", "Use the experiment environment"'
for m in 3 5; do
    "$RUN_VENV_DIR/bin/python" -m src.experiments.main.qwen_pipeline \
        1000 abstracts_only 32 1000 --augmentation-m "$m" --preflight
done
mkdir -p job_outputs
for m in 3 5; do
    sbatch --partition=capella --job-name="qwen-aug-M${m}-1000" \
        scripts/launch_capella.sh src.experiments.main.qwen_pipeline \
        1000 abstracts_only 32 1000 --augmentation-m "$m" --resume
done
