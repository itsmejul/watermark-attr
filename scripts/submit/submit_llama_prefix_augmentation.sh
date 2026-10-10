#!/usr/bin/env bash
# One command: four watermark jobs, then M=3/M=5 training + closed verification.
set -Eeuo pipefail
if [[ $# -ne 1 || "$1" != capella ]]; then
    echo "Usage: bash $0 capella (from repository root)" >&2
    exit 64
fi
if [[ ! -f data/t_ws/config_llama.json || ! -f scripts/launch_capella.sh ]]; then
    echo "Run from the repository root." >&2
    exit 1
fi
generation_env="${WATERMARK_VENV_DIR:-$PWD/.venv-watermark}"
experiment_env="${EXPERIMENT_VENV_DIR:-$PWD/.venv-experiment}"
if [[ -z "${WATERMARK_VENV_DIR:-}" && ! -x "$generation_env/bin/python" ]]; then
    generation_env="$PWD/.venv-qwen"
fi
if [[ -z "${EXPERIMENT_VENV_DIR:-}" && ! -x "$experiment_env/bin/python" ]]; then
    experiment_env="$PWD/.venv-qwen-experiments"
fi
for env_dir in "$generation_env" "$experiment_env"; do
    if [[ ! -x "$env_dir/bin/python" ]]; then
        echo "Missing environment: $env_dir" >&2
        exit 1
    fi
done
if ! "$generation_env/bin/python" -c 'pass' >/dev/null 2>&1 || \
   ! "$experiment_env/bin/python" -c 'pass' >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then source /etc/profile; fi
    module purge
    module load release/24.04 GCCcore/13.3.0 Python/3.12.3
fi
"$generation_env/bin/python" -c \
  'from importlib.metadata import version; assert version("waterfall") == "0.3.4"; assert version("transformers") == "5.17.0", "Use req-watermark.txt"'
"$experiment_env/bin/python" -c \
  'from importlib.metadata import version; assert version("waterfall") == "0.3.4"; assert version("transformers") == "5.5.0", "Use req-experiment.txt"'
export HF_HOME="$PWD/.cache/huggingface"
"$generation_env/bin/python" -m src.data_creation.create_llama_prefix_variants --preflight
# Catch gated-tokenizer access and BOS/prefix mismatches before submitting jobs.
"$generation_env/bin/python" - <<'PY'
from transformers import AutoTokenizer
from src.data_creation.create_qwen_prefix_variants import inputs
from src.util.shared_prefix import prefix_ids
config, rows = inputs(1000, 'llama')
tok = AutoTokenizer.from_pretrained(config['watermark_model'])
for row in rows:
    prefix_ids(tok, row['version_1'], row['prefix'])
    if row['k_p'] >= tok.vocab_size // 2:
        raise ValueError('Selected key is incompatible with legacy cosine-only generation')
print('Llama tokenizer access, saved prefixes, and Fourier key compatibility checked.')
PY
mkdir -p job_outputs
generation_jobs=()
for v in 2 3 4 5; do
    submission=$(RUN_VENV_DIR="$generation_env" sbatch --parsable --partition=capella \
        --job-name="llama-prefix-v${v}-1000" \
        scripts/launch_capella.sh src.data_creation.create_llama_prefix_variants \
        --n-samples 1000 --version "$v")
    job_id="${submission%%;*}"
    generation_jobs+=("$job_id")
    echo "Llama V${v} generation: ${job_id}"
done
for m in 3 5; do
    dependencies="afterok:${generation_jobs[0]}:${generation_jobs[1]}"
    if [[ "$m" == 5 ]]; then
        dependencies+=":${generation_jobs[2]}:${generation_jobs[3]}"
    fi
    submission=$(RUN_VENV_DIR="$experiment_env" sbatch --parsable --partition=capella \
        --dependency="$dependencies" --kill-on-invalid-dep=yes \
        --job-name="llama-aug-M${m}-1000" \
        scripts/launch_capella.sh src.experiments.main.llama_prefix_augmentation \
        1000 abstracts_only 32 1000 --augmentation-m "$m" --resume)
    echo "Llama M${m} training + evaluation: ${submission%%;*} (${dependencies})"
done
echo 'M=1 reuses results/llama/experiment1/prefix_10/1000/32/. No combine step required.'
