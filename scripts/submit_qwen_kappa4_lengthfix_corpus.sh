#!/usr/bin/env bash
# Submit the isolated 64k Qwen kappa=4, token-length-limited watermark corpus.
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

config="data/watermark_config_qwen3_5_9b_kappa4_lengthfix.json"
if [[ ! -f "$config" || ! -f data/keys.json ]]; then
    echo "Run this helper from the repository root." >&2
    exit 1
fi

mkdir -p job_outputs
for batch in $(seq 1 13); do
    printf -v job_name "qwen-wm-k4-b%02d" "$batch"
    sbatch --job-name="$job_name" \
        "$launcher" src.data_creation.create_t_ws "$batch" --config "$config"
done

echo "Submitted 13 independent 5,000-text batches (the final batch has 4,000)."
echo "After every batch completes, combine with:"
echo ".venv-qwen/bin/python -m src.data_creation.create_t_ws --combine --config $config"
