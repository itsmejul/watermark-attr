#!/usr/bin/env bash
# Submit the isolated Qwen-on-Llama-watermark control matrix.
set -Eeuo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <capella|horeka-green> [pipeline arguments, e.g. --resume]" >&2
    exit 64
fi
case "$1" in
    capella) launcher="scripts/launch_capella_qwen.sh" ;;
    horeka-green) launcher="scripts/launch_horeka_green.sh" ;;
    *) echo "Unknown cluster: $1" >&2; exit 64 ;;
esac
shift
for argument in "$@"; do
    case "$argument" in
        --smoke|--dry-run|--preflight)
            echo "Use the launcher directly for $argument; this helper submits 18 full configurations." >&2
            exit 64 ;;
    esac
done
if [[ ! -f data/experiment_config_qwen.json || ! -f data/t_ws/combined_t_ws.json ]]; then
    echo "Run from the repository root with the Llama-watermarked corpus available." >&2
    exit 1
fi
export QWEN_VENV_DIR="${QWEN_VENV_DIR:-${PWD}/.venv-qwen-experiments}"
if [[ ! -x "${QWEN_VENV_DIR}/bin/python" ]]; then
    echo "Missing training environment: ${QWEN_VENV_DIR}" >&2
    exit 1
fi
mkdir -p job_outputs
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        "${QWEN_VENV_DIR}/bin/python" -m src.experiments.main.qwen_on_llama_pipeline \
            "$n" "$sample_type" 32 1000 --preflight "$@" >/dev/null
    done
done
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --job-name="qwen-llamawm-${sample_type}-${n}" "$launcher" \
            src.experiments.main.qwen_on_llama_pipeline "$n" "$sample_type" 32 1000 "$@"
    done
done
