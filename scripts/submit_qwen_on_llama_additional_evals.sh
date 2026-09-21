#!/usr/bin/env bash
# Submit open-keyspace evaluation plus auxiliary metrics for Qwen-on-Llama.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <capella|horeka-green>" >&2
    exit 64
fi

cluster="$1"
case "$cluster" in
    capella) launcher="scripts/launch_capella_qwen.sh" ;;
    horeka-green) launcher="scripts/launch_horeka_green.sh" ;;
    *) echo "Unknown cluster: $cluster" >&2; exit 64 ;;
esac

if [[ ! -f data/experiment_config_qwen.json || ! -f data/t_ws/combined_t_ws.json ]]; then
    echo "Run from the repository root with the Llama-watermarked corpus available." >&2
    exit 1
fi

export QWEN_VENV_DIR="${QWEN_VENV_DIR:-${PWD}/.venv-qwen-experiments}"
if [[ ! -x "${QWEN_VENV_DIR}/bin/python" ]]; then
    echo "Missing Qwen experiment environment: ${QWEN_VENV_DIR}" >&2
    exit 1
fi
if ! "${QWEN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if [[ "$cluster" == "capella" ]]; then
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

# Validate the shared corpus/open prompts before creating any Slurm jobs.
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        "${QWEN_VENV_DIR}/bin/python" -m src.experiments.main.qwen_on_llama_open_pipeline \
            "$n" "$sample_type" 32 1000 --preflight >/dev/null
    done
done

# One open-keyspace job per trained configuration. --resume makes re-submission safe.
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --job-name="qol-open-${sample_type}-${n}" "$launcher" \
            src.experiments.main.qwen_on_llama_open_pipeline \
            "$n" "$sample_type" 32 1000 --resume
    done
done

# One auxiliary-metric sweep per experiment. It evaluates the best and final
# epochs and skips files already produced by an earlier/restarted job.
for sample_type in abstracts_only abstracts_and_titles questions; do
    sbatch --job-name="qol-metrics-${sample_type}" "$launcher" \
        src.experiments.main.similarity_eval \
        --profile qwen_on_llama \
        --metrics bm25 cosine bertscore bigram lcs \
        --sweep --sample_types "$sample_type" --skip-existing
done
