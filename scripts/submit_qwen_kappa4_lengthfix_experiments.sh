#!/usr/bin/env bash
# Submit the isolated Qwen-on-Qwen kappa=4, length-fixed 18-run matrix.
# Every task/corpus-size pair is one independent Slurm job.
set -Eeuo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <capella|horeka-green> [pipeline arguments, e.g. --resume]" >&2
    exit 64
fi
cluster="$1"
case "$cluster" in
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
readonly profile="qwen_kappa4_lengthfix"
readonly corpus_dir="data/t_ws_qwen3_5_9b_kappa4_lengthfix"
readonly prompts_dir="data/prompts_qwen3_5_9b_kappa4_lengthfix"
if [[ ! -f data/experiment_config_qwen.json || \
      ! -f "${corpus_dir}/combined_t_ws.json" || \
      ! -f "${corpus_dir}/combined_manifest.json" || \
      ! -f "${prompts_dir}/prefix_10.json" ]]; then
    echo "Missing the combined kappa=4 corpus or its tokenizer-specific prefix prompts." >&2
    echo "Run the prefix-generation command documented for this grid, then retry." >&2
    exit 1
fi
export QWEN_VENV_DIR="${QWEN_VENV_DIR:-${PWD}/.venv-qwen-experiments}"
if [[ ! -x "${QWEN_VENV_DIR}/bin/python" ]]; then
    echo "Missing training environment: ${QWEN_VENV_DIR}" >&2
    exit 1
fi
if ! "${QWEN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if [[ "$cluster" == "capella" ]]; then
        if ! type module >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            source /etc/profile
        fi
        if ! type module >/dev/null 2>&1; then
            echo "Capella's module command is unavailable; run this helper from a login shell." >&2
            exit 1
        fi
        module purge
        module load release/24.04 GCCcore/13.3.0 Python/3.12.3
    fi
fi
if ! "${QWEN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    echo "The training-environment Python cannot start: ${QWEN_VENV_DIR}/bin/python" >&2
    exit 1
fi
mkdir -p job_outputs

# Validate all 18 configurations before submitting any jobs.
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        "${QWEN_VENV_DIR}/bin/python" -m src.experiments.main.full_pipeline \
            "$n" "$sample_type" 32 1000 --profile "$profile" --preflight "$@" >/dev/null
    done
done

# Deliberately submit one independent job for each grid cell.
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --job-name="qwen-k4-${sample_type}-${n}" "$launcher" \
            src.experiments.main.full_pipeline "$n" "$sample_type" 32 1000 \
            --profile "$profile" "$@"
    done
done
