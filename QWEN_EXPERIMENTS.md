# Qwen3.5-9B experiments

Use `--profile qwen` on the existing main experiment commands. Omitting it
retains the Llama arm; do not run Llama training/verification in the new environment.
Neither `.venv` nor the running watermark environment `.venv-qwen` is modified.

## Full-parameter / continued-pretraining arm

`full_pipeline_pretrained.py` is the isolated full-parameter counterpart to the
LoRA pipeline. It defaults to the 1,000-example arm, trains for five epochs, and
saves an inference-ready full model plus a resumable Trainer checkpoint after
every epoch. The positional arguments and the Qwen operational flags match the
main pipeline:

```bash
python -m src.experiments.main.full_pipeline_pretrained \
  1000 abstracts_only 32 1000 --profile qwen --preflight

sbatch --time=20:00:00 --job-name=qwen-full-1000 \
  scripts/launch_capella_qwen.sh src.experiments.main.full_pipeline_pretrained \
  1000 abstracts_only 32 1000 --profile qwen --train-only
```

The full-parameter arm writes models below `full_models/qwen/` and results below
`results/experimentN-qwen-pretrained/`; it never reads or writes LoRA adapter or
LoRA result directories. Epoch models live at `<run>/<epoch>/model/`, while the
latest optimizer/scheduler/RNG state lives below `<run>/resume_checkpoints/`.
Use `--resume` after a stopped job and `--eval-only` to run generation and
verification separately from expensive training.

The single-H100 defaults are deliberately conservative: BF16 weights,
`paged_adamw_8bit`, microbatch 1, effective batch 32 via accumulation, and
Unsloth gradient checkpointing. Full fine-tuning uses a lower learning rate of
`1e-5`; LoRA's `2e-4` is generally too aggressive when every parameter is
trainable. This is still a borderline fit on an 80 GB H100. Run the isolated
`--smoke` job first and inspect peak VRAM; paging can make an otherwise fitting
run substantially slower. Five BF16 epoch models plus one resumable checkpoint
and the model cache require roughly 150--200 GB of free local storage.

For `abstracts_only`, this is causal continued pretraining on the watermarked
abstract text. The title and question variants remain supervised full-parameter
fine-tuning because they retain the chat-formatted prompt/answer examples used
by the matching LoRA experiments.

Slurm stdout/stderr from the launchers is saved as
`job_outputs/slurm-<job-name>-<job-id>.out` and `.err` in the repository.
Git includes the directory; if necessary, run `mkdir -p job_outputs` **before**
`sbatch` (Slurm opens logs before the script starts). The 18-job submission helper
also creates it. Existing jobs/logs are not moved; GPU utilization logs stay in `logs/`.

## What stays the same

- Six sizes: **100, 500, 1000, 5000, 10000, 50000** (the last Llama size was 63800).
- Three experiments: `abstracts_only`, `abstracts_and_titles`, `questions`.
- 100 epochs; adapters every 5 epochs; effective batch 32; learning rate 0.0002;
  constant schedule with 3% warmup; rank/alpha 16; no LoRA dropout.
- Same keys, seed-48 training subsets, held-out rows 63800–63999, seed-1234
  evaluation subsets (at most 1000), and open negatives at rows 64000–64999.
- Training truncation: 300 tokens for experiment 1; 512 for experiments 2/3.
- Greedy answer generation (`do_sample=false`, temperature 0, max 300 new tokens).
  **Watermark generation** separately uses sampling/temperature 0.5/kappa 6/ngram 2.
- Titles, perturbed titles, and questions are reused. Questions were generated
  from **original unwatermarked abstracts**, not Llama paraphrases.

## Deliberate differences and compatibility

- Qwen watermarked targets: `data/t_ws_qwen3_5_9b/combined_t_ws.json`.
- Qwen prefixes/control prefixes/open prefixes: `data/prompts_qwen3_5_9b/`.
- Results: `results/experiment{1,2,3}-qwen/`; adapter root: `lora_adapters/qwen/`.
  Controls retain the `_unwatermarked` variant suffix inside those roots.
- Qwen chat templates always receive `enable_thinking=False`. Experiment 1
  remains raw-text completion, not a new chat/reasoning task.
- BF16 LoRA with Unsloth, **not QLoRA**. Text-only loading excludes the vision
  tower. H100 training uses microbatch 32 and accumulation 1 for experiment 1;
  the longer chat-formatted experiments 2/3 use microbatch 16 and accumulation
  2. All three retain effective batch 32, loss-evaluation batch 32, and disabled
  gradient checkpointing. Inference batch is 64 and is a separate setting.
  These replace the initial memory-conservative A100 settings (microbatch 1,
  accumulation 32, eval batch 1, Unsloth gradient checkpointing). Test GPU memory
  fit before production; Qwen need not fit the same batch as Llama. Adjust with
  `--micro-batch-size` / `--inference-batch-size` as needed. Existing jobs do not
  pick up config edits, and existing run manifests reject changed settings;
  these defaults are not an automatic migration of already-started runs.
- Qwen LoRA covers both full-attention projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`) and Gated DeltaNet projections (`in_proj_qkv`, `in_proj_z`,
  `in_proj_a`, `in_proj_b`, `out_proj`). The run records exact trainable names
  and counts in `trainable_parameters.json`.
- Qwen training examples receive an explicit terminal EOS. The data collator
  masks positions using the attention mask, so genuine EOS labels remain
  supervised even though Qwen uses its EOS token as padding. This, together
  with the 300-token generation cap, directly addresses the old 512-token
  runaway tails.
- The EOS, generation-length, and LoRA-target changes produce a new run
  configuration. Existing Qwen adapter/result directories should be archived
  before starting fresh runs; do not use `--resume` across this change. Run
  manifests deliberately reject mixing the old and new settings.
- Published Unsloth **2026.9.5** requires Transformers **<=5.5.0**. Training uses
  **5.5.0**, Waterfall **0.3.4**, Torch **2.9.1**, and a resolved dependency lock.
  Watermarking remains on Transformers **5.17.0** in `.venv-qwen`. Do not install
  Unsloth into that environment or bypass its requirements with `--no-deps`.
- The training stack explicitly keeps AdamW rather than inheriting newer Torch's
  changed fused-optimizer default. Transformers 5's renamed length-grouping
  argument is handled. Unsloth loads before Transformers at pipeline entrypoints.
- Unsloth's selected release bundles linear-attention kernels; do not blindly add
  incompatible `flash-attn`/`causal-conv1d` builds to either environment. Inspect
  the GPU smoke log for fallback warnings and actual timing.
- The verifier follows Waterfall 0.3.4's Fourier sine-sign convention while
  preserving the legacy 0.2.13 calculation. Verification no longer needs to load
  the 9B weights in the Qwen runner. These are distinct watermark arms, not
  bitwise reproductions of the old corpus.
- Prefixes retain the historical tokenizer.encode/decode convention (including
  its special-token handling). Token-based lengths/LCS/bigrams are tokenizer
  dependent; use the Qwen bigram cache, never the Llama cache.

Sources: [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune),
[published Unsloth metadata](https://pypi.org/pypi/unsloth/2026.9.5/json).

## 1. Combine watermark jobs and prepare data

From the repository root, using the existing watermark environment:

```bash
source .venv-qwen/bin/activate
python -m src.data_creation.create_t_ws --config data/watermark_config_qwen3_5_9b.json --combine
python -m src.data_creation.create_prompt_dataset \
  --profile data/prompt_config_qwen3_5_9b.json --tasks prefix --set both \
  --include-unwatermarked-prefix
```

The combined corpus must contain all 64000 texts and its combined manifest.
Copy `data/seeded_dataset.jsonl` (at least 65000 rows), `data/keys.json`, and the
original `data/prompts/{titles_1,titles_2,titles_3,questions}.json` plus their
`_open.json` counterparts to a cleaned checkout if absent. Prefix preparation
validates reused files and **does not make OpenAI calls** with `--tasks prefix`.
Do not regenerate titles/questions just because they are missing on the new host.

## 2. Create a separate training environment

From the repository root. On Capella load the same modules as the launcher first:

```bash
module load release/24.04 GCCcore/13.3.0 Python/3.12.3  # Capella only
.venv-qwen/bin/python -m venv .venv-qwen-experiments
source .venv-qwen-experiments/bin/activate
python -m pip install -r requirements-qwen-experiments.lock.txt
python -m pip check
export QWEN_VENV_DIR="$PWD/.venv-qwen-experiments"
```

On HoreKa use the workspace-local Python already used by `.venv-qwen`, without
the Capella module command. Both launchers honor `QWEN_VENV_DIR`; without it they
still default to `.venv-qwen` for existing watermark jobs.

Preflight is CPU-only and checks paths, alignment, manifest hashes, and splits:

```bash
python -m src.experiments.main.full_pipeline 50000 questions 32 1000 --profile qwen --preflight
```

## 3. GPU smoke tests before submitting 18 jobs

One job per experiment, one epoch, 100 training texts, five generated answers
per prompt variant. Includes training, saving/reloading adapters, and verification:

```bash
export QWEN_VENV_DIR="$PWD/.venv-qwen-experiments"
for experiment in abstracts_only abstracts_and_titles questions; do
  sbatch --time=02:00:00 --job-name="qwen-smoke-${experiment}" \
    scripts/launch_capella_qwen.sh src.experiments.main.full_pipeline \
    100 "$experiment" 32 5 --profile qwen --smoke
done
```

On HoreKa substitute `scripts/launch_horeka_green.sh`. The production partition
is `accelerated`; for a short development test use
`--partition=dev_accelerated --time=01:00:00` instead. Compilation may make the
first training run slow. Smoke data goes to `results/experimentN-qwen-smoke/`
and `lora_adapters/qwen/smoke/`, never the production directories.

Check Slurm `COMPLETED`/exit `0:0`, finite training/eval losses, nonempty answers,
`verification_closed.json`, and saved tokenizer/adapter files for all three jobs.
Local checks do not substitute for these GPU tests.

## 4. Submit all main experiments (18 independent jobs)

After the smoke tests pass:

```bash
bash scripts/submit_qwen_experiments.sh capella
# Or:
bash scripts/submit_qwen_experiments.sh horeka-green
```

The helper preflights all configurations before submitting. It uses batch 32,
1000 evaluation samples, all six requested sizes, and all three experiments.
Current launcher limits: **Capella 20 hours; HoreKa Green 48 hours**. The large
training runs are not guaranteed to finish within those limits.

Full Trainer checkpoints (optimizer/scheduler/RNG) are saved each epoch, keeping
the last two. Every-fifth-epoch analysis adapters remain separately saved. To
continue a timed-out job, resubmit the **same configuration** with `--resume`:

```bash
sbatch scripts/launch_capella_qwen.sh src.experiments.main.full_pipeline \
  50000 questions 32 1000 --profile qwen --resume
```

This resumes training from the last completed checkpoint or skips completed
training, then skips completed generation/verification stages. An interrupted
generation stage restarts for that prompt/epoch. No automatic Slurm requeue is
enabled. Only resubmit after the previous job has stopped. Manifests and a
per-adapter job lock reject incompatible inputs/settings and concurrent writers.
`--train-only` and `--eval-only` are available if stages must be split later.

### Restarting runs after a microbatch change

The defaults are training microbatch 32/accumulation 1 for experiment 1 and
microbatch 16/accumulation 2 for experiments 2/3, with loss-evaluation batch 32,
gradient checkpointing disabled, and inference batch 64. These settings retain
effective batch 32. Microbatch changes alter the recorded run configuration. Do
not delete or edit manifests to bypass the guard. The procedure below describes
a complete restart while preserving old files in a recoverable archive.
No dependencies, Llama results, watermarked texts, prompts, control runs, or smoke
runs are changed. Open-keyspace results under the selected main-run directories
are archived with their corresponding adapters.

First inspect `squeue --me -o '%.18i %.12T %.80j'`, cancel only the old Qwen main
experiment job IDs with `scancel ID1 ID2 ...`, and wait until those IDs disappear
from `squeue` (including jobs in COMPLETING state). Cancel pending jobs too.
Do not archive directories while a job can still write into them.

After pulling this branch, run this block from the repository root on Capella:

```bash
(
  set -euo pipefail
  test -f data/experiment_config_qwen.json
  restart_archive=$(mktemp -d "$PWD/qwen-restart-backup-XXXXXXXX")
  for old_path in \
    lora_adapters/qwen/abstracts_only \
    lora_adapters/qwen/abstracts_and_titles \
    lora_adapters/qwen/questions \
    results/experiment1-qwen/prefix_10 \
    results/experiment2-qwen/titles \
    results/experiment2-qwen/titles_1 \
    results/experiment2-qwen/titles_2 \
    results/experiment2-qwen/titles_3 \
    results/experiment3-qwen/train_questions \
    results/experiment3-qwen/held_out_questions
  do
    if [[ -d "$old_path" ]]; then
      mkdir -p "$restart_archive/$(dirname "$old_path")"
      mv -- "$old_path" "$restart_archive/$old_path"
    fi
  done
  echo "Old runs preserved in: $restart_archive"
)
```

Archive only once, before submitting new runs. First use experiment 2 at size
100 as a full-pipeline H100 check; unlike the five-example smoke test, this also
exercises a full inference batch of 64. It is a real 100-epoch production run:

```bash
export QWEN_VENV_DIR="$PWD/.venv-qwen-experiments"
mkdir -p job_outputs
sbatch --job-name=qwen-h100-check-exp2-100 \
  scripts/launch_capella_qwen.sh src.experiments.main.full_pipeline \
  100 abstracts_and_titles 32 1000 --profile qwen
```

Check that this job finishes COMPLETED with exit code 0:0, not just training
completion. Memory fit at the requested settings is not guaranteed; stop here
if it runs out of memory. Passing this subset is not a guarantee for every
sequence-length mix in the larger runs. Then submit the remaining 17:

```bash
for experiment in abstracts_only abstracts_and_titles questions; do
  for size in 100 500 1000 5000 10000 50000; do
    if [[ "$experiment" == abstracts_and_titles && "$size" == 100 ]]; then
      continue
    fi
    sbatch --job-name="qwen-${experiment}-${size}" \
      scripts/launch_capella_qwen.sh src.experiments.main.full_pipeline \
      "$size" "$experiment" 32 1000 --profile qwen
  done
done
```

These are fresh runs: no `--resume`. Future restarts of these new runs can use
`--resume` with unchanged settings. The Capella launcher still requests 20 hours;
use the measured new throughput to decide if larger runs need checkpoint resumes.

## 5. Controls and downstream evaluation

Examples for one configuration (substitute any size/experiment; use the selected
launcher for GPU tasks). Keep `QWEN_VENV_DIR` exported as above:

```bash
# Unwatermarked training control
sbatch scripts/launch_capella_qwen.sh src.experiments.main.full_pipeline_unwatermarked \
  1000 abstracts_only 32 1000 --profile qwen

# Open-keyspace evaluation, after the corresponding main job finishes
sbatch scripts/launch_capella_qwen.sh src.experiments.main.open_keyspace_eval \
  1000 abstracts_only 32 1000 --profile qwen
# Add --unwatermarked to evaluate the corresponding control adapters instead.

# CPU-only corpus statistics/cache preparation
python -m src.experiments.main.count_t_w_lengths --profile qwen
python -m src.experiments.main.compute_t_w_bigrams --profile qwen
python -m src.experiments.main.baseline_t_w_verification 1000 1 --profile qwen

# All completed main configurations; best and final epoch by default
sbatch scripts/launch_capella_qwen.sh src.experiments.main.similarity_eval \
  --profile qwen --metrics bm25 cosine bertscore bigram lcs --sweep --skip-existing
# Add --all-epochs for every saved epoch (consider separate jobs per configuration).

# Control similarity, compared to original abstracts
sbatch scripts/launch_capella_qwen.sh src.experiments.main.similarity_eval \
  --profile qwen --unwatermarked --n_samples 1000 --sample_type abstracts_only \
  --batch_size 32 --metrics bm25 cosine lcs --target original

# Optional: title perturbation similarity is model-independent; same inputs as Llama
sbatch scripts/launch_capella_qwen.sh src.experiments.main.perturbed_titles_similarity --profile qwen
```

The last command writes `results/perturbed_titles_similarity-qwen/`; recomputing
it is optional because the title texts did not change. Ablation experiments
remain legacy-only; this migration targets the three main experiments/controls.

## Qwen trained on the Llama-watermarked corpus

`qwen_on_llama_pipeline.py` is the cross-model control. It trains the Qwen
model with the same Qwen LoRA configuration but reads the existing
`data/t_ws/combined_t_ws.json` targets and Llama-tokenized prefix prompts. It
also verifies generated text with the Llama tokenizer and the historical
Waterfall 0.2.13 Fourier convention, while still running inside the Qwen
training environment.

Its outputs cannot collide with either main arm:

- adapters: `lora_adapters/qwen_on_llama/{sample_type}/{size}/{batch}/`
- results: `results/experimentN-qwen-on-llama/{prompt_type}/{size}/{batch}/{epoch}/`

Run a small diagnostic first:

```bash
python -m src.experiments.main.qwen_on_llama_pipeline \
  100 abstracts_only 32 100 --preflight

sbatch --job-name=qwen-llamawm-exp1-100 \
  scripts/launch_capella_qwen.sh src.experiments.main.qwen_on_llama_pipeline \
  100 abstracts_only 32 100
```

The module accepts the same `--smoke`, `--train-only`, `--eval-only`,
`--resume`, microbatch, and inference-batch options as the Qwen main arm. To
submit the complete 18-run matrix after the diagnostic succeeds:

```bash
bash scripts/submit_qwen_on_llama_experiments.sh capella
```

## Notebooks and checks

Use `src/eval/experiment{1,2,3}_eval_qwen.ipynb` and
`src/eval/unwatermarked_control_eval_qwen.ipynb` from `src/eval/`. Original
notebooks are unchanged. Qwen copies have cleared outputs, Qwen result/adapter
paths, 50000 as the largest size, and separate `thesis/figures/results/qwen/`
outputs. Register the separate kernel if using these on the HPC:

```bash
python -m ipykernel install --user --name watermark-qwen --display-name 'Qwen experiments'
python -m unittest discover -s tests -v
```

Compatibility checks performed locally: dependency resolution, all 18 path/config
combinations, mocked train/save/reload/evaluate/resume/control/open orchestration,
legacy Fourier parity, modern scores against Waterfall 0.3.4, tiny Qwen hybrid
forward/backward and greedy generation under Transformers 5.5.0, and the
BERTScore encoder API. **Full 9B Unsloth GPU training is not yet
validated here.** Preserve the smoke logs and `run_manifest.json` with results.
