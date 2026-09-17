# Qwen3.5-9B experiments

Use `--profile qwen` on the existing main experiment commands. Omitting it
retains the Llama arm; do not run Llama training/verification in the new environment.
Neither `.venv` nor the running watermark environment `.venv-qwen` is modified.

## What stays the same

- Six sizes: **100, 500, 1000, 5000, 10000, 50000** (the last Llama size was 63800).
- Three experiments: `abstracts_only`, `abstracts_and_titles`, `questions`.
- 100 epochs; adapters every 5 epochs; effective batch 32; learning rate 0.0002;
  constant schedule with 3% warmup; rank/alpha 16; no LoRA dropout.
- Same keys, seed-48 training subsets, held-out rows 63800–63999, seed-1234
  evaluation subsets (at most 1000), and open negatives at rows 64000–64999.
- Training truncation: 300 tokens for experiment 1; 512 for experiments 2/3.
- Greedy answer generation (`do_sample=false`, temperature 0, max 512 new tokens).
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
  tower. Initial A100 settings: microbatch 1, accumulation 32, eval batch 1,
  inference batch 4, Unsloth gradient checkpointing. Memory fit needs GPU testing.
  Adjust with `--micro-batch-size` / `--inference-batch-size` before production.
- The exact old target list `q_proj,k_proj,v_proj,o_proj` is retained. Qwen's
  linear-attention blocks use other names, so this targets its full-attention
  blocks only. It is **not equivalent layer coverage** to Llama. The run records
  trainable names/counts in `trainable_parameters.json`; broadening the list
  would be a separate experimental change.
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
