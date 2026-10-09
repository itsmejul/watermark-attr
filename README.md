# Repository Structure
`data` contains the seeded, unwatermarked texts, as well as the llama-watermarked and qwen-watermarked texts alongside their prompts.
Generation parameters are in `data/t_ws/config_*.json`. The Llama parameters are defaults from the Waterfall paper, the Qwen parameters come from the ablations.

`lora_adapters/` contains runtime LoRA artifacts on HPC. Each saved epoch only needs its `adapter_config.json` and learned `adapter_model.safetensors`; it is gitignored except for `lora_adapters/configs/*.json`. Qwen may also keep up to two temporary `checkpoint-*` directories for interrupted-job recovery. 

`training_metadata/{trainmodel-on-watermarkmodel}/{task}/{corpus_size}/{batch_size}/` contains the loss trajectories for each run. 

`results/` has the structure: `results/{trainmodel-on-watermarkmodel}/{experiment}/{prompt_type}/{corpus_size}/{batch_size}/{epoch}/verification_closed.json`. Main results are in `verification_closed.json`, saved every five epochs. It mainly stores the correct keys for each prompt, the top predicted keys, and the q-score it gave for each of them. Based on these files we can compute accuracy and so on.

`scripts/` just the HPC runner scripts.

`src/eval/` the main eval notebooks, these contain our results in visual and table form.


# Shared-prefix Qwen watermark augmentation (generation only)

Generate four additional paraphrases for the canonical N=1,000 training subset:

```bash
bash scripts/submit/submit_qwen_prefix_variants.sh capella
```

This submits four independent jobs, versions 2--5, using `.venv-watermark`
(or the legacy `.venv-qwen` if the renamed environment is absent).
`RUN_VENV_DIR` can override this. Use the **generation** environment from
`req-watermark.txt`, not the experiment environment.

Each document keeps its existing key and exact saved ten-token evaluation
prefix. The original unwatermarked abstract remains the user input to the same
paraphrasing prompt; the saved prefix is appended to the assistant prefill.
Sampling is unchanged (kappa 4, temperature 1, top-p 1, top-k 0, thinking off).
Versions V2--V5 use base seeds 201, 202, 203, and 204, respectively.
Document index i uses base_seed + i, saved in the manifest, for reproducible
resumption. Seeds differ across versions for each document, but ranges overlap
across different documents.
The ten supplied tokens count against the 1.5-times-source-token generation
budget. The first continuation token uses the prefix as Waterfall bigram
context. Raw returned tokens preserve whitespace at the prefix boundary.

Outputs are isolated under `data/t_ws/qwen_prefix_augmentation/1000/version_{2,3,4,5}/`.
Each stores complete `watermarked_texts.json` (prefix included), aligned
`inputs.json`, `manifest.json`, and `progress.json`. Checkpoints are written
after every document, and repeat submissions resume. Inputs are fingerprinted;
mismatching existing outputs are never silently overwritten. Empty or
prefix-changing continuations fail visibly rather than being silently accepted.
Canonical corpora, prompts, adapters, and results are not changed.

After all four jobs complete, combine on the login node (no model/GPU needed):

```bash
python -m src.data_creation.create_qwen_prefix_variants --combine
```

This writes `data/t_ws/qwen_prefix_augmentation/1000/combined_versions.json`,
one record per document with original text, key, prefix, and versions V1--V5.
V1 is copied from the existing corpus. `summary.json` reports completeness and
exact duplicate counts. No training is submitted and no detection-success
filtering is performed. The Python entry point supports other `--n-samples`
values when a corresponding canonical closed-grid reference exists.

Optional one-document GPU smoke test before submitting all versions:

```bash
RUN_VENV_DIR="$PWD/.venv-qwen" sbatch --partition=capella \
  --job-name=qwen-prefix-smoke scripts/launch_capella.sh \
  src.data_creation.create_qwen_prefix_variants --version 2 --limit 1
```

# Direct-watermark oracle baseline

From the repository root on Capella, submit the twelve independent baseline jobs:

```bash
bash scripts/submit/submit_oracle_watermark_evals.sh capella
```

This uses `.venv-experiment` (override with `RUN_VENV_DIR`) and requires access
to the Llama tokenizer through the existing Hugging Face cache or authentication.
It loads no model weights, performs no training or generation, and uses the
same Waterfall verification as the main grids. There is one job per watermark
source (`llama`, `qwen`) and corpus size (100, 500, 1000, 5000, 10000, 50000).
Each evaluates the same min(N, 1000) documents as the grid against all N
training keys. Llama uses legacy Fourier scoring; Qwen uses current scoring.
The Llama baseline also applies to Qwen-on-Llama.

Before submission, all sample selections are checked against saved grid
verifications; the epoch-100 recitation reference must exist for each size.
Results are isolated under `results/oracle_watermark/<source>/<N>/`:
`verification_closed.json`, `summary.json` (Acc@1, Acc@5, margin and explicit
selected-top-1 accuracy), `manifest.json` (document indices, keys and input
hashes), `alignment.json`, and `latency.json`. Acc@1 follows the main grid's
rank convention, where tied maxima share rank 1; selected-top-1 accuracy
also checks the actual returned key. Empty source texts count as failures
and stay in the denominator; `grid_rank_acc_at_1` additionally reports the
unadjusted grid convention (which counts all-zero ties as rank 1).
Reruns reuse completed verification only
when the manifest matches. No main-grid results are modified.

Single-job example:

```bash
RUN_VENV_DIR="$PWD/.venv-experiment" sbatch --partition=capella --time=02:00:00 \
  --job-name=oracle-qwen-1000 scripts/launch_capella.sh \
  src.experiments.main.oracle_watermark_eval --source qwen --n-samples 1000
```

This is an oracle-text attribution baseline, not a binary watermark-presence
test or a guaranteed mathematical upper bound. It evaluates complete stored
watermarked abstracts without training truncation or prompt removal.

# Evaluation notebooks
Should be run in their directory.

## Main experiment notebooks
We evaluate three tasks (experiment 1-3) and three model combinations:
- `experiment{1,2,3}_eval_llama.ipynb`: Llama trained on the Llama-watermarked corpus.
- `experiment{1,2,3}_eval_qwen.ipynb`: Qwen trained on the Qwen-watermarked corpus.
- `experiment{1,2,3}_eval_qwen_on_llama.ipynb`: Qwen trained on the Llama-watermarked corpus.

Each notebook creates plots and saves them to `src/eval/figures`. They also create tables and save them to `src/eval/tables`.
They report primary metrics (closed-keyspace attribution accuracy), and supplementary metrics (BM25, correctQ, Margin, SemSim, BERTScore, Rouge-2, NormLCS) and potentially also additional control experiments like open-keyspace evaluation.

## Supporting notebooks
- `experiment1_model_comparison.ipynb`: epoch-100 top-1 comparison across all three
  model/data combinations and corpus sizes.
- `llama_unwatermarked_control_eval.ipynb`: Llama unwatermarked control trajectory.
- `watermark_ablations_eval.ipynb`: direct watermark-generation ablations, used for motivating hyperparameter choices..

# Create environments
Use Python 3.12 and create both environments from the repository root.
On HPC clusters, load the modules first, for example like so:

```bash
module purge
module load release/24.04 GCCcore/13.3.0 Python/3.12.3
```
Due to incompatible transformer version requirements between Unsloth and Waterfall,
two separate environments are need.
Create the watermarking environment:

```bash
python -m venv .venv-watermark
.venv-watermark/bin/python -m pip install --upgrade pip
.venv-watermark/bin/python -m pip install -r req-watermark.txt
.venv-watermark/bin/python -m pip check
```

Create the training/evaluation environment:

```bash
python -m venv .venv-experiment
.venv-experiment/bin/python -m pip install --upgrade pip
.venv-experiment/bin/python -m pip install -r req-experiment.lock.txt
.venv-experiment/bin/python -m pip check
.venv-experiment/bin/python -m ipykernel install --user \
  --name watermark-experiment --display-name ".venv-experiment"
```

## Which environment to use
Use `.venv-watermark` for:

- `src.data_creation.create_t_ws` and all watermark-corpus batches/combines
- `src.data_creation.create_prompt_dataset`
- `src.experiments.ablations.qwen_kappa_ablation`
- all `submit_qwen_*ablation.sh` and `submit_qwen_*corpus.sh` helpers

Use `.venv-experiment` for:

- all main LoRA experiments with fine-tuning and evaluation
- full-parameter/continued-pretraining runs
- closed/open-keyspace evaluation and additional similarity metrics
- the evaluation notebooks
