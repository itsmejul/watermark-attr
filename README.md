### Waterfall-based LLM Source Attribution
This repository contains the code needed to reproduce the results of the thesis "From Memorization to Generalization: On the Limits of Text Watermarks for Source Attribution in Large Language Models" authored by Julian Mosig von Aehrenfeld.
Evaluation results and dataset are included in `/results` and `/data`, while the actual trained LoRA adapters are not saved due to their size.

## Setup
All experiments were conducted using Python 3.12.3 on a single Nvidia H100 GPU. The seeded dataset is saved using Git LFS:
```
git lfs install
git lfs pull
```
Required packages are pinned in `requirements.txt`:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In order to use the huggingface models that were used for the experiments, you need to request access to them on Huggingface and create an access token on your huggingface account, then run
`$ huggingface-cli login`
to login and paste your token there. 
Now, you will have access to all the models that your account has access to.

Also, an OpenAI API is needed in `.env`, see `.env.example` to produce the prompts for further experiments (current the prompts are present in `/data`).

## Data preparation
Download the unarXive open subset from https://zenodo.org/records/7752754 and
extract it to `data/unarxive_open/`.  

Then run the scripts in `src/data_creation/` in the following order (each
expects the repo root on PYTHONPATH, e.g.
`PYTHONPATH=. python src/data_creation/sample_dataset.py`):

`sample_dataset.py`: samples a seeded subset of the dataset and saves it to
`data/seeded_dataset.jsonl`.

`create_keys.py`: creates `data/keys.json` with the watermark key values
according to `data/watermark_config.json`.

`create_t_ws.py`: runs the watermarking.
Due to high computation costs, optionally pass a batch number to only watermark one block of `batch_size` samples per process (multiple runs are needed then).  
Pass nothing to watermark all samples at once. 
Each block is written to `data/t_ws/t_ws_batch_samples_<start>_to_<end>/`. 
Once all blocks are done, run it again with `--combine` to merge them into `data/t_ws/combined_t_ws.json`.

### Qwen 3.5 watermarking on HoreKa Green

The Qwen corpus is a separate experimental arm. It uses the unchanged
`data/keys.json` keys and watermark parameters, disables Qwen reasoning, and
writes only below `data/t_ws_qwen3_5_9b/`. The Llama environment and
`data/t_ws/` corpus remain untouched. The Qwen config explicitly retains the
old Transformer's implicit `top_k=50` in addition to temperature 0.5 and
top-p 0.9; these intentionally differ from Qwen's generic recommendations so
the model is the changed variable.

Create its separate environment from the repository root (the system Python
on Green is too old for current Waterfall):

```bash
uv python install 3.12
uv venv --python 3.12 .venv-qwen
uv pip install --python .venv-qwen/bin/python -r requirements-qwen-watermark.txt
source .venv-qwen/bin/activate
export HF_HOME=/hkfs/work/workspace/scratch/id_qry6439-watermark_paper/.cache/huggingface
mkdir -p "$HF_HOME"
hf download Qwen/Qwen3.5-9B
```

Qwen3.5-9B is public; the explicit download populates the same workspace cache
used by the launcher and avoids 13 jobs trying to download the model together.

Run a 50-text timing test in batch 1. Its output is a valid partial checkpoint,
so the later full batch-1 job resumes at text 51:

```bash
sbatch --partition=dev_accelerated --time=01:00:00 \
  scripts/launch_horeka_green.sh src.data_creation.create_t_ws 1 \
  --config data/watermark_config_qwen3_5_9b.json --limit 50
```

After checking the timing, submit the 13 corpus batches:

```bash
for batch in $(seq 1 13); do
  sbatch scripts/launch_horeka_green.sh src.data_creation.create_t_ws "$batch" \
    --config data/watermark_config_qwen3_5_9b.json
done
```

Jobs atomically checkpoint every 100 texts. Re-running the same command safely
resumes a partial batch. After every batch is complete, validate hashes,
ranges, and item counts and create the combined corpus:

```bash
.venv-qwen/bin/python -m src.data_creation.create_t_ws --combine \
  --config data/watermark_config_qwen3_5_9b.json
```

`create_prompt_dataset.py`: creates all prompt files in
`data/prompts` (prefixes, perturbed titles, and questions) for both the closed and the open-keyspace set. Needs `OPENAI_API_KEY`. Run with `--set closed`,
`--set open`, or no parameter for both.

## Ablations
Requires a GPU. 
Run with `-m`, e.g.
`PYTHONPATH=. python -m src.experiments.ablations.1_id_kp_ablation`.
Each script takes one optional `sub_experiment_name` argument; with no argument it runs
all of its default sub-experiments in sequence.

`1_id_kp_ablation.py [sub_experiment_name]` (`unique_ids`, `unique_kps`, default: both).

`2_watermarkfn_ablation.py [sub_experiment_name]` (`fourier`, `square`, default: both).

`3_ngram_ablation.py [sub_experiment_name]` (`one` .. `five` by default).

`4_kappa_ablation.py [sub_experiment_name]` (`two`, `four`, `six`, `eight`,
default: all).

## Main experiments
Require a GPU. Run with `-m`, e.g.
`PYTHONPATH=. python -m src.experiments.main.full_pipeline`.

`full_pipeline.py [n_samples] [sample_type] [batch_size] [n_eval_samples] [--eval-only]`:
trains a LoRA adapter and evaluates it at every saved epoch. `sample_type` is
one of `abstracts_only` (Experiment 1), `abstracts_and_titles` (Experiment 2), `questions` (Experiment 3). `--eval-only`
skips training and only runs generation + verification on an adapter already
saved under `lora_adapters/`.

For example, to reproduce a run of the thesis' first experiment at the largest sample size, run 
`python -m src.experiments.main.full_pipeline 63800 abstracts_only 32 1000`.
And for experiment 2 on the smallest sample size, run:
`python -m src.experiments.main.full_pipeline 100 abstracts_and_titles 32 1000`.

`full_pipeline_unwatermarked.py [n_samples] [sample_type] [batch_size] [n_eval_samples]`:
Same as above, trained on the original unwatermarked abstracts.

`open_keyspace_eval.py [n_samples] [sample_type] [batch_size] [n_open_samples]`:
evaluates a trained adapter against held-out negative samples for the open keyspace evaluation.

`similarity_eval.py --metrics METRIC [METRIC ...] [--n_samples N] [--sample_type TYPE] [--batch_size BS] [--sweep] [--all-epochs] [--unwatermarked]`:
Similarity/retrieval metrics on generated answers; `--metrics` takes one or
more of `bm25`, `cosine`, `bertscore`, `bigram`, `lcs`. `--sweep` runs
every `(sample_type, n_samples, batch_size)` combination in
`results/`; `--unwatermarked` runs only the bm25 control instead.

`baseline_t_w_verification.py [n_samples] [top_k] [t_ws_dir] [keys_file_name]`:
watermark detection on the watermarked texts themselves.

`compute_t_w_bigrams.py`, `count_t_w_lengths.py`, `perturbed_titles_similarity.py`:
Compute data used by `similarity_eval.py`'s bigram
metric, token-length statistics, and the perturbed-title similarity check.
