### WMCite: Waterfall-based LLM Source Attribution

## Setup
In order to use the huggingface models that were used for the experiments, you need to request access to them on Huggingface and create an access token on your huggingface account, then run
$ huggingface-cli login
to login and paste your token there. 
Now, you will have access to all the models that your account has access to.

Also, an OpenAI API is needed in `.env`, see `.env.example`.

## Data preparation
Download the unarXive open subset from https://zenodo.org/records/7752754 and
extract it to `data/unarxive_open/`.  

Then run the scripts in src/data_creation/ in the following order (each
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

`create_prompt_dataset.py`: creates all prompt files in
`data/prompts` (prefix_10, perturbed titles, and questions) for both the closed and the open-keyspace set. Needs OPENAI_API_KEY. Run with `--set closed`,
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
one of `abstracts_only`, `abstracts_and_titles`, `questions`. `--eval-only`
skips training and only runs generation + verification on an adapter already
saved under `lora_adapters/`.

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