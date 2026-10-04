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

`req-experiment.txt` records the intentional top-level dependencies;
`req-experiment.lock.txt` is the Linux x86-64/Python 3.12 resolution used for
reproducible installation. Regenerate it only after deliberately changing the
top-level file:

```bash
uv pip compile req-experiment.txt --python-version 3.12 -o req-experiment.lock.txt
```

## Which environment to use

Use `.venv-watermark` for:

- `src.data_creation.create_t_ws` and all watermark-corpus batches/combines
- `src.data_creation.create_prompt_dataset`
- `src.experiments.ablations.qwen_kappa_ablation`
- all `submit_qwen_*ablation.sh` and `submit_qwen_*corpus.sh` helpers

Use `.venv-experiment` for:

- all LoRA grids: Qwen-on-Qwen, Qwen-on-Llama, and Llama EOS-fix
- full-parameter/continued-pretraining runs
- closed/open-keyspace evaluation and additional similarity metrics
- the evaluation notebooks

Submission helpers default to their correct environment; `RUN_VENV_DIR` can
override that default. For a direct Slurm call through
`scripts/launch_capella_qwen.sh` or `scripts/launch_horeka_green.sh`, select it
explicitly:

```bash
export RUN_VENV_DIR="$PWD/.venv-watermark"   # corpus/prompt/watermark work
# or
export RUN_VENV_DIR="$PWD/.venv-experiment"  # training/evaluation work
```

Examples:

```bash
# Watermark corpus generation
export RUN_VENV_DIR="$PWD/.venv-watermark"
sbatch scripts/launch_capella_qwen.sh src.data_creation.create_t_ws 1 \
  --config data/t_ws/config_qwen.json

# Maintained Qwen-on-Qwen grid
export RUN_VENV_DIR="$PWD/.venv-experiment"
bash scripts/submit_qwen_experiments.sh capella --resume

# Maintained Llama grid
bash scripts/submit_llama_eosfix_experiments.sh capella

# One direct evaluation command
export RUN_VENV_DIR="$PWD/.venv-experiment"
sbatch scripts/launch_capella_qwen.sh src.experiments.main.similarity_eval \
  --profile qwen --metrics cosine lcs --sweep --skip-existing
```
