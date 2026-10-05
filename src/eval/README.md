# Evaluation notebooks

Run notebooks from this directory so their relative result and figure paths resolve correctly.

## Main experiment notebooks

For each of the three training/data combinations, there is one notebook per experiment:

- `experiment{1,2,3}_eval_llama.ipynb`: Llama trained on the Llama-watermarked corpus.
- `experiment{1,2,3}_eval_qwen.ipynb`: Qwen trained on the corrected Qwen-watermarked corpus.
- `experiment{1,2,3}_eval_qwen_on_llama.ipynb`: Qwen trained on the Llama-watermarked corpus.

Each notebook retains the main analysis plots and ends with two paper-ready tables:

- **Primary results:** closed-keyspace attribution accuracy (and MRR for Experiment 1).
- **Supplementary results:** BM25 controls, `CorrectQ`, margin, semantic similarity,
  BERTScore, Rouge-2/bigram overlap, normalized LCS/rank/length, and open-keyspace
  metrics where applicable.

Metrics that have not been run or synced are represented as `NA`; they do not stop
the notebook. Rerunning after the corresponding result files arrive fills the cells
automatically. Full-precision CSV tables are written under `tables/<model>/`, and
figures are written under `figures/<model>/<experiment>/`. The displayed and printed
LaTeX tables are rounded to four decimal places.

## Supporting notebooks

- `experiment1_model_comparison.ipynb`: epoch-100 top-1 comparison across all three
  model/data combinations and corpus sizes.
- `llama_unwatermarked_control_eval.ipynb`: Llama unwatermarked control trajectory.
- `watermark_ablations_eval.ipynb`: direct watermark-generation ablations, including
  the corrected Qwen length-limit/kappa sweep and available semantic-similarity data.

The supporting notebooks write their numeric tables to
`tables/{comparison,controls,ablations}/`. Supporting figures are written to the
corresponding directories below `figures/`.
