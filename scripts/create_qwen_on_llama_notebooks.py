"""Create analysis notebooks for Qwen trained on Llama-watermarked text.

The Qwen notebooks are the source because the trained model, corpus-size grid,
and epoch schedule are Qwen's. Only the corpus/result/adapter namespaces differ.
Generated notebooks have no stored outputs and never modify their sources.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def convert(notebook, experiment):
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Qwen experiments", "language": "python", "name": "watermark-qwen",
    }
    replacements = {
        f"../../results/experiment{experiment}-qwen":
            f"../../results/experiment{experiment}-qwen-on-llama",
        "../../lora_adapters/qwen/": "../../lora_adapters/qwen_on_llama/",
        "../../data/t_ws_qwen3_5_9b/": "../../data/t_ws/",
        "../../data/prompts_qwen3_5_9b/": "../../data/prompts/",
        "../../thesis/figures/results/qwen/":
            "../../thesis/figures/results/qwen_on_llama/",
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)
        cell.pop("attachments", None)
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    if experiment == 1:
        # The open-keyspace study is a fixed-N ablation, not a corpus-size
        # sweep. Keep the main SAMPLE_SIZES grid intact for every later plot.
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if "OPEN_RESULTS_ROOT" in source and "EPOCHS" in source:
                source = source.replace(
                    "SAMPLE_SIZES = [100, 500, 1000, 5000, 10000, 50000]",
                    "OPEN_SAMPLE_SIZE = 1000",
                )
            elif "best_epoch_open = {}" in source:
                source = source.replace("for sz in SAMPLE_SIZES:",
                                        "for sz in [OPEN_SAMPLE_SIZE]:")
            elif "metrics = {sz: compute_metrics(sz) for sz in SAMPLE_SIZES}" in source:
                source = source.replace(
                    "metrics = {sz: compute_metrics(sz) for sz in SAMPLE_SIZES}",
                    "metrics = {OPEN_SAMPLE_SIZE: compute_metrics(OPEN_SAMPLE_SIZE)}",
                ).replace("for sz in SAMPLE_SIZES:", "for sz in [OPEN_SAMPLE_SIZE]:")
            elif "aurocs    = [metrics[sz]" in source:
                source = '''m = metrics.get(OPEN_SAMPLE_SIZE)
if m is not None:
    labels = ["AUROC", "TPR@FPR=0.05", "TPR@FPR=0.01"]
    values = [m["auroc"], m["tpr@0.05"], m["tpr@0.01"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["red", "#377eb8", "#377eb8"])
    ax.set_ylabel("metric", fontsize=FONT_SIZE)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(f"Open-keyspace evaluation (N={OPEN_SAMPLE_SIZE:,}, epoch={m['epoch']})",
                 fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp1_open_keyspace_n1000.pdf", format="pdf")
    plt.show()
else:
    print(f"No open-keyspace result found for N={OPEN_SAMPLE_SIZE:,}.")
'''
            cell["source"] = source.splitlines(keepends=True)
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Experiment {experiment}: Qwen3.5-9B on Llama-watermarked text\n",
            "This cross-model control trains Qwen on the corpus watermarked by Llama and "
            "uses the Llama detector.\n",
            "Run from `src/eval/`. Results, adapters, and figures use dedicated "
            "`qwen-on-llama` / `qwen_on_llama` paths.\n",
            "The notebook expects open-keyspace results where plotted and the auxiliary "
            "BM25, cosine, BERTScore, bigram-overlap, and LCS files.\n",
        ],
    })
    return notebook


def main():
    for experiment in (1, 2, 3):
        source = ROOT / "src/eval" / f"experiment{experiment}_eval_qwen.ipynb"
        destination = source.with_name(f"experiment{experiment}_eval_qwen_on_llama.ipynb")
        notebook = convert(json.loads(source.read_text()), experiment)
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
