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
