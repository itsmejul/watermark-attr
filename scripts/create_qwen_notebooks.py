"""Mechanically copy the Llama analysis notebooks for the isolated Qwen arm.

Outputs and execution counts are cleared; original notebooks are never edited.
Run from any directory with: python scripts/create_qwen_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def convert(notebook):
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Qwen experiments", "language": "python", "name": "watermark-qwen",
    }
    replacements = {
        "../../results/experiment1": "../../results/experiment1-qwen",
        "../../results/experiment2": "../../results/experiment2-qwen",
        "../../results/experiment3": "../../results/experiment3-qwen",
        "../../lora_adapters/": "../../lora_adapters/qwen/",
        "../../data/t_ws/": "../../data/t_ws_qwen3_5_9b/",
        "../../thesis/figures/results/": "../../thesis/figures/results/qwen/",
        "meta-llama/Llama-3.1-8B-Instruct": "Qwen/Qwen3.5-9B",
        "63800": "50000",
        "63,800": "50,000",
        "63.8k": "50k",
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)
        cell.pop("attachments", None)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["metadata"] = {}
    notebook["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {}, "source": [
            "# Qwen3.5-9B experiment arm\n",
            "Generated from the corresponding Llama notebook; all outputs are cleared.\n",
            "Run from `src/eval/`. Results and figures use Qwen-only directories.\n",
            "Six corpus sizes: 100, 500, 1000, 5000, 10000, 50000; 100 training epochs.\n",
            "This arm uses Waterfall 0.3.4 and a different tokenizer. Token-based metrics\n",
            "and LoRA layer coverage are not numerically identical to the Llama arm.\n",
        ],
    })
    return notebook


def main():
    for stem in ("experiment1_eval", "experiment2_eval", "experiment3_eval", "unwatermarked_control_eval"):
        source = ROOT / "src/eval" / f"{stem}.ipynb"
        destination = source.with_name(f"{stem}_qwen.ipynb")
        notebook = convert(json.loads(source.read_text()))
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
