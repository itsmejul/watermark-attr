"""Create clean analysis notebooks for the isolated Llama EOS-fix arm."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def convert(notebook, experiment):
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Qwen experiments", "language": "python", "name": "watermark-qwen",
    }
    replacements = {
        f"../../results/experiment{experiment}":
            f"../../results/experiment{experiment}-llamaeosfix",
        "../../lora_adapters/": "../../lora_adapters/llamaeosfix/",
        "../../thesis/figures/results/":
            "../../thesis/figures/results/llamaeosfix/",
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
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    if experiment == 1:
        # No EOS-fix open-keyspace run exists. Remove that block while keeping
        # an empty mapping for the final summary table's optional AUROC rows.
        start = next(i for i, cell in enumerate(notebook["cells"])
                     if cell["cell_type"] == "markdown"
                     and "## Open Keyspace Evaluation" in "".join(cell["source"]))
        end = next(i for i in range(start + 1, len(notebook["cells"]))
                   if notebook["cells"][i]["cell_type"] == "markdown"
                   and "".join(notebook["cells"][i]["source"]).startswith("## "))
        notebook["cells"][start:end] = [
            {
                "cell_type": "markdown", "metadata": {},
                "source": [
                    "## Open Keyspace Evaluation\n",
                    "Not run for the EOS-fix arm. The closed-set and auxiliary analyses below "
                    "remain fully runnable.\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": ["metrics = {}  # optional open-keyspace metrics\n"],
            },
        ]

    notebook["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {}, "source": [
            f"# Experiment {experiment}: Llama 3.1-8B with EOS fix\n",
            "This isolated rerun uses terminal-EOS supervision, EOS-aware generation, the "
            "maintained training stack, and a maximum corpus size of 50,000.\n",
            "Run from `src/eval/`. Results, adapters, and figures use dedicated "
            "`llamaeosfix` paths.\n",
        ],
    })
    return notebook


def main():
    for experiment in (1, 2, 3):
        source = ROOT / "src/eval" / f"experiment{experiment}_eval.ipynb"
        destination = source.with_name(f"experiment{experiment}_eval_llama_eosfix.ipynb")
        notebook = convert(json.loads(source.read_text()), experiment)
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
