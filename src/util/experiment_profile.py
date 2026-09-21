"""Explicit experiment routing. Importable without any ML dependencies."""
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path

from src.util.filereader import REPO_ROOT

SAMPLE_TYPES = ("abstracts_only", "abstracts_and_titles", "questions")
QWEN_SIZES = (100, 500, 1000, 5000, 10000, 50000)
PROMPT_TYPES = {
    "abstracts_only": ("prefix_10",),
    "abstracts_and_titles": ("titles", "titles_1", "titles_2", "titles_3"),
    "questions": ("train_questions", "held_out_questions"),
}


@dataclass(frozen=True)
class ExperimentProfile:
    name: str = "llama"

    def __post_init__(self):
        if self.name not in ("llama", "qwen", "qwen_on_llama"):
            raise ValueError(f"Unknown profile: {self.name}")

    @property
    def is_qwen_trained(self):
        return self.name in ("qwen", "qwen_on_llama")

    @property
    def suffix(self):
        return {"llama": "", "qwen": "-qwen", "qwen_on_llama": "-qwen-on-llama"}[self.name]

    @property
    def corpus_dir(self):
        return "data/t_ws_qwen3_5_9b" if self.name == "qwen" else "data/t_ws"

    @property
    def prompts_dir(self):
        return "data/prompts_qwen3_5_9b" if self.name == "qwen" else "data/prompts"

    @property
    def watermark_config_path(self):
        return "data/watermark_config_qwen3_5_9b.json" if self.name == "qwen" else "data/watermark_config.json"

    @property
    def watermark_config(self):
        return json.loads((REPO_ROOT / self.watermark_config_path).read_text())

    @property
    def model(self):
        return self.watermark_config["watermark_model"]

    def prompt_path(self, filename):
        # Titles and questions are model-independent and intentionally shared.
        directory = self.prompts_dir if filename.startswith("prefix_") else "data/prompts"
        return f"{directory}/{filename}"

    def subset_kwargs(self, unwatermarked=False):
        filename = "prefix_10_unwatermarked.json" if unwatermarked else "prefix_10.json"
        return dict(texts_path=f"{self.corpus_dir}/combined_t_ws.json",
                    keys_path="data/keys.json", prompts_path=self.prompt_path(filename))

    def experiment_dir(self, sample_type):
        return f"experiment{SAMPLE_TYPES.index(sample_type) + 1}{self.suffix}"

    def adapter_root(self, sample_type, unwatermarked=False):
        name = sample_type + ("_unwatermarked" if unwatermarked else "")
        namespace = {"llama": [], "qwen": ["qwen"], "qwen_on_llama": ["qwen_on_llama"]}[self.name]
        return ["lora_adapters", *namespace, name]

    def train_config(self, sample_type):
        path = REPO_ROOT / "lora_adapters" / sample_type / "train_config.json"
        config = json.loads(path.read_text())
        if self.is_qwen_trained:
            overrides = json.loads((REPO_ROOT / "data/experiment_config_qwen.json").read_text())
            micro_batch_overrides = overrides.pop("micro_batch_size_overrides", {})
            config.update(overrides)
            config["micro_batch_size"] = micro_batch_overrides.get(
                sample_type, config["micro_batch_size"]
            )
        return config

    def check_waterfall(self):
        expected = "0.3.4" if self.is_qwen_trained else "0.2.13"
        actual = version("waterfall")
        if actual != expected:
            raise RuntimeError(f"{self.name} requires waterfall=={expected}; found {actual}. Use its separate environment.")


def add_profile_argument(parser):
    parser.add_argument("--profile", choices=("llama", "qwen", "qwen_on_llama"), default="llama")


def dispatch_profile(mode):
    """Route opt-in Qwen runs before legacy scripts import transformers.

    Keeping the existing Llama entrypoints intact also preserves their historical
    config merge, paths, and training behavior.
    """
    import argparse
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    add_profile_argument(parser)
    args, remaining = parser.parse_known_args()
    if args.profile in ("qwen", "qwen_on_llama"):
        from src.experiments.main.qwen_pipeline import main
        main(mode, remaining, watermark_source="llama" if args.profile == "qwen_on_llama" else "qwen")
        raise SystemExit(0)
    # Never silently re-score old corpora under the new Fourier convention.
    if not any(x in remaining for x in ("--help", "-h")):
        ExperimentProfile("llama").check_waterfall()
    sys.argv[1:] = remaining
