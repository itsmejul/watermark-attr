"""Exercise orchestration/resume without GPU dependencies or real result writes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.experiments.main import qwen_pipeline as pipeline
from src.util.experiment_profile import ExperimentProfile, PROMPT_TYPES


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.calls = {"train": [], "ask": [], "verify": []}
        self.write(["data/t_ws_qwen3_5_9b"], "combined_manifest.json", {
            "combined_count": 64000, "config_sha256": "hash", "keys_sha256": "hash",
            "watermark_model": "Qwen/Qwen3.5-9B",
        })
        self.write(["data/prompts_qwen3_5_9b"], "prefix_10_unwatermarked.json", ["prefix"] * 64000)
        for p in ("prefix_10", "titles_1", "titles_2", "titles_3", "questions"):
            directory = "data/prompts_qwen3_5_9b" if p == "prefix_10" else "data/prompts"
            self.write([directory], f"{p}_open.json", [["q1", "q2"] if p == "questions" else "prompt"] * 1000)

        def read(parts, filename):
            if filename == "generation_config.json":
                return {"do_sample": False, "temperature": 0, "top_p": .9, "max_response_tokens": 512}
            return json.loads(self.root.joinpath(*parts, filename).read_text())

        def subset(n, **kwargs):
            self.assertIn("t_ws_qwen3_5_9b", kwargs["texts_path"])
            self.assertIn("prompts_qwen3_5_9b", kwargs["prompts_path"])
            return self.subset(n)

        def train(texts, heldout, config, path, **kwargs):
            self.calls["train"].append((texts, config, kwargs))
            for epoch in pipeline.saved_epochs(config):
                self.write(path + [str(epoch), "lora_adapter"], "adapter_config.json", {})

        def ask(prompts, config, path, adapter, save_file_name, **kwargs):
            self.calls["ask"].append((path, adapter, prompts))
            self.write(path, save_file_name, ["answer"] * len(prompts))

        def verify(answers, ids, keys, watermarker, path, **kwargs):
            self.calls["verify"].append((path, keys, kwargs))
            self.write(path, "verification_closed.json", {"correct_ranks": [1] * len(answers)})

        def verify_open(answers, ids, watermarker, path, **kwargs):
            self.calls["verify"].append((path, None, kwargs))
            self.write(path, "verification_open.json", {"n_candidates": len(kwargs["candidate_k_ps"])})

        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: json.dumps(messages))
        modules = {
            "unsloth": SimpleNamespace(FastLanguageModel=object),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer)),
            "src.util.finetune": SimpleNamespace(finetune=train),
            "src.util.llm": SimpleNamespace(ask_batched=ask),
            "src.util.watermark": SimpleNamespace(init_watermarker=lambda *a, **k: (None, None, object()),
                                                 verify_watermarks_full=verify,
                                                 verify_watermarks_open_keyspace=verify_open),
        }
        self.stack.enter_context(patch.dict("sys.modules", modules))
        self.stack.enter_context(patch.object(ExperimentProfile, "check_waterfall"))
        for name, value in {
            "REPO_ROOT": self.root, "write_path_file_atomic": self.write, "load_path_file": read,
            "sha256": lambda p: "hash", "load_subset": subset,
            "load_held_out_set": lambda **k: self.subset(2, offset=63800),
            "load_open_keyspace_set": lambda n: {"abstracts": ["abstract"] * n, "titles": ["title"] * n},
            "load_abstracts": lambda n: [f"original {i}" for i in range(n)],
            "version": lambda p: "5.5.0" if p == "transformers" else "test",
        }.items():
            self.stack.enter_context(patch.object(pipeline, name, value))

    def write(self, parts, filename, value):
        path = self.root.joinpath(*parts, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def subset(n, offset=0):
        result = {"T_ws": [f"watermarked {i}" for i in range(n)], "ids": [1] * n,
                  "k_ps": list(range(1 + offset, n + 1 + offset))}
        for field in ("titles", "titles_1", "titles_2", "titles_3", "train_questions", "held_out_questions", "prefix_10"):
            result[field] = [f"{field} {i}" for i in range(n)]
        return result

    def run_pipeline(self, mode="watermarked", sample_type="abstracts_only", extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(mode, ["10", sample_type, "32", "5", "--smoke", *extra])

    def test_train_resume_and_open_roundtrip_all_experiments(self):
        for sample_type, prompts in PROMPT_TYPES.items():
            with self.subTest(sample_type=sample_type):
                self.run_pipeline(sample_type=sample_type)
                n_train, n_ask = len(self.calls["train"]), len(self.calls["ask"])
                self.run_pipeline(sample_type=sample_type, extra=["--resume"])
                self.assertEqual(len(self.calls["train"]), n_train)
                self.assertEqual(len(self.calls["ask"]), n_ask)
                self.run_pipeline("open", sample_type)
                self.assertEqual(len(self.calls["ask"]), n_ask + len(prompts))
                for path, adapter, _ in self.calls["ask"]:
                    self.assertIn("-qwen-smoke", path[1])
                    self.assertEqual(adapter[:3], ["lora_adapters", "qwen", "smoke"])

    def test_control_uses_original_training_texts_and_separate_paths(self):
        self.run_pipeline("unwatermarked")
        texts = self.calls["train"][0][0]
        self.assertEqual(texts, [f"original {i}" for i in pipeline.subset_indices(10)])
        path, adapter, _ = self.calls["ask"][0]
        self.assertIn("abstracts_only_unwatermarked", adapter)
        self.assertEqual(path[2], "prefix_10_unwatermarked")
        self.run_pipeline("open", extra=["--unwatermarked"])

    def test_refuses_incompatible_manifest_and_accidental_retraining(self):
        self.run_pipeline()
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        with self.assertRaises(ValueError):
            self.run_pipeline(extra=["--micro-batch-size", "2", "--resume"])

    def test_missing_adapter_and_missing_input_fail(self):
        with self.assertRaises(FileNotFoundError):
            self.run_pipeline(extra=["--eval-only"])
        (self.root / "data/t_ws_qwen3_5_9b/combined_manifest.json").unlink()
        with self.assertRaises(ValueError):
            self.run_pipeline(extra=["--preflight"])


if __name__ == "__main__":
    unittest.main()
