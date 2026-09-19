import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.fft import rfft

from src.util.experiment_profile import ExperimentProfile, QWEN_SIZES, SAMPLE_TYPES
from src.util.fourier_scores import fourier_scores
from src.util.checkpoints import latest_complete_checkpoint
from src.experiments.main import qwen_pipeline as pipeline


class ProfileTests(unittest.TestCase):
    def test_llama_defaults_unchanged(self):
        p = ExperimentProfile()
        self.assertEqual(p.subset_kwargs(), dict(texts_path="data/t_ws/combined_t_ws.json",
                         keys_path="data/keys.json", prompts_path="data/prompts/prefix_10.json"))
        self.assertEqual(p.adapter_root("questions"), ["lora_adapters", "questions"])
        self.assertEqual(p.experiment_dir("questions"), "experiment3")

    def test_all_18_qwen_configs_are_isolated(self):
        p = ExperimentProfile("qwen")
        roots = set()
        for sample_type in SAMPLE_TYPES:
            for n in QWEN_SIZES:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    pipeline.main("watermarked", [str(n), sample_type, "32", "--dry-run"])
                resolved = json.loads(output.getvalue())
                roots.add(resolved["adapters"])
                self.assertIn("-qwen", resolved["results"])
                self.assertIn("t_ws_qwen3_5_9b", resolved["subset"]["texts_path"])
                self.assertEqual(resolved["train"]["epochs"], 100)
                self.assertEqual(resolved["train"]["batch_size"], 32)
                expected_micro_batch = 32 if sample_type == "abstracts_only" else 16
                self.assertEqual(resolved["train"]["micro_batch_size"], expected_micro_batch)
                self.assertEqual(
                    resolved["train"]["batch_size"] // resolved["train"]["micro_batch_size"],
                    1 if sample_type == "abstracts_only" else 2,
                )
                self.assertEqual(resolved["train"]["eval_batch_size"], 32)
                self.assertEqual(resolved["train"]["inference_batch_size"], 64)
                self.assertIs(resolved["train"]["use_gradient_checkpointing"], False)
                self.assertEqual(resolved["train"]["n_samples"], n)
                self.assertFalse(resolved["generation"]["do_sample"])
                self.assertEqual(resolved["generation"]["temperature"], 0)
                self.assertEqual(resolved["generation"]["max_response_tokens"], 512)
                self.assertFalse(resolved["train"]["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(p.train_config(sample_type)["target_modules"],
                                 ExperimentProfile().train_config(sample_type)["target_modules"])
        self.assertEqual(len(roots), 18)

    def test_shared_prompts_and_tokenizer_specific_prefixes(self):
        p = ExperimentProfile("qwen")
        for filename in ("questions.json", "titles_1.json", "questions_open.json", "titles_3_open.json"):
            self.assertEqual(p.prompt_path(filename), f"data/prompts/{filename}")
        for filename in ("prefix_10.json", "prefix_10_open.json", "prefix_10_unwatermarked.json"):
            self.assertEqual(p.prompt_path(filename), f"data/prompts_qwen3_5_9b/{filename}")

    def test_control_and_smoke_paths(self):
        for mode in ("watermarked", "unwatermarked", "open"):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                pipeline.main(mode, ["1000", "questions", "32", "--smoke", "--dry-run"])
            config = json.loads(out.getvalue())
            self.assertIn("-smoke", config["results"])
            self.assertIn("/smoke/", config["adapters"])
            self.assertEqual(config["train"]["epochs"], 1)
            self.assertEqual(config["train"]["n_samples"], 100)
            if mode == "unwatermarked":
                self.assertIn("questions_unwatermarked", config["adapters"])

    def test_selection_and_epochs_match_legacy(self):
        from src.experiments.main.similarity_eval import _subset_dataset_indices, _get_eval_indices
        for n in QWEN_SIZES:
            self.assertEqual(pipeline.subset_indices(n), _subset_dataset_indices(n))
            self.assertTrue(set(pipeline.subset_indices(n)).isdisjoint(range(63800, 64000)))
            self.assertEqual(pipeline.eval_indices(n, min(n, 1000)), _get_eval_indices(min(n, 1000), n))
        self.assertEqual(pipeline.saved_epochs({"epochs": 100, "save_every_n_epochs": 5}), list(range(5, 101, 5)))
        self.assertEqual(pipeline.saved_epochs({"epochs": 1, "save_every_n_epochs": 5}), [1])

    def test_reasoning_disabled_for_training_and_inference(self):
        calls = []
        tok = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)))
        pipeline.format_chat(tok, "question", "answer")
        pipeline.format_chat(tok, "question")
        self.assertFalse(calls[0][1]["enable_thinking"])
        self.assertFalse(calls[0][1]["add_generation_prompt"])
        self.assertTrue(calls[1][1]["add_generation_prompt"])
        self.assertFalse(calls[1][1]["enable_thinking"])

    def test_similarity_profile_routing_and_restore(self):
        from src.experiments.main import similarity_eval as sim
        try:
            sim.configure_profile("qwen")
            self.assertEqual(sim.SAMPLE_SIZES, QWEN_SIZES)
            self.assertEqual(sim.EXPERIMENT_DIR["abstracts_only"], "experiment1-qwen")
            self.assertIn("t_ws_qwen3_5_9b", str(sim.BIGRAMS_NPZ))
            self.assertEqual(sim.TOKENIZER_NAME, "Qwen/Qwen3.5-9B")
        finally:
            sim.configure_profile("llama")
        self.assertEqual(sim.EXPERIMENT_DIR["abstracts_only"], "experiment1")

    def test_qwen_notebooks_isolated_and_cleared(self):
        root = Path(__file__).resolve().parents[1]
        notebooks = list((root / "src/eval").glob("*_qwen.ipynb"))
        self.assertEqual(len(notebooks), 4)
        for path in notebooks:
            nb = json.loads(path.read_text())
            source = "".join("".join(c["source"]) for c in nb["cells"])
            self.assertNotIn("63800", source)
            self.assertNotIn('Path("../../results/experiment1")', source)
            self.assertIn("/qwen/", source)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    compile("".join(cell["source"]), str(path), "exec")


class FourierTests(unittest.TestCase):
    def test_legacy_scores_unchanged(self):
        dense = np.random.default_rng(12).random((4, 16), dtype=np.float32)
        wf = SimpleNamespace(N=16, scaling_factor=1)
        f = rfft(dense, axis=-1)[:, 1:-1].astype(np.complex64)
        np.testing.assert_array_equal(fourier_scores(dense, wf), np.concatenate((f.real, f.imag), axis=1))

    def test_modern_scores_equal_direct_basis_even_and_odd(self):
        for n in (8, 9):
            max_freq = (n - 1) // 2
            dense = np.random.default_rng(12).random((4, n), dtype=np.float32)
            wf = SimpleNamespace(N=n, scaling_factor=1, num_fns=2 * max_freq)
            angles = 2 * np.pi * np.arange(n)[None, :] * np.arange(1, max_freq + 1)[:, None] / n
            expected = dense @ np.concatenate((np.cos(angles), np.sin(angles))).T
            np.testing.assert_allclose(fourier_scores(dense, wf), expected, atol=1e-6)


class CheckpointTests(unittest.TestCase):
    def test_resume_ignores_incomplete_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(latest_complete_checkpoint(tmp))
            old = Path(tmp) / "checkpoint-10"
            old.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
                (old / name).write_text("test")
            (old / "trainer_state.json").write_text('{"global_step": 10}')
            incomplete = Path(tmp) / "checkpoint-20"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text('{"global_step": 20}')
            self.assertEqual(latest_complete_checkpoint(tmp), str(old))

    def test_full_model_resume_ignores_incomplete_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "checkpoint-10"
            complete.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json",
                         "model.safetensors.index.json"):
                (complete / name).write_text("test")
            (complete / "trainer_state.json").write_text('{"global_step": 10}')
            incomplete = Path(tmp) / "checkpoint-20"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text('{"global_step": 20}')
            self.assertEqual(latest_complete_checkpoint(tmp, full_model=True), str(complete))


if __name__ == "__main__":
    unittest.main()
