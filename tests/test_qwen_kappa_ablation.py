import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.ablations import qwen_kappa_ablation as ablation


class QwenKappaAblationTests(unittest.TestCase):
    def test_summary_metrics(self):
        verification = {
            "n_candidates": 3,
            "correct_k_ps": [11, 22, 33],
            "correct_ranks": [1, 2, 3],
            "correct_scores": [0.9, 0.7, 0.1],
            "top_k_p": [[11, 22, 33], [11, 22, 33], [11, 22, 33]],
            "top_scores": [[0.9, 0.5, 0.2], [0.8, 0.7, 0.1], [0.8, 0.5, 0.1]],
        }
        summary = ablation.summarize_verification(verification)
        self.assertEqual(summary["top1_accuracy"], 1 / 3)
        self.assertEqual(summary["top10_accuracy"], 1.0)
        self.assertAlmostEqual(summary["mean_reciprocal_rank"], (1 + 1 / 2 + 1 / 3) / 3)
        self.assertEqual(summary["median_rank"], 2)
        self.assertAlmostEqual(
            summary["mean_correct_vs_best_wrong_margin"],
            (0.4 - 0.1 - 0.7) / 3,
        )

    def test_condition_config_changes_only_ablation_fields(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "n_samples": 64000,
            "batch_size": 5000,
        }
        with patch.object(ablation, "_load_json", return_value=base):
            config = ablation._condition_config(14)
        self.assertEqual(config["kappa"], 14.0)
        self.assertEqual(config["n_samples"], 100)
        self.assertEqual(config["batch_size"], 100)
        self.assertEqual(config["generation_seed"], ablation.GENERATION_SEED)
        self.assertEqual(base["kappa"], 6.0)

    def test_write_or_validate_rejects_changed_condition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("condition")
            (root / relative).mkdir()
            (root / relative / "config.json").write_text(json.dumps({"kappa": 6}))
            with patch.object(ablation, "REPO_ROOT", root):
                with self.assertRaises(ValueError):
                    ablation._write_or_validate(relative, "config.json", {"kappa": 10})


if __name__ == "__main__":
    unittest.main()
