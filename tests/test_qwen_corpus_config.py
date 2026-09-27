import json
from pathlib import Path
import unittest


class QwenCorpusConfigTests(unittest.TestCase):
    def test_kappa4_lengthfix_corpus_is_complete_and_isolated(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads(
            (
                repo
                / "data/watermark_config_qwen3_5_9b_kappa4_lengthfix.json"
            ).read_text()
        )
        self.assertEqual(config["n_samples"], 64000)
        self.assertEqual(config["batch_size"], 5000)
        self.assertEqual(config["kappa"], 4.0)
        self.assertEqual(config["temperature_watermark"], 1.0)
        self.assertEqual(config["top_p_watermark"], 1.0)
        self.assertEqual(config["top_k_watermark"], 0)
        self.assertEqual(config["max_new_tokens_ratio_watermark"], 1.5)
        self.assertFalse(config["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(config["keys_file"], "data/keys.json")
        self.assertEqual(
            config["output_dir"],
            "data/t_ws_qwen3_5_9b_kappa4_lengthfix",
        )
        self.assertNotEqual(config["output_dir"], "data/t_ws_qwen3_5_9b")


if __name__ == "__main__":
    unittest.main()
