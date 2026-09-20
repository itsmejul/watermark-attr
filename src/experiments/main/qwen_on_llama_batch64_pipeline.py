"""Effective-batch-64 Qwen training on the Llama-watermarked corpus.

This is an isolated optimization ablation.  It retains the established
microbatches (32 for raw abstracts, 16 for chat-formatted tasks), so an
effective batch of 64 gives gradient accumulation of 2 and 4 respectively.
Adapters and evaluations use dedicated ``qwen_on_llama_batch64`` and
``qwen-on-llama-batch64`` roots.
"""

from src.experiments.main.qwen_pipeline import main as _qwen_main


def main(argv=None):
    return _qwen_main(
        "watermarked",
        argv,
        watermark_source="llama",
        experiment_variant="batch64",
    )


if __name__ == "__main__":
    main()
