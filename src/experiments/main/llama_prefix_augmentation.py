"""Isolated M=3/5 Llama recitation, matching the canonical EOS-fixed baseline."""
from src.experiments.main.qwen_pipeline import main

if __name__ == '__main__':
    main(watermark_source='llama', training_source='llama')
