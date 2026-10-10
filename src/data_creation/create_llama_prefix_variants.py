"""Generate same-key Llama paraphrases using the existing Llama prefixes."""
from src.data_creation.create_qwen_prefix_variants import main

if __name__ == '__main__':
    main(source='llama')
