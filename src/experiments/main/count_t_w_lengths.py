"""Count the token length of every watermarked text and save the list to
data/t_ws/token_lengths.json.

Usage:
    python -m src.experiments.main.count_t_w_lengths
"""

from transformers import AutoTokenizer

from src.util.filereader import load_path_file, write_path_file

T_ws = load_path_file(["data", "t_ws"], "combined_t_ws.json")
print(len(T_ws))

config = load_path_file(["data"], "watermark_config.json")
model_name = config["watermark_model"]
print(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

lengths = [len(tokenizer.encode(t_w)) for t_w in T_ws]
write_path_file(["data", "t_ws"], "token_lengths.json", lengths)
