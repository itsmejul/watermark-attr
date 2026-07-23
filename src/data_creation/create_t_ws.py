"""Watermark the original abstracts (T_o -> T_w) in batches, writes each batch to
data/t_ws/t_ws_batch_samples_<start>_to_<end>/.

Watermarking all n_samples at once is expensive, so we split it into
batches of config["batch_size"] that are run in parallel. 
Pass a batch number to process just that batch.
Passing nothing will process everything in one run. 
Afterwards, combine the batches into data/t_ws/combined_t_ws.json using --combine.

usage:
    PYTHONPATH=. python src/data_creation/create_t_ws.py 3    # 3rd batch only
    PYTHONPATH=. python src/data_creation/create_t_ws.py      # all samples
    PYTHONPATH=. python src/data_creation/create_t_ws.py --combine
"""
import argparse
import json
import math
import os
import re
import sys

sys.stdout.reconfigure(line_buffering=True)

T_WS_DIR = "data/t_ws"
COMBINED_FILE = "data/t_ws/combined_t_ws.json"


def combine():
    """Concatenate all batch outputs into COMBINED_FILE, ordered by start index."""
    def get_start_idx(dirname):
        match = re.search(r"t_ws_batch_samples_(\d+)_to_\d+", dirname)
        return int(match.group(1)) if match else -1

    subdirs = sorted(
        [d for d in os.listdir(T_WS_DIR) if os.path.isdir(os.path.join(T_WS_DIR, d))],
        key=get_start_idx,
    )

    combined = []
    for subdir in subdirs:
        filepath = os.path.join(T_WS_DIR, subdir, "watermarked_texts.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            combined.extend(data)
            print(f"Added {len(data)} items from {subdir}")
        else:
            print(f"Warning: No watermarked_texts.json found in {subdir}")

    with open(COMBINED_FILE, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nDone. {len(combined)} total items saved to {COMBINED_FILE}")


def create(batch_num):
    from src.util.filereader import load_abstracts, load_path_file
    from src.util.watermark import watermark

    config = load_path_file(["data"], "watermark_config.json")
    n_samples = config["n_samples"]
    batch_size = config["batch_size"]

    keys = load_path_file(["data"], "keys.json")
    T_os = load_abstracts(n_samples)
    ids = keys["ids"]
    k_ps = keys["k_ps"]

    n_possible_batches = math.ceil(n_samples / batch_size)
    if batch_num == -1:
        start_idx, end_idx = 0, n_samples
    elif 1 <= batch_num <= n_possible_batches:
        start_idx = (batch_num - 1) * batch_size
        end_idx = batch_num * batch_size
        T_os = T_os[start_idx:end_idx]
        ids = ids[start_idx:end_idx]
        k_ps = k_ps[start_idx:end_idx]
    else:
        raise ValueError(
            f"batch_num must be -1 or in [1, {n_possible_batches}] "
            f"(batch_size={batch_size}, n_samples={n_samples}); got {batch_num}"
        )

    print(f"start_idx {start_idx}, end_idx {end_idx}, num texts: {len(T_os)}")

    batch_dir_name = f"t_ws_batch_samples_{start_idx}_to_{end_idx - 1}"
    watermark(
        T_os,
        ids,
        k_ps,
        config,
        experiment_path=["data", "t_ws", batch_dir_name],
        return_sts_scores=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "batch_num",
        nargs="?",
        type=int,
        default=-1,
        help="1-indexed batch of batch_size samples; -1 (default) processes all.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="merge all finished batches into combined_t_ws.json instead of watermarking",
    )
    args = parser.parse_args()

    if args.combine:
        combine()
    else:
        create(args.batch_num)


if __name__ == "__main__":
    main()
