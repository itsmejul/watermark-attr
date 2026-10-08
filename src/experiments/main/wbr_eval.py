"""Backfill WBR beside every existing bigrams.json in the three final grids.

CPU-only: python -m src.experiments.main.wbr_eval [--qwen-tokenizer-json PATH]
Reuses saved correct-key alignment and checks recomputed bigram overlap against
the existing metric before writing anything for a checkpoint.
"""
import argparse
import json
from pathlib import Path

from src.util.filereader import REPO_ROOT, load_abstracts, write_path_file_atomic
from src.util.wbr import bigram_set, wbr_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-tokenizer-json', type=Path)
    args = parser.parse_args()
    keys = json.loads((REPO_ROOT / 'data/keys.json').read_text())['k_ps']
    if len(set(keys)) != len(keys):
        raise ValueError('Backfill requires globally unique document keys')
    by_key = {k: i for i, k in enumerate(keys)}
    originals = load_abstracts(64000)
    tokenizers = {}
    target_cache = {}
    for profile in ('llama', 'qwen', 'qwen_on_llama'):
        source = 'qwen' if profile == 'qwen' else 'llama'
        watermarked = json.loads((REPO_ROOT / f'data/t_ws/{source}/combined_t_ws.json').read_text())
        paths = sorted((REPO_ROOT / 'results' / profile).glob('experiment*/**/bigrams.json'))
        for counter, path in enumerate(paths, 1):
            big = json.loads(path.read_text())
            name = big['tokenizer']
            if name not in tokenizers:
                if source == 'qwen' and args.qwen_tokenizer_json:
                    from tokenizers import Tokenizer
                    raw = Tokenizer.from_file(str(args.qwen_tokenizer_json))
                    tokenizers[name] = lambda texts, raw=raw: [
                        e.ids for e in raw.encode_batch(texts, add_special_tokens=False)]
                else:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(name)
                    tokenizers[name] = lambda texts, tok=tok: tok(
                        texts, add_special_tokens=False)['input_ids']
            encode = tokenizers[name]
            kp = big['correct_k_ps']
            answers = json.loads(path.with_name('answers.json').read_text())
            if len(answers) != len(kp):
                raise ValueError(f'Misaligned answers: {path}')
            missing = [k for k in kp if (source, name, k) not in target_cache]
            wm_ids = encode([watermarked[by_key[k]] for k in missing]) if missing else []
            org_ids = encode([originals[by_key[k]] for k in missing]) if missing else []
            for k, w, o in zip(missing, wm_ids, org_ids):
                wb = bigram_set(w)
                target_cache[source, name, k] = (wb, wb - bigram_set(o))
            answer_ids = encode(answers)
            eligible = []
            for i, (k, a) in enumerate(zip(kp, answer_ids)):
                wb, carrier = target_cache[source, name, k]
                if (len(wb) != big['n_bigrams_target_unique'][i]
                        or len(wb & bigram_set(a)) != big['overlap_unique'][i]):
                    raise ValueError(f'Tokenization/alignment differs from saved bigram metric: {path}, row {i}')
                eligible.append(carrier)
            result = wbr_result(eligible, answer_ids, kp, name)
            result.update(watermarked_source=f'data/t_ws/{source}/combined_t_ws.json',
                          original_source='data/seeded_dataset.jsonl')
            write_path_file_atomic(list(path.parent.relative_to(REPO_ROOT).parts), 'wbr.json', result)
            print(f'{profile} {counter}/{len(paths)}: {path.parent.relative_to(REPO_ROOT)} WBR={result["mean_wbr"]}', flush=True)


if __name__ == '__main__':
    main()
