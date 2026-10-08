"""Detect directly on the final grids' watermarked evaluation abstracts.

No generation, adapters, or model weights are needed. Acc@1 uses the same
rank convention as the main grid (ties share rank); selected-top-1 accuracy
is also reported to make any ties visible.
"""
import argparse
import hashlib
import json
import random
from importlib.metadata import version

from src.util.filereader import (
    REPO_ROOT, HELDOUT_START, HELDOUT_END, write_path_file_atomic,
)

SIZES = (100, 500, 1000, 5000, 10000, 50000)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     sort_keys=True).encode()).hexdigest()


def select_indices(total, n):
    """Exactly match load_subset(seed=48) and grid evaluation seed=1234."""
    pool = [i for i in range(total) if not HELDOUT_START <= i < HELDOUT_END]
    if not 0 < n <= len(pool):
        raise ValueError(f'Invalid corpus size {n} for pool of {len(pool)}')
    training = sorted(random.Random(48).sample(pool, len(pool))[:n])
    evaluation = (sorted(random.Random(1234).sample(range(n), 1000))
                  if n > 1000 else list(range(n)))
    return training, [training[i] for i in evaluation]


def prepare(source, n):
    corpus = read(REPO_ROOT / f'data/t_ws/{source}/combined_t_ws.json')
    keys = read(REPO_ROOT / 'data/keys.json')
    config = read(REPO_ROOT / f'data/t_ws/config_{source}.json')
    if len(corpus) != 64000 or len(keys['ids']) != len(corpus) or len(keys['k_ps']) != len(corpus):
        raise ValueError('Expected aligned 64,000-document corpus and keys')
    if len(set(keys['k_ps'])) != len(keys['k_ps']) or len(set(keys['ids'])) != 1:
        raise ValueError('Expected globally unique keys and one shared Waterfall ID')
    training, evaluation = select_indices(len(corpus), n)
    candidates = [keys['k_ps'][i] for i in training]
    correct = [keys['k_ps'][i] for i in evaluation]
    texts = [corpus[i] for i in evaluation]
    if any(not isinstance(t, str) for t in texts):
        raise ValueError('Non-string watermarked evaluation text')
    # Fail closed if the expected reference is absent or differs. Also check
    # every available task/epoch, including Qwen-on-Llama for Llama sources.
    profiles = ('llama', 'qwen_on_llama') if source == 'llama' else ('qwen',)
    reference = REPO_ROOT / f'results/{source}/experiment1/prefix_10/{n}/32/100/verification_closed.json'
    if not reference.is_file():
        raise FileNotFoundError(f'Required grid alignment reference missing: {reference}')
    references = []
    for profile in profiles:
        for path in sorted((REPO_ROOT / 'results' / profile).glob(
                f'experiment*/*/{n}/32/*/verification_closed.json')):
            saved = read(path)
            if saved['correct_k_ps'] != correct or saved['n_candidates'] != n:
                raise ValueError(f'Evaluation sample/candidate mismatch: {path}')
            references.append(str(path.relative_to(REPO_ROOT)))
    manifest = dict(
        schema_version=1, source=source, n_candidates=n, n_evaluation=len(texts),
        corpus_path=f'data/t_ws/{source}/combined_t_ws.json',
        corpus_sha256=digest(corpus), keys_sha256=digest(keys), config=config,
        subset_seed=48, evaluation_seed=1234, legacy_fourier=(source == 'llama'),
        training_document_indices=training, evaluation_document_indices=evaluation,
        candidate_k_ps=candidates, correct_k_ps=correct,
        evaluation_texts_sha256=digest(texts),
    )
    return config, texts, keys['ids'][0], manifest, references


def summarize(result, texts=None):
    ranks = result['correct_ranks']
    n = len(ranks)
    if not n:
        raise ValueError('Empty verification')
    valid = [bool(t.strip()) for t in texts] if texts is not None else [True] * n
    if len(valid) != n:
        raise ValueError('Text/verification count mismatch')
    return dict(
        n_evaluation=n, n_candidates=result['n_candidates'],
        n_empty_texts=valid.count(False),
        acc_at_1=sum(ok and r == 1 for ok, r in zip(valid, ranks)) / n,
        acc_at_5=sum(ok and r <= 5 for ok, r in zip(valid, ranks)) / n,
        grid_rank_acc_at_1=sum(r == 1 for r in ranks) / n,
        selected_top1_accuracy=sum(ok and pred[0] == true for ok, pred, true in
                                  zip(valid, result['top_k_p'], result['correct_k_ps'])) / n,
        mean_margin=sum(score - (top[1] if rank == 1 else top[0])
                        for score, top, rank in zip(result['correct_scores'],
                                                   result['top_scores'], ranks)) / n,
        rank_convention='1 + number of candidate scores strictly greater than correct score',
        empty_text_policy='Count as incorrect; retain in denominator',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('llama', 'qwen'), required=True)
    parser.add_argument('--n-samples', type=int, choices=SIZES, required=True)
    parser.add_argument('--verify-batch-size', type=int, default=32)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.verify_batch_size < 1:
        parser.error('--verify-batch-size must be positive')
    config, texts, watermark_id, manifest, references = prepare(args.source, args.n_samples)
    print(f'Aligned {len(texts)} oracle texts against {args.n_samples} candidates; '
          f'checked {len(references)} saved grid evaluations.', flush=True)
    if args.preflight:
        return
    if version('waterfall') != '0.3.4':
        raise RuntimeError('Use the maintained environment with waterfall==0.3.4')
    from src.util.watermark import init_watermarker, verify_watermarks_full
    output = ['results', 'oracle_watermark', args.source, str(args.n_samples)]
    root = REPO_ROOT.joinpath(*output)
    # Lock prevents simultaneous submissions from mixing artifacts. Never
    # overwrite an output generated from different inputs.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / 'manifest.json'
        if not manifest_path.exists() and (root / 'verification_closed.json').exists():
            raise ValueError('Existing verification has no provenance manifest; refusing reuse')
        if manifest_path.exists() and read(manifest_path) != manifest:
            raise ValueError(f'Existing oracle inputs differ: {manifest_path}')
        write_path_file_atomic(output, 'manifest.json', manifest)
        write_path_file_atomic(output, 'alignment.json', dict(references=references))
        verification = root / 'verification_closed.json'
        if not verification.exists():
            _, _, watermarker = init_watermarker(config, load_model=False)
            verify_watermarks_full(
                texts, watermark_id, manifest['correct_k_ps'], watermarker, output,
                candidate_k_ps=manifest['candidate_k_ps'], top_k=100,
                batch_size=args.verify_batch_size, legacy_fourier=manifest['legacy_fourier'],
            )
        result = read(verification)
        if (result['correct_k_ps'] != manifest['correct_k_ps']
                or result['n_candidates'] != args.n_samples):
            raise ValueError('Saved oracle verification is misaligned')
        summary = summarize(result, texts)
        summary.update(source=args.source, waterfall_version=version('waterfall'),
                       transformers_version=version('transformers'))
        write_path_file_atomic(output, 'summary.json', summary)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
