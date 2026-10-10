"""Four additional same-key, shared-prefix paraphrases for a closed-grid subset.

Each version is an independent resumable job. --combine validates all versions
and writes document-major records; it never modifies the canonical corpus.
"""
import argparse
import fcntl
import json
from importlib.metadata import version

from src.experiments.main.oracle_watermark_eval import digest, read, select_indices
from src.util.filereader import REPO_ROOT, load_abstracts, write_path_file_atomic


def inputs(n, source='qwen'):
    if source not in ('qwen', 'llama'):
        raise ValueError(f'Unsupported watermark source: {source}')
    config = read(REPO_ROOT / f'data/t_ws/config_{source}.json')
    corpus = read(REPO_ROOT / f'data/t_ws/{source}/combined_t_ws.json')
    keys = read(REPO_ROOT / 'data/keys.json')
    prompts = read(REPO_ROOT / f'data/prompts/{source}/prefix_10.json')
    originals = load_abstracts(64000)
    if not len(corpus) == len(keys['ids']) == len(keys['k_ps']) == len(prompts) == len(originals) == 64000:
        raise ValueError('Canonical inputs must contain 64,000 aligned documents')
    train, evaluation = select_indices(len(corpus), n)
    if len(set(keys['ids'])) != 1 or len(set(keys['k_ps'])) != len(corpus):
        raise ValueError('Expected one Waterfall ID and unique document keys')
    reference = read(REPO_ROOT / f'results/{source}/experiment1/prefix_10/{n}/32/100/verification_closed.json')
    if reference['correct_k_ps'] != [keys['k_ps'][i] for i in evaluation] or reference['n_candidates'] != n:
        raise ValueError('Subset differs from the existing closed evaluation')
    rows = [dict(document_index=i, watermark_id=keys['ids'][i], k_p=keys['k_ps'][i],
                 original=originals[i], prefix=prompts[i], version_1=corpus[i]) for i in train]
    for row in rows:
        if not row['prefix'] or not row['version_1'].startswith(row['prefix']):
            raise ValueError(f'Invalid saved prefix for document {row["document_index"]}')
    return config, rows


def variant_manifest(config, rows, v, source='qwen'):
    # User-selected bases. Per-document reseeding preserves resumability.
    # Ranges overlap across documents, but each document has four distinct seeds.
    seed = {2: 201, 3: 202, 4: 203, 5: 204}[v]
    manifest = dict(schema_version=1, n_documents=len(rows), version=v,
                source=source, subset_seed=48, prefix_tokens=10,
                config=config, inputs_sha256=digest(rows),
                document_indices=[r['document_index'] for r in rows],
                generation_seed_base=seed,
                generation_seeds=[seed + i for i in range(len(rows))],
                seed_rule='generation_seed_base + index in sorted selected subset',
                length_policy='ceil(1.5 * original token count) minus 10 new tokens',
                prefix_context_watermark=True)
    if source == 'llama':
        manifest.update(
            length_policy='legacy prompt-character budget minus supplied prefix content tokens',
            prefix_policy='reuse saved 10-token prefix including BOS; 9 visible text tokens',
            fourier_policy='legacy scoring; require cosine-only keys compatible with Waterfall 0.3.4',
            generation_environment=dict(waterfall='0.3.4', transformers='5.17.0'),
        )
    return manifest


def check_texts(texts, rows, complete=False):
    if not isinstance(texts, list) or len(texts) > len(rows) or (complete and len(texts) != len(rows)):
        raise ValueError('Incorrect variant document count')
    for text, row in zip(texts, rows):
        if not isinstance(text, str) or not text.startswith(row['prefix']) or not text[len(row['prefix']):].strip():
            raise ValueError(f'Invalid continuation for document {row["document_index"]}')


def main(source='qwen'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=1000)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--version', type=int, choices=(2, 3, 4, 5))
    mode.add_argument('--combine', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--limit', type=int, help='Maximum NEW documents this invocation (smoke test)')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    config, rows = inputs(args.n_samples, source)
    print(f'Aligned {len(rows)} documents with the canonical {source} grid.', flush=True)
    if args.preflight:
        return
    base = ['data', 't_ws', f'{source}_prefix_augmentation', str(args.n_samples)]
    if args.combine:
        variants = [[r['version_1'] for r in rows]]
        for v in (2, 3, 4, 5):
            root = REPO_ROOT.joinpath(*base, f'version_{v}')
            if read(root / 'manifest.json') != variant_manifest(config, rows, v, source):
                raise ValueError(f'Input manifest differs for version {v}')
            texts = read(root / 'watermarked_texts.json')
            check_texts(texts, rows, complete=True)
            variants.append(texts)
        records = [dict(row, versions=[texts[i] for texts in variants]) for i, row in enumerate(rows)]
        write_path_file_atomic(base, 'combined_versions.json', records)
        unique = [len(set(r['versions'])) for r in records]
        write_path_file_atomic(base, 'summary.json', dict(
            n_documents=len(rows), n_versions=5, n_new_texts=4 * len(rows),
            complete=True, inputs_sha256=digest(rows),
            documents_with_duplicate_versions=sum(n < 5 for n in unique),
            mean_unique_versions=sum(unique) / len(unique)))
        print(f'Combined all five versions into {"/".join(base)}/combined_versions.json')
        return
    if version('waterfall') != '0.3.4':
        raise RuntimeError('Use the watermark environment with waterfall==0.3.4')
    if version('transformers') != '5.17.0':
        raise RuntimeError('Use req-watermark.txt (transformers==5.17.0), not the training environment')
    from transformers import AutoTokenizer
    from src.util.shared_prefix import prefix_ids
    tokenizer = AutoTokenizer.from_pretrained(config['watermark_model'])
    if source == 'llama':
        # The old and new Fourier generators agree on these cosine keys.
        # Refuse sine/Nyquist keys rather than silently change their meaning.
        if any(r['k_p'] >= tokenizer.vocab_size // 2 for r in rows):
            raise ValueError('Llama compatibility requires keys below vocab_size // 2')
    for row in rows:
        prefix_ids(tokenizer, row['version_1'], row['prefix'])
    output = base + [f'version_{args.version}']
    root = REPO_ROOT.joinpath(*output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = variant_manifest(config, rows, args.version, source)
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / 'manifest.json'
        if manifest_path.exists() and read(manifest_path) != manifest:
            raise ValueError('Existing version has different inputs; refusing to overwrite')
        saved = root / 'watermarked_texts.json'
        if saved.exists():
            if not manifest_path.exists():
                raise ValueError('Existing texts have no provenance manifest')
            texts = read(saved)
            check_texts(texts, rows)
            for text, row in zip(texts, rows):
                prefix_ids(tokenizer, text, row['prefix'])
        write_path_file_atomic(output, 'manifest.json', manifest)
        write_path_file_atomic(output, 'inputs.json', rows)
        from src.util.watermark import watermark
        run_config = dict(config, generation_seed=manifest['generation_seed_base'])
        if source == 'llama':
            run_config['fixed_prefix_content_tokens'] = True
        watermark([r['original'] for r in rows], [r['watermark_id'] for r in rows],
                  [r['k_p'] for r in rows], run_config, output,
                  fixed_prefixes=[r['prefix'] for r in rows], resume=True,
                  checkpoint_every=1, max_new_texts=args.limit)
        check_texts(read(saved), rows, complete=args.limit is None)


if __name__ == '__main__':
    main()
