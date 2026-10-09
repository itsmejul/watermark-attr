"""Validate fixed-prefix variants before admitting them to a training run."""
from src.data_creation.create_qwen_prefix_variants import inputs, variant_manifest, check_texts
from src.experiments.main.oracle_watermark_eval import read
from src.util.filereader import REPO_ROOT


def load_augmentation(n, m, subset):
    if m not in (3, 5):
        raise ValueError('Only M=3 and M=5 need new training; reuse M=1')
    config, rows = inputs(n)
    if ([r['k_p'] for r in rows] != subset['k_ps']
            or [r['version_1'] for r in rows] != subset['T_ws']
            or [r['prefix'] for r in rows] != subset['prefix_10']):
        raise ValueError('Augmentation inputs differ from canonical training subset')
    base = f'data/t_ws/qwen_prefix_augmentation/{n}'
    versions = [subset['T_ws']]
    required = []
    for v in range(2, m + 1):
        directory = f'{base}/version_{v}'
        manifest = f'{directory}/manifest.json'
        texts_path = f'{directory}/watermarked_texts.json'
        if read(REPO_ROOT / manifest) != variant_manifest(config, rows, v):
            raise ValueError(f'Incorrect variant manifest: {manifest}')
        texts = read(REPO_ROOT / texts_path)
        check_texts(texts, rows, complete=True)
        versions.append(texts)
        required.extend((manifest, texts_path))
    # Document-major ordering; Trainer shuffles examples during training.
    texts = [versions[v][i] for i in range(n) for v in range(m)]
    return texts, required
