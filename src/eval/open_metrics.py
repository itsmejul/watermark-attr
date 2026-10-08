"""Empirical raw and standardized MaxQ detection and attribution metrics.

Thresholds use negatives only. Accept iff score > threshold; ties at the
threshold are rejected. These are in-sample ROC operating points, not estimates
of performance at a threshold calibrated on an independent negative set.
"""
import math

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_open_set(closed, opened, fpr_targets=(0.01, 0.05), score_type='raw_maxq'):
    """OpenAcc divides jointly detected/correctly attributed positives by ALL positives."""
    if closed['n_candidates'] != opened['n_candidates']:
        raise ValueError('Positive and negative candidate counts differ')
    pos = np.asarray(closed['top_scores'], dtype=np.float64)[:, 0]
    neg = np.asarray(opened['top_scores'], dtype=np.float64)[:, 0]
    ranks = np.asarray(closed['correct_ranks'])
    if not pos.size or not neg.size or ranks.shape != pos.shape:
        raise ValueError('Empty scores or misaligned positive ranks')
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError('Nonfinite detector scores')
    if score_type == 'standardized_maxq':
        transformed = []
        for scores, verification in ((pos, closed), (neg, opened)):
            mean = np.asarray(verification['score_mean'], dtype=np.float64)
            std = np.asarray(verification['score_std'], dtype=np.float64)
            if (mean.shape != scores.shape or std.shape != scores.shape
                    or not np.isfinite(mean).all() or not np.isfinite(std).all()
                    or (std <= 0).any()):
                raise ValueError('Standardization requires aligned finite means and positive stds')
            transformed.append((scores - mean) / std)
        pos, neg = transformed
    elif score_type != 'raw_maxq':
        raise ValueError(f'Unknown score type: {score_type}')
    # Check the actual returned top-1 key as well: rank=1 alone can include ties.
    correct = (ranks == 1) & (
        np.asarray(closed['top_k_p'])[:, 0] == np.asarray(closed['correct_k_ps'])
    )
    result = {
        'score_type': score_type,
        'n_pos': int(pos.size), 'n_neg': int(neg.size),
        'auroc': float(roc_auc_score(
            np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg])),
        'closed_top1_accuracy': float(correct.mean()),
    }
    descending = np.sort(neg)[::-1]
    for target in fpr_targets:
        if not 0 <= target < 1:
            raise ValueError('FPR targets must be in [0, 1)')
        budget = math.floor(target * neg.size)
        threshold = float(descending[budget])
        detected = pos > threshold
        suffix = f'{target:g}'
        result.update({
            f'threshold@{suffix}': threshold,
            f'fpr@{suffix}': float((neg > threshold).mean()),
            f'tpr@{suffix}': float(detected.mean()),
            f'openacc@{suffix}': float((detected & correct).mean()),
            f'n_detected@{suffix}': int(detected.sum()),
            f'n_correct_attributed@{suffix}': int((detected & correct).sum()),
        })
    return result


def open_evaluation_report(results_root, figure_dir, epoch=100, n=1000, batch=32):
    """Export both score definitions and shared-threshold metrics for one notebook."""
    import json
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    base = results_root / 'prefix_10' / str(n) / str(batch) / str(epoch)
    closed = json.loads((base / 'verification_closed.json').read_text())
    opened = json.loads((base / 'verification_open.json').read_text())
    by_score = {kind: dict(evaluate_open_set(closed, opened, score_type=kind), epoch=epoch)
                for kind in ('raw_maxq', 'standardized_maxq')}
    rows = []
    for kind, result in by_score.items():
        for target in (0.01, 0.05):
            suffix = f'{target:g}'
            rows.append({
                'model': results_root.parent.name, 'corpus_size': n, 'epoch': epoch,
                'score_type': kind, 'n_positive': result['n_pos'], 'n_negative': result['n_neg'],
                'auroc': result['auroc'], 'closed_top1_accuracy': result['closed_top1_accuracy'],
                'target_fpr': target, 'achieved_fpr': result[f'fpr@{suffix}'],
                'threshold': result[f'threshold@{suffix}'], 'threshold_rule': 'score > threshold',
                'tpr': result[f'tpr@{suffix}'], 'open_acc': result[f'openacc@{suffix}'],
                'n_detected': result[f'n_detected@{suffix}'],
                'n_correct_attributed': result[f'n_correct_attributed@{suffix}'],
            })
    table = pd.DataFrame(rows)
    table_dir = Path('tables') / results_root.parent.name
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / 'experiment1_open_keyspace.csv', index=False)
    (table_dir / 'experiment1_open_keyspace.tex').write_text(
        table[['score_type', 'target_fpr', 'achieved_fpr', 'auroc', 'tpr', 'open_acc']]
        .to_latex(index=False, float_format=lambda x: f'{x:.4f}'))
    display(table)
    fields = ['auroc', 'tpr@0.01', 'tpr@0.05', 'openacc@0.01', 'openacc@0.05']
    x = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, kind, label in [(-.18, 'raw_maxq', 'Raw MaxQ'),
                                (.18, 'standardized_maxq', 'Standardized MaxQ')]:
        ax.bar(x + offset, [by_score[kind][k] for k in fields], width=.36, label=label)
    ax.set_xticks(x, ['AUROC', 'TPR@1%', 'TPR@5%', 'OpenAcc@1%', 'OpenAcc@5%'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Open keyspace: N={n:,}, epoch {epoch}')
    ax.legend()
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'exp1_open_keyspace_raw_standardized.pdf')
    plt.show()
    return {n: by_score['raw_maxq']}, {n: by_score['standardized_maxq']}
