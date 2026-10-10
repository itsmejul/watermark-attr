"""Export the M=1/3/5 recitation trajectory, reusing the canonical M=1 grid."""
import argparse
import csv
import io
import json

from src.experiments.main.oracle_watermark_eval import read, select_indices, summarize
from src.util.filereader import REPO_ROOT, write_path_file_atomic


def collect(n):
    keys = read(REPO_ROOT / 'data/keys.json')['k_ps']
    _, indices = select_indices(len(keys), n)
    expected = [keys[i] for i in indices]
    rows = []
    for m in (1, 3, 5):
        directory = ('qwen' if m == 1 else f'_ablations/qwen_prefix_augmentation/M{m}')
        for epoch in range(5, 101, 5):
            path = REPO_ROOT / f'results/{directory}/experiment1/prefix_10/{n}/32/{epoch}/verification_closed.json'
            row = dict(m=m, n_documents=n, n_training_texts=n*m, epoch=epoch,
                       status='missing', acc_at_1=None, acc_at_5=None, mean_margin=None,
                       verification_path=str(path.relative_to(REPO_ROOT)))
            if path.exists():
                result = read(path)
                if result['correct_k_ps'] != expected or result['n_candidates'] != n:
                    raise ValueError(f'Misaligned evaluation: {path}')
                scores = summarize(result)
                row.update(status='complete', **{k: scores[k] for k in ('acc_at_1', 'acc_at_5', 'mean_margin')})
            rows.append(row)
    return rows


def collect_losses(n):
    """Return saved train/held-out losses without choosing a best test epoch."""
    rows = []
    for m in (1, 3, 5):
        directory = ('qwen-on-qwen' if m == 1 else f'_ablations/qwen_prefix_augmentation/M{m}')
        path = REPO_ROOT / f'training_metadata/{directory}/abstracts_only/{n}/32/loss_history.json'
        if not path.exists():
            continue
        by_epoch = {}
        for entry in read(path).get('log_history', []):
            if not any(k in entry for k in ('eval_train_loss', 'eval_heldout_loss')):
                continue
            epoch = float(entry['epoch'])
            row = by_epoch.setdefault(epoch, dict(m=m, n_documents=n, epoch=epoch,
                global_step=None, train_loss=None, heldout_loss=None))
            row['global_step'] = entry.get('step')
            for source, dest in (('eval_train_loss', 'train_loss'), ('eval_heldout_loss', 'heldout_loss')):
                if source in entry:
                    row[dest] = entry[source]
        rows.extend(by_epoch[e] for e in sorted(by_epoch))
    return rows


def export_rows(rows, name):
    output = ['src', 'eval', 'tables', 'ablations']
    write_path_file_atomic(output, name + '.json', rows)
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        REPO_ROOT.joinpath(*output, name + '.csv').write_text(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=1000)
    args = parser.parse_args()
    rows = collect(args.n_samples)
    name = f'qwen_prefix_augmentation_{args.n_samples}'
    export_rows(rows, name)
    export_rows(collect_losses(args.n_samples), name + '_losses')
    print(json.dumps([row for row in rows if row['epoch'] == 100], indent=2))
    print(f'Complete checkpoints: {sum(r["status"] == "complete" for r in rows)}/60')


if __name__ == '__main__':
    main()
