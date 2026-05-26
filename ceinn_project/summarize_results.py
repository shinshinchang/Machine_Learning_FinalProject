import csv
import glob
import json
from pathlib import Path


OUTPUT_DIR = Path('./outputs')


def infer_from_filename(path: Path):
    stem = path.stem  # metrics_xxx
    suffix = stem[len('metrics_'):] if stem.startswith('metrics_') else stem

    if suffix in {'sasrec', 'gru4rec'}:
        return suffix, 'none'

    if suffix == 'none' or suffix.startswith('w_o_'):
        ablation = 'none' if suffix == 'none' else suffix.replace('w_o_', 'w/o_', 1)
        return 'ceinn', ablation

    return 'unknown', suffix


def safe_get_metrics(metrics, split):
    block = metrics.get(split, {}) if isinstance(metrics, dict) else {}
    if not isinstance(block, dict):
        return '', ''
    return block.get('HR@10', ''), block.get('NDCG@10', '')


def main():
    files = sorted(glob.glob(str(OUTPUT_DIR / 'metrics_*.json')))
    rows = []

    for file_path in files:
        path = Path(file_path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

        inferred_model, inferred_ablation = infer_from_filename(path)
        model = metrics.get('model', inferred_model)
        ablation = metrics.get('ablation', inferred_ablation)

        valid_hr, valid_ndcg = safe_get_metrics(metrics, 'valid')
        test_hr, test_ndcg = safe_get_metrics(metrics, 'test')

        rows.append(
            {
                'file': path.name,
                'model': model,
                'ablation': ablation,
                'valid_HR@10': valid_hr,
                'valid_NDCG@10': valid_ndcg,
                'test_HR@10': test_hr,
                'test_NDCG@10': test_ndcg,
                'epoch': metrics.get('epoch', ''),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / 'summary_results.csv'
    fieldnames = [
        'file',
        'model',
        'ablation',
        'valid_HR@10',
        'valid_NDCG@10',
        'test_HR@10',
        'test_NDCG@10',
        'epoch',
    ]

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved CSV to: {out_csv}')
    print('| Model | Ablation | HR@10 | NDCG@10 |')
    print('| --- | --- | --- | --- |')
    for row in rows:
        print(
            f"| {row['model']} | {row['ablation']} | {row['test_HR@10']} | {row['test_NDCG@10']} |"
        )


if __name__ == '__main__':
    main()
