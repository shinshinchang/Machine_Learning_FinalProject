"""
One-shot driver for §3.3.3 — builds and persists item popularity groups
for both datasets after Phase 2 has produced `train_seqs.pkl`.

Usage
-----
    python scripts/build_popularity_groups.py          # both datasets
    python scripts/build_popularity_groups.py --only amazon
    python scripts/build_popularity_groups.py --only movielens

Output
------
    data/processed/amazon_beauty/item_popularity_group.pkl
    data/processed/movielens_10m/item_popularity_group.pkl

Each pickle holds a dict with two keys:
    - 'labels' : {item_idx: 'head' | 'torso' | 'tail'}
    - 'summary': quantile thresholds + group sizes (for the README)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Make the `utils` package importable when this script is invoked from
# the repo root: `python scripts/build_popularity_groups.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.popularity import build_popularity_groups, save_popularity_groups


DATASETS = {
    "amazon": "amazon_beauty",
    "movielens": "movielens_10m",
}


def _process(dataset_key: str, processed_root: Path) -> None:
    folder = processed_root / DATASETS[dataset_key]
    train_path = folder / "train_seqs.pkl"
    vocab_path = folder / "vocab_sizes.json"
    out_path = folder / "item_popularity_group.pkl"

    if not train_path.exists():
        raise FileNotFoundError(f"missing {train_path}; run Phase 2 first")
    if not vocab_path.exists():
        raise FileNotFoundError(f"missing {vocab_path}; run Phase 2 first")

    with train_path.open("rb") as f:
        train_seqs = pickle.load(f)
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    n_items = int(vocab["n_items"])
    labels, summary = build_popularity_groups(train_seqs, n_items)
    save_popularity_groups(labels, summary, out_path)

    print(
        f"[{dataset_key:>9s}] head={summary['n_head']:>6d}  "
        f"torso={summary['n_torso']:>6d}  tail={summary['n_tail']:>6d}  "
        f"thresholds=(head>{summary['head_threshold_count']:.1f}, "
        f"tail<{summary['tail_threshold_count']:.1f})  →  {out_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=["amazon", "movielens"], default=None,
        help="Process only one dataset.",
    )
    parser.add_argument(
        "--processed-root", type=Path, default=REPO_ROOT / "data" / "processed",
        help="Root containing the per-dataset processed folders.",
    )
    args = parser.parse_args()

    keys = [args.only] if args.only else list(DATASETS.keys())
    for k in keys:
        _process(k, args.processed_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
