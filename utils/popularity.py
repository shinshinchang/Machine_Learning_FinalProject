"""
Item popularity grouping (Plan §3.3.3).

Builds a mapping  item_idx → {'head', 'torso', 'tail'}  based on the
*training-set* interaction count of each item. Train-only counts avoid
leaking validation / test label distribution into the group definition.

Thresholds
----------
The plan specifies P80 (top 20%) and P20 (bottom 20%) on the item
frequency distribution. We compute these as quantiles over the per-item
training counts, INCLUDING items that appear 0 times in train (which
would be unusual after 5-core filtering but is defensive against
preprocessing drift).

Why not put this in preprocess.py?
----------------------------------
preprocess.py is the gatekeeper for Phase 2 outputs and is now frozen.
Popularity grouping is a Phase 3 artefact (§3.3.3) consumed only by
metrics, so it lives in its own helper and is driven by a one-shot
script under `scripts/build_popularity_groups.py`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np


def build_popularity_groups(
    train_seqs: Dict[int, Iterable[Tuple[int, int, int]]],
    n_items: int,
    *,
    head_quantile: float = 0.80,
    tail_quantile: float = 0.20,
    pad_index: int = 0,
) -> Tuple[Dict[int, str], Dict[str, float]]:
    """
    Parameters
    ----------
    train_seqs : {user_idx: [(item_idx, rating_bin, dt_bin), ...]} — the
                 Phase-2 `train_seqs.pkl` artefact for either dataset.
    n_items    : total number of real items (NOT including PAD). Items
                 are indexed 1..n_items.
    head_quantile / tail_quantile : float in (0, 1). Items with TRAIN
                 frequency above the head_quantile go to 'head'; those
                 below the tail_quantile go to 'tail'; the rest 'torso'.
    pad_index  : item index reserved as PAD (=0 in Phase 2). Skipped.

    Returns
    -------
    group_labels : {item_idx: 'head' | 'torso' | 'tail'}, one entry per
                   real item (i.e. n_items entries).
    summary      : dict with the computed quantile thresholds and group
                   sizes, useful for the validation log.
    """
    if not (0.0 < tail_quantile < head_quantile < 1.0):
        raise ValueError(
            f"build_popularity_groups: bad quantiles "
            f"tail={tail_quantile}, head={head_quantile}"
        )

    # Per-item interaction count.
    counts = np.zeros(n_items + 1, dtype=np.int64)  # +1 to leave [0] = PAD
    for _u, seq in train_seqs.items():
        for triple in seq:
            iid = int(triple[0])
            if iid == pad_index:
                continue
            if 0 < iid <= n_items:
                counts[iid] += 1

    # Quantiles over the real-item counts only.
    real_counts = counts[1:]
    if real_counts.size == 0:
        raise ValueError("build_popularity_groups: no items in train_seqs")

    head_thr = float(np.quantile(real_counts, head_quantile))
    tail_thr = float(np.quantile(real_counts, tail_quantile))

    group_labels: Dict[int, str] = {}
    for iid in range(1, n_items + 1):
        c = int(counts[iid])
        if c > head_thr:
            group_labels[iid] = "head"
        elif c < tail_thr:
            group_labels[iid] = "tail"
        else:
            group_labels[iid] = "torso"

    summary = {
        "head_quantile": head_quantile,
        "tail_quantile": tail_quantile,
        "head_threshold_count": head_thr,
        "tail_threshold_count": tail_thr,
        "n_head": sum(1 for g in group_labels.values() if g == "head"),
        "n_torso": sum(1 for g in group_labels.values() if g == "torso"),
        "n_tail": sum(1 for g in group_labels.values() if g == "tail"),
        "n_items": n_items,
    }
    return group_labels, summary


def save_popularity_groups(
    group_labels: Dict[int, str],
    summary: Dict[str, float],
    output_path: Union[str, Path],
) -> None:
    """
    Pickle the (labels, summary) pair to disk for `metrics.py` to load.
    The path convention is `data/processed/<dataset>/item_popularity_group.pkl`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"labels": group_labels, "summary": summary}, f, protocol=4)


def load_popularity_groups(path: Union[str, Path]) -> Tuple[Dict[int, str], Dict[str, float]]:
    """Load a previously saved popularity-group pickle."""
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    return obj["labels"], obj["summary"]
