"""
Evaluation metrics for CEINN (Plan §3.3).

All metrics here are pure NumPy / scikit-learn — no torch dependency.
This is deliberate: evaluation runs on the CPU after scoring, scores
having been moved off-GPU in `evaluate.py`. Decoupling keeps metric
unit tests fast and lets the same code path serve both interactive
inspection (Jupyter) and CI.

Two-layer design
----------------
1) `compute_full_ranking(scores, target, seen)` reduces a per-user
   prediction over the full catalogue to a single integer rank (1-indexed).
2) Indicator metrics (`ndcg_at_k`, `hr_at_k`, `mrr`) consume vectors of
   ranks and return per-user values, plus the mean across users.

This separation matters because:
  * §3.3.3 (group analysis) reuses ranks with different group labels
    without recomputing scores.
  * §3.3.4 (confounding AUC) doesn't use ranks at all.
  * Phase 4+ baselines (SASRec, BPR-MF, ...) reuse `compute_full_ranking`
    verbatim — fairness of comparison hinges on identical ranking logic.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np


# =============================================================================
# §3.3.1  Full-ranking primitive
# =============================================================================

def compute_full_ranking(
    scores: np.ndarray,
    target_item: int,
    seen_items: Optional[Set[int]] = None,
    *,
    pad_index: int = 0,
) -> int:
    """
    Given a per-item score vector for ONE user at ONE supervised position,
    return the 1-indexed rank of `target_item` after excluding:
      * PAD slot (index `pad_index`),
      * items the user has already seen in TRAIN (`seen_items`).

    Tie-breaking: scores equal to the target's score are NOT counted as
    beating it ("≥" semantics would systematically over-rank popular
    items at score collisions; "strict >" is the conventional choice for
    Full Ranking evaluation in recsys papers and matches the baseline
    convention in SASRec / BERT4Rec official code).

    Parameters
    ----------
    scores       : (V,) float — model's predicted scores for each item id.
    target_item  : int — the ground-truth item id; must be > 0.
    seen_items   : iterable of ints, or None — items to mask out before
                   ranking. The target_item itself is NEVER masked out
                   even if it happens to appear in `seen_items` (which
                   it shouldn't for held-out targets, but be safe).
    pad_index    : the PAD slot to always exclude.

    Returns
    -------
    rank : 1-indexed rank in {1, 2, ..., V - n_excluded}.
    """
    if target_item == pad_index:
        raise ValueError("compute_full_ranking: target_item must not be PAD")

    s = np.asarray(scores, dtype=np.float64).copy()

    # Mask PAD and seen items by setting their score to -inf so they
    # never beat the target. We re-set the target score afterwards in
    # case the caller accidentally included it.
    s[pad_index] = -np.inf
    if seen_items:
        # Convert to array for fancy-indexing; drop the target if present.
        seen_arr = np.fromiter(
            (i for i in seen_items if i != target_item and 0 <= i < s.shape[0]),
            dtype=np.int64,
        )
        if seen_arr.size:
            s[seen_arr] = -np.inf

    target_score = s[target_item]
    # Strictly more than the target_score → ranked above the target.
    n_above = int(np.sum(s > target_score))
    return n_above + 1


# =============================================================================
# §3.3.2  Indicator metrics
# =============================================================================

def ndcg_at_k(ranks: Sequence[int], k: int) -> np.ndarray:
    """
    NDCG@K for single-relevant-item retrieval.

    With exactly one relevant item per user at rank r, the IDCG is 1 and
    the DCG is 1 / log2(r + 1) if r <= K else 0.

    Returns a per-user array of length len(ranks). The caller can take
    .mean() to obtain the corpus-level NDCG@K, or pass it to
    group-wise aggregators.
    """
    r = np.asarray(ranks, dtype=np.float64)
    score = np.zeros_like(r)
    hit = r <= k
    score[hit] = 1.0 / np.log2(r[hit] + 1.0)
    return score


def hr_at_k(ranks: Sequence[int], k: int) -> np.ndarray:
    """
    Hit Ratio @ K. Per-user, returns 1.0 if the target was in the top K
    (rank <= K), else 0.0.
    """
    r = np.asarray(ranks, dtype=np.int64)
    return (r <= k).astype(np.float64)


def mrr(ranks: Sequence[int]) -> np.ndarray:
    """
    Mean Reciprocal Rank — but returned per-user. Take .mean() at the
    aggregation step.
    """
    r = np.asarray(ranks, dtype=np.float64)
    return 1.0 / r


# =============================================================================
# §3.3.3  Group metrics (Head / Torso / Tail)
# =============================================================================

def group_metrics(
    ranks: Sequence[int],
    item_ids: Sequence[int],
    group_labels: Dict[int, str],
    *,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Slice (ranks, items) into the popularity groups defined by
    `group_labels` and report NDCG@K and HR@K per group.

    Parameters
    ----------
    ranks        : (N,) per-user ranks of the held-out target.
    item_ids     : (N,) the target item id corresponding to each rank.
    group_labels : dict mapping item_id → group name (e.g. 'head' / 'torso' / 'tail').
                   Built by `utils/popularity.py` once per dataset.
    k            : top-K cutoff.

    Returns
    -------
    {
        'head':  {'ndcg@k': float, 'hr@k': float, 'n_users': int},
        'torso': {...},
        'tail':  {...},
    }

    Items whose id is missing from `group_labels` are dropped (with a
    silent count; they typically correspond to UNK slots).
    """
    r = np.asarray(ranks, dtype=np.int64)
    it = np.asarray(item_ids, dtype=np.int64)

    group_of = np.array(
        [group_labels.get(int(i), None) for i in it], dtype=object
    )

    out: Dict[str, Dict[str, float]] = {}
    for group in ("head", "torso", "tail"):
        sel = group_of == group
        n = int(sel.sum())
        if n == 0:
            out[group] = {"ndcg@k": float("nan"), "hr@k": float("nan"), "n_users": 0}
            continue
        out[group] = {
            "ndcg@k": float(ndcg_at_k(r[sel], k).mean()),
            "hr@k": float(hr_at_k(r[sel], k).mean()),
            "n_users": n,
        }
    return out


# =============================================================================
# §3.3.4  Confounding AUC (deconfounding quality probe)
# =============================================================================

def confounding_auc(
    h_states: np.ndarray,
    z_labels: np.ndarray,
    *,
    n_folds: int = 5,
    random_state: int = 0,
) -> float:
    """
    Train a Logistic Regression to predict the confounder bucket Z from
    the latent state h_t, with 5-fold stratified CV. Return the macro
    OvR AUC. (Plan §3.3.4)

    Interpretation (H2b)
    --------------------
    - AUC near (1 / K) = 1 / n_classes random baseline  → h ⊥ Z achieved.
    - AUC near 1.0  → h still encodes the confounder.

    Parameters
    ----------
    h_states   : (N, d) float — latent representations.
    z_labels   : (N,)   int   — bucket labels in {0, ..., K-1}.
    n_folds    : CV folds; default 5 as the plan specifies.
    random_state: passed to the splitter and the classifier.

    Returns
    -------
    macro_auc : float in [0, 1].

    Notes
    -----
    * Uses sklearn — added to requirements at Phase 3.
    * Classes that don't appear in `z_labels` are dropped before fitting
      so LR doesn't choke on degenerate buckets. The macro AUC is then
      computed only over represented classes.
    """
    # Lazy import: keep sklearn out of import-time cost for code paths
    # (data loaders, etc.) that never call this.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.multiclass import OneVsRestClassifier

    h = np.asarray(h_states, dtype=np.float64)
    z = np.asarray(z_labels, dtype=np.int64)
    if h.shape[0] != z.shape[0]:
        raise ValueError(
            f"confounding_auc: size mismatch h={h.shape}, z={z.shape}"
        )

    # Restrict to classes with at least n_folds members (StratifiedKFold
    # requires this). Drop the rest from the data entirely.
    counts = np.bincount(z)
    keep_classes = np.where(counts >= n_folds)[0]
    keep_mask = np.isin(z, keep_classes)
    h = h[keep_mask]
    z = z[keep_mask]
    if len(np.unique(z)) < 2:
        # Degenerate: cannot define AUC with a single class.
        return float("nan")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, test_idx in skf.split(h, z):
        # OneVsRest explicitly per the plan ("Logistic Regression ... 5-fold
        # CV ... 多分類 AUC"). sklearn 1.5+ removed the `multi_class` kwarg
        # from `LogisticRegression`, so OvR must be requested via wrapper.
        base = LogisticRegression(max_iter=1000, random_state=random_state)
        clf = OneVsRestClassifier(base)
        clf.fit(h[train_idx], z[train_idx])
        proba = clf.predict_proba(h[test_idx])
        try:
            auc = roc_auc_score(
                z[test_idx],
                proba,
                multi_class="ovr",
                average="macro",
                labels=clf.classes_,
            )
            aucs.append(auc)
        except ValueError:
            # Fold-level degeneracy → skip.
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


# =============================================================================
# Convenience: aggregate the standard CEINN evaluation
# =============================================================================

def standard_topk_report(
    ranks: Sequence[int],
    ks: Tuple[int, ...] = (5, 10, 20),
) -> Dict[str, float]:
    """
    Convenience aggregator returning NDCG@K, HR@K for each K in `ks`
    plus MRR — the canonical Phase 3 / Phase 6 reporting block.

    All metrics are CORPUS-LEVEL means.
    """
    out: Dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(ndcg_at_k(ranks, k).mean())
        out[f"hr@{k}"] = float(hr_at_k(ranks, k).mean())
    out["mrr"] = float(mrr(ranks).mean())
    return out
