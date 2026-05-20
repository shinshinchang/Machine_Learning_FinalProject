"""
Pure statistical / mathematical utilities reused across phases.

Why this module exists
----------------------
Several pieces of CEINN's preprocessing AND training need the same primitive:
- Phase 2 Δt log-bucketing (Amazon: 32 bins, MovieLens: 64 bins)
- Phase 2 salesRank log-quantile bucketing (Amazon: 10 bins)
- Phase 2 GenreRed Jaccard (standard + IDF-weighted)
- Phase 3 batched Jaccard for in-graph cost estimation
- Phase 4+ identical bucketing at inference time (frozen edges)

Re-implementing bucketing across `preprocess.py` and the per-dataset
loaders would invite subtle disagreements (different edge handling on
ties, different log bases, different open/closed interval conventions).
This module owns those decisions once, and is covered by inline unit
checks at import-time / tested via Phase 2's internal invariants and
the Phase 3 unit tests under `tests/`.

Design notes
------------
* All bucketing functions are FIT/TRANSFORM separated. You fit on the
  training distribution and persist the edges, then transform val/test
  samples with the same frozen edges — strictly preventing leakage.
* "Edges" are right-closed: bucket i covers (edges[i], edges[i+1]].
  Values <= edges[0] go to bucket 0; values > edges[-1] go to the last
  bucket. This makes the bucketing total over the real line.
* The Phase 3 execution plan (§3.1.1, §3.1.3) names some of these
  functions differently from Phase 2 (`log_quantile_bucket`, `log_time_bucket`,
  `jaccard_similarity`, `weighted_jaccard`). Rather than duplicate the
  logic and risk drift, the Phase 3 names are exposed as thin wrappers
  over the Phase 2 implementations at the bottom of this file. Both
  names refer to the same code path.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Log-quantile bucketing  (Plan §2.1.4, §2.2.4, and §3.1.1 / §3.1.3)
# =============================================================================

def fit_log_quantile_edges(
    values: Sequence[float],
    n_bins: int,
    *,
    base: float = 10.0,
    add_one: bool = True,
) -> np.ndarray:
    """
    Fit equal-frequency bucket edges in *log* space.

    Parameters
    ----------
    values   : positive numbers from which to estimate the quantiles. NaN /
               non-positive entries are silently dropped before fitting;
               callers should check input quality first if needed.
    n_bins   : number of buckets to produce. Edges returned have length
               n_bins + 1.
    base     : logarithm base (10 for salesRank per the plan, 10 for Δt too;
               kept as a parameter for reuse).
    add_one  : if True, use log(x + 1) (the canonical guard for salesRank
               which can take value 0 in theory; harmless for Δt).

    Returns
    -------
    edges : 1-D float64 array of length n_bins + 1, monotonically increasing.
            edges[0] = log of the min observed value,
            edges[-1] = log of the max. Quantile boundaries fall in between.

    Notes
    -----
    Persist the returned edges to disk. Pass them to `apply_bucket_edges`
    at inference time to guarantee identical bucket assignments.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError("fit_log_quantile_edges: no positive finite values")
    if n_bins < 1:
        raise ValueError("fit_log_quantile_edges: n_bins must be >= 1")

    if add_one:
        arr = arr + 1.0
    logs = np.log(arr) / np.log(base)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(logs, qs)

    # Resolve ties: if a heavy point mass causes adjacent edges to collide,
    # nudge subsequent edges up by a tiny epsilon to keep them strictly
    # increasing. Without this, np.digitize would emit empty buckets.
    eps = np.finfo(np.float64).eps * max(1.0, abs(edges[-1]))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def apply_bucket_edges(
    values: Sequence[float],
    edges: np.ndarray,
    *,
    base: float = 10.0,
    add_one: bool = True,
) -> np.ndarray:
    """
    Map values to bucket indices in {0, 1, ..., n_bins - 1} using frozen
    edges from `fit_log_quantile_edges`.

    Values <= edges[0]  → bucket 0.
    Values >  edges[-1] → bucket n_bins - 1 (the last bucket).
    Non-positive or NaN values are mapped to bucket 0 by convention.
    """
    arr = np.asarray(values, dtype=np.float64)
    n_bins = len(edges) - 1

    out = np.zeros(arr.shape, dtype=np.int64)
    valid = np.isfinite(arr) & (arr > 0)

    if valid.any():
        if add_one:
            logs = np.log(arr[valid] + 1.0) / np.log(base)
        else:
            logs = np.log(arr[valid]) / np.log(base)
        # np.digitize with right=True puts values equal to an edge into the
        # *lower* bucket, matching the right-closed interval definition.
        idx = np.digitize(logs, edges[1:-1], right=True)
        out[valid] = idx
    np.clip(out, 0, n_bins - 1, out=out)
    return out


# =============================================================================
# Jaccard similarity over binary multi-hot vectors  (Plan §2.3.4, §3.1.2)
# =============================================================================

def jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    """
    Jaccard similarity between two binary vectors of identical length.

    |A ∩ B| / |A ∪ B|. Returns 0.0 if both are empty (the convention used
    by §3.3.4: "no overlap → no redundancy contribution").
    """
    if a.shape != b.shape:
        raise ValueError(f"jaccard_binary: shape mismatch {a.shape} vs {b.shape}")
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    if union == 0.0:
        return 0.0
    return inter / union


def jaccard_against_accumulated(
    candidate: np.ndarray,
    history_union: np.ndarray,
) -> float:
    """
    Specialised variant of jaccard_binary used in the GenreRed sweep.

    `candidate`     : 19-dim binary vector for the next movie's genres.
    `history_union` : 19-dim binary vector for the union of all genres
                      seen by the user strictly before this interaction.

    Returns the standard Jaccard score in [0, 1]. If `history_union` is
    all zeros (first interaction of the user), returns 0.0 by definition.
    """
    return jaccard_binary(candidate, history_union)


def jaccard_idf_weighted(
    candidate: np.ndarray,
    history_union: np.ndarray,
    idf: np.ndarray,
) -> float:
    """
    IDF-weighted Jaccard, used by ablation ML4 (§7.3).

    Standard Jaccard treats every genre as equally distinctive. EDA §4.1
    shows Drama and Comedy together cover >40% of movies, so they shouldn't
    weigh as much as Film-Noir or IMAX when measuring "genre fatigue".

    Formula
    -------
        J_idf(A, B) = sum(idf[g] * 1[g in A ∩ B])
                    / sum(idf[g] * 1[g in A ∪ B])

    where `idf[g] = log(N_movies / df_g)` with df_g the document
    frequency of genre g across the movie corpus. See
    `compute_genre_idf` below.
    """
    if candidate.shape != history_union.shape or candidate.shape != idf.shape:
        raise ValueError("jaccard_idf_weighted: shape mismatch")
    inter_mask = np.logical_and(candidate > 0, history_union > 0)
    union_mask = np.logical_or(candidate > 0, history_union > 0)
    num = float((idf * inter_mask).sum())
    den = float((idf * union_mask).sum())
    if den == 0.0:
        return 0.0
    return num / den


def batch_jaccard(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    *,
    idf_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorised pair-wise Jaccard for two batches of binary vectors.

    Used by:
      * Phase 3 unit tests (correctness sanity)
      * Phase 4+ in-graph GenreRed lookups when computing cost-per-candidate
        for the discrete choice model. Although the offline sweep in
        `preprocess.py` persists GenreRed for the (u, t, true_item) triples,
        the choice model needs GenreRed for ALL candidate items in the
        scoring batch at training time, which is what this function
        accelerates.

    Parameters
    ----------
    mat_a : array of shape (N, G), binary or {0, 1}-cast-able.
    mat_b : array of shape (M, G), binary or {0, 1}-cast-able.
            For element-wise (N == M) pairing, just pass mat_b of shape
            (N, G); this returns an (N,) vector. For all-pairs, pass
            (M, G) where M != N and use a broadcast-friendly variant.

    idf_weights : optional (G,) vector of per-feature weights for the
                  IDF-weighted variant. When None, computes the
                  un-weighted standard Jaccard.

    Returns
    -------
    sims : 1-D float64 array of length N (element-wise pairing) OR
           2-D float64 array of shape (N, M) (all-pairs, when N != M).

    Notes
    -----
    Both inputs are cast to bool internally. Rows that are all-zero on
    both sides of a pair yield 0.0 (the §3.3.4 convention).
    """
    a = np.asarray(mat_a, dtype=bool)
    b = np.asarray(mat_b, dtype=bool)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"batch_jaccard: incompatible shapes {a.shape} vs {b.shape}"
        )

    G = a.shape[1]
    if idf_weights is None:
        w = np.ones(G, dtype=np.float64)
    else:
        w = np.asarray(idf_weights, dtype=np.float64)
        if w.shape != (G,):
            raise ValueError(
                f"batch_jaccard: idf_weights shape {w.shape} != ({G},)"
            )

    # Element-wise pairing branch (N == M and the caller wants per-row).
    # We can't perfectly distinguish this from all-pairs on shape alone, so
    # we follow the explicit convention: equal first dim → element-wise.
    if a.shape[0] == b.shape[0]:
        af = a.astype(np.float64) * w  # (N, G)
        bf = b.astype(np.float64) * w  # (N, G)
        inter = np.minimum(af, bf).sum(axis=1)  # weighted intersection
        union = np.maximum(af, bf).sum(axis=1)  # weighted union
        out = np.zeros(a.shape[0], dtype=np.float64)
        nz = union > 0.0
        out[nz] = inter[nz] / union[nz]
        return out

    # All-pairs branch: returns (N, M).
    af = a.astype(np.float64) * w  # (N, G)
    bf = b.astype(np.float64) * w  # (M, G)
    # Broadcast: inter[n, m] = sum_g min(af[n, g], bf[m, g])
    inter = np.minimum(af[:, None, :], bf[None, :, :]).sum(axis=2)
    union = np.maximum(af[:, None, :], bf[None, :, :]).sum(axis=2)
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    nz = union > 0.0
    out[nz] = inter[nz] / union[nz]
    return out


def compute_genre_idf(
    genre_vectors: Iterable[np.ndarray],
    n_genres: int,
    *,
    smooth: bool = True,
) -> np.ndarray:
    """
    Build the per-genre IDF weights from the movie corpus.

    Parameters
    ----------
    genre_vectors : iterable of 19-dim binary vectors, one per movie.
    n_genres      : 19 for MovieLens 10M.
    smooth        : if True, use log((N + 1) / (df + 1)) + 1 (sklearn
                    convention), which avoids zero IDF for ubiquitous
                    genres and division-by-zero for unseen ones.

    Returns
    -------
    idf : 1-D float64 array of length `n_genres`.
    """
    df = np.zeros(n_genres, dtype=np.float64)
    n_movies = 0
    for v in genre_vectors:
        if v.shape != (n_genres,):
            raise ValueError(
                f"compute_genre_idf: expected vectors of shape ({n_genres},), "
                f"got {v.shape}"
            )
        df += (v > 0).astype(np.float64)
        n_movies += 1
    if n_movies == 0:
        raise ValueError("compute_genre_idf: empty corpus")

    if smooth:
        return np.log((n_movies + 1.0) / (df + 1.0)) + 1.0
    return np.log(n_movies / np.maximum(df, 1.0))


# =============================================================================
# Phase 3 execution-plan name aliases  (§3.1.1, §3.1.2, §3.1.3)
# =============================================================================
#
# The execution plan asks for these specific function names. They are thin
# wrappers — using the Phase 2 implementations underneath — so the bucket
# edges produced at preprocessing time and consumed at inference time go
# through exactly one code path.

def log_quantile_bucket(
    values: Sequence[float],
    n_bins: int,
    *,
    base: float = 10.0,
    add_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase 3 §3.1.1 alias. Fits edges AND returns the per-value bucket
    indices in a single call (the plan's prescribed interface).

    Returns
    -------
    (edges, indices) tuple.
        edges  : length n_bins + 1, monotone increasing.
        indices: same length as `values`.
    """
    edges = fit_log_quantile_edges(values, n_bins, base=base, add_one=add_one)
    indices = apply_bucket_edges(values, edges, base=base, add_one=add_one)
    return edges, indices


def log_time_bucket(
    delta_t_seconds: Sequence[float],
    edges: np.ndarray,
    *,
    base: float = 10.0,
) -> np.ndarray:
    """
    Phase 3 §3.1.3 alias. Maps Δt (in seconds) to bucket indices using
    *frozen* edges fit on TRAIN-set Δt. Defensive against Δt <= 0
    (collapsed-session intra-events or same-second ties) by mapping them
    to bucket 0 (which the preprocessing pipeline reserves as PAD).
    """
    return apply_bucket_edges(delta_t_seconds, edges, base=base, add_one=True)


def jaccard_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Phase 3 §3.1.2 alias for `jaccard_binary`."""
    return jaccard_binary(vec_a, vec_b)


def weighted_jaccard(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    idf_weights: np.ndarray,
) -> float:
    """Phase 3 §3.1.2 alias for `jaccard_idf_weighted`."""
    return jaccard_idf_weighted(vec_a, vec_b, idf_weights)


# =============================================================================
# Self-checks (run on `import`; cheap and silent unless they fail)
# =============================================================================

def _self_check() -> None:
    rng = np.random.default_rng(0)
    vals = rng.uniform(1, 1e5, size=10_000)
    edges = fit_log_quantile_edges(vals, n_bins=10)
    assigned = apply_bucket_edges(vals, edges)
    counts = np.bincount(assigned, minlength=10)
    assert counts.min() > 0, f"degenerate buckets: {counts}"
    target = vals.size // 10
    assert counts.max() < target * 1.3, f"buckets too imbalanced: {counts}"

    a = np.array([1, 1, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 1, 0], dtype=np.float32)
    assert math.isclose(jaccard_binary(a, b), 1 / 3), "jaccard wrong"
    assert math.isclose(jaccard_binary(a, np.zeros_like(a)), 0.0)

    corpus = [
        np.array([1, 1, 0], dtype=np.float32),
        np.array([1, 1, 0], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.float32),
    ]
    idf = compute_genre_idf(corpus, n_genres=3)
    assert idf[0] < idf[1] < idf[2], f"IDF monotonicity violated: {idf}"

    # batch_jaccard (element-wise) consistency with per-pair jaccard_binary
    A = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    B = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)
    sims = batch_jaccard(A, B)
    assert math.isclose(sims[0], 1 / 3), f"batch_jaccard mismatch: {sims[0]}"
    assert math.isclose(sims[1], 1 / 2), f"batch_jaccard mismatch: {sims[1]}"

    # Phase 3 aliases
    e2, idx2 = log_quantile_bucket(vals, n_bins=10)
    assert np.array_equal(e2, edges)
    assert np.array_equal(idx2, assigned)


_self_check()
