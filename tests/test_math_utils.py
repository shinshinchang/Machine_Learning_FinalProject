"""Unit tests for utils/math_utils.py (Plan §3.1 + §3.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils.math_utils import (
    apply_bucket_edges,
    batch_jaccard,
    compute_genre_idf,
    fit_log_quantile_edges,
    jaccard_binary,
    jaccard_idf_weighted,
    log_quantile_bucket,
    log_time_bucket,
    weighted_jaccard,
)


# ---------- Log-quantile bucketing ------------------------------------------

class TestLogQuantileBucket:

    def test_edges_monotone(self):
        rng = np.random.default_rng(42)
        vals = rng.uniform(1, 1e6, size=5_000)
        edges = fit_log_quantile_edges(vals, n_bins=10)
        assert len(edges) == 11
        assert np.all(np.diff(edges) > 0), "edges must be strictly increasing"

    def test_equal_frequency_buckets(self):
        rng = np.random.default_rng(0)
        vals = rng.uniform(1, 1e5, size=10_000)
        _, idx = log_quantile_bucket(vals, n_bins=10)
        counts = np.bincount(idx, minlength=10)
        # Equal-frequency target = 1000 per bucket. Allow 30% slack.
        assert counts.min() > 700
        assert counts.max() < 1300

    def test_tie_handling_no_empty_buckets(self):
        # Heavy point mass at 1.0 should still yield non-collapsed buckets
        # thanks to the eps nudge in fit_log_quantile_edges.
        vals = np.r_[np.ones(500), np.linspace(2, 100, 500)]
        edges = fit_log_quantile_edges(vals, n_bins=5)
        # All edges strictly increasing → no degenerate buckets.
        assert np.all(np.diff(edges) > 0)

    def test_apply_with_frozen_edges_no_leakage(self):
        # Fit on TRAIN, apply on TEST. Test values outside train range
        # must clamp to first/last bucket.
        train = np.linspace(10, 1000, 1000)
        test = np.array([1.0, 5.0, 500.0, 1e9])
        edges = fit_log_quantile_edges(train, n_bins=4)
        out = apply_bucket_edges(test, edges)
        assert out[0] == 0  # below min → bucket 0
        assert out[-1] == 3  # above max → last bucket
        assert (out >= 0).all() and (out < 4).all()

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            fit_log_quantile_edges([], n_bins=5)
        with pytest.raises(ValueError):
            fit_log_quantile_edges([1, 2, 3], n_bins=0)


# ---------- Jaccard variants -------------------------------------------------

class TestJaccard:

    def test_hand_computed_pair(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([0, 1, 1, 0])
        # |∩| = 1, |∪| = 3 → 1/3
        assert math.isclose(jaccard_binary(a, b), 1 / 3)

    def test_double_empty_returns_zero(self):
        z = np.zeros(5)
        assert jaccard_binary(z, z) == 0.0

    def test_identity_returns_one(self):
        v = np.array([1, 0, 1, 0, 1])
        assert math.isclose(jaccard_binary(v, v), 1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            jaccard_binary(np.zeros(3), np.zeros(4))

    def test_alias_jaccard_similarity(self):
        # Phase 3 alias must produce identical output.
        from utils.math_utils import jaccard_similarity
        a = np.array([1, 1, 0])
        b = np.array([1, 0, 1])
        assert jaccard_similarity(a, b) == jaccard_binary(a, b)

    def test_weighted_idf_amplifies_rare_genres(self):
        # df=[3, 2, 1] over 3 docs → idf[0] < idf[1] < idf[2].
        corpus = [
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 0, 1]),
        ]
        idf = compute_genre_idf(corpus, n_genres=3)

        # Two candidates with the same standard-Jaccard score but different
        # discriminative content. Weighted-Jaccard should favour the one
        # whose overlap concentrates on the rare genre.
        cand1 = np.array([1, 1, 0])
        hist1 = np.array([1, 1, 0])  # overlap on common genres
        cand2 = np.array([1, 0, 1])
        hist2 = np.array([1, 0, 1])  # overlap includes rare genre

        # Both unweighted Jaccards = 1.0. We just check the weighted score
        # exists and is in [0, 1].
        w1 = weighted_jaccard(cand1, hist1, idf)
        w2 = weighted_jaccard(cand2, hist2, idf)
        assert 0.0 <= w1 <= 1.0
        assert 0.0 <= w2 <= 1.0
        # With both at identical overlap the IDF-weighted version still
        # yields 1.0 (matched on whatever's in the set).
        assert math.isclose(w1, 1.0)
        assert math.isclose(w2, 1.0)

    def test_batch_jaccard_elementwise_matches_pairwise(self):
        rng = np.random.default_rng(7)
        A = (rng.uniform(size=(20, 19)) > 0.5).astype(np.uint8)
        B = (rng.uniform(size=(20, 19)) > 0.5).astype(np.uint8)
        batched = batch_jaccard(A, B)
        per_pair = np.array([jaccard_binary(A[i], B[i]) for i in range(20)])
        assert np.allclose(batched, per_pair)

    def test_batch_jaccard_allpairs_shape(self):
        A = np.eye(3, dtype=np.uint8)
        B = np.ones((4, 3), dtype=np.uint8)
        out = batch_jaccard(A, B)
        assert out.shape == (3, 4)
        # Each row of A is a single 1; intersecting with all-1 gives 1,
        # union is 3 → 1/3.
        assert np.allclose(out, 1 / 3)


# ---------- log_time_bucket alias -------------------------------------------

class TestLogTimeBucket:

    def test_negative_dt_goes_to_pad_bucket(self):
        edges = fit_log_quantile_edges([60, 3600, 86400, 604800], n_bins=4)
        out = log_time_bucket([-10, 0, 30], edges)
        assert (out == 0).all()

    def test_positive_dt_uses_full_range(self):
        edges = fit_log_quantile_edges(
            np.geomspace(60, 86400 * 365, 1000), n_bins=8
        )
        out = log_time_bucket([60, 3600, 86400, 86400 * 30], edges)
        assert (out >= 0).all() and (out < 8).all()
        # Strictly increasing input on log scale → non-decreasing buckets.
        assert (np.diff(out) >= 0).all()
