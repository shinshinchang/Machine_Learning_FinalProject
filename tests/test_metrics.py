"""
Unit tests for utils/metrics.py (Plan §3.3 + §3.4.2).

For each metric, we hand-compute the expected value on a tiny toy
example (3 users × 5 items style) and check the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils.metrics import (
    compute_full_ranking,
    confounding_auc,
    group_metrics,
    hr_at_k,
    mrr,
    ndcg_at_k,
    standard_topk_report,
)


# ---------- compute_full_ranking --------------------------------------------

class TestFullRanking:

    def test_basic_no_seen(self):
        # 5 items (idx 0..4). PAD = 0.
        # scores: idx 1 = 0.1, idx 2 = 0.9, idx 3 = 0.5, idx 4 = 0.3
        scores = np.array([0.0, 0.1, 0.9, 0.5, 0.3])
        # Target idx 3 → only idx 2 ranks above (0.9 > 0.5).
        rank = compute_full_ranking(scores, target_item=3)
        assert rank == 2

    def test_target_first(self):
        scores = np.array([0.0, 0.9, 0.1, 0.05])
        rank = compute_full_ranking(scores, target_item=1)
        assert rank == 1

    def test_seen_items_masked(self):
        scores = np.array([0.0, 0.9, 0.8, 0.5])
        # Without masking, target=3 would rank 3rd (0.9, 0.8 above).
        # Mask {1, 2} → target=3 ranks 1st.
        rank = compute_full_ranking(scores, target_item=3, seen_items={1, 2})
        assert rank == 1

    def test_pad_always_excluded(self):
        scores = np.array([1e9, 0.1, 0.2])  # PAD has the highest score
        rank = compute_full_ranking(scores, target_item=2)
        assert rank == 1  # PAD must be ignored

    def test_ties_strict_above(self):
        scores = np.array([0.0, 0.5, 0.5, 0.5])
        # Target gets the same score as two others — no strict beats → rank 1.
        rank = compute_full_ranking(scores, target_item=2)
        assert rank == 1

    def test_pad_target_raises(self):
        scores = np.array([0.0, 0.5])
        with pytest.raises(ValueError):
            compute_full_ranking(scores, target_item=0)


# ---------- ndcg_at_k --------------------------------------------------------

class TestNDCG:

    def test_hand_computed(self):
        # 3 users with ranks 1, 2, 11. NDCG@10:
        # 1/log2(1+1) = 1.0
        # 1/log2(2+1) = 0.6309...
        # rank 11 > 10 → 0
        ranks = [1, 2, 11]
        per_user = ndcg_at_k(ranks, k=10)
        assert math.isclose(per_user[0], 1.0)
        assert math.isclose(per_user[1], 1.0 / math.log2(3))
        assert per_user[2] == 0.0

    def test_at_k_5(self):
        ranks = [3, 5, 6]
        per_user = ndcg_at_k(ranks, k=5)
        # rank=3 → 1/log2(4), rank=5 → 1/log2(6), rank=6 → 0
        assert math.isclose(per_user[0], 1.0 / math.log2(4))
        assert math.isclose(per_user[1], 1.0 / math.log2(6))
        assert per_user[2] == 0.0

    def test_perfect_ranks(self):
        # All targets in position 1 → NDCG@K = 1.0 for any K >= 1.
        ranks = [1, 1, 1]
        for k in (5, 10, 20):
            assert np.allclose(ndcg_at_k(ranks, k=k), 1.0)


# ---------- hr_at_k ----------------------------------------------------------

class TestHR:

    def test_hand_computed(self):
        ranks = [1, 5, 10, 11, 100]
        # HR@10 → 4 hits out of 5 → 0.8
        per_user = hr_at_k(ranks, k=10)
        assert per_user.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
        assert per_user.mean() == 0.6  # 3 out of 5

    def test_at_k_5(self):
        ranks = [3, 4, 5, 6]
        per_user = hr_at_k(ranks, k=5)
        assert per_user.tolist() == [1.0, 1.0, 1.0, 0.0]


# ---------- mrr --------------------------------------------------------------

class TestMRR:

    def test_hand_computed(self):
        ranks = [1, 2, 4]
        per_user = mrr(ranks)
        # Mean = (1 + 0.5 + 0.25) / 3 = 0.5833...
        assert math.isclose(per_user.mean(), (1 + 0.5 + 0.25) / 3, abs_tol=1e-6)


# ---------- group_metrics ----------------------------------------------------

class TestGroupMetrics:

    def test_grouping_splits_correctly(self):
        ranks = [1, 5, 50, 2, 100]
        items = [10, 20, 30, 40, 50]
        labels = {10: "head", 20: "head", 30: "tail", 40: "torso", 50: "tail"}
        out = group_metrics(ranks, items, labels, k=10)
        assert out["head"]["n_users"] == 2
        assert out["torso"]["n_users"] == 1
        assert out["tail"]["n_users"] == 2

    def test_missing_group_returns_nan(self):
        ranks = [1, 2]
        items = [10, 20]
        # Both items map to head only.
        labels = {10: "head", 20: "head"}
        out = group_metrics(ranks, items, labels, k=10)
        assert out["head"]["n_users"] == 2
        assert math.isnan(out["torso"]["hr@k"])
        assert math.isnan(out["tail"]["hr@k"])
        assert out["torso"]["n_users"] == 0


# ---------- standard_topk_report --------------------------------------------

class TestStandardReport:

    def test_keys_present(self):
        ranks = [1, 3, 7]
        out = standard_topk_report(ranks, ks=(5, 10, 20))
        for k in (5, 10, 20):
            assert f"ndcg@{k}" in out
            assert f"hr@{k}" in out
        assert "mrr" in out


# ---------- confounding_auc -------------------------------------------------

class TestConfoundingAUC:

    def test_perfectly_predictive_features(self):
        # If h_states encode Z perfectly, AUC should be ≈ 1.
        rng = np.random.default_rng(0)
        n_per_class = 50
        z = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class), np.full(n_per_class, 2)]).astype(np.int64)
        # h = one-hot of z + small noise → strongly predictive.
        h = np.eye(3)[z] + 0.01 * rng.standard_normal((150, 3))
        auc = confounding_auc(h, z, n_folds=5)
        assert auc > 0.95

    def test_random_features_near_chance(self):
        # If h is independent of Z, AUC should be near 1/K random baseline,
        # but macro-OvR AUC actually approaches 0.5 (not 1/K), because
        # OvR collapses each class to binary.
        rng = np.random.default_rng(123)
        z = rng.integers(0, 3, size=300).astype(np.int64)
        h = rng.standard_normal((300, 8))
        auc = confounding_auc(h, z, n_folds=5)
        assert 0.35 < auc < 0.65  # near chance

    def test_handles_rare_class_dropping(self):
        # Class 2 appears only 3 times → dropped from CV (needs >=5).
        z = np.array([0] * 50 + [1] * 50 + [2] * 3, dtype=np.int64)
        h = np.eye(3)[z] + 0.1 * np.random.default_rng(0).standard_normal((103, 3))
        # Should not raise. The remaining binary problem may yield a
        # finite AUC, or NaN if every fold ends up degenerate. Either is
        # acceptable behaviour — what matters is no exception.
        auc = confounding_auc(h, z, n_folds=5)
        assert math.isnan(auc) or (0.0 <= auc <= 1.0)
