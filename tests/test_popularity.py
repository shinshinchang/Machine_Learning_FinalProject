"""Unit tests for utils/popularity.py."""

from __future__ import annotations

import pytest

from utils.popularity import build_popularity_groups


class TestBuildPopularityGroups:

    def test_basic_three_way_split(self):
        # 10 items, with counts 1..10. P80 = 8.2 → items {9, 10} are head;
        # P20 = 2.8 → items {1, 2} are tail; rest is torso.
        train_seqs = {}
        for iid in range(1, 11):
            # Generate `iid` interactions of item `iid` by a single fake user.
            train_seqs[iid] = [(iid, 1, 1)] * iid

        labels, summary = build_popularity_groups(train_seqs, n_items=10)
        assert summary["n_items"] == 10
        assert summary["n_head"] >= 1
        assert summary["n_tail"] >= 1
        # Highest-count item must be head.
        assert labels[10] == "head"
        # Lowest-count item must be tail.
        assert labels[1] == "tail"

    def test_pad_index_skipped(self):
        # Even if PAD slot accidentally appears in a sequence, it must be
        # ignored.
        train_seqs = {1: [(0, 1, 1), (1, 1, 1), (2, 1, 1)]}
        labels, _ = build_popularity_groups(train_seqs, n_items=2)
        # No KeyError; PAD doesn't appear in labels (only real items).
        assert set(labels.keys()) == {1, 2}

    def test_bad_quantiles_raise(self):
        with pytest.raises(ValueError):
            build_popularity_groups(
                {1: [(1, 1, 1)]}, n_items=1,
                head_quantile=0.5, tail_quantile=0.8,
            )

    def test_label_coverage(self):
        # Every real item in 1..n_items must receive a label.
        train_seqs = {1: [(1, 1, 1), (2, 1, 1), (3, 1, 1)] * 5}
        labels, _ = build_popularity_groups(train_seqs, n_items=3)
        for iid in (1, 2, 3):
            assert iid in labels
            assert labels[iid] in ("head", "torso", "tail")
