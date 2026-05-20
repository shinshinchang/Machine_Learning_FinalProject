"""
Amazon Beauty loader  —  reads Phase-2 artefacts and exposes the
training-time lookups needed by §3.3 (explicit economic cost) and §3.2
(salesRank confounder Z).

This module performs NO further preprocessing. It is a thin, well-typed
wrapper over the pickles in `data/processed/amazon_beauty/`. Each
lookup is O(1) per item.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class AmazonBeautyLoader:
    """
    Eager loader for Amazon Beauty's Phase-2 artefacts.

    Attributes
    ----------
    train_seqs : {user_idx: [(item_idx, rating_bin, dt_bin), ...]}
        Sequences truncated to `max_seq_len`, sorted oldest-to-newest.
    val_seqs   : {user_idx: (item_idx, rating_bin, dt_bin)}
    test_seqs  : {user_idx: (item_idx, rating_bin, dt_bin)}

    item_meta  : {item_idx: {"cat": int, "brand": int,
                              "log_price": float, "Z": int}}
        Fully-populated for every item id; missing values were imputed
        during Phase 2 (so callers never have to handle None).

    id_maps    : full bidirectional remap dictionaries.
    vocab      : flat dict of vocab sizes (n_users, n_items, n_cats, ...).
    bucket_edges : Δt and salesRank edges used at fit time. Saved so that
                   any future inference-time bucketisation matches Phase 2.
    """

    train_seqs: Dict[int, List[Tuple[int, int, int]]]
    val_seqs:   Dict[int, Tuple[int, int, int]]
    test_seqs:  Dict[int, Tuple[int, int, int]]
    item_meta:  Dict[int, Dict[str, Any]]
    id_maps:    Dict[str, Any]
    vocab:      Dict[str, int]
    bucket_edges: Dict[str, Any]

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_directory(cls, processed_dir: str | Path) -> "AmazonBeautyLoader":
        """Materialise the loader by reading all pickles + the vocab JSON."""
        d = Path(processed_dir)

        def _pkl(name: str) -> Any:
            with open(d / name, "rb") as f:
                return pickle.load(f)

        train_seqs   = _pkl("train_seqs.pkl")
        val_seqs     = _pkl("val_seqs.pkl")
        test_seqs    = _pkl("test_seqs.pkl")
        item_meta    = _pkl("item_meta.pkl")
        id_maps      = _pkl("id_maps.pkl")
        bucket_edges = _pkl("bucket_edges.pkl")
        with open(d / "vocab_sizes.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

        loader = cls(
            train_seqs=train_seqs, val_seqs=val_seqs, test_seqs=test_seqs,
            item_meta=item_meta, id_maps=id_maps, vocab=vocab,
            bucket_edges=bucket_edges,
        )
        loader._sanity_check()
        return loader

    # -------------------------------------------------------------------------
    # Per-item lookups (used by §3.3 economic-cost computation)
    # -------------------------------------------------------------------------
    def get_meta(self, item_idx: int) -> Dict[str, Any]:
        """Return the full meta dict for one item; raises on PAD or unknown id."""
        if item_idx <= 0:
            raise KeyError(f"item_idx {item_idx} is PAD/invalid")
        return self.item_meta[item_idx]

    def category_of(self, item_idx: int) -> int:
        return self.item_meta[item_idx]["cat"]

    def brand_of(self, item_idx: int) -> int:
        return self.item_meta[item_idx]["brand"]

    def log_price_of(self, item_idx: int) -> float:
        return self.item_meta[item_idx]["log_price"]

    def Z_of(self, item_idx: int) -> int:
        """Return the salesRank-derived confounder bucket Z_i ∈ {0, ..., n_Z_bins-1}."""
        return self.item_meta[item_idx]["Z"]

    # -------------------------------------------------------------------------
    # Vectorised bulk lookups (preferred in training loops)
    # -------------------------------------------------------------------------
    def bulk_meta_arrays(self) -> Dict[str, np.ndarray]:
        """
        Materialise four arrays indexed by item_idx (0 = PAD slot).

        Shapes: (n_items + 1,). The PAD row stores zeros / 0.0 which is
        safe to embed because layers (Embedding / LayerNorm / ...) tolerate
        zeros at the PAD index.
        """
        n = int(self.vocab["n_items"])
        cat   = np.zeros(n + 1, dtype=np.int64)
        brand = np.zeros(n + 1, dtype=np.int64)
        lp    = np.zeros(n + 1, dtype=np.float32)
        Z     = np.zeros(n + 1, dtype=np.int64)
        for idx, m in self.item_meta.items():
            cat[idx]   = int(m["cat"])
            brand[idx] = int(m["brand"])
            lp[idx]    = float(m["log_price"])
            Z[idx]     = int(m["Z"])
        return {"cat": cat, "brand": brand, "log_price": lp, "Z": Z}

    # -------------------------------------------------------------------------
    # Sequence iteration helpers
    # -------------------------------------------------------------------------
    def iter_train_users(self):
        """Yield (user_idx, sequence) for every user with a non-empty train seq."""
        for u, seq in self.train_seqs.items():
            if seq:
                yield u, seq

    def n_train_users(self) -> int:
        return sum(1 for _u, s in self.train_seqs.items() if s)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def _sanity_check(self) -> None:
        n_items = int(self.vocab["n_items"])
        assert len(self.item_meta) == n_items, (
            f"item_meta size {len(self.item_meta)} != n_items {n_items}"
        )
        for idx, m in self.item_meta.items():
            assert 0 < idx <= n_items, f"item_idx {idx} out of range"
            assert {"cat", "brand", "log_price", "Z"} <= m.keys()
        # Sequence ids must respect [1, n_items].
        for u, seq in self.train_seqs.items():
            for (iid, _r, _d) in seq:
                assert 1 <= iid <= n_items, f"train iid {iid} out of range"
