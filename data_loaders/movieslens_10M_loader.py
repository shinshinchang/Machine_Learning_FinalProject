"""
MovieLens 10M loader  —  reads Phase-2 artefacts and exposes the
training-time lookups needed by §3.2 (dynamic Z_i(t)) and §3.3 (Genre
Jaccard for GenreRed).

Storage convention
------------------
dynamic_Z, genre_red, and genre_red_idf were stored in Phase 2 as
"parallel arrays" plus a `row_index: {(user_idx, position): array_idx}`
dict. This makes them O(1) lookup by user/position while staying compact
in memory (versus a Python dict of 10M floats which would blow past
600 MB of overhead).

The loader provides typed accessors hiding this representation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class MovieLens10MLoader:
    """Eager loader for MovieLens 10M's Phase-2 artefacts."""

    train_seqs: Dict[int, List[Tuple[int, int, int]]]
    val_seqs:   Dict[int, Tuple[int, int, int]]
    test_seqs:  Dict[int, Tuple[int, int, int]]
    item_genre: Dict[int, np.ndarray]      # {item_idx: 19-dim float32 binary}

    # Parallel arrays + row index (Phase-2 storage choice)
    _dynZ_values:    np.ndarray            # int64, len = n_train_rows
    _genreRed_values: np.ndarray           # float32, len = n_train_rows
    _genreRedIdf_values: np.ndarray        # float32, len = n_train_rows
    _row_index:      Dict[Tuple[int, int], int]
    idf_weights:     np.ndarray            # float64, length n_genres
    genre_vocab:     List[str]             # ordered by index

    id_maps:    Dict[str, Any]
    vocab:      Dict[str, int]
    bucket_edges: Dict[str, Any]

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_directory(cls, processed_dir: str | Path) -> "MovieLens10MLoader":
        d = Path(processed_dir)

        def _pkl(name: str) -> Any:
            with open(d / name, "rb") as f:
                return pickle.load(f)

        train_seqs   = _pkl("train_seqs.pkl")
        val_seqs     = _pkl("val_seqs.pkl")
        test_seqs    = _pkl("test_seqs.pkl")
        item_genre   = _pkl("item_genre.pkl")
        dynZ         = _pkl("dynamic_Z.pkl")
        gr_std       = _pkl("genre_red.pkl")
        gr_idf       = _pkl("genre_red_idf.pkl")
        id_maps      = _pkl("id_maps.pkl")
        bucket_edges = _pkl("bucket_edges.pkl")
        with open(d / "vocab_sizes.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # All three precomputed arrays share the same row_index.
        assert dynZ["row_index"] is gr_std["row_index"] or \
               dynZ["row_index"] == gr_std["row_index"], \
               "Phase-2 row_index alignment broken between dyn_Z and genre_red"

        loader = cls(
            train_seqs=train_seqs, val_seqs=val_seqs, test_seqs=test_seqs,
            item_genre=item_genre,
            _dynZ_values=dynZ["values"],
            _genreRed_values=gr_std["values"],
            _genreRedIdf_values=gr_idf["values"],
            _row_index=dynZ["row_index"],
            idf_weights=gr_idf["idf_weights"],
            genre_vocab=gr_idf["genre_vocab"],
            id_maps=id_maps, vocab=vocab, bucket_edges=bucket_edges,
        )
        loader._sanity_check()
        return loader

    # -------------------------------------------------------------------------
    # Item-level lookups
    # -------------------------------------------------------------------------
    def genre_vector_of(self, item_idx: int) -> np.ndarray:
        """Return the 19-dim binary genre vector for one item."""
        if item_idx <= 0:
            raise KeyError(f"item_idx {item_idx} is PAD/invalid")
        return self.item_genre[item_idx]

    def bulk_genre_matrix(self) -> np.ndarray:
        """
        Stack all item genre vectors into a (n_items + 1, n_genres) matrix
        indexed by item_idx (0 = PAD row, all zeros). Useful for vectorised
        GenreRed computations at inference time.
        """
        n = int(self.vocab["n_items"])
        g = int(self.vocab["n_genres"])
        out = np.zeros((n + 1, g), dtype=np.float32)
        for idx, v in self.item_genre.items():
            out[idx] = v
        return out

    # -------------------------------------------------------------------------
    # (user, position) → precomputed scalars
    # -------------------------------------------------------------------------
    def Z_at(self, user_idx: int, position: int) -> int:
        """Cumulative interaction count of the chosen item strictly BEFORE
        position `position` in user `user_idx`'s training sequence."""
        return int(self._dynZ_values[self._row_index[(user_idx, position)]])

    def genre_red_at(self, user_idx: int, position: int) -> float:
        """Standard Jaccard similarity between the candidate's genre vector
        and the user's history-union at this position."""
        return float(self._genreRed_values[self._row_index[(user_idx, position)]])

    def genre_red_idf_at(self, user_idx: int, position: int) -> float:
        """IDF-weighted Jaccard variant (used by ablation ML4)."""
        return float(self._genreRedIdf_values[self._row_index[(user_idx, position)]])

    # -------------------------------------------------------------------------
    # Bulk access for a single user (preferred in training; avoids dict
    # lookups in inner loops)
    # -------------------------------------------------------------------------
    def Z_array_for_user(self, user_idx: int) -> np.ndarray:
        """1-D array aligned with `train_seqs[user_idx]`."""
        seq = self.train_seqs[user_idx]
        out = np.empty(len(seq), dtype=np.int64)
        for pos in range(len(seq)):
            out[pos] = self._dynZ_values[self._row_index[(user_idx, pos)]]
        return out

    def genre_red_array_for_user(
        self, user_idx: int, idf: bool = False,
    ) -> np.ndarray:
        seq = self.train_seqs[user_idx]
        out = np.empty(len(seq), dtype=np.float32)
        src = self._genreRedIdf_values if idf else self._genreRed_values
        for pos in range(len(seq)):
            out[pos] = src[self._row_index[(user_idx, pos)]]
        return out

    # -------------------------------------------------------------------------
    # Sequence iteration helpers
    # -------------------------------------------------------------------------
    def iter_train_users(self):
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
        n_genres = int(self.vocab["n_genres"])
        for idx, v in self.item_genre.items():
            assert 0 < idx <= n_items, f"item_idx {idx} out of range"
            assert v.shape == (n_genres,), \
                f"genre vector for {idx} has shape {v.shape}, expected ({n_genres},)"

        # row_index size matches the parallel arrays.
        assert len(self._row_index) == len(self._dynZ_values), (
            f"row_index size {len(self._row_index)} "
            f"!= dyn_Z len {len(self._dynZ_values)}"
        )
        assert len(self._dynZ_values) == len(self._genreRed_values), (
            "dyn_Z and genre_red array lengths must match"
        )
        assert len(self._genreRedIdf_values) == len(self._genreRed_values), (
            "genre_red and genre_red_idf array lengths must match"
        )
        # GenreRed must be in [0, 1].
        if self._genreRed_values.size > 0:
            assert float(self._genreRed_values.min()) >= 0.0
            assert float(self._genreRed_values.max()) <= 1.0 + 1e-6
            assert float(self._genreRedIdf_values.min()) >= 0.0
            assert float(self._genreRedIdf_values.max()) <= 1.0 + 1e-6
