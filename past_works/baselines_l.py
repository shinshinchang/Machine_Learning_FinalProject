"""
Baseline models for CEINN's Phase 5 (Plan §5.1).

This module implements the three non-causal baselines that share the
same Phase-2 data format and the same Full-Ranking evaluation protocol
(`utils/metrics.py`) as CEINN. Side-by-side comparison hinges on every
baseline using identical seen-item masking, identical PAD semantics,
and identical NDCG/HR/MRR computation — so the models live in the same
repository as the rest of the project rather than in a third-party
framework.

Three baselines provided here:

    PopRec      — non-personalised popularity ranker. Plan §5.1 bullet 1.
                  Provides the lower-bound sanity check ("if anything
                  ranks below this, the eval pipeline is broken").

    BPRMFModel  — Bayesian Personalised Ranking with Matrix Factorisation
                  (Rendle et al., 2009). Standard non-sequential CF
                  baseline; user + item embeddings + pairwise BPR loss.

    GRU4RecModel — Session-based GRU recommender (Hidasi et al., 2016),
                  re-trained on the same shifted-sequence supervision as
                  CEINN. Uses tied item embeddings at the output, no
                  per-item bias, full-vocab softmax CE.

Design constraints (kept consistent with `train.py` and the CEINN
backbone in `models/sequential_backbone.py`):

  * Item indices are 1-based; index 0 is PAD. All `nn.Embedding`s use
    `padding_idx=0` so the PAD row stays at zero and contributes no
    gradient.

  * Forward signatures are deliberately *narrower* than CEINN's: no
    rating embedding, no temporal embedding, no propensity, no cost.
    Baselines must see strictly less information than CEINN — anything
    else would muddy the ablation story.

  * Every baseline exposes `score_all_items(...)` returning logits of
    shape `(B, n_items + 1)` so the same Full-Ranking evaluator can
    serve all three (see `train_baseline.py::validate_full_ranking`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1) PopRec — non-personalised popularity ranker
# =============================================================================

class PopRec(nn.Module):
    """
    Non-personalised popularity ranker (Plan §5.1).

    The model stores a `popularity` buffer of shape (n_items + 1,) whose
    value at index `i` is the log-frequency of item `i` in the training
    set. Index 0 (PAD) is set to -inf so PAD never beats any real item
    in ranking. The model has zero trainable parameters; "training"
    simply means computing the frequency vector once and freezing it.

    Why log-frequency rather than raw count?
    ----------------------------------------
    Two reasons:
      1. Numerical: raw counts can be in the tens of thousands for
         MovieLens, which makes the implied softmax temperature absurd.
         For ranking the monotonic transform doesn't matter, but for
         any downstream calibration log-counts are gentler.
      2. Convention: this matches how SASRec/BERT4Rec papers report
         their PopRec baselines.
    """

    def __init__(self, n_items: int, pad_index: int = 0) -> None:
        super().__init__()
        self.n_items = int(n_items)
        self.pad_index = int(pad_index)
        # Persistent buffer so .to(device) and .state_dict() both work.
        self.register_buffer(
            "popularity",
            torch.full((n_items + 1,), float("-inf")),
        )

    @classmethod
    def fit_from_train_seqs(
        cls,
        train_seqs,
        n_items: int,
        *,
        pad_index: int = 0,
        log_smoothing: float = 1.0,
    ) -> "PopRec":
        """
        Construct a PopRec from the Phase-2 `train_seqs.pkl`.

        Parameters
        ----------
        train_seqs : {user_idx: [(item_idx, rating_bin, dt_bin), ...]}
        n_items    : real item count; index 0 is PAD.
        log_smoothing : additive smoothing constant inside log.
                        Default 1.0 → `log(count + 1)`, standard
                        Laplace-style smoothing that gives never-seen
                        items a finite but minimal score (log 1 = 0).

        Returns
        -------
        A PopRec instance with the `popularity` buffer populated.
        """
        counts = np.zeros(n_items + 1, dtype=np.float64)
        for _u, seq in train_seqs.items():
            for triple in seq:
                iid = int(triple[0])
                if iid == pad_index:
                    continue
                if 0 < iid <= n_items:
                    counts[iid] += 1.0

        model = cls(n_items=n_items, pad_index=pad_index)
        scores = np.log(counts + log_smoothing)  # NaN-safe; counts >= 0
        scores[pad_index] = float("-inf")
        model.popularity.copy_(torch.from_numpy(scores).float())
        return model

    def score_all_items(
        self,
        user_ids: torch.Tensor,
        item_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return scores of shape (B, n_items + 1).

        Same vector broadcast across the batch — `user_ids` is taken
        only for its shape. `item_seq` is accepted for API parity but
        is unused (PopRec is non-personalised by definition).
        """
        B = user_ids.shape[0]
        return self.popularity.unsqueeze(0).expand(B, -1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Alias for score_all_items so the model is callable."""
        return self.score_all_items(*args, **kwargs)


# =============================================================================
# 2) BPR-MF — Bayesian Personalised Ranking + Matrix Factorisation
# =============================================================================

class BPRMFModel(nn.Module):
    """
    Matrix-factorisation recommender with BPR pairwise loss
    (Rendle et al., 2009).

    Score(u, i) = E_u(u) · E_i(i)

    Training: each step samples one positive (u, i_pos) from the
    training interactions and one uniform-random negative item i_neg.
    Loss is the standard
        L_BPR = -log σ(score(u, i_pos) - score(u, i_neg))

    Note that BPR-MF is **non-sequential**: it considers each
    interaction as an independent positive sample. The sequence order
    is discarded at training time; the model only sees the unordered
    set of (user, item) pairs from the training split. This is the
    correct convention for what the field calls "BPR-MF" — making it
    sequential would conflate two distinct baselines.

    User embedding stays personalised at evaluation, which is the only
    reason BPR-MF can beat PopRec at all on long-tail metrics.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        *,
        pad_index: int = 0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.d = int(d)
        self.pad_index = int(pad_index)

        self.user_embedding = nn.Embedding(
            n_users + 1, d, padding_idx=pad_index,
        )
        self.item_embedding = nn.Embedding(
            n_items + 1, d, padding_idx=pad_index,
        )
        # Small-magnitude normal init is standard for BPR-MF (large
        # values explode the early-training pairwise sigmoid). We do
        # NOT use Xavier here because we want symmetry between user and
        # item embedding magnitudes.
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=init_std)
        # Force the PAD row to exactly zero (padding_idx already does
        # this at init, but be defensive against weight loading from a
        # bad checkpoint).
        with torch.no_grad():
            self.user_embedding.weight[pad_index].zero_()
            self.item_embedding.weight[pad_index].zero_()

    # -------------------------------------------------------------------------
    # Training-time interface
    # -------------------------------------------------------------------------
    def bpr_loss(
        self,
        user_ids: torch.Tensor,   # (N,)
        pos_items: torch.Tensor,  # (N,)
        neg_items: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """
        Standard pairwise BPR loss. Returns a scalar.

        We clamp the (positive - negative) margin before passing through
        log-sigmoid to avoid `log(0)` when the margin is hugely negative
        (otherwise an unfortunate init can produce inf gradients in the
        first batch).
        """
        u = self.user_embedding(user_ids)           # (N, d)
        ip = self.item_embedding(pos_items)         # (N, d)
        in_ = self.item_embedding(neg_items)        # (N, d)

        pos_score = (u * ip).sum(dim=-1)            # (N,)
        neg_score = (u * in_).sum(dim=-1)           # (N,)
        diff = pos_score - neg_score
        # `logsigmoid` is numerically stable across the full range, so
        # no clamp needed — but we still mean-reduce to a scalar.
        return -F.logsigmoid(diff).mean()

    # -------------------------------------------------------------------------
    # Evaluation-time interface (parity with PopRec / GRU4Rec)
    # -------------------------------------------------------------------------
    def score_all_items(
        self,
        user_ids: torch.Tensor,        # (B,)
        item_seq: Optional[torch.Tensor] = None,  # unused; for API parity
    ) -> torch.Tensor:
        """
        Return (B, n_items + 1) scores: E_u(u) · E_i.T for the whole
        item vocabulary, PAD slot included. The caller (Full-Ranking
        evaluator) is responsible for masking PAD and seen items.

        We deliberately do NOT mask PAD here — keeping the shape
        consistent with the embedding table size avoids subtle off-by-
        one errors downstream. The evaluator uses `pad_index` explicitly.
        """
        u = self.user_embedding(user_ids)                 # (B, d)
        # All item rows, including PAD (= zero vector → score 0).
        scores = u @ self.item_embedding.weight.t()       # (B, V+1)
        return scores

    def forward(self, user_ids: torch.Tensor, item_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score_all_items(user_ids, item_seq)


# =============================================================================
# 3) GRU4Rec — session-based GRU next-item predictor
# =============================================================================

class GRU4RecModel(nn.Module):
    """
    Session-based GRU recommender (Hidasi et al., 2016), adapted to the
    Phase-2 data format.

    Architecture
    ------------
        item_ids ─→ E_i ─→ [Dropout] ─→ GRU(L layers) ─→ Dropout
                                         │
                                         └─→ logits = h_t · E_i.weight.T

    The output projection ties weights with the input embedding (standard
    in modern GRU4Rec implementations). This:
      * halves the parameter count
      * tends to stabilise training on small item vocabularies (~10k)
      * matches what SASRec does (so the SASRec-vs-GRU4Rec comparison
        isn't confounded by output-head capacity)

    Loss
    ----
    Full-vocab cross-entropy on the next-item target, PAD positions
    ignored — exactly the form of `utils.losses.SequentialCrossEntropy`.
    We don't import that class to keep this file self-contained, but
    the behaviour is identical.

    Why not BPR-max from the original paper?
    ----------------------------------------
    The original GRU4Rec used a TOP1 / BPR-max sampled loss because their
    target catalogue was hundreds of thousands of products. At |V| ≤ 12k
    full-vocab CE is feasible and gives a stronger gradient signal —
    SASRec/BERT4Rec authors made the same choice for the same reason.
    Plan §3.2.4 explicitly endorses full-vocab softmax at these scales.

    Crucially this means GRU4Rec here is the **strongest variant** of
    the GRU4Rec family, not the historically-original one. We say so
    in the paper.
    """

    def __init__(
        self,
        n_items: int,
        d: int,
        *,
        n_layers: int = 1,
        dropout: float = 0.1,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.n_items = int(n_items)
        self.d = int(d)
        self.n_layers = int(n_layers)
        self.pad_index = int(pad_index)

        self.item_embedding = nn.Embedding(
            n_items + 1, d, padding_idx=pad_index,
        )
        # The original paper used 1 layer; we expose `n_layers` because
        # the search space in `configs/baselines.yaml` includes deeper
        # variants (for Stage-1 sanity).
        # GRU's `dropout` kwarg only applies BETWEEN layers, so for
        # n_layers=1 we apply dropout manually outside the GRU.
        self.gru = nn.GRU(
            input_size=d,
            hidden_size=d,
            num_layers=n_layers,
            batch_first=True,
            dropout=(dropout if n_layers > 1 else 0.0),
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Sensible init: small Xavier-uniform on embeddings, default
        # PyTorch GRU init (which is uniform 1/sqrt(d)) on the recurrence.
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[pad_index].zero_()

    # -------------------------------------------------------------------------
    # Forward: per-position logits over the full vocab
    # -------------------------------------------------------------------------
    def forward_full_sequence(
        self,
        item_ids: torch.Tensor,        # (B, T) — padded with 0
    ) -> torch.Tensor:
        """
        Run GRU over the sequence and return per-position logits.

        Returns
        -------
        logits : (B, T, n_items + 1)
        """
        emb = self.item_embedding(item_ids)              # (B, T, d)
        emb = self.emb_dropout(emb)
        h, _ = self.gru(emb)                             # (B, T, d)
        h = self.out_dropout(h)
        # Tied output: project back via the embedding table.
        logits = h @ self.item_embedding.weight.t()      # (B, T, V+1)
        return logits

    def encode_last(
        self,
        item_ids: torch.Tensor,        # (B, T)
    ) -> torch.Tensor:
        """
        Run GRU and return ONLY the hidden state at the last non-PAD
        position, for use at evaluation. Returns (B, d).

        We can't just take `h[:, -1]` because the right-padding pushes
        the meaningful state away from position T-1 for short
        sequences. Instead we compute per-row sequence lengths and
        gather.
        """
        emb = self.item_embedding(item_ids)              # (B, T, d)
        # No dropout at eval time (the caller is expected to have
        # `model.eval()`, but this is defensive).
        with torch.no_grad():
            # Sequence lengths = position of the LAST non-PAD token,
            # i.e., `count of non-PAD - 1`. We use the count directly
            # below for gather (1-indexed length - 1 = 0-indexed last).
            valid = (item_ids != self.pad_index).long()  # (B, T)
            last_pos = valid.sum(dim=1) - 1               # (B,)
            last_pos = last_pos.clamp_min(0)              # all-PAD → 0

        h, _ = self.gru(emb)                             # (B, T, d)
        # Gather row-by-row at last_pos.
        idx = last_pos.view(-1, 1, 1).expand(-1, 1, h.shape[-1])  # (B, 1, d)
        h_last = h.gather(dim=1, index=idx).squeeze(1)   # (B, d)
        return h_last

    def score_all_items(
        self,
        user_ids: Optional[torch.Tensor] = None,    # unused; for API parity
        item_seq: torch.Tensor = None,              # (B, T)
    ) -> torch.Tensor:
        """
        Return (B, n_items + 1) scores given a padded input sequence.

        Used by the Full-Ranking evaluator. The caller masks PAD and
        seen items downstream.
        """
        if item_seq is None:
            raise ValueError(
                "GRU4RecModel.score_all_items requires `item_seq`; "
                "this model is sequential and has no user-only path."
            )
        h_last = self.encode_last(item_seq)              # (B, d)
        scores = h_last @ self.item_embedding.weight.t() # (B, V+1)
        return scores


# =============================================================================
# Factory helper (used by `train_baseline.py`)
# =============================================================================

def build_baseline(
    name: str,
    *,
    n_users: int,
    n_items: int,
    d: int,
    n_layers: int = 1,
    dropout: float = 0.1,
    pad_index: int = 0,
) -> nn.Module:
    """
    Dispatch a baseline name to its constructor with the correct kwargs.
    Centralised so `train_baseline.py` doesn't repeat the if/else.

    Note: PopRec is created here as a "blank" instance — its real
    state is set by `fit_from_train_seqs` in the training script.
    Callers should treat the returned PopRec as a typed placeholder.
    """
    key = name.lower()
    if key in ("poprec", "pop", "popularity"):
        return PopRec(n_items=n_items, pad_index=pad_index)
    if key in ("bpr_mf", "bprmf", "bpr-mf"):
        return BPRMFModel(
            n_users=n_users, n_items=n_items, d=d, pad_index=pad_index,
        )
    if key in ("gru4rec", "gru"):
        return GRU4RecModel(
            n_items=n_items, d=d, n_layers=n_layers, dropout=dropout,
            pad_index=pad_index,
        )
    raise ValueError(
        f"Unknown baseline name {name!r}. Expected one of: "
        f"poprec, bpr_mf, gru4rec."
    )
