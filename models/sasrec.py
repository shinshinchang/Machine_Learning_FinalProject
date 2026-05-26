"""
SASRec (Kang & McAuley, 2018): Self-Attentive Sequential Recommendation.

This is the Phase-5 sequential-Transformer baseline. It shares the
shift-by-one supervision, the tied-output convention, and the
`score_all_items(...)` Full-Ranking interface with `GRU4RecModel`, so
`train_baseline.py::run_sasrec` can reuse the GRU4Rec training step
(both models expose `forward_full_sequence(item_ids) → (B, T, V+1)`).

Architectural choices (deliberate, documented):

  * **Pre-norm Transformer** (`norm_first=True`). The original SASRec
    code uses post-norm with manual layer placement, but pre-norm is
    materially more stable for small datasets and is what RecBole /
    most 2022+ reimplementations use. The reported NDCG@10 on Amazon
    Beauty / MovieLens differs by ~0.5% between the two — well within
    seed variance.

  * **Feed-forward dim = 4·d**. The original 2018 paper used FFN inner
    size = d (a "point-wise FFN with same width as model"). Modern
    reproducibility studies and RecBole's reference config use 4·d,
    matching standard Transformer convention. We follow the latter so
    SASRec and BERT4Rec have comparable FFN capacity (this matters for
    the SASRec-vs-BERT4Rec head-to-head being a fair architectural
    comparison rather than a capacity comparison).

  * **Learned (not sinusoidal) position embeddings**, max length
    `max_seq_len`. Matches the original paper and is standard for
    sequential-recommendation Transformers (small vocab, short
    sequences → learned positions train fine).

  * **Tied output embedding**. `logits = h @ E_i.weight.T`. Halves
    parameters and lets SASRec and GRU4Rec be compared without a
    confounding output-head capacity difference (see GRU4RecModel
    docstring for the same argument).

  * **Causal mask + key-padding mask**. The causal mask enforces
    autoregressive attention (position i can only see ≤ i). The
    key-padding mask removes PAD positions from being attended to as
    keys. With both masks, NaN softmax is impossible because every
    real query position can still attend to itself, and the dataset
    filters users with <2 real items (see SequentialNextItemDataset).

  * **PAD row zeroing**. We use `padding_idx=0` on the item embedding,
    AND explicitly zero `E_i.weight[0]` at init. Both are needed:
    `padding_idx` blocks gradient updates to that row but does not
    re-zero it after the random init, so we force-zero it once.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SASRecModel(nn.Module):
    """
    Self-Attentive Sequential Recommender (Kang & McAuley, 2018) with
    full-vocab cross-entropy training.

    Parameters
    ----------
    n_items     : int   — number of items (PAD=0 is added internally; vocab is n_items+1)
    d           : int   — embedding / model dimension (Plan §5.2: 64 / 128)
    n_heads     : int   — number of attention heads (default 2, paper's setting)
    n_layers    : int   — number of Transformer encoder blocks (default 2)
    max_seq_len : int   — max input sequence length (Plan: 50 Amazon / 200 ML)
    dropout     : float — applied to embeddings, attention, and FFN
    ff_mult     : int   — FFN inner dim = ff_mult × d (default 4)
    pad_index   : int   — PAD token id (default 0)

    Public methods (used by train_baseline.py and Full-Ranking eval)
    ----------------------------------------------------------------
    forward_full_sequence(item_ids)
        → (B, T, n_items + 1) logits — used for shift-by-one CE training
    score_all_items(user_ids, item_seq)
        → (B, n_items + 1) logits at the last non-PAD position — used at eval
    """

    def __init__(
        self,
        n_items: int,
        d: int,
        *,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        ff_mult: int = 4,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.n_items = int(n_items)
        self.d = int(d)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.pad_index = int(pad_index)
        self.vocab_size = n_items + 1  # PAD + items

        if d % n_heads != 0:
            raise ValueError(
                f"SASRec: d ({d}) must be divisible by n_heads ({n_heads})."
            )

        # --- Embeddings ---------------------------------------------------
        self.item_embedding = nn.Embedding(
            self.vocab_size, d, padding_idx=pad_index,
        )
        self.pos_embedding = nn.Embedding(max_seq_len, d)
        self.emb_layernorm = nn.LayerNorm(d, eps=1e-8)
        self.emb_dropout = nn.Dropout(dropout)

        # --- Transformer encoder (pre-norm, causal) ---------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — see docstring
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        # Final LayerNorm AFTER the stack (standard pre-norm Transformer pattern).
        self.final_layernorm = nn.LayerNorm(d, eps=1e-8)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Truncated-normal-ish init for embeddings (std=0.02, BERT/GPT convention).
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        # Force-zero the PAD row (padding_idx only blocks the gradient,
        # not the init draw).
        with torch.no_grad():
            self.item_embedding.weight[self.pad_index].zero_()

    # -------------------------------------------------------------------------
    # Internal: shared input pipeline + masks
    # -------------------------------------------------------------------------
    def _embed_with_position(self, item_ids: torch.Tensor) -> torch.Tensor:
        """item_ids: (B, T) long → (B, T, d) float."""
        B, T = item_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"SASRec: input length {T} exceeds max_seq_len={self.max_seq_len}."
            )
        positions = torch.arange(T, device=item_ids.device).unsqueeze(0).expand(B, T)
        x = self.item_embedding(item_ids) + self.pos_embedding(positions)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)
        return x

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Boolean upper-triangular mask: True (= masked-out) above the diagonal,
        False on/below. We use a bool dtype rather than a float -inf mask so
        it matches the dtype of `src_key_padding_mask` — PyTorch 1.13+ warns
        and future versions will require both masks to share a dtype.
        """
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1,
        )

    def _encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder (causal self-attention + FFN stack) and
        return per-position hidden states.

        Returns (B, T, d).
        """
        x = self._embed_with_position(item_ids)
        attn_mask = self._causal_mask(item_ids.size(1), item_ids.device)
        pad_mask = item_ids == self.pad_index  # (B, T) — True at PAD
        h = self.transformer(
            x, mask=attn_mask, src_key_padding_mask=pad_mask,
        )
        h = self.final_layernorm(h)
        return h

    # -------------------------------------------------------------------------
    # Training-time forward: per-position next-item logits
    # -------------------------------------------------------------------------
    def forward_full_sequence(
        self,
        item_ids: torch.Tensor,        # (B, T) — right-padded with 0
    ) -> torch.Tensor:
        """
        Returns (B, T, n_items + 1) — logits over the full item vocabulary
        at every position. Loss is computed by the caller with shift-by-one
        target and `ignore_index=pad_index`.
        """
        h = self._encode(item_ids)
        # Tied output: project via the item embedding table.
        logits = h @ self.item_embedding.weight.t()  # (B, T, V+1)
        return logits

    # -------------------------------------------------------------------------
    # Eval-time scoring: read the LAST non-PAD position's logits
    # -------------------------------------------------------------------------
    def encode_last(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, d) — hidden state at each row's last non-PAD position.

        Right-padding pushes the meaningful state away from position T-1
        for short sequences, so we gather per-row.
        """
        h = self._encode(item_ids)                          # (B, T, d)
        valid = (item_ids != self.pad_index).long()         # (B, T)
        last_pos = (valid.sum(dim=1) - 1).clamp_min(0)      # (B,)
        idx = last_pos.view(-1, 1, 1).expand(-1, 1, h.size(-1))   # (B, 1, d)
        h_last = h.gather(dim=1, index=idx).squeeze(1)      # (B, d)
        return h_last

    def score_all_items(
        self,
        user_ids: Optional[torch.Tensor] = None,    # unused; for API parity
        item_seq: Optional[torch.Tensor] = None,    # (B, T)
    ) -> torch.Tensor:
        """
        Return (B, n_items + 1) scores given a padded input sequence.
        Used by `validate_full_ranking`; the caller masks PAD/seen items.
        """
        if item_seq is None:
            raise ValueError(
                "SASRecModel.score_all_items requires `item_seq`; this is a "
                "sequential model with no user-only path."
            )
        h_last = self.encode_last(item_seq)                 # (B, d)
        scores = h_last @ self.item_embedding.weight.t()    # (B, V+1)
        return scores
