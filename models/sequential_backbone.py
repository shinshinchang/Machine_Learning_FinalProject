"""
Module 3.1 — Sequential Backbone (Plan §3.1, Phase 4 §4.1).

Architectural commitment
------------------------
This module enforces the *feature isolation principle* from §3.1.2: the
latent state h_t is built EXCLUSIVELY from
    (item_ids, rating_ids, dt_ids)
i.e. interaction identity, explicit preference intensity, and temporal
log-bucket. Price, salesRank, category, brand, genres, and ANY other
semantic / economic side-feature are deliberately excluded so that the
downstream causal deconfounder (§3.2) has a clean canvas on which to
enforce h_t ⊥ Z.

The same `item_embedding` table is later reused — by reference — in
`economics_utility.py` to compute V(u, i, t) = f(h_t, E_i(i)). This
sharing is a deliberate design choice (§3.3.5): the value head uses
the pure ID embedding learned by the backbone, NOT a freshly-allocated
copy. Re-using the same `nn.Embedding` object means parameters are
tied at the storage level and gradients flow through both paths during
backprop.

Phase 4 acceptance criteria (§4.1.4)
------------------------------------
1. Output shape (batch_size, d) when `return_full_sequence=False`.
2. Causal masking: modifying input at positions t+1..n MUST NOT change
   the encoder output at position t. The unit test in
   `tests/test_models_phase4.py` verifies this by patching the second
   half of a batch and asserting bitwise equality on the first half.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# =============================================================================
# Embedding helpers
# =============================================================================

def _xavier_init_with_pad(emb: nn.Embedding, pad_index: int = 0) -> None:
    """
    Xavier-uniform initialise an Embedding table while keeping the PAD
    row pinned to zero. nn.Embedding's own `padding_idx` argument already
    enforces a zero row at construction time AND blocks its gradient,
    but Xavier-resetting `weight` afterwards re-randomises the pad row.
    We therefore re-zero it explicitly here.
    """
    nn.init.xavier_uniform_(emb.weight)
    with torch.no_grad():
        emb.weight[pad_index].zero_()


# =============================================================================
# §3.1.3  Joint embedding: e_k = E_i(i_k) + E_r(r_k) + E_t(Δt_k)
# =============================================================================

class JointEmbedding(nn.Module):
    """
    Builds the per-token interaction embedding e_k as the SUM of three
    aligned-dimension lookups (Plan §3.1.3):

        e_k = E_i(i_k) + E_r(r_k) + E_t(Δt_k)

    Sizes (PAD index = 0 included):
      - ItemEmbedding   : (n_items + 1, d)
      - RatingEmbedding : (n_rating_bins + 1, d)        [+1 = PAD]
      - TemporalEmbedding: (n_dt_bins + 1, d)           [+1 = PAD]

    Parameters
    ----------
    n_items        : total real items (without PAD). The table allocates
                     n_items + 1 rows.
    n_rating_bins  : as recorded in `vocab_sizes.json["n_rating_bins"]`.
                     For Amazon Beauty this is 6 (0=PAD, 1..5=stars);
                     for MovieLens 11 (0=PAD, 1..10=half-stars).
    n_dt_bins      : as recorded in `vocab_sizes.json["n_dt_bins"]`
                     (Amazon=32, MovieLens=64). Bin 0 = PAD bin.
    d              : embedding dimension, shared across the three tables
                     (required because they are summed).
    pad_index      : PAD index, default 0 across all three.

    Notes
    -----
    All three sub-tables use `padding_idx=pad_index`, which zeros the
    PAD row and blocks its gradient. After Xavier-uniform init we
    re-zero the PAD row explicitly (see `_xavier_init_with_pad`) because
    nn.init.xavier_uniform_ overwrites every row including PAD.
    """

    def __init__(
        self,
        n_items: int,
        n_rating_bins: int,
        n_dt_bins: int,
        d: int,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.d = d

        # Note: nn.Embedding's first arg is `num_embeddings` which already
        # accounts for the PAD slot — we MUST pass n_items + 1 (n_rating + 1,
        # n_dt + 1) because Phase-2 reserves index 0 as PAD AND stores rating
        # bins in 1..n_rating_bins. The total number of valid indices is
        # therefore n_rating_bins + 1.
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items + 1,
            embedding_dim=d,
            padding_idx=pad_index,
        )
        self.rating_embedding = nn.Embedding(
            num_embeddings=n_rating_bins + 1,
            embedding_dim=d,
            padding_idx=pad_index,
        )
        self.dt_embedding = nn.Embedding(
            num_embeddings=n_dt_bins + 1,
            embedding_dim=d,
            padding_idx=pad_index,
        )

        _xavier_init_with_pad(self.item_embedding, pad_index)
        _xavier_init_with_pad(self.rating_embedding, pad_index)
        _xavier_init_with_pad(self.dt_embedding, pad_index)

    def forward(
        self,
        item_ids: torch.Tensor,
        rating_ids: torch.Tensor,
        dt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        item_ids, rating_ids, dt_ids : LongTensor of shape (B, T).
                                       Padded positions hold pad_index (=0).

        Returns
        -------
        e : FloatTensor of shape (B, T, d). PAD positions are exactly the
            zero vector (every sub-table has its PAD row = 0).
        """
        if item_ids.shape != rating_ids.shape or item_ids.shape != dt_ids.shape:
            raise ValueError(
                f"JointEmbedding: shape mismatch — item={item_ids.shape}, "
                f"rating={rating_ids.shape}, dt={dt_ids.shape}"
            )
        return (
            self.item_embedding(item_ids)
            + self.rating_embedding(rating_ids)
            + self.dt_embedding(dt_ids)
        )


# =============================================================================
# §3.1.4  Causal-masked Transformer encoder
# =============================================================================

class SequentialBackbone(nn.Module):
    """
    Pure-ID Transformer encoder with strict causal masking
    (Plan §3.1.4, §3.1.5).

    Pipeline:
        (item_ids, rating_ids, dt_ids)
            → JointEmbedding → e ∈ R^{B×T×d}
            → L stacked TransformerEncoder layers with causal mask M
            → take output at last valid (non-PAD) position → h_t ∈ R^{B×d}

    Parameters
    ----------
    n_items, n_rating_bins, n_dt_bins, pad_index : as in JointEmbedding.
    d        : model dimension (shared by embeddings and Transformer).
    n_heads  : number of attention heads. d must be divisible by n_heads.
    n_layers : number of stacked TransformerEncoder layers (L in §3.1.4).
    d_ff     : feed-forward hidden dim inside each layer (default 4*d).
    dropout  : applied inside each layer.
    max_seq_len : optional positional embedding cap; required for the
                  *learned* positional table. Pass the value from
                  `vocab_sizes.json["max_seq_len"]` (Amazon=50, ML=200).
                  If None, no positional embedding is added — the
                  attention then relies solely on E_t(Δt) for ordering,
                  which is acceptable per §3.1.3 (Δt encodes order).
                  Adding a learned positional embedding is, however,
                  the standard SASRec setup and recommended.

    Causal mask
    -----------
    PyTorch's `nn.MultiheadAttention` / `TransformerEncoderLayer`
    accepts an *additive* float mask of shape (T, T): entries that
    should be allowed = 0.0, entries that should be blocked = -inf.
    We build it once on the first forward call and cache by tensor
    size and device. Re-emitted on shape / device change.

    Output extraction
    -----------------
    h_t is the output at each batch's LAST NON-PAD position. We
    determine this from the input `item_ids` (PAD positions have
    item_id == pad_index). This generalises gracefully to right-padded
    batches with variable sequence lengths.
    """

    def __init__(
        self,
        n_items: int,
        n_rating_bins: int,
        n_dt_bins: int,
        d: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: Optional[int] = None,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
        self.d = d
        self.pad_index = pad_index
        self.max_seq_len = max_seq_len

        self.joint_embedding = JointEmbedding(
            n_items=n_items,
            n_rating_bins=n_rating_bins,
            n_dt_bins=n_dt_bins,
            d=d,
            pad_index=pad_index,
        )

        # Optional learned positional embedding (Phase-4 default ON when
        # max_seq_len is given). PAD position (index 0) is NOT special at
        # the positional table level — every position contributes; PAD
        # tokens still get neutralised by the token-side zero embedding.
        if max_seq_len is not None:
            self.pos_embedding = nn.Embedding(max_seq_len, d)
            nn.init.xavier_uniform_(self.pos_embedding.weight)
        else:
            self.pos_embedding = None

        # ---------------------------------------------------------------
        # batch_first=True so we work in (B, T, d) the whole time.
        # norm_first=False (the default Pre-LN vs Post-LN question — the
        # plan specifies the Post-LN order "Attention → Add & Norm →
        # FFN → Add & Norm" in §4.1.2, which corresponds to norm_first=False).
        # ---------------------------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=(d_ff if d_ff is not None else 4 * d),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

        # Cached causal mask, rebuilt on shape / device / dtype change.
        self._cached_mask_size: int = -1
        self._cached_mask_device: torch.device = torch.device("cpu")
        self._cached_mask_dtype: torch.dtype = torch.float32
        self.register_buffer("_causal_mask", torch.zeros(0), persistent=False)

    # -------------------------------------------------------------------------
    # Convenience accessor — used by economics_utility.py to share the
    # SAME item embedding table without re-creating it.
    # -------------------------------------------------------------------------
    @property
    def item_embedding(self) -> nn.Embedding:
        return self.joint_embedding.item_embedding

    # -------------------------------------------------------------------------
    # Internal: causal mask
    # -------------------------------------------------------------------------
    def _get_causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Return an additive causal mask of shape (T, T):
            mask[i, j] = 0     if j ≤ i  (allowed)
            mask[i, j] = -inf  if j > i  (future, blocked)

        Cached across calls when (T, device, dtype) are unchanged.
        """
        if (
            self._cached_mask_size == T
            and self._cached_mask_device == device
            and self._cached_mask_dtype == dtype
            and self._causal_mask.numel() > 0
        ):
            return self._causal_mask

        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        # triu with diagonal=1 keeps upper triangle strictly above the
        # main diagonal as -inf; everything on or below diagonal stays 0.
        mask = torch.triu(mask, diagonal=1)
        # Cache as a buffer attribute (no register_buffer here because we
        # need to mutate it; the buffer registration above is just a
        # placeholder so it's tracked by `.to(device)` semantics).
        self._causal_mask = mask
        self._cached_mask_size = T
        self._cached_mask_device = device
        self._cached_mask_dtype = dtype
        return mask

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(
        self,
        item_ids: torch.Tensor,
        rating_ids: torch.Tensor,
        dt_ids: torch.Tensor,
        *,
        return_full_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        item_ids, rating_ids, dt_ids : (B, T) LongTensor.
        return_full_sequence : if True, return the (B, T, d) tensor for
                               all positions (useful for next-item
                               prediction over the whole sequence during
                               training). If False (default), return
                               h_t ∈ R^{B×d} at each batch's last
                               non-PAD position.

        Returns
        -------
        Either (B, d)  — if return_full_sequence is False
        Or     (B, T, d) — if True
        """
        if item_ids.dim() != 2:
            raise ValueError(f"Expected 2-D (B, T) input; got {item_ids.shape}")

        B, T = item_ids.shape
        device = item_ids.device

        # 1) Joint embedding e_k.
        x = self.joint_embedding(item_ids, rating_ids, dt_ids)  # (B, T, d)

        # 2) Add positional embedding if configured.
        if self.pos_embedding is not None:
            if T > self.pos_embedding.num_embeddings:
                raise ValueError(
                    f"Sequence length T={T} exceeds positional table "
                    f"capacity {self.pos_embedding.num_embeddings}"
                )
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            x = x + self.pos_embedding(pos_ids)

        x = self.dropout(x)

        # 3) Build the key_padding_mask from PAD positions.
        # PyTorch convention (bool form): True = IGNORED. We emit a
        # FLOAT form (0.0 / -inf) so that it shares dtype with attn_mask
        # and avoids the "mismatched mask types" deprecation warning
        # introduced in torch >=2.0.
        pad_bool = (item_ids == self.pad_index)  # (B, T) bool

        # If a row is entirely PAD, masking every key produces softmax-NaNs.
        # Leave the first slot unmasked in such rows; the resulting h_t
        # at position 0 is meaningless for all-PAD rows but downstream
        # last-valid-position extraction picks position 0 anyway and the
        # caller is responsible for filtering out all-PAD rows.
        all_pad_rows = pad_bool.all(dim=1)
        if all_pad_rows.any():
            pad_bool = pad_bool.clone()
            pad_bool[all_pad_rows, 0] = False

        key_padding_mask = torch.zeros_like(pad_bool, dtype=x.dtype)
        key_padding_mask.masked_fill_(pad_bool, float("-inf"))

        # 4) Causal mask (additive, float, (T, T)).
        causal = self._get_causal_mask(T, device, x.dtype)

        # 5) Transformer encoder stack.
        h = self.encoder(
            x,
            mask=causal,
            src_key_padding_mask=key_padding_mask,
        )  # (B, T, d)

        h = self.layer_norm(h)

        if return_full_sequence:
            return h

        # 6) Take h at each batch's LAST non-PAD position.
        # If all-PAD row, fall back to position 0 (we already kept it
        # unmasked above to avoid NaN propagation).
        valid = (item_ids != self.pad_index).to(torch.int64)  # (B, T)
        # +1 then subtract 1 because cumsum starts at 1 from first valid;
        # but we want the actual last-valid index, so use argmax of
        # reversed validity.
        last_pos = valid.sum(dim=1) - 1  # (B,)
        last_pos = last_pos.clamp(min=0)  # all-PAD rows → use position 0

        batch_idx = torch.arange(B, device=device)
        h_t = h[batch_idx, last_pos]  # (B, d)
        return h_t
