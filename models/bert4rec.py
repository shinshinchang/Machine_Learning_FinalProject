"""
BERT4Rec (Sun et al., 2019): Bidirectional Encoder Representations for
Sequential Recommendation.

A bidirectional Transformer trained with the Cloze (Masked LM) objective:
random positions in the user's history are replaced with [MASK], and the
model predicts the original item at those positions. At evaluation time,
[MASK] is placed at the position where the next item would go, and we
read the logits there.

Differences from SASRec
-----------------------
  * **No causal mask** — bidirectional attention over the full sequence.
  * **Extended vocabulary**: indices `0..n_items` are PAD + real items
    (same as everywhere else), and index `n_items + 1` is the special
    `[MASK]` token. Output `score_all_items(...)` returns logits over
    indices `0..n_items` only (we slice off the MASK column), keeping
    the same `(B, n_items + 1)` shape as every other baseline so the
    Full-Ranking evaluator doesn't need a BERT4Rec-specific branch.
  * **Training**: MLM cross-entropy. The label tensor has the original
    item id at masked positions and PAD (=0) elsewhere, so a single
    `F.cross_entropy(... ignore_index=0)` call gives the right loss
    (PAD positions in the *target* are ignored, but the input contains
    [MASK] tokens at those same positions which DO contribute to the
    forward pass — they're just not supervised).
  * **Evaluation**: `score_all_items` places [MASK] at the position
    immediately AFTER the user's last real item in the input (i.e., at
    the first PAD slot). For users whose history already fills
    `max_seq_len`, we drop the oldest item from the front and put
    [MASK] at position `max_seq_len - 1`. This is the standard "append
    MASK and truncate" eval protocol in the BERT4Rec literature.

Hyperparameters at training time
--------------------------------
  * `mask_prob`: probability of masking each non-PAD position
    (Plan §5.2 defers to the paper's recommended 0.2). Set per-baseline
    in `configs/baselines.yaml`.

  * Each row is guaranteed at least one masked position (the train
    step in `train_baseline.py::random_mask_batch` falls back to a
    single random non-PAD position if Bernoulli masking happened to
    produce zero masks for a row). Without this guard, zero-mask rows
    silently consume compute without contributing to the loss, and
    rare degenerate batches can produce NaN.

Why the [MASK]-on-validate gets slotted into score_all_items, not
validate_full_ranking
---------------------------------------------------------------------
`validate_full_ranking` is shared with PopRec, BPR-MF, GRU4Rec, and
SASRec. Adding a BERT4Rec branch there would couple the eval primitive
to one model's quirk. Keeping the [MASK] placement inside this class
preserves the "every baseline implements `score_all_items` the same
way" invariant: input is a padded item sequence, output is
`(B, n_items + 1)` logits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BERT4RecModel(nn.Module):
    """
    Bidirectional Transformer for sequential recommendation, with a
    special [MASK] token at vocab index `n_items + 1`.

    Parameters
    ----------
    n_items     : int   — number of items (PAD=0 is added internally; vocab is n_items+2 with MASK)
    d           : int   — embedding / model dimension (Plan §5.2: 64 / 128)
    n_heads     : int   — number of attention heads (default 2, paper's setting)
    n_layers    : int   — number of Transformer encoder blocks (default 2)
    max_seq_len : int   — max input sequence length (Plan: 50 Amazon / 200 ML)
    dropout     : float — applied to embeddings, attention, and FFN
    ff_mult     : int   — FFN inner dim = ff_mult × d (default 4, BERT convention)
    pad_index   : int   — PAD token id (default 0)

    Public methods
    --------------
    forward_masked(item_ids)
        → (B, T, vocab_size) logits — used internally and during MLM training
    score_all_items(user_ids, item_seq)
        → (B, n_items + 1) logits at the MASK position — used at eval

    Attributes
    ----------
    mask_token_id : int — the [MASK] vocab index (= n_items + 1)
    vocab_size    : int — n_items + 2 (PAD + items + MASK)
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
        self.mask_token_id = int(n_items) + 1   # last vocab index
        self.vocab_size = int(n_items) + 2      # PAD + items + MASK

        if d % n_heads != 0:
            raise ValueError(
                f"BERT4Rec: d ({d}) must be divisible by n_heads ({n_heads})."
            )

        # --- Embeddings ---------------------------------------------------
        self.item_embedding = nn.Embedding(
            self.vocab_size, d, padding_idx=pad_index,
        )
        self.pos_embedding = nn.Embedding(max_seq_len, d)
        self.emb_layernorm = nn.LayerNorm(d, eps=1e-8)
        self.emb_dropout = nn.Dropout(dropout)

        # --- Transformer encoder (pre-norm, NO causal mask) -------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.final_layernorm = nn.LayerNorm(d, eps=1e-8)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Truncated-normal-ish init (std=0.02, BERT convention).
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        # PAD row → zero (padding_idx blocks gradient but doesn't zero init).
        with torch.no_grad():
            self.item_embedding.weight[self.pad_index].zero_()

    # -------------------------------------------------------------------------
    # Internal: embedding + transformer pass
    # -------------------------------------------------------------------------
    def _embed_with_position(self, item_ids: torch.Tensor) -> torch.Tensor:
        B, T = item_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"BERT4Rec: input length {T} exceeds max_seq_len={self.max_seq_len}."
            )
        positions = torch.arange(T, device=item_ids.device).unsqueeze(0).expand(B, T)
        x = self.item_embedding(item_ids) + self.pos_embedding(positions)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)
        return x

    def forward_masked(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional encode → per-position logits over the full vocabulary
        (including [MASK]). Used both during MLM training (loss is computed
        at masked positions only) and internally by `score_all_items`.

        Returns (B, T, vocab_size).
        """
        x = self._embed_with_position(item_ids)
        pad_mask = item_ids == self.pad_index   # True at PAD
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        h = self.final_layernorm(h)
        logits = h @ self.item_embedding.weight.t()   # (B, T, vocab_size)
        return logits

    # -------------------------------------------------------------------------
    # Eval-time scoring: place [MASK] at the prediction slot
    # -------------------------------------------------------------------------
    def _build_eval_input(
        self, item_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a right-padded history of shape (B, T), build the [MASK]ed
        input the bidirectional encoder should see at eval time.

        Two cases (handled jointly by torch.where):

        (A) Row has at least one PAD position:
            place [MASK] at the FIRST pad index (= "right after the
            user's last real item"). Real history is preserved verbatim.

            Example with T=8, pad=0:
                input  : [3, 7, 2, 0, 0, 0, 0, 0]
                output : [3, 7, 2, M, 0, 0, 0, 0]   (predict at idx 3)

        (B) Row has no PAD (history fills T):
            drop position 0 (oldest item), shift everything left, place
            [MASK] at position T-1. This is the standard "append MASK and
            truncate" protocol from the BERT4Rec paper.

            Example with T=4, pad=0:
                input  : [3, 7, 2, 9]
                output : [7, 2, 9, M]                (predict at idx 3)

        Returns
        -------
        eval_seq    : (B, T) — masked input ready for `forward_masked`
        predict_pos : (B,)   — the index in each row where [MASK] sits
        """
        B, T = item_seq.shape
        device = item_seq.device
        batch_idx = torch.arange(B, device=device)

        is_pad = item_seq == self.pad_index                    # (B, T)
        row_has_pad = is_pad.any(dim=1)                        # (B,)
        first_pad = is_pad.float().argmax(dim=1)               # (B,) — 0 if no pad

        # Shifted-left version for the "no pad" branch.
        shifted = torch.cat(
            [item_seq[:, 1:], torch.full_like(item_seq[:, :1], self.pad_index)],
            dim=1,
        )

        # Pick per-row variant: original if row_has_pad else shifted.
        eval_seq = torch.where(row_has_pad.unsqueeze(1), item_seq, shifted).clone()

        # Predict position: first_pad if has_pad else T-1.
        predict_pos = torch.where(
            row_has_pad, first_pad,
            torch.full_like(first_pad, T - 1),
        )

        # Place [MASK] at predict_pos.
        eval_seq[batch_idx, predict_pos] = self.mask_token_id

        return eval_seq, predict_pos

    def score_all_items(
        self,
        user_ids: Optional[torch.Tensor] = None,    # unused; for API parity
        item_seq: Optional[torch.Tensor] = None,    # (B, T) — right-padded history
    ) -> torch.Tensor:
        """
        Return (B, n_items + 1) scores at the [MASK] position.

        Slices off the [MASK] column from the (B, vocab_size) logits so
        the Full-Ranking evaluator (which expects an n_items+1 axis) is
        agnostic to BERT4Rec's extended vocabulary.
        """
        if item_seq is None:
            raise ValueError(
                "BERT4RecModel.score_all_items requires `item_seq`; this is "
                "a sequential model with no user-only path."
            )
        B = item_seq.size(0)
        device = item_seq.device
        batch_idx = torch.arange(B, device=device)

        eval_seq, predict_pos = self._build_eval_input(item_seq)
        logits = self.forward_masked(eval_seq)                   # (B, T, vocab_size)
        pred_logits = logits[batch_idx, predict_pos]             # (B, vocab_size)
        # Return only the PAD + items columns (drop the [MASK] column).
        return pred_logits[:, : self.n_items + 1]                # (B, n_items + 1)
