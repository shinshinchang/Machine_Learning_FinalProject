"""
CausalRec (Qiu et al., MM '21): Causal Inference for Visual Debiasing in
Visually-Aware Recommendation.

This is CEINN's Phase-5 *causal-recommendation* baseline (Tier 2). Unlike
the Tier-1 baselines (PopRec … BERT4Rec), CausalRec removes a bias via an
explicit counterfactual correction at INFERENCE time. The bias it targets
is *visual* bias — a user may interact with an item because of its
appearance even when the item is not a real-preference match. CEINN, by
contrast, targets *popularity* bias from an economics-informed angle; the
two are the natural methodological pair ("here is another way the field
does causal debiasing in recommendation").

Domain note (CEINN integration)
--------------------------------
CausalRec is visually-aware: it requires per-item image features `V_i`.
Amazon Beauty has them (it is one of CausalRec's own datasets), so we load
McAuley's precomputed 4096-d CNN features. MovieLens-10M has no item
images, so CausalRec is reported on Amazon Beauty only (Plan decision
"option A"); the MovieLens cell is marked "—" in the results table.

Model (paper Eq. 19–22)
-----------------------
Three sigmoid-of-dot-product branches, fused by multiplication:

    M_iu   = σ(γ_u · γ_i)                  # pure ID preference match
    M_ivu  = σ(γ_u · (γ_i ∘ Eφ(V_i)))      # visual-modulated match (indirect visual effect)
    N_vu   = σ(θ_u · Eφ(V_i))              # visual notice (direct visual effect)
    Y_ivu  = M_iu · M_ivu · N_vu

where γ_u / γ_i are the ID embeddings, θ_u is the user's visual factor,
and Eφ(V_i) is the (frozen) 4096-d CNN feature linearly projected to d by
a learnable matrix E. ∘ is the Hadamard product.

Training (paper Eq. 23–24): multitask BPR over three branches
-------------------------------------------------------------
    ℓ = ℓ_BPR(Y_ivu) + ℓ_BPR(N_vu) + ℓ_BPR(M_iu · M_ivu)

each ℓ_BPR being the standard pairwise -log σ(score⁺ − score⁻). The L2
regularisation λ1 of the original paper is folded into the optimiser's
weight_decay. Training is *biased* — no counterfactual correction is
applied here, exactly as in the paper ("after the biased training
process").

Inference (paper Eq. 27–28): Total Indirect Effect debiasing
------------------------------------------------------------
    ŷ_ivu = M_iu · M_ivu · N_vu  −  λ2 · M_{i*}u · M_{i*}{v*}u · N_vu

The reference ("no-treatment") item i* and visual v* are the catalogue
*averages* (paper: "a null value or an average value"). Note N_vu keeps
the REAL visual feature in BOTH terms — only the match branches see the
reference. Hence the reference product M_{i*}u · M_{i*}{v*}u is a per-user
constant C_u, and

    ŷ_ivu = N_vu · (M_iu · M_ivu − λ2 · C_u),

which down-weights items that score highly *only* through the visual
notice — i.e., visually-biased items. λ2 ∈ [0, ~1.2] controls how much of
the direct visual effect is removed (λ2 = 0 → biased; λ2 = 1 → fully
removed).

Interface parity
----------------
`score_all_items(user_ids, item_seq=None) → (B, n_items+1)` returns the
TIE-corrected full-ranking scores, so CEINN's `validate_full_ranking` is
used unchanged. `item_seq` is ignored — CausalRec is non-sequential
(it slots beside BPR-MF), so the sequence is not consumed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalRecModel(nn.Module):
    """
    Visually-aware causal recommender (Qiu et al., 2021) with multitask
    BPR training and Total-Indirect-Effect inference.

    Parameters
    ----------
    n_users         : int   — number of users (PAD=0 added internally)
    n_items         : int   — number of items (PAD=0 added internally)
    d               : int   — embedding dimension (aligned to BPR-MF: 64 on Amazon)
    visual_features : Tensor — (n_items+1, visual_dim) frozen CNN features;
                               row `pad_index` MUST be all-zero
    visual_dim      : int   — dimensionality of the raw CNN features (default 4096)
    lambda2         : float — TIE debiasing strength (default 1.0)
    pad_index       : int   — PAD id (default 0)
    init_std        : float — std of the Normal init for embeddings

    Public methods
    --------------
    bpr_multitask_loss(user_ids, pos_items, neg_items) → scalar   # training (Eq. 23–24)
    score_all_items(user_ids, item_seq=None)           → (B, n_items+1)  # TIE eval (Eq. 28)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        *,
        visual_features: torch.Tensor,
        visual_dim: int = 4096,
        lambda2: float = 1.0,
        pad_index: int = 0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.d = int(d)
        self.visual_dim = int(visual_dim)
        self.lambda2 = float(lambda2)
        self.pad_index = int(pad_index)

        # --- Validate & register the frozen visual features -------------
        if visual_features is None:
            raise ValueError(
                "CausalRecModel requires `visual_features` of shape "
                f"(n_items+1={n_items + 1}, visual_dim={visual_dim}). CausalRec "
                "is visually-aware; on a dataset without item images (e.g. "
                "MovieLens-10M) it should not be instantiated."
            )
        vf = visual_features.detach().clone().to(torch.get_default_dtype())
        if vf.shape != (n_items + 1, visual_dim):
            raise ValueError(
                f"visual_features shape {tuple(vf.shape)} != "
                f"({n_items + 1}, {visual_dim})."
            )
        # PAD row must be zero so the PAD item produces a neutral visual signal.
        vf[pad_index].zero_()
        # Frozen: registered as a buffer, never updated (matches VBPR/CausalRec —
        # the pretrained CNN features are fixed; only the projection E learns).
        self.register_buffer("visual_features", vf)

        # --- Learnable parameters ---------------------------------------
        # γ_u (match) and θ_u (visual notice) are SEPARATE user factors,
        # mirroring VBPR's split between collaborative and visual user
        # vectors (the paper allows θ_u = γ_u; we keep them distinct so the
        # visual-notice branch can specialise).
        self.user_embedding = nn.Embedding(n_users + 1, d, padding_idx=pad_index)
        self.user_visual_embedding = nn.Embedding(n_users + 1, d, padding_idx=pad_index)
        self.item_embedding = nn.Embedding(n_items + 1, d, padding_idx=pad_index)
        # E: linear projection of the raw CNN feature to the embedding space.
        # No bias — it is a pure transform matrix (paper notation Eφ(V_i)),
        # which also keeps the all-zero PAD feature mapping to a zero vector.
        self.visual_proj = nn.Linear(visual_dim, d, bias=False)

        # Small-magnitude normal init (BPR family convention — large values
        # saturate the pairwise sigmoid in the first batches).
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.user_visual_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.visual_proj.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            self.user_embedding.weight[pad_index].zero_()
            self.user_visual_embedding.weight[pad_index].zero_()
            self.item_embedding.weight[pad_index].zero_()

    # -------------------------------------------------------------------------
    # Visual projection helpers
    # -------------------------------------------------------------------------
    def _project_visual(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Eφ(V_i) for a set of item ids. item_ids: (...) → (..., d)."""
        raw = self.visual_features[item_ids]          # (..., visual_dim)
        return self.visual_proj(raw)                  # (..., d)

    def _project_visual_all(self) -> torch.Tensor:
        """Eφ(V) for every item. → (n_items+1, d)."""
        return self.visual_proj(self.visual_features)  # (V+1, d)

    # -------------------------------------------------------------------------
    # Training: three branch scores for explicit (user, item) pairs
    # -------------------------------------------------------------------------
    def _branch_scores(
        self,
        user_ids: torch.Tensor,    # (N,)
        item_ids: torch.Tensor,    # (N,)
    ):
        """
        Returns the three pre-fusion branch scores for each (u, i) pair:
            m_iu, m_ivu, n_vu   — each (N,), all in (0, 1).
        """
        gu = self.user_embedding(user_ids)            # (N, d)
        tu = self.user_visual_embedding(user_ids)     # (N, d)
        gi = self.item_embedding(item_ids)            # (N, d)
        ev = self._project_visual(item_ids)           # (N, d)

        m_iu = torch.sigmoid((gu * gi).sum(-1))                   # σ(γ_u·γ_i)
        m_ivu = torch.sigmoid((gu * (gi * ev)).sum(-1))          # σ(γ_u·(γ_i∘Ev))
        n_vu = torch.sigmoid((tu * ev).sum(-1))                  # σ(θ_u·Ev)
        return m_iu, m_ivu, n_vu

    def bpr_multitask_loss(
        self,
        user_ids: torch.Tensor,    # (N,)
        pos_items: torch.Tensor,   # (N,)
        neg_items: torch.Tensor,   # (N,)
    ) -> torch.Tensor:
        """
        Multitask BPR over the three branches (paper Eq. 23–24).
        L2 (λ1) is supplied by the optimiser's weight_decay.
        """
        mp_iu, mp_ivu, np_vu = self._branch_scores(user_ids, pos_items)
        mn_iu, mn_ivu, nn_vu = self._branch_scores(user_ids, neg_items)

        # Branch 1: full fused score Y = M_iu · M_ivu · N_vu
        y_pos = mp_iu * mp_ivu * np_vu
        y_neg = mn_iu * mn_ivu * nn_vu
        # Branch 2: visual notice N_vu
        # Branch 3: match · visual-match M_iu · M_ivu
        mm_pos = mp_iu * mp_ivu
        mm_neg = mn_iu * mn_ivu

        loss = (
            -F.logsigmoid(y_pos - y_neg)
            - F.logsigmoid(np_vu - nn_vu)
            - F.logsigmoid(mm_pos - mm_neg)
        ).mean()
        return loss

    # -------------------------------------------------------------------------
    # Inference: TIE-corrected full-ranking scores
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _reference_constant(self, gu: torch.Tensor) -> torch.Tensor:
        """
        Per-user reference constant C_u = M_{i*}u · M_{i*}{v*}u, where i*/v*
        are the catalogue averages over REAL items (PAD excluded).
        gu: (B, d) user match-embeddings → returns (B,).
        """
        # Average item embedding γ_{i*} and average projected visual Ev_{i*}.
        gi_star = self.item_embedding.weight[1:].mean(dim=0)       # (d,)
        ev_all = self._project_visual_all()                       # (V+1, d)
        ev_star = ev_all[1:].mean(dim=0)                          # (d,)

        m_istar_u = torch.sigmoid(gu @ gi_star)                  # (B,)
        m_istar_vstar_u = torch.sigmoid(gu @ (gi_star * ev_star))  # (B,)
        return m_istar_u * m_istar_vstar_u                       # (B,)

    @torch.no_grad()
    def score_all_items(
        self,
        user_ids: torch.Tensor,                     # (B,)
        item_seq: Optional[torch.Tensor] = None,    # ignored (non-sequential)
    ) -> torch.Tensor:
        """
        TIE-corrected scores over the whole catalogue (paper Eq. 28):

            ŷ(i) = M_iu(i)·M_ivu(i)·N_vu(i) − λ2 · C_u · N_vu(i)

        Returns (B, n_items+1). PAD (index 0) is left unmasked — the
        Full-Ranking evaluator masks it via `pad_index` (same convention
        as BPR-MF).
        """
        gu = self.user_embedding(user_ids)            # (B, d)
        tu = self.user_visual_embedding(user_ids)     # (B, d)
        gi = self.item_embedding.weight               # (V+1, d)
        ev = self._project_visual_all()               # (V+1, d)

        # Per-(user, item) branch scores over the full catalogue.
        m_iu = torch.sigmoid(gu @ gi.t())             # (B, V+1)
        m_ivu = torch.sigmoid(gu @ (gi * ev).t())     # (B, V+1)
        n_vu = torch.sigmoid(tu @ ev.t())             # (B, V+1)

        y = m_iu * m_ivu * n_vu                        # (B, V+1) — biased score
        c_u = self._reference_constant(gu).unsqueeze(1)  # (B, 1)
        tie = y - self.lambda2 * c_u * n_vu            # (B, V+1) — debiased
        return tie

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Alias so the model is callable like the other baselines."""
        return self.score_all_items(*args, **kwargs)
