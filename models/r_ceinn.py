"""
Module 3.4 — CEINN main model (Plan §4.4).

Wires the three modules from Phase 4 (§4.1 backbone, §4.2 deconfounder,
§4.3 economics utility) into a single forward pass:

    inputs (item_ids, rating_ids, dt_ids)
        │
        ▼
    SequentialBackbone (§3.1)
        ├── h_t  ──────────────────────────────────┐
        │                                           │
        ▼                                           ▼
    CausalDeconfounder (§3.2)            EconomicsUtility (§3.3)
        ├── propensity p_i = P(E=1 | Z)      ├── V(u, i, t)  via bilinear
        └── z_logits = D(GRL(h_t, α))        ├── C(i, t)      dataset-specific
                                              ├── λ_u         from history pool
                                              └── U = V − λ_u·C

The item embedding table E_i is allocated ONCE in the backbone and
shared by reference with both the value head (V) and the λ_u network.
This is the "weight tying" required by §3.3.5: V uses *the same*
purely-ID embeddings that the backbone learned, not a copy.

Forward signature
-----------------
The two datasets diverge in cost computation, so we expose two
front-end entry points:
    `forward_amazon(...)`   — Amazon Beauty path
    `forward_movielens(...)` — MovieLens path
both returning a dict of tensors that `train.py` will pipe into the
relevant loss modules (`utils/losses.py`).

Dataset-aware factory
---------------------
`CEINNModel.from_loader(loader, config)` inspects the loader type and
returns a CEINN instance wired with the right cost backend, n_z_buckets,
and embedding sizes. This is the recommended construction path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .causal_deconfounder import CausalDeconfounder, alpha_schedule
from .economics_utility import (
    AmazonBeautyCost,
    EconomicsUtility,
    MovieLensCost,
    build_amazon_cost_from_loader,
)
from .sequential_backbone import SequentialBackbone


class CEINNModel(nn.Module):
    """
    The CEINN main model — wires the three Phase-4 submodules.

    Parameters
    ----------
    n_items       : total real items (without PAD).
    n_rating_bins : as stored in vocab_sizes.json.
    n_dt_bins     : as stored in vocab_sizes.json.
    n_z_buckets   : K for the discriminator + propensity estimator.
                    Amazon = n_Z_bins from vocab (10). MovieLens
                    typically uses dynamic-Z discretised (caller picks,
                    e.g. 10 quantile bins on the precomputed values).
    d, n_heads, n_layers, d_ff, dropout, max_seq_len, pad_index :
                    forwarded to SequentialBackbone.
    dataset_name  : "amazon_beauty" or "movielens".
    cost_backend  : already-constructed cost module (AmazonBeautyCost
                    or MovieLensCost). Use the `from_loader` factory
                    below for the standard wiring path.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        cost_backend: nn.Module,
        n_items: int,
        n_rating_bins: int,
        n_dt_bins: int,
        n_z_buckets: int,
        d: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: Optional[int] = None,
        pad_index: int = 0,
        propensity_hidden: int = 64,
        propensity_z_emb_dim: int = 16,
        discriminator_hidden: Optional[int] = None,
        # New: forwarded to EconomicsUtility. See its docstring.
        value_mode: str = "bilinear",
        cost_scale_mode: str = "none",
        cost_scale_target_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.n_items = n_items
        self.n_z_buckets = n_z_buckets
        self.pad_index = pad_index
        self.d = d

        # §3.1 backbone — also owns the shared item_embedding table.
        self.backbone = SequentialBackbone(
            n_items=n_items,
            n_rating_bins=n_rating_bins,
            n_dt_bins=n_dt_bins,
            d=d,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_index=pad_index,
        )

        # §3.2 deconfounder
        self.deconfounder = CausalDeconfounder(
            d=d,
            n_z_buckets=n_z_buckets,
            propensity_hidden=propensity_hidden,
            propensity_z_emb_dim=propensity_z_emb_dim,
            discriminator_hidden=discriminator_hidden,
            dropout=dropout,
        )

        # §3.3 economics utility — wires SHARED item embedding by reference.
        self.economics = EconomicsUtility(
            dataset_name=dataset_name,
            d=d,
            item_embedding=self.backbone.item_embedding,
            cost_backend=cost_backend,
            pad_index=pad_index,
            value_mode=value_mode,
            cost_scale_mode=cost_scale_mode,
            cost_scale_target_ratio=cost_scale_target_ratio,
        )

    # =========================================================================
    # Convenience: encode an input sequence → h_t (no economics, no
    # deconfounder). Useful for evaluation-time scoring.
    # =========================================================================
    def encode(
        self,
        item_ids: torch.Tensor,
        rating_ids: torch.Tensor,
        dt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return h_t ∈ R^{B×d}."""
        return self.backbone(item_ids, rating_ids, dt_ids)

    # =========================================================================
    # Full forward — Amazon Beauty
    # =========================================================================
    def forward_amazon(
        self,
        item_ids: torch.Tensor,
        rating_ids: torch.Tensor,
        dt_ids: torch.Tensor,
        z_buckets_target: torch.Tensor,
        z_buckets_candidates: Optional[torch.Tensor] = None,
        *,
        grl_alpha: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        item_ids, rating_ids, dt_ids : (B, T) input sequence.
        z_buckets_target : (B,) confounder bucket Z of the TARGET item
                           (used by both the discriminator's CE target
                           and the propensity estimator).
        z_buckets_candidates : (B, V_cat) or (V_cat,) — Z bucket of
                           every candidate item, for the propensity to
                           be applied as IPS weight in the choice loss.
                           Optional: if None, the caller computes IPS
                           weights externally.
        grl_alpha       : current GRL strength (set per epoch).

        Returns
        -------
        dict with:
          h_t                : (B, d)
          U                  : (B, V_cat)  utility = V − λ_u·C
          V                  : (B, V_cat)
          C                  : (V_cat,)
          lambda_u           : (B,)
          z_logits           : (B, K)  discriminator output
          propensity_target  : (B,)    p_i for the supervised target
          propensity_logit_target : (B,) raw logit for BCE training
          propensity_candidates : (B, V_cat) or None
        """
        h_t = self.backbone(item_ids, rating_ids, dt_ids)
        z_logits = self.deconfounder.discriminate(h_t, alpha=grl_alpha)

        econ = self.economics.compute_utility_amazon(h_t, item_history=item_ids)

        p_target_logit = self.deconfounder.propensity_logit(z_buckets_target)
        p_target = torch.sigmoid(p_target_logit)

        out: Dict[str, torch.Tensor] = {
            "h_t": h_t,
            "U": econ["U"],
            "V": econ["V"],
            "C": econ["C"],
            "lambda_u": econ["lambda_u"],
            "z_logits": z_logits,
            "propensity_target": p_target,
            "propensity_logit_target": p_target_logit,
        }
        if z_buckets_candidates is not None:
            # Vectorise: flatten, run estimator, reshape back.
            shape = z_buckets_candidates.shape
            flat = z_buckets_candidates.reshape(-1)
            p_flat = torch.sigmoid(self.deconfounder.propensity_logit(flat))
            out["propensity_candidates"] = p_flat.reshape(shape)
        return out

    # =========================================================================
    # Full forward — MovieLens
    # =========================================================================
    def forward_movielens(
        self,
        item_ids: torch.Tensor,
        rating_ids: torch.Tensor,
        dt_ids: torch.Tensor,
        z_buckets_target: torch.Tensor,
        genre_red: torch.Tensor,
        recency_press: torch.Tensor,
        pop_press: torch.Tensor,
        z_buckets_candidates: Optional[torch.Tensor] = None,
        *,
        grl_alpha: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        item_ids, rating_ids, dt_ids : (B, T) input sequence.
        z_buckets_target : (B,) discretised Z(t) bucket of the target.
        genre_red, recency_press, pop_press : (B, V_cat) per-candidate
                       MovieLens cost signals (precomputed in Phase 2).
        z_buckets_candidates : (B, V_cat) or (V_cat,) — Z bucket of
                       every candidate (optional).
        grl_alpha   : current GRL strength.

        Returns
        -------
        Same keys as `forward_amazon`, but with C of shape (B, V_cat).
        """
        h_t = self.backbone(item_ids, rating_ids, dt_ids)
        z_logits = self.deconfounder.discriminate(h_t, alpha=grl_alpha)

        econ = self.economics.compute_utility_movielens(
            h_t,
            item_history=item_ids,
            genre_red=genre_red,
            recency_press=recency_press,
            pop_press=pop_press,
        )

        p_target_logit = self.deconfounder.propensity_logit(z_buckets_target)
        p_target = torch.sigmoid(p_target_logit)

        out: Dict[str, torch.Tensor] = {
            "h_t": h_t,
            "U": econ["U"],
            "V": econ["V"],
            "C": econ["C"],
            "lambda_u": econ["lambda_u"],
            "z_logits": z_logits,
            "propensity_target": p_target,
            "propensity_logit_target": p_target_logit,
        }
        if z_buckets_candidates is not None:
            shape = z_buckets_candidates.shape
            flat = z_buckets_candidates.reshape(-1)
            p_flat = torch.sigmoid(self.deconfounder.propensity_logit(flat))
            out["propensity_candidates"] = p_flat.reshape(shape)
        return out

    # =========================================================================
    # Generic forward dispatcher (lets the caller stay dataset-agnostic
    # if all the inputs are bundled in a dict).
    # =========================================================================
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        grl_alpha: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Dispatch on `self.dataset_name`. The batch dict must contain
        the expected keys for the active dataset path. See
        `forward_amazon` and `forward_movielens` for the required keys.
        """
        if self.dataset_name == "amazon_beauty":
            return self.forward_amazon(
                item_ids=batch["item_ids"],
                rating_ids=batch["rating_ids"],
                dt_ids=batch["dt_ids"],
                z_buckets_target=batch["z_buckets_target"],
                z_buckets_candidates=batch.get("z_buckets_candidates"),
                grl_alpha=grl_alpha,
            )
        elif self.dataset_name == "movielens":
            return self.forward_movielens(
                item_ids=batch["item_ids"],
                rating_ids=batch["rating_ids"],
                dt_ids=batch["dt_ids"],
                z_buckets_target=batch["z_buckets_target"],
                genre_red=batch["genre_red"],
                recency_press=batch["recency_press"],
                pop_press=batch["pop_press"],
                z_buckets_candidates=batch.get("z_buckets_candidates"),
                grl_alpha=grl_alpha,
            )
        else:
            raise RuntimeError(f"Unknown dataset_name {self.dataset_name!r}")


# =============================================================================
# Factory: construct a CEINN from a Phase-2 loader + small config dict
# =============================================================================

def build_ceinn_amazon(
    loader,
    *,
    d: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    propensity_hidden: int = 64,
    discriminator_hidden: Optional[int] = None,
    value_mode: str = "bilinear",
    cost_scale_mode: str = "none",
    cost_scale_target_ratio: float = 1.0,
) -> CEINNModel:
    """
    Build a CEINN model for Amazon Beauty from an `AmazonBeautyLoader`.

    The loader supplies all vocab sizes AND the per-item meta arrays
    (cat, brand, log_price, Z) that the cost backend needs.

    Post-fix knobs (default values reproduce the pre-fix baseline EXCEPT
    for `value_mode`, which now defaults to identity-initialised
    bilinear — see `BilinearValue` docstring):
      - value_mode      : "bilinear" | "dot" | "bilinear_xavier"
      - cost_scale_mode : "none" | "auto"
      - cost_scale_target_ratio : only used when cost_scale_mode='auto'.
    """
    cost_backend = build_amazon_cost_from_loader(loader)
    return CEINNModel(
        dataset_name="amazon_beauty",
        cost_backend=cost_backend,
        n_items=int(loader.vocab["n_items"]),
        n_rating_bins=int(loader.vocab["n_rating_bins"]),
        n_dt_bins=int(loader.vocab["n_dt_bins"]),
        n_z_buckets=int(loader.vocab["n_Z_bins"]),
        d=d,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_len=int(loader.vocab["max_seq_len"]),
        pad_index=int(loader.vocab["pad_index"]),
        propensity_hidden=propensity_hidden,
        discriminator_hidden=discriminator_hidden,
        value_mode=value_mode,
        cost_scale_mode=cost_scale_mode,
        cost_scale_target_ratio=cost_scale_target_ratio,
    )


def build_ceinn_movielens(
    loader,
    *,
    n_z_buckets: int = 10,
    d: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    propensity_hidden: int = 64,
    discriminator_hidden: Optional[int] = None,
    value_mode: str = "bilinear",
    cost_scale_mode: str = "none",
    cost_scale_target_ratio: float = 1.0,
) -> CEINNModel:
    """
    Build a CEINN model for MovieLens 10M from a `MovieLens10MLoader`.

    See `build_ceinn_amazon` for the new value_mode / cost_scale knobs.
    """
    cost_backend = MovieLensCost()
    return CEINNModel(
        dataset_name="movielens",
        cost_backend=cost_backend,
        n_items=int(loader.vocab["n_items"]),
        n_rating_bins=int(loader.vocab["n_rating_bins"]),
        n_dt_bins=int(loader.vocab["n_dt_bins"]),
        n_z_buckets=n_z_buckets,
        d=d,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_len=int(loader.vocab["max_seq_len"]),
        pad_index=int(loader.vocab["pad_index"]),
        propensity_hidden=propensity_hidden,
        discriminator_hidden=discriminator_hidden,
        value_mode=value_mode,
        cost_scale_mode=cost_scale_mode,
        cost_scale_target_ratio=cost_scale_target_ratio,
    )
