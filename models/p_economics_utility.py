"""
Module 3.3 — Economics-Informed Utility Module (Plan §3.3, Phase 4 §4.3).

Outputs the per-(user-state, candidate-item) utility
    U(u, i, t) = V(u, i, t) − λ_u · C(i, t)
which `utils/losses.py::UtilityChoiceLoss` consumes to compute L_C
(the NLL of the actual chosen item under a full-catalogue softmax).

Two dataset-specific cost backends
----------------------------------
Per §3.3.3 / §3.3.4 the cost function takes a different functional form
in each scenario:

  * Amazon Beauty (explicit economic cost, §3.3.3):
        C(i, t) = α1·log(price_i)
                + α2·φ(category_i)
                + α3·ψ(brand_i)
                + α4·η(salesRank_i)            ← uses Z bucket
                + α5·( log(price_i) × φ(category_i) )
    α1..α5 are learnable nn.Parameters initialised U(0, 0.1).

  * MovieLens (implicit behavioural friction, §3.3.4):
        C(i, t) = β1·GenreRed(i, H_{u,<t})
                + β2·RecencyPress(t)
                + β3·PopPress(i, t)
    All three signals are precomputed Phase-2 scalars, queried per
    (user, position) — no extra learned lookups required. β1..β3 are
    learnable nn.Parameters initialised to small positive values.

Both backends expose the same interface:
    cost_backend.compute(...) → (B, V) cost tensor
so the main model is agnostic to dataset.

Value head and λ_u network
--------------------------
Per §3.3.5 / §4.3.3, V is a bilinear over h_t and E_i(i):
    V(u, i, t) = h_t^T W_V E_i(i)
We default to the bilinear form (cheap; closest to traditional CF).
A small MLP variant is provided behind a flag if future ablation needs it.

Per §3.3.5 / §4.3.4, λ_u is a scalar in (0, 1) computed by
    z_u = mean_k E_i(i_k)          (mean-pool of historical ID embeddings)
    λ_u = σ(w^T z_u)               (linear projection + sigmoid)

Shared item embedding (CRITICAL)
--------------------------------
The bilinear V head AND the λ_u network BOTH use the SAME
nn.Embedding table that the backbone trained — they are passed in at
construction time by reference, not copied. See `models/ceinn.py` for
how this is wired (the backbone exposes `item_embedding` as a
property).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared helpers
# =============================================================================

def _init_uniform_positive(p: nn.Parameter, lo: float = 0.0, hi: float = 0.1) -> None:
    """In-place uniform initialisation in [lo, hi] for cost coefficients."""
    with torch.no_grad():
        p.uniform_(lo, hi)


# =============================================================================
# §4.3.1  Amazon Beauty: explicit economic cost
# =============================================================================

class AmazonBeautyCost(nn.Module):
    """
    Explicit economic cost (Plan §3.3.3):

        C(i, t) = α1 log(price_i)
                + α2 φ(category_i)
                + α3 ψ(brand_i)
                + α4 η(salesRank_i)
                + α5 ( log(price_i) × φ(category_i) )

    Parameters / coefficients
    -------------------------
    All five α's are nn.Parameter scalars initialised U(0, 0.1). No
    sign constraint — the optimiser is free to set them negative if
    the data favours that (and the unit test allows it).

    Implementation of φ, ψ, η
    -------------------------
    Following §4.3.1, the categorical lookups (φ for category, ψ for
    brand, η for salesRank-Z bucket) are implemented as one-hot
    embedding tables of dim 1: i.e. an nn.Embedding(n_cats, 1) that
    maps each category id directly to a learnable scalar. This is
    exactly equivalent to a one-hot lookup followed by a learned
    linear projection to a scalar, just stored more compactly.

    Item-meta buffers
    -----------------
    The four per-item features (cat_id, brand_id, log_price, Z_bucket)
    are stored as `register_buffer` tensors so they move with the
    model when `.to(device)` is called. They come from
    `loaders.AmazonBeautyLoader.bulk_meta_arrays()` and are indexed by
    item_idx — index 0 is the PAD row (all zeros) which guarantees
    PAD slots contribute zero cost.
    """

    def __init__(
        self,
        n_items: int,
        n_cats: int,
        n_brands: int,
        n_z_bins: int,
        cat_ids: torch.Tensor,        # (n_items + 1,) long
        brand_ids: torch.Tensor,      # (n_items + 1,) long
        log_price: torch.Tensor,      # (n_items + 1,) float
        z_bins: torch.Tensor,         # (n_items + 1,) long
    ) -> None:
        super().__init__()
        self.n_items = n_items

        # Per-item meta tables — buffers so they participate in
        # `.to(device)` but receive no gradient.
        if cat_ids.shape[0] != n_items + 1:
            raise ValueError(f"cat_ids must be (n_items+1,), got {cat_ids.shape}")
        if brand_ids.shape[0] != n_items + 1:
            raise ValueError(f"brand_ids must be (n_items+1,), got {brand_ids.shape}")
        if log_price.shape[0] != n_items + 1:
            raise ValueError(f"log_price must be (n_items+1,), got {log_price.shape}")
        if z_bins.shape[0] != n_items + 1:
            raise ValueError(f"z_bins must be (n_items+1,), got {z_bins.shape}")

        self.register_buffer("cat_ids", cat_ids.long())
        self.register_buffer("brand_ids", brand_ids.long())
        self.register_buffer("log_price", log_price.float())
        self.register_buffer("z_bins", z_bins.long())

        # Scalar embedding tables (one-hot → scalar lookups).
        self.category_scalar = nn.Embedding(n_cats, 1)
        self.brand_scalar    = nn.Embedding(n_brands, 1)
        self.z_scalar        = nn.Embedding(n_z_bins, 1)
        nn.init.uniform_(self.category_scalar.weight, 0.0, 0.1)
        nn.init.uniform_(self.brand_scalar.weight, 0.0, 0.1)
        nn.init.uniform_(self.z_scalar.weight, 0.0, 0.1)

        # Five learnable α coefficients.
        self.alpha1 = nn.Parameter(torch.empty(1))
        self.alpha2 = nn.Parameter(torch.empty(1))
        self.alpha3 = nn.Parameter(torch.empty(1))
        self.alpha4 = nn.Parameter(torch.empty(1))
        self.alpha5 = nn.Parameter(torch.empty(1))
        for p in (self.alpha1, self.alpha2, self.alpha3, self.alpha4, self.alpha5):
            _init_uniform_positive(p, 0.0, 0.1)

    # -------------------------------------------------------------------------
    # Vectorised "per-catalogue-slot" cost. Returns the cost for EVERY
    # item index (0..n_items). At training the main model broadcasts
    # this to (B, V_cat) by selecting the relevant slice (or sharing it
    # across the batch when no per-batch state alters cost, as is the
    # case for Amazon — cost is a pure item property).
    # -------------------------------------------------------------------------
    def cost_all_items(self) -> torch.Tensor:
        """
        Returns (n_items + 1,) cost vector. Index 0 = PAD row, which
        evaluates to 0 because all four lookup tables hold zeros at
        slot 0 by Phase-2 convention (PAD index of cat, brand, Z are
        all 0 with embedding row 0 explicitly kept at small uniform
        values — fine because the PAD candidate is masked out of the
        softmax in UtilityChoiceLoss anyway).
        """
        # phi(category), psi(brand), eta(salesRank-Z) — each (n_items+1,)
        phi = self.category_scalar(self.cat_ids).squeeze(-1)
        psi = self.brand_scalar(self.brand_ids).squeeze(-1)
        eta = self.z_scalar(self.z_bins).squeeze(-1)

        lp = self.log_price  # (n_items + 1,)
        cost = (
            self.alpha1 * lp
            + self.alpha2 * phi
            + self.alpha3 * psi
            + self.alpha4 * eta
            + self.alpha5 * (lp * phi)
        )
        return cost  # (n_items + 1,)

    def forward(
        self,
        item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        If `item_ids` is None, returns (n_items + 1,) cost for all
        catalogue items. Otherwise, returns a tensor of the same shape
        as `item_ids` containing the cost at each id.
        """
        c_all = self.cost_all_items()
        if item_ids is None:
            return c_all
        return c_all[item_ids]


# =============================================================================
# §4.3.2  MovieLens: implicit behavioural friction
# =============================================================================

class MovieLensCost(nn.Module):
    """
    Implicit behavioural friction (Plan §3.3.4):

        C(i, t) = β1 · GenreRed(i, H_{u,<t})
                + β2 · RecencyPress(t)
                + β3 · PopPress(i, t)

    Inputs are PER-(USER-POSITION) — not per-item-id. We therefore
    cannot pre-compute "cost for every item id" as Amazon does. The
    main model passes in the precomputed scalars for the supervised
    position, then optionally broadcasts to the full catalogue for
    the choice loss.

    Coefficients
    ------------
    β1, β2, β3 are nn.Parameter scalars initialised at 0.01 (small
    positive, per §4.3.2). The plan warns to monitor β1 — if the
    EDA's "GenreRed saturation hypothesis" holds, β1 may drift toward
    zero or negative; that is empirically interesting and not a bug.

    PAD candidate
    -------------
    For PAD items (id 0), GenreRed is undefined. Callers should mask
    these out — the UtilityChoiceLoss does so via `-inf` on the PAD
    slot, so even an arbitrary cost at PAD is harmless.
    """

    def __init__(self) -> None:
        super().__init__()
        self.beta1 = nn.Parameter(torch.tensor(0.01))
        self.beta2 = nn.Parameter(torch.tensor(0.01))
        self.beta3 = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        genre_red: torch.Tensor,
        recency_press: torch.Tensor,
        pop_press: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        genre_red    : float tensor of any shape — the precomputed
                       GenreRed value(s).
        recency_press: same shape — normalised position-in-history.
        pop_press    : same shape — normalised dynamic Z(t).

        Returns
        -------
        cost : same shape as the inputs.
        """
        return (
            self.beta1 * genre_red
            + self.beta2 * recency_press
            + self.beta3 * pop_press
        )


# =============================================================================
# §4.3.3  Subjective Value V(u, i, t)
# =============================================================================

class BilinearValue(nn.Module):
    """
    Bilinear value head (Plan §4.3.3 default):

        V(u, i, t) = h_t^T W_V E_i(i)

    Given h_t ∈ R^{B×d} and the item embedding table E_i, we form
    h̃ = h_t @ W_V ∈ R^{B×d}, then take the inner product with every
    item embedding:
        V[b, j] = h̃[b, :] · E_i.weight[j, :]

    This yields a (B, V_cat) score matrix — the value at each batch's
    state against every candidate item — which is exactly what the
    choice loss (full-vocab softmax) expects.

    PAD slot (item id 0)
    --------------------
    The PAD row of E_i is zero, so V[:, 0] = 0 for all batches.
    Combined with UtilityChoiceLoss's -inf mask on PAD, the PAD slot
    never wins.
    """

    def __init__(self, d: int, item_embedding: nn.Embedding) -> None:
        super().__init__()
        self.d = d
        # Tied reference (NOT a copy) — see the module docstring.
        self.item_embedding = item_embedding
        self.W_V = nn.Linear(d, d, bias=False)
        nn.init.xavier_uniform_(self.W_V.weight)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        h_t : (B, d)
        Returns
        -------
        V : (B, V_cat)
        """
        h_proj = self.W_V(h_t)                       # (B, d)
        # E_i.weight: (V_cat, d). V_proj @ E^T → (B, V_cat).
        scores = h_proj @ self.item_embedding.weight.t()
        return scores


# =============================================================================
# §4.3.4  User cost-sensitivity λ_u network
# =============================================================================

class LambdaUNet(nn.Module):
    """
    λ_u = σ(w^T z_u),    z_u = mean_k E_i(i_k) over the user's history.

    The mean-pool is computed in `forward()`, taking PAD positions
    (item_id == 0) out of the denominator so they contribute zero
    weight. For the special case of an all-PAD row (no history),
    z_u falls back to the zero vector and λ_u = σ(0) = 0.5.

    Regularisation hook
    -------------------
    The plan flags possible λ_u → 0 degeneracy. We monitor that via
    the `lambda_log_penalty(...)` helper, which returns
    `- mean(log(λ_u))`. The training loop multiplies it by a small
    coefficient and adds to L_total. We don't apply it inside this
    module to keep the loss-bookkeeping in one place.

    Parameters
    ----------
    d              : embedding dimension (must match item_embedding.d).
    item_embedding : the shared nn.Embedding table from the backbone.
    """

    def __init__(self, d: int, item_embedding: nn.Embedding, pad_index: int = 0) -> None:
        super().__init__()
        self.d = d
        self.item_embedding = item_embedding  # tied reference
        self.pad_index = pad_index
        # Linear projection w: d -> 1 (no bias, per §4.3.4 spec).
        self.w = nn.Linear(d, 1, bias=False)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, item_ids_history: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        item_ids_history : (B, T) long — full user history (or
                           current input sequence). PAD positions are 0.

        Returns
        -------
        lambda_u : (B,) float in (0, 1).
        """
        # (B, T, d), PAD rows are zero already due to padding_idx=0.
        emb = self.item_embedding(item_ids_history)
        valid = (item_ids_history != self.pad_index).float().unsqueeze(-1)  # (B, T, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)  # (B, 1); avoids div-by-zero
        z_u = (emb * valid).sum(dim=1) / denom    # (B, d)
        logit = self.w(z_u).squeeze(-1)           # (B,)
        return torch.sigmoid(logit)

    @staticmethod
    def lambda_log_penalty(lambda_u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        -mean(log(λ_u + eps)). Increases as λ_u → 0. Add to L_total
        scaled by a small coefficient (≈ 1e-3) if monitoring shows
        λ_u collapsing.
        """
        return -(lambda_u.clamp_min(eps).log()).mean()


# =============================================================================
# §4.3 wrapper: EconomicsUtility
# =============================================================================

class EconomicsUtility(nn.Module):
    """
    Bundles the value head, the λ_u network, and a dataset-specific
    cost backend into one object exposed to the main CEINN model.

    The dataset is fixed at construction time via `dataset_name`:
      - "amazon_beauty" → `cost_backend = AmazonBeautyCost(...)`
      - "movielens"     → `cost_backend = MovieLensCost(...)`

    The two backends differ in their forward signatures, so
    `compute_utility(...)` exposes two distinct paths:
      `compute_utility_amazon(h_t, item_history)`
      `compute_utility_movielens(h_t, item_history, genre_red, recency_press, pop_press)`

    Both return (B, V_cat) utility tensors ready to feed into
    UtilityChoiceLoss.

    Build helpers
    -------------
    See `build_amazon_economics(loader, item_embedding, d)` and
    `build_movielens_economics(item_embedding, d)` further below —
    they construct the right wiring from the per-dataset Loader.
    """

    def __init__(
        self,
        dataset_name: str,
        d: int,
        item_embedding: nn.Embedding,
        cost_backend: nn.Module,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        if dataset_name not in ("amazon_beauty", "movielens"):
            raise ValueError(
                f"dataset_name must be 'amazon_beauty' or 'movielens', got {dataset_name!r}"
            )
        self.dataset_name = dataset_name
        self.d = d
        self.pad_index = pad_index
        self.value_head = BilinearValue(d=d, item_embedding=item_embedding)
        self.lambda_net = LambdaUNet(d=d, item_embedding=item_embedding, pad_index=pad_index)
        self.cost_backend = cost_backend

    # -------------------------------------------------------------------------
    # Amazon: cost is a property of the item id (no per-(u,t) state).
    # -------------------------------------------------------------------------
    def compute_utility_amazon(
        self,
        h_t: torch.Tensor,
        item_history: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        h_t          : (B, d) — latent state from the backbone.
        item_history : (B, T) — the input item ID sequence (used to
                       compute z_u for λ_u).

        Returns
        -------
        dict with keys 'V', 'C', 'lambda_u', 'U':
          V        : (B, V_cat) value scores per candidate
          C        : (V_cat,)    cost per item id (shared across batch)
          lambda_u : (B,)        user cost-sensitivity
          U        : (B, V_cat)  utility = V - λ_u * C
        """
        if not isinstance(self.cost_backend, AmazonBeautyCost):
            raise RuntimeError("compute_utility_amazon called but cost backend is not Amazon")

        V = self.value_head(h_t)                       # (B, V_cat)
        C = self.cost_backend.cost_all_items()         # (V_cat,)
        lambda_u = self.lambda_net(item_history)       # (B,)

        # Broadcast: (B, V_cat) − (B, 1) * (V_cat,) → (B, V_cat)
        U = V - lambda_u.unsqueeze(-1) * C.unsqueeze(0)
        return {"V": V, "C": C, "lambda_u": lambda_u, "U": U}

    # -------------------------------------------------------------------------
    # MovieLens: cost depends on (u, t) AND the candidate item, but the
    # three signals (GenreRed, RecencyPress, PopPress) are precomputed
    # in Phase-2 and indexed per (user, position) at training time. For
    # the candidate-loop they are usually broadcast over candidates by
    # the training loop (since for a fixed user-position, RecencyPress
    # is constant; GenreRed varies by candidate because it depends on
    # the candidate's genre vector; PopPress varies by candidate's
    # dynamic Z(t)). The training loop is responsible for forming the
    # (B, V_cat) shaped inputs; this module just plugs them in.
    # -------------------------------------------------------------------------
    def compute_utility_movielens(
        self,
        h_t: torch.Tensor,
        item_history: torch.Tensor,
        genre_red: torch.Tensor,
        recency_press: torch.Tensor,
        pop_press: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        h_t           : (B, d)
        item_history  : (B, T)  — for λ_u's mean-pool
        genre_red     : (B, V_cat) — Jaccard similarity per candidate
        recency_press : (B, V_cat) — same value across V_cat (broadcast OK)
        pop_press     : (B, V_cat) — normalised Z(t) per candidate

        Returns
        -------
        dict {'V', 'C', 'lambda_u', 'U'} all (B, V_cat).
        """
        if not isinstance(self.cost_backend, MovieLensCost):
            raise RuntimeError("compute_utility_movielens called but cost backend is not MovieLens")

        V = self.value_head(h_t)                                # (B, V_cat)
        C = self.cost_backend(genre_red, recency_press, pop_press)  # (B, V_cat)
        lambda_u = self.lambda_net(item_history)                # (B,)

        U = V - lambda_u.unsqueeze(-1) * C
        return {"V": V, "C": C, "lambda_u": lambda_u, "U": U}


# =============================================================================
# Build helpers (called by `models/ceinn.py` / `train.py`)
# =============================================================================

def build_amazon_cost_from_loader(loader) -> AmazonBeautyCost:
    """
    Construct AmazonBeautyCost from an AmazonBeautyLoader instance.

    Uses `loader.bulk_meta_arrays()` to pull (cat, brand, log_price, Z)
    arrays of shape (n_items + 1,) — index 0 is PAD with zeros.
    """
    arrs = loader.bulk_meta_arrays()
    return AmazonBeautyCost(
        n_items=int(loader.vocab["n_items"]),
        n_cats=int(loader.vocab["n_cats"]),
        n_brands=int(loader.vocab["n_brands"]),
        n_z_bins=int(loader.vocab["n_Z_bins"]),
        cat_ids=torch.from_numpy(arrs["cat"]),
        brand_ids=torch.from_numpy(arrs["brand"]),
        log_price=torch.from_numpy(arrs["log_price"]),
        z_bins=torch.from_numpy(arrs["Z"]),
    )
