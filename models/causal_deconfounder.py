"""
Module 3.2 — Causal Deconfounder (Plan §3.2, Phase 4 §4.2).

Three subcomponents, each independently testable:

  1. `GradientReversalLayer` (GRL): torch.autograd.Function that is the
     identity on forward and multiplies the gradient by -alpha on
     backward. Implements the adversarial coupling between the
     backbone (which wants h_t ⊥ Z) and the discriminator (which wants
     to recover Z from h_t).

  2. `PropensityEstimator`: shallow MLP that estimates the propensity
     score p_i = P(E_i = 1 | Z_i). The exposure-vs-not binary form is
     adopted per §4.2.1 recommendation: training labels are 1 for
     observed (u, i) interactions in the train set, 0 for uniformly
     sampled negatives. The estimator takes the confounder bucket Z_i
     as its sole input and returns σ(MLP(E_Z(Z_i))).

  3. `Discriminator`: 2-layer MLP that predicts the K-way confounder
     bucket from h_t. The forward signature accepts pre-GRL h_t or
     applies GRL internally — see `forward()` doc.

GRL alpha schedule
------------------
Per §4.2.2, alpha follows the DANN-style sigmoid-warmup schedule:

    alpha(p) = 2 / (1 + exp(-10 * p)) - 1,
    p = current_epoch / total_epochs

`alpha_schedule(p)` exposes this as a pure function so the training
loop can pre-compute alpha each epoch and pass it in.

Responsibility boundaries
-------------------------
* This module DOES NOT compute losses. It produces logits / probabilities
  only. The adversarial CE is in `utils/losses.py::AdversarialCE`.
* This module DOES NOT own the confounder bucket extraction. For Amazon
  Beauty `Z_i` is the static salesRank bucket loaded from `item_meta`;
  for MovieLens it's the discrete bucketization of dynamic Z_i(t).
  Both come pre-computed from Phase 2; this module just consumes them.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# =============================================================================
# §4.2.2  Gradient Reversal Layer
# =============================================================================

class _GradientReversalFunction(torch.autograd.Function):
    """
    Forward: identity (returns input unchanged).
    Backward: multiplies the incoming gradient by `-alpha`.

    Reference: Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation
    by Backpropagation".
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        # Stash alpha on the context for the backward pass.
        ctx.alpha = float(alpha)
        # Use `x.view_as(x)` (not `x.clone()`) so the forward is a true
        # no-op on the data tensor; this is the canonical idiom that
        # PyTorch's own examples follow. `clone` would be correct too,
        # but `view_as` is cheaper.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Note: `alpha` is a Python float here, NOT a tensor — we don't
        # need a gradient w.r.t. it, so the second return is None.
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Functional handle to the GRL. Use this inside `forward()`:

        h_rev = grad_reverse(h_t, alpha=current_alpha)
        z_logits = discriminator(h_rev)

    The discriminator's gradients still update its own parameters
    normally; only the gradient that flows back *into h_t* gets
    sign-flipped (and scaled by alpha). This is what produces the
    adversarial pressure on the backbone.
    """
    return _GradientReversalFunction.apply(x, alpha)


def alpha_schedule(p: float) -> float:
    """
    DANN-style sigmoid-warmup schedule (Plan §4.2.2):

        alpha(p) = 2 / (1 + exp(-10 * p)) - 1

    p is expected in [0, 1]; the function is well-defined outside this
    range but only [0, 1] is meaningful. At p=0 alpha=0 (no GRL pressure);
    at p=1 alpha ≈ 0.9999 (≈ full pressure).
    """
    return 2.0 / (1.0 + math.exp(-10.0 * float(p))) - 1.0


# =============================================================================
# §4.2.1  Propensity Estimator: p_i = P(E_i = 1 | Z_i)
# =============================================================================

class PropensityEstimator(nn.Module):
    """
    Shallow 2-layer MLP that estimates the propensity score p_i.

    Per §4.2.1 we recommend the Sigmoid (binary exposure) form:
        - input: Z_i ∈ {0, 1, ..., K-1} via a small embedding (E_Z)
        - hidden: 1 hidden layer of size `hidden_dim` (default 64)
                  with ReLU activation
        - output: scalar sigmoid → exposure probability in (0, 1)

    The propensity is a PROPERTY OF THE ITEM (given its confounder
    bucket), not of a (user, item) pair. So this estimator is queried
    per-item-id during training and the resulting p_i is broadcast to
    all training positions whose target item is i.

    Training labels (handled by `train.py`, not here)
    -------------------------------------------------
    Positive: item-ids that appear in the training set (any user).
    Negative: uniformly random item-ids not present (random sampling).
    Binary cross-entropy against this label gives the propensity model
    its supervision signal. This module simply outputs the predicted
    probability; the loss is computed by `train.py` using a standard
    BCE-with-logits formulation (we return both `logit` and `prob`
    accessors for that purpose).

    Parameters
    ----------
    n_z_buckets : the K in P(E=1 | Z). For Amazon Beauty K=10
                   (salesRank deciles); for MovieLens K should match
                   whatever discretisation `train.py` chooses for the
                   dynamic Z(t) values.
    z_emb_dim   : the dimension of the Z-bucket embedding lookup.
                   Default 16 — the input is low-cardinality so this
                   need not be large.
    hidden_dim  : MLP hidden width (Plan §4.2.1 suggests 64).
    """

    def __init__(
        self,
        n_z_buckets: int,
        z_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_z_buckets = n_z_buckets
        self.z_embedding = nn.Embedding(n_z_buckets, z_emb_dim)
        nn.init.xavier_uniform_(self.z_embedding.weight)

        self.mlp = nn.Sequential(
            nn.Linear(z_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_buckets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_buckets : LongTensor of shape (N,) — integer confounder labels
                    in [0, n_z_buckets).

        Returns
        -------
        logit : (N,) FloatTensor — pre-sigmoid logits. Use
                `torch.sigmoid(logit)` to get the probability, or pass
                directly to F.binary_cross_entropy_with_logits at
                training time.
        """
        z_emb = self.z_embedding(z_buckets)
        return self.mlp(z_emb).squeeze(-1)

    def predict_proba(self, z_buckets: torch.Tensor) -> torch.Tensor:
        """Convenience: sigmoid of the logit. Used at evaluation time."""
        return torch.sigmoid(self.forward(z_buckets))


# =============================================================================
# §4.2.3  Discriminator: D(h_t) → Z bucket logits
# =============================================================================

class Discriminator(nn.Module):
    """
    2-layer MLP that tries to recover the confounder bucket Z from the
    latent state h_t (Plan §4.2.3).

    Architecture:
        Linear(d → hidden) → ReLU → (Dropout) → Linear(hidden → K)

    Parameters
    ----------
    d          : input dim (= backbone d).
    n_z_buckets: K, number of confounder bucket classes (Amazon=10;
                  MovieLens chooses its own discretisation).
    hidden_dim : MLP hidden width (default 2*d, capped at 128).
    dropout    : optional dropout between the two linear layers.

    Forward behaviour
    -----------------
    By default the discriminator does NOT apply GRL internally — the
    caller is expected to wrap h_t with `grad_reverse(h_t, alpha)`
    before passing it in. We expose the GRL-baking option behind an
    explicit `apply_grl=True` flag for convenience when the model is
    used as a single black box.

    Returns
    -------
    z_logits : (N, K) FloatTensor — unnormalised logits over the K
               confounder buckets. Feed into
               `utils.losses.AdversarialCE` together with the true Z
               labels to compute L_adv.
    """

    def __init__(
        self,
        d: int,
        n_z_buckets: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_z_buckets = n_z_buckets
        h = hidden_dim if hidden_dim is not None else min(128, max(64, 2 * d))
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h, n_z_buckets),
        )

    def forward(
        self,
        h: torch.Tensor,
        *,
        apply_grl: bool = False,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        h         : (N, d) latent states.
        apply_grl : if True, wrap h in GRL(alpha) before the MLP. If
                    False (default), the caller is responsible.
        alpha     : GRL strength (only used when apply_grl=True).
        """
        if apply_grl:
            h = grad_reverse(h, alpha=alpha)
        return self.net(h)


# =============================================================================
# §4.4.1  Container — bundles the three subcomponents for `CEINNModel`
# =============================================================================

class CausalDeconfounder(nn.Module):
    """
    Convenience container that holds the discriminator, the propensity
    estimator, and the GRL alpha state in one place. The CEINN main
    model owns one of these.

    The propensity estimator is included here for organisational
    purposes — conceptually p_i is part of the deconfounding mechanism
    even though its forward signature is independent of h_t.

    Public methods
    --------------
    discriminate(h_t, alpha) → z_logits  (applies GRL with `alpha`)
    propensity(z_buckets)    → p_i ∈ (0, 1)
    propensity_logit(z_buckets) → raw logit (use with BCE-with-logits)
    """

    def __init__(
        self,
        d: int,
        n_z_buckets: int,
        propensity_hidden: int = 64,
        propensity_z_emb_dim: int = 16,
        discriminator_hidden: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.propensity_estimator = PropensityEstimator(
            n_z_buckets=n_z_buckets,
            z_emb_dim=propensity_z_emb_dim,
            hidden_dim=propensity_hidden,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            d=d,
            n_z_buckets=n_z_buckets,
            hidden_dim=discriminator_hidden,
            dropout=dropout,
        )
        self.n_z_buckets = n_z_buckets

    # ---- propensity -----------------------------------------------------
    def propensity_logit(self, z_buckets: torch.Tensor) -> torch.Tensor:
        return self.propensity_estimator(z_buckets)

    def propensity(self, z_buckets: torch.Tensor) -> torch.Tensor:
        return self.propensity_estimator.predict_proba(z_buckets)

    # ---- discriminator with GRL ----------------------------------------
    def discriminate(self, h_t: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward h_t through GRL(alpha) → discriminator. Use the current
        epoch's alpha (see `alpha_schedule`).
        """
        return self.discriminator(h_t, apply_grl=True, alpha=alpha)
