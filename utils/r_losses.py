"""
Loss functions for CEINN training (Plan §3.2).

Responsibility boundaries
-------------------------
This module computes losses **given inputs**. It does NOT:
  * estimate propensity scores p_i  → that's `models/causal_deconfounder.py`
    (Phase 4)
  * implement the gradient reversal layer  → also `causal_deconfounder.py`
    (the adversarial CE here is the raw cross-entropy; GRL flips the
    gradient sign before it ever reaches this module)
  * pull anything off disk → all configuration arrives via constructor args
    populated by `train.py` from YAML

Each loss returns a SCALAR torch.Tensor with grad enabled. `CombinedLoss`
takes the already-computed scalars (rather than recomputing them) so the
caller can log each term independently — vital for monitoring whether
λ_adv has flipped the optimisation into adversarial collapse.

PAD masking (critical)
----------------------
Phase 2 reserves index 0 as PAD for both `item_idx` and `dt_bin`. Any
loss that does a full-vocab softmax over items MUST be passed logits
that include the PAD slot (so dimension matches the embedding table) and
must IGNORE positions whose target == 0. `CrossEntropyLoss(ignore_index=0)`
is the canonical PyTorch idiom for this and is used throughout.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# §3.2.1  Sequential cross-entropy  L_seq
# =============================================================================

class SequentialCrossEntropy(nn.Module):
    """
    Standard next-item prediction loss (Plan §3.2.1).

    This is the baseline for ablation A2 (w/o IPS) and the natural
    sanity-check target during early training.

    Parameters
    ----------
    pad_index : the item ID reserved as padding (Phase 2 uses 0).
    label_smoothing : if > 0, applies torch's built-in label smoothing
                      to the full-vocab cross-entropy. Off by default.
    """

    def __init__(self, pad_index: int = 0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.ce = nn.CrossEntropyLoss(
            ignore_index=pad_index,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, T, V) or (B, V) — last dim is the full item vocabulary,
                 PAD slot included.
        target : (B, T) or (B,) — int64 item indices; PAD positions == 0.
        """
        if logits.dim() == 3:
            B, T, V = logits.shape
            return self.ce(logits.reshape(B * T, V), target.reshape(B * T))
        return self.ce(logits, target)


# =============================================================================
# §3.2.2  Inverse-propensity weighted loss  L_IPS
# =============================================================================

class IPSLoss(nn.Module):
    """
    Inverse Propensity Scoring loss for sequential recommendation
    (Plan §3.2.2 + §3.2.7 variance reduction).

    The caller is responsible for providing per-sample propensities
    `p` ∈ (0, 1], one per supervised position. This module:
      1. forms the IPS weights w = 1 / p
      2. applies one of three variance-reduction strategies:
           - "clipped"             : w = min(1/p, τ)
           - "self_normalized"     : w = (1/p) / sum_batch(1/p)   (SNIPS)
           - "clipped_self_normalized" : clip first, then SN.
             Robust to extreme propensities AND batch-size effects;
             this is the recommended default for MovieLens, where
             max(1/p) was empirically ~7000 per EDA and a single
             outlier can otherwise dominate the SNIPS denominator
             (root cause §3.6). On a stable propensity distribution
             this variant reduces to plain SNIPS.
      3. computes per-sample cross-entropy and returns the weighted mean

    Why not estimate p here?
    ------------------------
    The propensity estimator (§3.2.5) is a small MLP trained alongside
    the backbone in `models/causal_deconfounder.py`. Keeping its outputs
    flowing in from outside lets us swap "true" propensities, oracle
    propensities, or even uniform propensities (=ablation A4 control)
    without touching this file.

    Notes on PAD
    ------------
    Positions whose target == `pad_index` contribute zero weight and are
    explicitly excluded from the normalisation denominator. This matches
    the behaviour of `nn.CrossEntropyLoss(ignore_index=...)` on the
    underlying CE term and prevents PAD positions from diluting the IPS
    mean.
    """

    _ALLOWED_VARIANTS = ("clipped", "self_normalized", "clipped_self_normalized")

    def __init__(
        self,
        variant: Literal["clipped", "self_normalized", "clipped_self_normalized"] = "clipped",
        clip_tau: float = 30.0,
        pad_index: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if variant not in self._ALLOWED_VARIANTS:
            raise ValueError(
                f"IPSLoss: unknown variant {variant!r}; "
                f"expected one of {self._ALLOWED_VARIANTS}"
            )
        self.variant = variant
        self.clip_tau = float(clip_tau)
        self.pad_index = pad_index
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        propensity: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits     : (B*T, V) flat — per-position scores over full vocab.
        target     : (B*T,)   flat — int64 item ids; PAD == pad_index.
        propensity : (B*T,)   flat — float in (0, 1]; ignored at PAD slots.

        For convenience the inputs can also be (B, T, V), (B, T), (B, T):
        they are flattened internally.
        """
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)
            target = target.reshape(B * T)
            propensity = propensity.reshape(B * T)

        # Per-sample cross-entropy with PAD already excluded by mask below.
        ce_per_sample = F.cross_entropy(
            logits, target, reduction="none", ignore_index=self.pad_index
        )
        # NOTE: when ignore_index fires, cross_entropy returns 0 for that
        # position. We re-derive the mask to keep the denominator honest.
        valid_mask = (target != self.pad_index).to(logits.dtype)

        # Clamp propensity to avoid division explosions before any clipping.
        p_safe = torch.clamp(propensity, min=self.eps).to(logits.dtype)
        raw_w = 1.0 / p_safe

        if self.variant == "clipped":
            w = torch.clamp(raw_w, max=self.clip_tau)
        elif self.variant == "self_normalized":
            masked_raw = raw_w * valid_mask
            denom = masked_raw.sum().clamp_min(self.eps)
            # Multiply by valid_count to bring the SN weights back onto the
            # same numerical scale as the clipped variant (their average is
            # 1.0 over valid positions). Without this, the SNIPS loss would
            # be ~1/N smaller than the clipped one and silently dominated
            # by L_adv and L_C in CombinedLoss.
            valid_count = valid_mask.sum().clamp_min(1.0)
            w = (raw_w / denom) * valid_count
        else:  # clipped_self_normalized — §3.6 fix.
            # First cap raw_w to suppress extreme outliers, THEN self-
            # normalise. With clip_tau ≈ 100 this rules out a single
            # 1/p ≈ 7000 outlier dominating the SNIPS denominator while
            # preserving the unbiasedness improvement of SNIPS over
            # pure clipping on the bulk of the distribution.
            capped = torch.clamp(raw_w, max=self.clip_tau)
            masked_capped = capped * valid_mask
            denom = masked_capped.sum().clamp_min(self.eps)
            valid_count = valid_mask.sum().clamp_min(1.0)
            w = (capped / denom) * valid_count

        w = w * valid_mask
        weighted = (ce_per_sample * w).sum()
        denom = valid_mask.sum().clamp_min(1.0)
        return weighted / denom


# =============================================================================
# §3.2.3  Adversarial cross-entropy  L_adv
# =============================================================================

class AdversarialCE(nn.Module):
    """
    Cross-entropy between the discriminator's prediction of the confounder
    Z (popularity / salesRank bucket) from the latent state h_t, and the
    true bucket label (Plan §3.2.3).

    GRL is NOT here — it lives in `models/causal_deconfounder.py`. This
    module is intentionally agnostic about gradient direction; whoever
    wires up the discriminator decides whether the upstream gets flipped
    gradients or not.

    Parameters
    ----------
    pad_label : optional Z label to ignore (used when the dynamic Z is
                undefined for the very first interaction of a user). If
                Phase 2's dynamic Z uses 0 to mean "no history", pass
                pad_label=0 here. None disables masking entirely.
    """

    def __init__(self, pad_label: Optional[int] = None) -> None:
        super().__init__()
        self.pad_label = pad_label
        if pad_label is None:
            self.ce = nn.CrossEntropyLoss(reduction="mean")
        else:
            self.ce = nn.CrossEntropyLoss(
                ignore_index=int(pad_label), reduction="mean"
            )

    def forward(self, z_logits: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        z_logits : (N, K) — discriminator logits over K Z-buckets.
        z_target : (N,)   — int64 true Z-bucket labels.
        """
        if z_logits.dim() == 3:
            N1, N2, K = z_logits.shape
            z_logits = z_logits.reshape(N1 * N2, K)
            z_target = z_target.reshape(N1 * N2)
        return self.ce(z_logits, z_target)


# =============================================================================
# §3.2.4  Utility-based discrete choice loss  L_C
# =============================================================================

class UtilityChoiceLoss(nn.Module):
    """
    Negative log-likelihood of the chosen item under a softmax over the
    *full* item catalogue (Plan §3.2.4 and §3.3.6).

    Inputs
    ------
    utilities : (B*T, V) per-(user-state, candidate-item) utility scores
                U(u, i, t) = V(u, i, t) - λ_u * C(i, t). The caller
                (Phase 4's `models/economics_utility.py`) computes these
                for the full V-sized catalogue at each supervised
                position. PAD slot at index 0 is included for dimension
                consistency but masked out via -inf so it never wins the
                softmax.
    target    : (B*T,) int64 — the true chosen item index; PAD == 0.

    Implementation notes
    --------------------
    * We set utilities[:, pad_index] = -inf before the softmax so the PAD
      slot is mass-less. This is the correct way to "exclude PAD from
      the candidate set" without changing the softmax denominator size.
    * The PAD *target* positions are still skipped via ignore_index.
    * The plan explicitly notes (Phase 3 §3.2.4) that full-vocab softmax
      is affordable at V ≈ 12k / 10k. We do NOT subsample.
    """

    def __init__(self, pad_index: int = 0) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_index, reduction="mean")

    def forward(self, utilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if utilities.dim() == 3:
            B, T, V = utilities.shape
            utilities = utilities.reshape(B * T, V)
            target = target.reshape(B * T)

        # Mask the PAD candidate from the choice set without touching the
        # tensor shape.
        masked = utilities.clone()
        masked[:, self.pad_index] = float("-inf")
        return self.ce(masked, target)


# =============================================================================
# §3.2.5  Combined objective
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Weighted sum of the four CEINN losses (Plan §3.2.5):

        L_total = L_IPS + λ_adv * L_adv + L_C + λ_reg * ||θ||²

    Notes
    -----
    * `lambda_reg` is applied here for bookkeeping, but in practice you
      should rely on `torch.optim.Optimizer(weight_decay=...)` to
      compute the L2 term — it's substantially cheaper and equivalent
      up to the optimiser update rule. The `l2_norm_sq` argument is
      provided for completeness so the term shows up in logs when needed.
    * `L_seq` is not part of the combined objective at the main training
      branch; it is used for ablation A2 (substitute L_IPS with L_seq).
      The training loop chooses which to feed in via the `ips` slot.
    """

    def __init__(self, lambda_adv: float, lambda_reg: float = 0.0) -> None:
        super().__init__()
        self.lambda_adv = float(lambda_adv)
        self.lambda_reg = float(lambda_reg)

    def forward(
        self,
        *,
        ips: torch.Tensor,
        adv: torch.Tensor,
        choice: torch.Tensor,
        l2_norm_sq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with the scalar `total` and each component for
        logging. Each component appears in the dict UNWEIGHTED, while
        `total` carries the λ-weighting.
        """
        total = ips + self.lambda_adv * adv + choice
        if l2_norm_sq is not None and self.lambda_reg > 0.0:
            total = total + self.lambda_reg * l2_norm_sq

        out: Dict[str, torch.Tensor] = {
            "total": total,
            "L_IPS": ips.detach(),
            "L_adv": adv.detach(),
            "L_C": choice.detach(),
        }
        if l2_norm_sq is not None:
            out["L_reg"] = l2_norm_sq.detach()
        return out
