"""
Phase 4 unit tests.

Per the Phase 4 quality gate, three properties must be verified:
  1. (§4.1.4) Causal mask isolates future information from past outputs.
  2. (§4.2.4) GRL flips the gradient direction on the backbone w.r.t.
              the discriminator's parameter gradients.
  3. (§4.4.2) End-to-end forward pass produces finite outputs; backward
              pass populates non-zero gradients on every learnable param.

Run with:
    pytest tests/test_models_phase4.py -q
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

# Make the repository root importable when pytest is invoked from
# anywhere — same trick as `scripts/build_popularity_groups.py`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.causal_deconfounder import (  # noqa: E402
    CausalDeconfounder,
    Discriminator,
    PropensityEstimator,
    alpha_schedule,
    grad_reverse,
)
from models.ceinn import CEINNModel  # noqa: E402
from models.economics_utility import (  # noqa: E402
    AmazonBeautyCost,
    BilinearValue,
    LambdaUNet,
    MovieLensCost,
)
from models.sequential_backbone import (  # noqa: E402
    JointEmbedding,
    SequentialBackbone,
)


# =============================================================================
# Small fixture builders
# =============================================================================

def _amazon_dummy_cost(n_items: int = 50, n_cats: int = 8, n_brands: int = 6, n_z: int = 10) -> AmazonBeautyCost:
    """A toy AmazonBeautyCost with random per-item meta arrays."""
    g = torch.Generator().manual_seed(0)
    cat_ids   = torch.randint(0, n_cats, (n_items + 1,), generator=g)
    brand_ids = torch.randint(0, n_brands, (n_items + 1,), generator=g)
    log_price = torch.randn(n_items + 1, generator=g)
    z_bins    = torch.randint(0, n_z, (n_items + 1,), generator=g)
    # Set the PAD row to defaults (0/0/0/0) for determinism.
    cat_ids[0] = 0; brand_ids[0] = 0; log_price[0] = 0.0; z_bins[0] = 0
    return AmazonBeautyCost(
        n_items=n_items, n_cats=n_cats, n_brands=n_brands, n_z_bins=n_z,
        cat_ids=cat_ids, brand_ids=brand_ids, log_price=log_price, z_bins=z_bins,
    )


def _toy_ceinn(dataset: str = "amazon_beauty") -> CEINNModel:
    torch.manual_seed(0)
    n_items = 50
    if dataset == "amazon_beauty":
        cost = _amazon_dummy_cost(n_items=n_items)
        return CEINNModel(
            dataset_name="amazon_beauty",
            cost_backend=cost,
            n_items=n_items,
            n_rating_bins=6,
            n_dt_bins=32,
            n_z_buckets=10,
            d=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            max_seq_len=20,
        )
    else:
        return CEINNModel(
            dataset_name="movielens",
            cost_backend=MovieLensCost(),
            n_items=n_items,
            n_rating_bins=11,
            n_dt_bins=64,
            n_z_buckets=10,
            d=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            max_seq_len=20,
        )


def _toy_batch(
    B: int = 4, T: int = 10,
    n_items: int = 50, n_rating: int = 6, n_dt: int = 32,
    seed: int = 0,
):
    """A right-padded batch: each row has its last K positions = PAD."""
    g = torch.Generator().manual_seed(seed)
    item_ids = torch.randint(1, n_items + 1, (B, T), generator=g)
    rating_ids = torch.randint(1, n_rating + 1, (B, T), generator=g)
    dt_ids = torch.randint(1, n_dt + 1, (B, T), generator=g)
    return item_ids, rating_ids, dt_ids


# =============================================================================
# §4.1.4  Backbone tests
# =============================================================================

class TestBackbone:

    def test_output_shape(self):
        bb = SequentialBackbone(
            n_items=50, n_rating_bins=6, n_dt_bins=32,
            d=64, n_heads=4, n_layers=2, dropout=0.0, max_seq_len=20,
        )
        item_ids, rating_ids, dt_ids = _toy_batch()
        h = bb(item_ids, rating_ids, dt_ids)
        assert h.shape == (4, 64), f"unexpected h shape: {h.shape}"
        assert torch.isfinite(h).all()

    def test_returns_full_sequence_when_requested(self):
        bb = SequentialBackbone(
            n_items=50, n_rating_bins=6, n_dt_bins=32,
            d=64, n_heads=4, n_layers=2, dropout=0.0, max_seq_len=20,
        )
        item_ids, rating_ids, dt_ids = _toy_batch()
        h_full = bb(item_ids, rating_ids, dt_ids, return_full_sequence=True)
        assert h_full.shape == (4, 10, 64)

    def test_causal_mask_isolates_future(self):
        """
        Per §4.1.4: modifying input at positions t+1..n must not change
        the encoder output at position t. We verify this on the *full
        sequence* (the per-position outputs).

        Build two batches that agree on the first half (positions 0..4)
        and differ on the second half (positions 5..9). Under strict
        causal masking, outputs at positions 0..4 must be identical
        in eval mode.
        """
        bb = SequentialBackbone(
            n_items=50, n_rating_bins=6, n_dt_bins=32,
            d=64, n_heads=4, n_layers=2, dropout=0.0, max_seq_len=20,
        )
        bb.eval()  # disable dropout

        torch.manual_seed(0)
        B, T = 2, 10
        item1 = torch.randint(1, 51, (B, T))
        rating1 = torch.randint(1, 7, (B, T))
        dt1 = torch.randint(1, 33, (B, T))

        # Perturb the second half (positions 5..9) only.
        item2 = item1.clone()
        rating2 = rating1.clone()
        dt2 = dt1.clone()
        torch.manual_seed(99)
        item2[:, 5:]   = torch.randint(1, 51, (B, T - 5))
        rating2[:, 5:] = torch.randint(1, 7, (B, T - 5))
        dt2[:, 5:]     = torch.randint(1, 33, (B, T - 5))

        with torch.no_grad():
            h1 = bb(item1, rating1, dt1, return_full_sequence=True)
            h2 = bb(item2, rating2, dt2, return_full_sequence=True)

        # Positions 0..4 must match exactly under causal masking.
        first_half_max_diff = (h1[:, :5] - h2[:, :5]).abs().max().item()
        assert first_half_max_diff < 1e-6, (
            f"Causal mask violated: first-half outputs differ by "
            f"max abs {first_half_max_diff:.2e} (expected ~0)"
        )

        # Sanity: the second half DOES differ.
        second_half_max_diff = (h1[:, 5:] - h2[:, 5:]).abs().max().item()
        assert second_half_max_diff > 1e-6, (
            "Second-half outputs accidentally identical — test is "
            "ineffective. Check perturbation."
        )

    def test_last_valid_position_extraction(self):
        """h_t should be picked at the last non-PAD position of each row."""
        bb = SequentialBackbone(
            n_items=50, n_rating_bins=6, n_dt_bins=32,
            d=64, n_heads=4, n_layers=2, dropout=0.0, max_seq_len=20,
        )
        bb.eval()
        B, T = 3, 10
        # Row 0: full (length 10) — last pos = 9
        # Row 1: length 6 — pad positions 6..9
        # Row 2: length 3 — pad positions 3..9
        item_ids = torch.randint(1, 51, (B, T))
        rating_ids = torch.randint(1, 7, (B, T))
        dt_ids = torch.randint(1, 33, (B, T))
        item_ids[1, 6:] = 0; rating_ids[1, 6:] = 0; dt_ids[1, 6:] = 0
        item_ids[2, 3:] = 0; rating_ids[2, 3:] = 0; dt_ids[2, 3:] = 0

        with torch.no_grad():
            h_full = bb(item_ids, rating_ids, dt_ids, return_full_sequence=True)
            h_last = bb(item_ids, rating_ids, dt_ids, return_full_sequence=False)

        # Expected last positions: 9, 5, 2.
        assert torch.allclose(h_last[0], h_full[0, 9])
        assert torch.allclose(h_last[1], h_full[1, 5])
        assert torch.allclose(h_last[2], h_full[2, 2])

    def test_pad_embedding_is_zero(self):
        """JointEmbedding must return exactly zero at PAD positions."""
        je = JointEmbedding(
            n_items=50, n_rating_bins=6, n_dt_bins=32, d=64, pad_index=0,
        )
        item_ids   = torch.tensor([[1, 2, 0, 0]])
        rating_ids = torch.tensor([[1, 1, 0, 0]])
        dt_ids     = torch.tensor([[1, 2, 0, 0]])
        e = je(item_ids, rating_ids, dt_ids)
        # PAD positions exactly zero.
        assert torch.allclose(e[0, 2], torch.zeros(64))
        assert torch.allclose(e[0, 3], torch.zeros(64))
        # Real positions non-zero.
        assert e[0, 0].abs().sum() > 0
        assert e[0, 1].abs().sum() > 0


# =============================================================================
# §4.2.4  GRL / discriminator tests
# =============================================================================

class TestDeconfounder:

    def test_grl_forward_is_identity(self):
        x = torch.randn(8, 16, requires_grad=True)
        y = grad_reverse(x, alpha=0.7)
        assert torch.allclose(y, x), "GRL forward must be identity"

    def test_grl_backward_flips_sign(self):
        """
        Build two identical inputs:
          - x1 goes straight through a linear layer
          - x2 goes through GRL(alpha=0.5) then the SAME linear layer
        With the same downstream loss, dL/dx2 should equal
        −alpha · dL/dx1.
        """
        torch.manual_seed(0)
        linear = nn.Linear(8, 4)
        alpha = 0.5

        x1 = torch.randn(3, 8, requires_grad=True)
        y1 = linear(x1)
        loss1 = y1.sum()
        loss1.backward()
        g1 = x1.grad.clone()

        # Re-do with GRL inserted.
        x2 = x1.detach().clone().requires_grad_(True)
        y2 = linear(grad_reverse(x2, alpha=alpha))
        loss2 = y2.sum()
        loss2.backward()
        g2 = x2.grad.clone()

        assert torch.allclose(g2, -alpha * g1, atol=1e-6), (
            f"GRL gradient mismatch: max diff "
            f"{(g2 - (-alpha * g1)).abs().max().item():.2e}"
        )

    def test_grl_zero_alpha_blocks_backbone_gradient(self):
        """alpha=0 → backbone gradient from adversarial path is zero."""
        torch.manual_seed(0)
        D = Discriminator(d=16, n_z_buckets=5)
        h = torch.randn(4, 16, requires_grad=True)
        z = torch.randint(0, 5, (4,))
        z_logits = D(h, apply_grl=True, alpha=0.0)
        loss = nn.functional.cross_entropy(z_logits, z)
        loss.backward()
        # The disc's params should still receive gradient — only the
        # signal flowing back into h is zeroed.
        assert h.grad is not None
        assert torch.allclose(h.grad, torch.zeros_like(h.grad)), (
            "alpha=0 must zero the gradient flowing into the backbone"
        )

    def test_alpha_schedule_monotone(self):
        """alpha(0) = 0, alpha(1) ≈ 1, monotone non-decreasing."""
        assert alpha_schedule(0.0) == pytest.approx(0.0)
        assert alpha_schedule(1.0) == pytest.approx(0.999909, abs=1e-4)
        vals = [alpha_schedule(p) for p in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]]
        for a, b in zip(vals, vals[1:]):
            assert a <= b + 1e-9

    def test_propensity_estimator_shape_and_range(self):
        pe = PropensityEstimator(n_z_buckets=10)
        z = torch.randint(0, 10, (16,))
        logit = pe(z)
        prob = pe.predict_proba(z)
        assert logit.shape == (16,)
        assert prob.shape == (16,)
        assert (prob > 0).all() and (prob < 1).all()

    def test_discriminator_grl_flips_backbone_gradient_direction(self):
        """
        End-to-end §4.2.4 check: when we compute L_adv and backprop,
        the gradient flowing into h_t (the backbone output) must point
        in the OPPOSITE direction to the gradient flowing into the
        discriminator's first-layer weights.

        Operationally: insert GRL(alpha=1.0), compute CE loss, run
        backward, verify sign relationship.
        """
        torch.manual_seed(42)
        D = Discriminator(d=16, n_z_buckets=5)
        h = torch.randn(8, 16, requires_grad=True)
        z = torch.randint(0, 5, (8,))

        # 1) Without GRL: control case (no gradient reversal).
        D_ctrl = Discriminator(d=16, n_z_buckets=5)
        D_ctrl.load_state_dict(D.state_dict())
        h_ctrl = h.detach().clone().requires_grad_(True)
        z_logits_ctrl = D_ctrl(h_ctrl, apply_grl=False)
        ce_ctrl = nn.functional.cross_entropy(z_logits_ctrl, z)
        ce_ctrl.backward()
        g_h_ctrl = h_ctrl.grad.clone()

        # 2) With GRL(alpha=1.0): gradient on h must be the negation.
        z_logits = D(h, apply_grl=True, alpha=1.0)
        ce = nn.functional.cross_entropy(z_logits, z)
        ce.backward()
        g_h = h.grad.clone()

        assert torch.allclose(g_h, -g_h_ctrl, atol=1e-6), (
            "GRL(alpha=1) should produce backbone gradient opposite to "
            "the un-reversed case."
        )


# =============================================================================
# §4.3  Economics utility tests
# =============================================================================

class TestEconomicsUtility:

    def test_amazon_cost_finite_and_correct_shape(self):
        cost = _amazon_dummy_cost()
        c_all = cost.cost_all_items()
        assert c_all.shape == (51,)
        assert torch.isfinite(c_all).all()

    def test_movielens_cost_pure_linear(self):
        mc = MovieLensCost()
        g = torch.tensor([0.0, 0.5, 1.0])
        r = torch.tensor([0.1, 0.5, 0.9])
        p = torch.tensor([0.2, 0.5, 0.8])
        c = mc(g, r, p)
        # By construction: c = β1*g + β2*r + β3*p, with the three β's
        # initialised at 0.01 each.
        expected = 0.01 * (g + r + p)
        assert torch.allclose(c, expected, atol=1e-6)

    def test_bilinear_value_shape(self):
        torch.manual_seed(0)
        item_emb = nn.Embedding(51, 16, padding_idx=0)
        bv = BilinearValue(d=16, item_embedding=item_emb)
        h = torch.randn(4, 16)
        v = bv(h)
        assert v.shape == (4, 51)

    def test_bilinear_value_pad_slot_zero(self):
        """V[:, 0] must be zero because E_i[0] is zero (PAD row)."""
        item_emb = nn.Embedding(51, 16, padding_idx=0)
        # Re-initialise weights but keep PAD row zero.
        nn.init.xavier_uniform_(item_emb.weight)
        with torch.no_grad():
            item_emb.weight[0].zero_()
        bv = BilinearValue(d=16, item_embedding=item_emb)
        h = torch.randn(4, 16)
        v = bv(h)
        assert torch.allclose(v[:, 0], torch.zeros(4), atol=1e-6)

    def test_lambda_u_in_range(self):
        item_emb = nn.Embedding(51, 16, padding_idx=0)
        nn.init.xavier_uniform_(item_emb.weight)
        with torch.no_grad():
            item_emb.weight[0].zero_()
        ln = LambdaUNet(d=16, item_embedding=item_emb)
        hist = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0],
            [0, 0, 0, 0, 0],  # empty history
        ])
        lam = ln(hist)
        assert lam.shape == (3,)
        assert (lam > 0).all() and (lam < 1).all()
        # Empty-history row should be exactly σ(0) = 0.5 because z_u = 0.
        assert lam[2].item() == pytest.approx(0.5, abs=1e-6)


# =============================================================================
# §4.4.2  CEINN integration test
# =============================================================================

class TestCEINNIntegration:

    def _amazon_batch(self, n_items=50, B=4, T=10):
        item_ids, rating_ids, dt_ids = _toy_batch(
            B=B, T=T, n_items=n_items, n_rating=6, n_dt=32,
        )
        z_target = torch.randint(0, 10, (B,))
        z_cand = torch.randint(0, 10, (n_items + 1,))
        return {
            "item_ids": item_ids,
            "rating_ids": rating_ids,
            "dt_ids": dt_ids,
            "z_buckets_target": z_target,
            "z_buckets_candidates": z_cand,
        }, item_ids

    def test_amazon_forward_finite(self):
        model = _toy_ceinn("amazon_beauty")
        batch, _ = self._amazon_batch()
        out = model(batch, grl_alpha=0.5)
        for k in ("h_t", "U", "V", "C", "lambda_u",
                  "z_logits", "propensity_target", "propensity_candidates"):
            assert k in out, f"missing key {k!r}"
            assert torch.isfinite(out[k]).all(), f"NaN/Inf in {k}"
        assert out["U"].shape == (4, 51)
        assert out["lambda_u"].shape == (4,)
        assert out["z_logits"].shape == (4, 10)

    def test_amazon_backward_populates_all_gradients(self):
        """
        §4.4.2 — combined loss (using ALL three losses) backwards must
        populate non-zero grads on every learnable param.
        """
        from utils.losses import (
            AdversarialCE, CombinedLoss, IPSLoss, UtilityChoiceLoss,
        )

        model = _toy_ceinn("amazon_beauty")
        batch, item_ids = self._amazon_batch()

        # Forward.
        out = model(batch, grl_alpha=1.0)

        # Pick targets (random for the toy test).
        B = item_ids.shape[0]
        target = torch.randint(1, 51, (B,))

        # L_IPS uses V (raw item softmax over the full catalogue) as a
        # stand-in for the sequential-CE branch (per Phase-3
        # `IPSLoss` signature). For this integration test we feed the
        # bilinear V scores as logits.
        ips_loss = IPSLoss(variant="clipped", clip_tau=30.0, pad_index=0)
        # propensity broadcast: just gather propensity of target items.
        # Use the candidate-table to look up p_i for the target's id.
        p_target_per_sample = out["propensity_candidates"][target]
        L_ips = ips_loss(out["V"], target, p_target_per_sample)

        adv_loss = AdversarialCE(pad_label=None)
        L_adv = adv_loss(out["z_logits"], batch["z_buckets_target"])

        choice_loss = UtilityChoiceLoss(pad_index=0)
        L_C = choice_loss(out["U"], target)

        combined = CombinedLoss(lambda_adv=0.1)
        total = combined(ips=L_ips, adv=L_adv, choice=L_C)["total"]

        assert torch.isfinite(total)
        total.backward()

        # Walk every parameter; check at least one element has
        # |grad| > 0. For embedding-PAD rows, grad is zero by design
        # (padding_idx=0); we exclude those.
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"param {name} has no grad"
            # For embeddings, ignore the PAD row when checking non-zero.
            if "embedding" in name and p.dim() >= 2 and p.shape[0] > 1:
                check = p.grad[1:]
            else:
                check = p.grad
            assert check.abs().sum().item() > 0, (
                f"param {name} received zero gradient — possible "
                f"disconnection from the loss"
            )

    def test_movielens_forward_finite(self):
        model = _toy_ceinn("movielens")
        B, T = 4, 10
        n_items = 50
        item_ids, rating_ids, dt_ids = _toy_batch(
            B=B, T=T, n_items=n_items, n_rating=11, n_dt=64,
        )
        # MovieLens-specific per-candidate signals.
        # (B, V_cat) tensors with V_cat = n_items + 1 = 51.
        V_cat = n_items + 1
        torch.manual_seed(1)
        genre_red = torch.rand(B, V_cat)
        recency = torch.rand(B, V_cat) * 0.5
        pop_press = torch.rand(B, V_cat)
        batch = {
            "item_ids": item_ids,
            "rating_ids": rating_ids,
            "dt_ids": dt_ids,
            "z_buckets_target": torch.randint(0, 10, (B,)),
            "genre_red": genre_red,
            "recency_press": recency,
            "pop_press": pop_press,
            "z_buckets_candidates": torch.randint(0, 10, (B, V_cat)),
        }
        out = model(batch, grl_alpha=0.7)
        assert out["U"].shape == (B, V_cat)
        assert out["C"].shape == (B, V_cat)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"

    def test_shared_item_embedding_storage(self):
        """
        §3.3.5: the item embedding used by BilinearValue and LambdaUNet
        must be the SAME tensor as in the backbone (not a copy).
        """
        model = _toy_ceinn("amazon_beauty")
        assert (
            model.backbone.item_embedding.weight.data_ptr()
            == model.economics.value_head.item_embedding.weight.data_ptr()
        ), "BilinearValue's item_embedding is not tied to the backbone"
        assert (
            model.backbone.item_embedding.weight.data_ptr()
            == model.economics.lambda_net.item_embedding.weight.data_ptr()
        ), "LambdaUNet's item_embedding is not tied to the backbone"

    def test_propensity_is_in_unit_interval(self):
        model = _toy_ceinn("amazon_beauty")
        batch, _ = self._amazon_batch()
        out = model(batch, grl_alpha=0.0)
        p = out["propensity_candidates"]
        assert (p > 0).all() and (p < 1).all()
