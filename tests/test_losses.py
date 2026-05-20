"""
Unit tests for utils/losses.py (Plan §3.2 + §3.4.1).

Each loss has at least 3 cases, including the edge-cases the plan calls
out explicitly:
  * IPSLoss: behaviour as p_i → 0 (clipping should kick in)
  * UtilityChoiceLoss: behaviour on all-zero logits (softmax → uniform)
  * Both: PAD positions must NOT contribute to the loss or dilute the mean
  * All: gradients must be finite (no NaN / Inf) on a tiny toy model
"""

from __future__ import annotations

import math

import pytest
import torch

from utils.losses import (
    AdversarialCE,
    CombinedLoss,
    IPSLoss,
    SequentialCrossEntropy,
    UtilityChoiceLoss,
)


# ---------- SequentialCrossEntropy ------------------------------------------

class TestSequentialCrossEntropy:

    def test_hand_computed_loss(self):
        # 2 samples, 4 classes (incl PAD=0). Build logits so the true class
        # gets the highest score; loss should be small.
        logits = torch.tensor([[0.0, 5.0, 0.0, 0.0],
                               [0.0, 0.0, 5.0, 0.0]])
        target = torch.tensor([1, 2])
        loss = SequentialCrossEntropy(pad_index=0)(logits, target)
        # ~ -log(softmax(5)[i]) ≈ log(1 + 3*exp(-5)) ≈ 0.0202
        assert loss.item() < 0.05

    def test_pad_positions_ignored(self):
        logits = torch.tensor([[1.0, 2.0, 3.0],
                               [9.0, 9.0, 9.0]])  # PAD: any logits
        target = torch.tensor([1, 0])             # second is PAD
        # Loss should equal the per-sample CE of the first row alone.
        loss = SequentialCrossEntropy(pad_index=0)(logits, target)
        # Expected = -log(softmax([1,2,3])[1])
        expected = -math.log(math.exp(2) / (math.exp(1) + math.exp(2) + math.exp(3)))
        assert math.isclose(loss.item(), expected, abs_tol=1e-5)

    def test_grad_flows(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 10, requires_grad=True)
        target = torch.tensor([1, 2, 3, 4])
        loss = SequentialCrossEntropy(pad_index=0)(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_uniform_logits_gives_log_vocab(self):
        # Uniform → CE = log(V - 1) since PAD is excluded from softmax?
        # No — pytorch's CE still softmaxes over ALL V slots even with
        # ignore_index. So expected = log(V).
        V = 5
        logits = torch.zeros(3, V)
        target = torch.tensor([1, 2, 3])
        loss = SequentialCrossEntropy(pad_index=0)(logits, target)
        assert math.isclose(loss.item(), math.log(V), abs_tol=1e-5)


# ---------- IPSLoss ----------------------------------------------------------

class TestIPSLoss:

    def test_clipped_caps_low_propensity(self):
        # Make CE deliberately ≈ 1 by putting equal logits, then check
        # the IPS weight is exactly tau when p << 1/tau.
        V = 4
        # CE = log(V) ≈ 1.386 on uniform logits.
        logits = torch.zeros(2, V)
        target = torch.tensor([1, 2])
        propensity = torch.tensor([1e-9, 1e-9])  # 1/p ≈ 1e9
        loss = IPSLoss(variant="clipped", clip_tau=10.0)(logits, target, propensity)
        expected = 10.0 * math.log(V)
        assert math.isclose(loss.item(), expected, rel_tol=1e-4)

    def test_clipped_unclipped_at_normal_propensity(self):
        V = 4
        logits = torch.zeros(2, V)
        target = torch.tensor([1, 2])
        propensity = torch.tensor([0.5, 0.5])  # 1/p = 2 < tau
        loss = IPSLoss(variant="clipped", clip_tau=30.0)(logits, target, propensity)
        expected = 2.0 * math.log(V)
        assert math.isclose(loss.item(), expected, rel_tol=1e-4)

    def test_self_normalized_recovers_unit_scale(self):
        # SNIPS weights average to 1.0 across valid positions, so the
        # weighted CE should ≈ raw CE when all p are equal.
        V = 5
        logits = torch.zeros(8, V)
        target = torch.arange(1, 5).repeat(2)  # 1,2,3,4,1,2,3,4
        propensity = torch.full((8,), 0.3)
        loss = IPSLoss(variant="self_normalized")(logits, target, propensity)
        # ≈ log(V) since uniform weights.
        assert math.isclose(loss.item(), math.log(V), rel_tol=1e-3)

    def test_pad_positions_zero_contribution(self):
        V = 4
        logits = torch.zeros(3, V)
        target = torch.tensor([1, 0, 2])  # middle is PAD
        propensity = torch.tensor([1.0, 1e-9, 1.0])  # PAD has crazy p
        loss = IPSLoss(variant="clipped", clip_tau=30.0)(logits, target, propensity)
        # PAD must not blow up the loss. With p=1 on valid positions and
        # clipped tau, the value is 1.0 * log(V) regardless of PAD's p.
        assert math.isclose(loss.item(), math.log(V), rel_tol=1e-4)

    def test_grad_finite_with_tiny_propensity(self):
        torch.manual_seed(1)
        logits = torch.randn(4, 8, requires_grad=True)
        target = torch.tensor([1, 2, 3, 4])
        propensity = torch.tensor([1e-12, 0.5, 0.1, 0.9])
        loss = IPSLoss(variant="clipped", clip_tau=50.0)(logits, target, propensity)
        loss.backward()
        assert torch.isfinite(logits.grad).all()


# ---------- AdversarialCE ----------------------------------------------------

class TestAdversarialCE:

    def test_basic_ce(self):
        # 4 samples over K=3 Z-buckets, perfect predictions → low loss.
        logits = torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ])
        target = torch.tensor([0, 1, 2, 0])
        loss = AdversarialCE()(logits, target)
        assert loss.item() < 0.05

    def test_pad_label_ignored(self):
        logits = torch.tensor([[1.0, 2.0, 3.0],
                               [9.0, 9.0, 9.0]])
        target = torch.tensor([2, 0])  # 0 is the pad label here
        loss = AdversarialCE(pad_label=0)(logits, target)
        # Only the first row contributes.
        expected = -math.log(math.exp(3) / (math.exp(1) + math.exp(2) + math.exp(3)))
        assert math.isclose(loss.item(), expected, abs_tol=1e-5)

    def test_uniform_logits(self):
        K = 5
        logits = torch.zeros(10, K)
        target = torch.randint(0, K, (10,))
        loss = AdversarialCE()(logits, target)
        assert math.isclose(loss.item(), math.log(K), abs_tol=1e-5)


# ---------- UtilityChoiceLoss -----------------------------------------------

class TestUtilityChoiceLoss:

    def test_zero_utilities_uniform_softmax(self):
        # All-zero utilities, PAD masked → softmax over V-1 real items.
        V = 5
        utilities = torch.zeros(3, V)
        target = torch.tensor([1, 2, 3])
        loss = UtilityChoiceLoss(pad_index=0)(utilities, target)
        # PAD slot pushed to -inf → only V-1 candidates contribute.
        assert math.isclose(loss.item(), math.log(V - 1), abs_tol=1e-5)

    def test_high_utility_on_target_low_loss(self):
        V = 6
        utilities = torch.zeros(2, V)
        utilities[0, 3] = 10.0
        utilities[1, 5] = 10.0
        target = torch.tensor([3, 5])
        loss = UtilityChoiceLoss(pad_index=0)(utilities, target)
        assert loss.item() < 0.01

    def test_pad_target_ignored(self):
        V = 4
        utilities = torch.zeros(2, V)
        utilities[0, 2] = 5.0
        target = torch.tensor([2, 0])  # second is PAD
        loss = UtilityChoiceLoss(pad_index=0)(utilities, target)
        # Loss is the per-sample CE of the first row only.
        # Logits after PAD-masking: [-inf, 0, 5, 0].
        # softmax → exp(5) / (exp(0) + exp(5) + exp(0))
        expected = -math.log(math.exp(5) / (math.exp(5) + 2))
        assert math.isclose(loss.item(), expected, abs_tol=1e-5)

    def test_grad_flows(self):
        torch.manual_seed(2)
        util = torch.randn(4, 12, requires_grad=True)
        target = torch.tensor([3, 5, 7, 9])
        loss = UtilityChoiceLoss(pad_index=0)(util, target)
        loss.backward()
        assert torch.isfinite(util.grad).all()


# ---------- CombinedLoss -----------------------------------------------------

class TestCombinedLoss:

    def test_weighted_sum(self):
        ips = torch.tensor(2.0)
        adv = torch.tensor(1.5)
        choice = torch.tensor(0.5)
        cl = CombinedLoss(lambda_adv=0.2)
        out = cl(ips=ips, adv=adv, choice=choice)
        # 2.0 + 0.2 * 1.5 + 0.5 = 2.8
        assert math.isclose(out["total"].item(), 2.8, abs_tol=1e-6)

    def test_components_detached_in_log(self):
        ips = torch.tensor(2.0, requires_grad=True)
        adv = torch.tensor(1.5, requires_grad=True)
        choice = torch.tensor(0.5, requires_grad=True)
        cl = CombinedLoss(lambda_adv=0.5)
        out = cl(ips=ips, adv=adv, choice=choice)
        # Logged components must NOT carry grad.
        assert not out["L_IPS"].requires_grad
        assert not out["L_adv"].requires_grad
        # But the total must be differentiable end-to-end.
        out["total"].backward()
        assert ips.grad is not None and torch.isfinite(ips.grad)

    def test_l2_added_when_provided(self):
        ips = torch.tensor(1.0)
        adv = torch.tensor(1.0)
        choice = torch.tensor(1.0)
        cl = CombinedLoss(lambda_adv=1.0, lambda_reg=0.1)
        out = cl(ips=ips, adv=adv, choice=choice, l2_norm_sq=torch.tensor(10.0))
        # 1 + 1 + 1 + 0.1*10 = 4
        assert math.isclose(out["total"].item(), 4.0, abs_tol=1e-6)
