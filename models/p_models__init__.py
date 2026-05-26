"""
CEINN model package (Phase 4).

Public surface:
    SequentialBackbone      — §3.1
    JointEmbedding          — §3.1.3
    GradientReversal / GRL  — §3.2.8 / §4.2.2
    alpha_schedule          — DANN-style warm-up
    PropensityEstimator     — §3.2.5 / §4.2.1
    Discriminator           — §4.2.3
    CausalDeconfounder      — bundle of the three above
    AmazonBeautyCost        — §3.3.3
    MovieLensCost           — §3.3.4
    BilinearValue           — §3.3.5 / §4.3.3
    LambdaUNet              — §3.3.5 / §4.3.4
    EconomicsUtility        — §3.3 bundle
    CEINNModel              — §4.4 main wiring
    build_ceinn_amazon(...) — factory from AmazonBeautyLoader
    build_ceinn_movielens(...) — factory from MovieLens10MLoader
"""

from .ceinn import (
    CEINNModel,
    build_ceinn_amazon,
    build_ceinn_movielens,
)
from .causal_deconfounder import (
    CausalDeconfounder,
    Discriminator,
    PropensityEstimator,
    alpha_schedule,
    grad_reverse,
)
from .economics_utility import (
    AmazonBeautyCost,
    BilinearValue,
    EconomicsUtility,
    LambdaUNet,
    MovieLensCost,
    build_amazon_cost_from_loader,
)
from .sequential_backbone import (
    JointEmbedding,
    SequentialBackbone,
)

__all__ = [
    "CEINNModel",
    "build_ceinn_amazon",
    "build_ceinn_movielens",
    "CausalDeconfounder",
    "Discriminator",
    "PropensityEstimator",
    "alpha_schedule",
    "grad_reverse",
    "AmazonBeautyCost",
    "BilinearValue",
    "EconomicsUtility",
    "LambdaUNet",
    "MovieLensCost",
    "build_amazon_cost_from_loader",
    "JointEmbedding",
    "SequentialBackbone",
]
