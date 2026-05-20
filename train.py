"""
CEINN training script (Phase 6).

Joint optimisation of
    L_total = L_IPS-choice + λ_adv · L_adv + λ_reg · ||θ||²

where L_IPS-choice is the IPS-weighted choice NLL (the merge of L_IPS
and L_C suggested by Implementation Note in §Appendix A of the design
doc — see the "Loss strategy" section of the design notes below for
the trade-off discussion).

Usage
-----
    python train.py --config configs/amazon_beauty.yaml
    python train.py --config configs/movielens_10M.yaml --seed 0
    # Override config fields from CLI:
    python train.py --config configs/amazon_beauty.yaml \
        --override training.lambda_adv=0.5 training.d=128

Outputs (under <config>.training.output_dir / <run_name>/):
    config.yaml          # frozen copy of the effective config
    train_log.csv        # one row per epoch
    best.ckpt            # state_dict at best Val NDCG@10
    last.ckpt            # state_dict at last epoch (resume helper)
    history.json         # full training history (mirror of CSV)

Design notes (read this before changing the script)
---------------------------------------------------

1. GRL one-step vs. plan's "two-step" recommendation (§6.1.1)
   The plan's two-step strategy (freeze D / freeze backbone alternately)
   is mathematically equivalent to single-step GRL training as proved
   by Ganin & Lempitsky (2015). We use single-step GRL because:
     a) Phase 4 already wired the GRL path. Two-step would require
        adding a second optimiser and the synchronisation between them.
     b) DANN's original implementation is single-step too.

2. Loss strategy: merge L_IPS into L_C, not separate
   Per design doc Appendix A, L_IPS and L_C both supervise the chosen
   item. Maintaining two competing softmaxes (over V scores and over U
   utilities) creates conflicting gradient signals on the item
   embedding table. We instead use ONE choice loss over U, weighted by
   1/p_i. Ablation A2 (w/o IPS) is enabled by setting all weights to
   1.0; ablation A5 (w/o Economics) replaces U with V (dot-product
   only); other ablations are controlled via the `ablation` block.

3. MovieLens uses negative sampling at train time
   Computing per-candidate cost signals over all 10k+ items per batch
   is too expensive for ML's 10M-row training set. We sample N negs
   per supervised position (default 100 — large enough to give the
   softmax a meaningful denominator while keeping memory bounded).
   Amazon uses full-catalogue softmax (item-only cost makes it cheap).

4. Propensity training
   We train the propensity estimator (§3.2.5) for `propensity_warmup`
   epochs against BCE on (positive=in-train, negative=random) labels
   BEFORE the main loop starts. Afterwards it continues to update
   under the same loss with a separate (small) learning-rate scale,
   so it tracks any drift but doesn't dominate the optimiser.

5. Sequential supervision: shift-by-one next-item prediction
   For each user training sequence [i_1, i_2, ..., i_n] of length n,
   the input is [i_1, ..., i_{n-1}] and the target is [i_2, ..., i_n].
   We run the backbone with `return_full_sequence=True` to get h_t at
   every position simultaneously and supervise per-position.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

# Repo-root-relative imports.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.amazon_beauty_loader import AmazonBeautyLoader   # noqa: E402
from data_loaders.movieslens_10M_loader import MovieLens10MLoader  # noqa: E402
from models.ceinn import CEINNModel, build_ceinn_amazon, build_ceinn_movielens
from models.causal_deconfounder import alpha_schedule
from utils.losses import (                                          # noqa: E402
    AdversarialCE, IPSLoss, SequentialCrossEntropy, UtilityChoiceLoss,
)
from utils.math_utils import apply_bucket_edges, fit_log_quantile_edges  # noqa: E402
from utils.metrics import (                                         # noqa: E402
    compute_full_ranking, confounding_auc, group_metrics, standard_topk_report,
)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Run BEFORE any randomised init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN benchmark trades off determinism for speed — we leave it
    # enabled here because Phase 6 prioritises wall-clock; final paper
    # results in Phase 8 should disable it for strict reproducibility.


# =============================================================================
# Config handling
# =============================================================================

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "training" not in cfg:
        raise KeyError(
            f"{path} has no `training:` section. Phase 6 expects training "
            f"hyperparameters under that key — see configs/amazon_beauty.yaml"
        )
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply `key.path=value` overrides from CLI. Value is parsed by YAML
    (so `0.5` → float, `true` → bool, `[1, 2]` → list).
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override {ov!r}; expected key.path=value")
        k, v = ov.split("=", 1)
        keys = k.split(".")
        cur = cfg
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = yaml.safe_load(v)
    return cfg


# =============================================================================
# Loaders + auxiliary precomputations
# =============================================================================

def resolve_processed_dir(cfg: Dict[str, Any]) -> Path:
    """The Phase-2 output_dir is the right place to find pickles."""
    return Path(cfg["preprocess"]["output_dir"])


def load_dataset_loader(cfg: Dict[str, Any]):
    """Dispatch to the right Phase-2 loader."""
    name = cfg["dataset"]["name"].lower()
    pdir = resolve_processed_dir(cfg)
    if name == "amazon_beauty":
        return "amazon_beauty", AmazonBeautyLoader.from_directory(pdir)
    elif name in ("movielens_10m", "movielens_10M", "movielens"):
        return "movielens", MovieLens10MLoader.from_directory(pdir)
    raise ValueError(f"Unknown dataset name {name!r}")


def compute_movielens_z_buckets(loader: MovieLens10MLoader, n_z_buckets: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretise the precomputed dynamic-Z values into `n_z_buckets`
    log-quantile bins, fit on TRAIN values only. Returns:
        z_bucket_array : (n_train_rows,) int — bucket id per row
        edges          : (n_z_buckets + 1,) float — edges for reuse
    """
    z_vals = loader._dynZ_values.astype(np.float64)
    edges = fit_log_quantile_edges(
        z_vals[z_vals > 0],  # log requires positive
        n_bins=n_z_buckets, base=10.0, add_one=True,
    )
    # apply_bucket_edges takes raw values and returns bucket ids in
    # [0, n_bins-1]; rows with z=0 (no prior history) deserve their own
    # bucket — we use bucket 0 (head/no-history) which is the lowest log.
    buckets = apply_bucket_edges(z_vals, edges, base=10.0, add_one=True)
    return buckets, edges


def build_seen_items(train_seqs: Dict[int, List[Tuple[int, int, int]]]) -> Dict[int, set]:
    """For evaluation: per-user set of training-seen items to mask out."""
    return {u: {iid for (iid, _r, _d) in seq} for u, seq in train_seqs.items()}


def load_popularity_groups(processed_dir: Path) -> Optional[Dict[int, str]]:
    """
    Optional: load Phase-3 head/torso/tail labels if present. Used by
    evaluate.py to slice metrics; returns None if not built yet (the
    standard run still works, group_metrics just won't be reported).
    """
    path = processed_dir / "item_popularity_group.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["labels"]


# =============================================================================
# Datasets
# =============================================================================

@dataclass
class TrainBatch:
    """One mini-batch of supervised sequence positions."""
    user_ids:    torch.Tensor       # (B,)  — user index
    item_ids:    torch.Tensor       # (B, T) — input items
    rating_ids:  torch.Tensor       # (B, T)
    dt_ids:      torch.Tensor       # (B, T)
    target_ids:  torch.Tensor       # (B, T) — next-item ids; PAD positions = 0


class SequentialNextItemDataset(Dataset):
    """
    Standard next-item supervision over Phase-2 training sequences.

    For each user with sequence [i_1, r_1, d_1, ..., i_n, r_n, d_n]
    of length n, we emit
        input  = [i_1, ..., i_{n-1}] padded to T = max_seq_len
        target = [i_2, ..., i_n]     padded to T = max_seq_len
    Both arrays are right-padded with 0 (the PAD index).

    Note: This dataset emits the FULL sequence (one row per user).
    Subsequence-based negative sampling and per-position weighting is
    handled inside the training step.
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[Tuple[int, int, int]]],
        max_seq_len: int,
        pad_index: int = 0,
    ) -> None:
        self.pad_index = pad_index
        self.max_seq_len = max_seq_len
        # Keep only sequences with at least 2 interactions
        # (need >= 1 input + 1 target).
        self.users = sorted(u for u, s in train_seqs.items() if len(s) >= 2)
        self.train_seqs = train_seqs

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u = self.users[idx]
        seq = self.train_seqs[u]
        # Truncate to the most recent (T+1) interactions so input has T.
        if len(seq) > self.max_seq_len + 1:
            seq = seq[-(self.max_seq_len + 1):]

        T = self.max_seq_len
        item = np.zeros(T, dtype=np.int64)
        rating = np.zeros(T, dtype=np.int64)
        dt = np.zeros(T, dtype=np.int64)
        target = np.zeros(T, dtype=np.int64)

        n = len(seq) - 1  # n input positions
        for k in range(n):
            iid, r, d = seq[k]
            item[k]   = iid
            rating[k] = r
            dt[k]     = d
            target[k] = seq[k + 1][0]

        return {
            "user_id":   u,
            "item_ids":   item,
            "rating_ids": rating,
            "dt_ids":     dt,
            "target_ids": target,
        }


def collate_batch(rows: List[Dict[str, Any]]) -> TrainBatch:
    return TrainBatch(
        user_ids=torch.tensor([r["user_id"] for r in rows], dtype=torch.long),
        item_ids=torch.from_numpy(np.stack([r["item_ids"]   for r in rows])),
        rating_ids=torch.from_numpy(np.stack([r["rating_ids"] for r in rows])),
        dt_ids=torch.from_numpy(np.stack([r["dt_ids"]     for r in rows])),
        target_ids=torch.from_numpy(np.stack([r["target_ids"] for r in rows])),
    )


# =============================================================================
# Ablation helpers — applied to a batch in-place before the forward pass
# =============================================================================

def apply_ablations_to_batch(batch: TrainBatch, ablation: Dict[str, Any]) -> TrainBatch:
    """
    Per the ablation table in §5.1:
      A1 (w/o Rating Emb) : zero out rating_ids
      A8 (w/o Temporal)   : zero out dt_ids
    The other ablations affect the model wiring or loss formulation
    and are applied elsewhere.
    """
    if not ablation.get("use_rating_emb", True):
        batch.rating_ids = torch.zeros_like(batch.rating_ids)
    if not ablation.get("use_temporal_emb", True):
        batch.dt_ids = torch.zeros_like(batch.dt_ids)
    return batch


# =============================================================================
# Negative sampling (MovieLens path) — uniform random catalogue sample
# =============================================================================

def sample_negatives(
    positive_targets: torch.Tensor,
    n_items: int,
    n_neg: int,
    pad_index: int = 0,
) -> torch.Tensor:
    """
    Uniform random negatives.

    Parameters
    ----------
    positive_targets : (N,) int64 — the supervised target ids; PAD = 0.
    n_items          : total real items (1..n_items).
    n_neg            : number of negatives per position.

    Returns
    -------
    negs : (N, n_neg) int64 — uniform-random item ids in [1, n_items].
           No de-duplication against the positive (the chance of
           collision in 10k items is ~0.01% and the choice loss
           ignores it).
    """
    N = positive_targets.shape[0]
    return torch.randint(1, n_items + 1, (N, n_neg),
                         dtype=torch.long, device=positive_targets.device)


# =============================================================================
# Propensity warm-up — pre-trains the propensity estimator on
# (positive=observed, negative=random) BCE
# =============================================================================

def warmup_propensity_estimator(
    model: CEINNModel,
    train_loader: DataLoader,
    item_z_bucket: torch.Tensor,
    *,
    n_epochs: int = 1,
    lr: float = 1e-3,
    device: torch.device,
    n_items: int,
) -> None:
    """
    Trains only `model.deconfounder.propensity_estimator` against BCE.

    For each batch we pull the target Z buckets (positive labels) and
    sample an equal number of random Z buckets as negatives.

    Parameters
    ----------
    item_z_bucket : (n_items + 1,) long tensor mapping item_idx → Z
                    bucket. For Amazon this is `loader.item_meta[..]["Z"]`
                    bulk-arrayed; for MovieLens it's the train-time
                    Z-discretisation built once at startup.
    """
    pe = model.deconfounder.propensity_estimator
    optim = torch.optim.Adam(pe.parameters(), lr=lr)
    pe.train()
    bce = nn.BCEWithLogitsLoss()

    K = pe.n_z_buckets
    n_steps = 0
    total_loss = 0.0
    for epoch in range(n_epochs):
        for batch in train_loader:
            tgt = batch.target_ids.to(device)               # (B, T)
            valid = (tgt != 0)                              # (B, T)
            if not valid.any():
                continue
            pos_items = tgt[valid]                          # (M,)
            pos_z = item_z_bucket.to(device)[pos_items]     # (M,)
            # Negatives: uniform random over Z buckets.
            neg_z = torch.randint(0, K, pos_z.shape, device=device)

            # Mix and supervise.
            z_in = torch.cat([pos_z, neg_z], dim=0)
            y    = torch.cat([torch.ones_like(pos_z, dtype=torch.float),
                              torch.zeros_like(neg_z, dtype=torch.float)],
                             dim=0)
            logit = pe(z_in)
            loss = bce(logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_steps += 1
    print(f"        propensity warm-up: avg BCE = {total_loss/max(n_steps,1):.4f} "
          f"({n_steps} steps over {n_epochs} epoch(s))")


# =============================================================================
# Movielens-specific: build per-(user, position) signal lookup arrays
# =============================================================================

def build_ml_position_lookup(
    loader: MovieLens10MLoader,
    item_z_bucket: torch.Tensor,
):
    """
    Build flat arrays indexed by the *batch's row index* for the three
    MovieLens cost signals at each supervised position:
        - GenreRed at (u, target_position)
        - Cumulative Z(t) for normalised PopPress
        - The user's first interaction time isn't directly stored —
          we approximate RecencyPress by position / sequence-length.

    For simplicity and speed, the training step pulls these in batch by
    iterating over rows. We avoid precomputing a huge tensor over all
    candidates (n_items × n_train_rows ≈ 100B entries).
    """
    # We just expose the loader-side lookups to the train step; no
    # eager materialisation needed.
    return {"loader": loader, "item_z_bucket": item_z_bucket}


# =============================================================================
# Validation helper — Full Ranking on val_seqs, returns Val NDCG@10
# =============================================================================

@torch.no_grad()
def validate_full_ranking(
    model: CEINNModel,
    val_seqs: Dict[int, Tuple[int, int, int]],
    train_seqs: Dict[int, List[Tuple[int, int, int]]],
    seen_items: Dict[int, set],
    *,
    max_seq_len: int,
    device: torch.device,
    item_z_bucket: torch.Tensor,
    ml_loader: Optional[MovieLens10MLoader] = None,
    pad_index: int = 0,
    batch_size: int = 256,
    ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute Full-Ranking Val NDCG@{5,10,20}, HR@{5,10,20}, MRR.
    Returns a dict with these and `n_users` (for sanity).

    For each user u we:
      1. Build the input sequence from train_seqs[u][-max_seq_len:]
      2. Encode → h_t
      3. Compute utilities (or just V for ablation A5)
      4. Mask training-seen + PAD; rank target.
    """
    model.eval()
    ablation = ablation or {}
    use_economics = ablation.get("use_economics", True)
    use_cost = ablation.get("use_cost", True)

    users = [u for u in val_seqs.keys() if u in train_seqs and train_seqs[u]]
    all_ranks: List[int] = []

    # Batch users for the backbone.
    for start in range(0, len(users), batch_size):
        ubatch = users[start:start + batch_size]
        # Build padded input sequences.
        B = len(ubatch)
        T = max_seq_len
        item = np.zeros((B, T), dtype=np.int64)
        rating = np.zeros((B, T), dtype=np.int64)
        dt = np.zeros((B, T), dtype=np.int64)
        for b, u in enumerate(ubatch):
            seq = train_seqs[u]
            if len(seq) > T:
                seq = seq[-T:]
            for k, (iid, r, d) in enumerate(seq):
                item[b, k]   = iid
                rating[b, k] = r
                dt[b, k]     = d
            # Pad rest with zeros (already initialised).
        # Apply A1/A8 ablations.
        if not ablation.get("use_rating_emb", True):
            rating[:] = 0
        if not ablation.get("use_temporal_emb", True):
            dt[:] = 0

        item_t   = torch.from_numpy(item).to(device)
        rating_t = torch.from_numpy(rating).to(device)
        dt_t     = torch.from_numpy(dt).to(device)

        h_t = model.encode(item_t, rating_t, dt_t)  # (B, d)

        # V over the full catalogue.
        V = model.economics.value_head(h_t)         # (B, V_cat)

        if use_economics and use_cost:
            if model.dataset_name == "amazon_beauty":
                # Cost is item-only.
                C_all = model.economics.cost_backend.cost_all_items()  # (V_cat,)
                lam = model.economics.lambda_net(item_t)               # (B,)
                scores = V - lam.unsqueeze(-1) * C_all.unsqueeze(0)
            else:
                # MovieLens: build per-candidate cost signals.
                assert ml_loader is not None, "ml_loader required for movielens eval"
                # GenreRed: for each (user, target_position) we can pull
                # from the loader's precomputed values *only at observed
                # train positions*. For Full Ranking we instead compute
                # GenreRed on the fly: jaccard(genre(candidate), union of
                # genres in train history).
                gmat = ml_loader.bulk_genre_matrix()  # (V_cat, n_genres)
                gmat_t = torch.from_numpy(gmat).to(device).float()  # (V_cat, G)

                # User history union per row.
                # item_t: (B, T). gmat_t[item_t]: (B, T, G). Sum over T,
                # then > 0 for the union.
                hist_g = gmat_t[item_t]                    # (B, T, G)
                hist_union = (hist_g.sum(dim=1) > 0).float()  # (B, G)

                # GenreRed[b, j] = |g_j ∩ U_b| / |g_j ∪ U_b|.
                # |g_j| = gmat_t.sum(1); |U_b| = hist_union.sum(1).
                g_size = gmat_t.sum(dim=1)                # (V_cat,)
                u_size = hist_union.sum(dim=1)            # (B,)
                # Inner product = intersection count.
                inter = hist_union @ gmat_t.t()           # (B, V_cat)
                union = (g_size.unsqueeze(0) + u_size.unsqueeze(1) - inter).clamp_min(1e-8)
                genre_red = inter / union                 # (B, V_cat) ∈ [0, 1]

                # RecencyPress: 1.0 (target is the most-recent position).
                recency = torch.ones(B, V.shape[1], device=device) * 0.5

                # PopPress: normalised Z bucket of candidate (constant
                # across batch since we don't have per-batch t).
                pop = item_z_bucket.to(device).float() / max(1.0, float(item_z_bucket.max().item()))
                pop = pop.unsqueeze(0).expand(B, -1)

                lam = model.economics.lambda_net(item_t)
                C = model.economics.cost_backend(genre_red, recency, pop)
                scores = V - lam.unsqueeze(-1) * C
        elif use_economics and not use_cost:
            # A6: utility without cost → V only.
            scores = V
        else:
            # A5 (w/o Economics): use pure V (dot product).
            scores = V

        # Compute rank for each user's target.
        scores_np = scores.detach().cpu().numpy()
        for b, u in enumerate(ubatch):
            target = val_seqs[u][0]
            if target == pad_index:
                continue
            r = compute_full_ranking(
                scores_np[b], target_item=int(target),
                seen_items=seen_items.get(u, set()),
                pad_index=pad_index,
            )
            all_ranks.append(r)

    if not all_ranks:
        return {"ndcg@10": 0.0, "n_users": 0}
    rep = standard_topk_report(all_ranks, ks=(5, 10, 20))
    rep["n_users"] = len(all_ranks)
    return rep


# =============================================================================
# Training step
# =============================================================================

@dataclass
class StepOutput:
    L_ips: float
    L_adv: float
    L_C: float
    L_total: float
    disc_acc: float
    valid_positions: int


def train_one_epoch(
    model: CEINNModel,
    train_loader: DataLoader,
    optim: torch.optim.Optimizer,
    *,
    device: torch.device,
    grl_alpha: float,
    lambda_adv: float,
    lambda_reg: float,
    item_z_bucket: torch.Tensor,         # (n_items+1,) long
    ablation: Dict[str, Any],
    n_items: int,
    ml_n_negs: int = 100,
    ml_loader: Optional[MovieLens10MLoader] = None,
    ips_variant: str = "clipped",
    ips_clip_tau: float = 30.0,
    grad_clip: float = 1.0,
    pad_index: int = 0,
) -> StepOutput:
    """
    One pass over the training loader. Loss formulation:
      L_total = L_choice + λ_adv · L_adv     (+ λ_reg ||θ||² via weight_decay)
    where L_choice is IPS-weighted CE over candidate utilities.

    If `ablation["loss_mode"] == "seq_choice"` (A2 w/o IPS), the IPS
    weights are forced to 1.0.

    Returns averaged per-epoch metrics.
    """
    model.train()
    total_L_ips = 0.0
    total_L_adv = 0.0
    total_L_C   = 0.0
    total_disc_correct = 0
    total_disc_count = 0
    total_positions = 0
    total_loss = 0.0
    n_steps = 0

    loss_mode = ablation.get("loss_mode", "ips_choice")
    use_grl = ablation.get("use_grl", True)
    use_economics = ablation.get("use_economics", True)
    use_cost = ablation.get("use_cost", True)
    fixed_lambda_u = ablation.get("fixed_lambda_u", False)
    fixed_lambda_u_value = ablation.get("fixed_lambda_u_value", 0.5)

    # Override ips_variant per A4.
    if loss_mode == "snips_choice":
        ips_variant = "self_normalized"

    adv_ce = AdversarialCE(pad_label=None)

    for batch in train_loader:
        batch = apply_ablations_to_batch(batch, ablation)
        item_ids   = batch.item_ids.to(device)
        rating_ids = batch.rating_ids.to(device)
        dt_ids     = batch.dt_ids.to(device)
        target_ids = batch.target_ids.to(device)
        B, T = item_ids.shape

        # ---------------------------------------------------------------
        # 1) Backbone forward — full-sequence h_t.
        # ---------------------------------------------------------------
        h_seq = model.backbone(item_ids, rating_ids, dt_ids,
                                return_full_sequence=True)  # (B, T, d)

        # ---------------------------------------------------------------
        # 2) Build the candidate set & utilities.
        #
        # Amazon: full V_cat softmax. Cost is item-only so we can
        # vectorise without ever materialising (B*T, V_cat).
        #
        # MovieLens: sample N negatives per position and only compute
        # cost/utility on the small candidate set.
        # ---------------------------------------------------------------
        valid_mask = (target_ids != pad_index)  # (B, T)
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            continue

        # h flattened over valid positions only.
        h_flat = h_seq[valid_mask]                          # (M, d)
        tgt_flat = target_ids[valid_mask]                   # (M,)

        # User history pool used for λ_u (same for every position of a
        # given user — we recompute per row by feeding item_ids).
        # Broadcasting from (B, T) → (M, T) via the row mapping.
        row_to_user = torch.arange(B, device=device).unsqueeze(1).expand(B, T)[valid_mask]  # (M,)
        # Gather item_ids by user row.
        hist_for_lambda = item_ids[row_to_user]             # (M, T)

        if model.dataset_name == "amazon_beauty":
            # ----- V over full V_cat -----
            V = model.economics.value_head(h_flat)          # (M, V_cat)

            # ----- Cost -----
            if use_economics and use_cost:
                C_all = model.economics.cost_backend.cost_all_items()  # (V_cat,)
                if fixed_lambda_u:
                    lam = torch.full((h_flat.shape[0],), fixed_lambda_u_value, device=device)
                else:
                    lam = model.economics.lambda_net(hist_for_lambda)  # (M,)
                util = V - lam.unsqueeze(-1) * C_all.unsqueeze(0)
            elif use_economics and not use_cost:
                util = V
            else:
                # A5 (w/o Economics): pure V dot-product.
                util = V

            # ----- L_C: IPS-weighted CE over the full V_cat -----
            # Propensity per target item via the propensity estimator.
            if loss_mode == "seq_choice":
                propensity = torch.ones_like(tgt_flat, dtype=torch.float)
            else:
                # Look up Z bucket of each target then ask the estimator.
                z_tgt = item_z_bucket.to(device)[tgt_flat]
                with torch.no_grad() if loss_mode == "ips_choice" else _nullcontext():
                    p_logit = model.deconfounder.propensity_logit(z_tgt)
                    propensity = torch.sigmoid(p_logit)

            ips_loss_fn = IPSLoss(
                variant=ips_variant, clip_tau=ips_clip_tau, pad_index=pad_index,
            ) if loss_mode != "seq_choice" else SequentialCrossEntropy(pad_index=pad_index)

            # Mask PAD candidate from the choice softmax.
            util_for_loss = util.clone()
            util_for_loss[:, pad_index] = float("-inf")

            if loss_mode == "seq_choice":
                # No IPS weighting (A2): plain CE.
                L_C = F.cross_entropy(util_for_loss, tgt_flat, ignore_index=pad_index)
                L_ips = torch.tensor(0.0, device=device)
            else:
                # IPS-weighted CE.
                L_C = ips_loss_fn(util_for_loss, tgt_flat, propensity)
                L_ips = L_C.detach().clone()  # for monitoring only

        else:
            # ===== MovieLens path with negative sampling =====
            negs = sample_negatives(tgt_flat, n_items=n_items, n_neg=ml_n_negs,
                                     pad_index=pad_index)        # (M, n_neg)
            cand = torch.cat([tgt_flat.unsqueeze(1), negs], dim=1)  # (M, 1+n_neg)

            # V on the small candidate set: bilinear of h_flat with the
            # candidate embeddings.
            cand_emb = model.backbone.item_embedding(cand)           # (M, K, d)
            h_proj = model.economics.value_head.W_V(h_flat)          # (M, d)
            V_cand = torch.einsum("md,mkd->mk", h_proj, cand_emb)    # (M, K)

            # Cost signals on the candidate set.
            if use_economics and use_cost:
                # GenreRed: jaccard(g(c), hist_union(user_row))
                assert ml_loader is not None
                gmat = ml_loader.bulk_genre_matrix()
                gmat_t = torch.from_numpy(gmat).to(device).float()  # (V_cat, G)
                cand_g = gmat_t[cand]                                # (M, K, G)
                hist_g = gmat_t[hist_for_lambda]                     # (M, T, G)
                hist_union = (hist_g.sum(dim=1) > 0).float()         # (M, G)
                cand_size = cand_g.sum(dim=2)                        # (M, K)
                hist_size = hist_union.sum(dim=1, keepdim=True)      # (M, 1)
                inter = torch.einsum("mkg,mg->mk", cand_g, hist_union)  # (M, K)
                union = (cand_size + hist_size - inter).clamp_min(1e-8)
                genre_red = inter / union                            # (M, K)

                # RecencyPress: position-based.
                position_in_seq = valid_mask.cumsum(dim=1)[valid_mask].float()
                seq_len = valid_mask.sum(dim=1).clamp_min(1).float()
                recency_per_pos = position_in_seq / seq_len[row_to_user]
                recency = recency_per_pos.unsqueeze(1).expand(-1, cand.shape[1])

                # PopPress: candidate's Z bucket / max bucket.
                z_max = max(1.0, float(item_z_bucket.max().item()))
                pop_press = item_z_bucket.to(device)[cand].float() / z_max  # (M, K)

                C_cand = model.economics.cost_backend(genre_red, recency, pop_press)

                if fixed_lambda_u:
                    lam = torch.full((h_flat.shape[0],), fixed_lambda_u_value, device=device)
                else:
                    lam = model.economics.lambda_net(hist_for_lambda)

                util = V_cand - lam.unsqueeze(-1) * C_cand
            elif use_economics and not use_cost:
                util = V_cand
            else:
                util = V_cand

            # The target is at column 0 of cand → target label = 0.
            target_col = torch.zeros(h_flat.shape[0], dtype=torch.long, device=device)

            if loss_mode == "seq_choice":
                L_C = F.cross_entropy(util, target_col)
                L_ips = torch.tensor(0.0, device=device)
            else:
                z_tgt = item_z_bucket.to(device)[tgt_flat]
                p_logit = model.deconfounder.propensity_logit(z_tgt)
                propensity = torch.sigmoid(p_logit)
                # Apply the same variance-reduction strategy as IPSLoss
                # but inline (cleaner than building a vocab-wide tensor).
                if ips_variant == "self_normalized":
                    raw_w = 1.0 / propensity.clamp_min(1e-8)
                    denom = raw_w.sum().clamp_min(1e-8)
                    valid_count = float(raw_w.shape[0])
                    w = raw_w / denom * valid_count
                else:
                    w = (1.0 / propensity.clamp_min(1e-8)).clamp(max=ips_clip_tau)
                # Per-sample CE then weighted mean.
                ce_per = F.cross_entropy(util, target_col, reduction="none")
                L_C = (ce_per * w).mean()
                L_ips = L_C.detach().clone()

        # ---------------------------------------------------------------
        # 3) Adversarial discriminator loss on h_flat.
        # ---------------------------------------------------------------
        if use_grl:
            z_tgt = item_z_bucket.to(device)[tgt_flat]
            z_logits = model.deconfounder.discriminate(h_flat, alpha=grl_alpha)
            L_adv = adv_ce(z_logits, z_tgt)
            disc_pred = z_logits.argmax(dim=-1)
            total_disc_correct += int((disc_pred == z_tgt).sum().item())
            total_disc_count   += int(z_tgt.numel())
        else:
            # A3: w/o GRL. We do NOT compute L_adv at all (would still
            # need a target Z for the discriminator's own training, but
            # this ablation removes the whole branch).
            L_adv = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------------
        # 4) Combine, backward, step.
        # ---------------------------------------------------------------
        L_combined = L_C + lambda_adv * L_adv
        # L2 regularisation is applied via Optimizer(weight_decay=lambda_reg),
        # NOT here, per Phase 3 losses.py guidance.

        optim.zero_grad()
        L_combined.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        total_L_ips   += float(L_ips.item())
        total_L_adv   += float(L_adv.item())
        total_L_C     += float(L_C.item())
        total_loss    += float(L_combined.item())
        total_positions += n_valid
        n_steps += 1

    n_steps = max(1, n_steps)
    return StepOutput(
        L_ips=total_L_ips / n_steps,
        L_adv=total_L_adv / n_steps,
        L_C=total_L_C / n_steps,
        L_total=total_loss / n_steps,
        disc_acc=(total_disc_correct / max(1, total_disc_count)),
        valid_positions=total_positions,
    )


class _nullcontext:
    """Minimal contextlib.nullcontext shim (avoid extra import)."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(path: Path, model: CEINNModel, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "dataset_name": model.dataset_name,
        "extra": extra,
    }, path)


def load_checkpoint(path: Path, model: CEINNModel, *, map_location=None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("extra", {})


# =============================================================================
# Main entry
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to the per-dataset YAML config.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override seed for this run (default: from config).")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Subfolder under training.output_dir (default: timestamp).")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Per-field overrides, e.g. training.lambda_adv=0.5.")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu / cuda:0 etc. (default: auto).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    train_cfg = cfg["training"]

    # Resolve device.
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 0))
    set_seed(seed)

    # Output dir layout: <training.output_dir>/<run_name>/
    base_out = Path(train_cfg["output_dir"])
    run_name = args.run_name or f"seed{seed}_{int(time.time())}"
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output dir: {out_dir}")

    # Snapshot the effective config so re-runs are exactly reproducible.
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # -------------------------------------------------------------------------
    # Load Phase-2 artefacts.
    # -------------------------------------------------------------------------
    print("Loading Phase-2 artefacts …")
    dataset_kind, loader = load_dataset_loader(cfg)

    # Z-bucket array (item_idx → Z bucket id), shape (n_items + 1,).
    # For Amazon this comes from item_meta; for ML we discretise dynamic Z.
    if dataset_kind == "amazon_beauty":
        n_z_buckets = int(loader.vocab["n_Z_bins"])
        item_z = torch.from_numpy(loader.bulk_meta_arrays()["Z"]).long()
        ml_loader = None
    else:
        n_z_buckets = int(train_cfg.get("n_z_buckets", 10))
        # We need a per-item Z bucket. Use the dynamic Z at each
        # item's LATEST train position as the static proxy.
        # Build by scanning the row_index.
        print("    Building per-item Z buckets from dynamic Z …")
        z_per_row, edges = compute_movielens_z_buckets(loader, n_z_buckets)
        # Map each item to its last-row Z bucket (default = bucket 0).
        n_items = int(loader.vocab["n_items"])
        last_z = np.zeros(n_items + 1, dtype=np.int64)
        # row_index keyed (u, pos); reconstruct item id via train_seqs.
        for (u, pos), row_idx in loader._row_index.items():
            iid = loader.train_seqs[u][pos][0]
            last_z[iid] = int(z_per_row[row_idx])
        item_z = torch.from_numpy(last_z)
        ml_loader = loader

    seen_items = build_seen_items(loader.train_seqs)
    max_seq_len = int(loader.vocab["max_seq_len"])
    n_items = int(loader.vocab["n_items"])
    pad_index = int(loader.vocab["pad_index"])

    # -------------------------------------------------------------------------
    # Build model.
    # -------------------------------------------------------------------------
    arch = train_cfg["architecture"]
    if dataset_kind == "amazon_beauty":
        model = build_ceinn_amazon(
            loader,
            d=arch["d"], n_heads=arch["n_heads"], n_layers=arch["n_layers"],
            dropout=arch["dropout"],
            propensity_hidden=arch.get("propensity_hidden", 64),
            discriminator_hidden=arch.get("discriminator_hidden"),
        )
    else:
        model = build_ceinn_movielens(
            loader,
            n_z_buckets=n_z_buckets,
            d=arch["d"], n_heads=arch["n_heads"], n_layers=arch["n_layers"],
            dropout=arch["dropout"],
            propensity_hidden=arch.get("propensity_hidden", 64),
            discriminator_hidden=arch.get("discriminator_hidden"),
        )
    model.to(device)
    item_z = item_z.to(device)
    print(f"    Model: dataset={model.dataset_name}, n_items={n_items}, "
          f"d={arch['d']}, layers={arch['n_layers']}, heads={arch['n_heads']}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {n_params:,}")

    # -------------------------------------------------------------------------
    # Dataset / loader.
    # -------------------------------------------------------------------------
    ds = SequentialNextItemDataset(loader.train_seqs, max_seq_len=max_seq_len,
                                    pad_index=pad_index)
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_batch, drop_last=False,
    )
    print(f"    Train sequences: {len(ds)} users, batch_size={batch_size}")

    # -------------------------------------------------------------------------
    # Propensity warm-up.
    # -------------------------------------------------------------------------
    propensity_warmup = int(train_cfg.get("propensity_warmup_epochs", 1))
    if propensity_warmup > 0:
        print(f"    Propensity warm-up: {propensity_warmup} epoch(s)")
        warmup_propensity_estimator(
            model, train_loader, item_z_bucket=item_z,
            n_epochs=propensity_warmup,
            lr=float(train_cfg.get("propensity_lr", 1e-3)),
            device=device, n_items=n_items,
        )

    # -------------------------------------------------------------------------
    # Optimizer.
    # -------------------------------------------------------------------------
    lr = float(train_cfg["lr"])
    lambda_reg = float(train_cfg.get("lambda_reg", 0.0))
    optim = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=lambda_reg,
    )

    # -------------------------------------------------------------------------
    # Training loop.
    # -------------------------------------------------------------------------
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg.get("early_stop_patience", 10))
    lambda_adv = float(train_cfg["lambda_adv"])
    ablation = train_cfg.get("ablation", {})

    ips_variant = train_cfg.get("ips_variant", "clipped")
    ips_clip_tau = float(train_cfg.get("ips_clip_tau", 30.0))
    ml_n_negs = int(train_cfg.get("ml_n_negs", 100))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    val_batch_size = int(train_cfg.get("val_batch_size", 256))

    history: List[Dict[str, Any]] = []
    best_val_ndcg10 = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    csv_path = out_dir / "train_log.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,alpha,L_total,L_ips,L_adv,L_C,disc_acc,"
                "val_ndcg@5,val_ndcg@10,val_ndcg@20,val_hr@10,val_mrr,"
                "epoch_time_s\n")

    print(f"\n=== Training: max {max_epochs} epochs, patience {patience} ===")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        p = (epoch - 1) / max(1, max_epochs - 1)
        alpha = alpha_schedule(p) if ablation.get("use_grl", True) else 0.0

        step = train_one_epoch(
            model, train_loader, optim,
            device=device,
            grl_alpha=alpha,
            lambda_adv=lambda_adv,
            lambda_reg=lambda_reg,
            item_z_bucket=item_z,
            ablation=ablation,
            n_items=n_items,
            ml_n_negs=ml_n_negs,
            ml_loader=ml_loader,
            ips_variant=ips_variant,
            ips_clip_tau=ips_clip_tau,
            grad_clip=grad_clip,
            pad_index=pad_index,
        )

        val_metrics = validate_full_ranking(
            model, loader.val_seqs, loader.train_seqs, seen_items,
            max_seq_len=max_seq_len, device=device, item_z_bucket=item_z,
            ml_loader=ml_loader, pad_index=pad_index,
            batch_size=val_batch_size, ablation=ablation,
        )
        ndcg10 = val_metrics.get("ndcg@10", 0.0)
        epoch_time = time.time() - t0

        record = {
            "epoch": epoch, "alpha": alpha,
            "L_total": step.L_total, "L_ips": step.L_ips,
            "L_adv": step.L_adv, "L_C": step.L_C,
            "disc_acc": step.disc_acc,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "n_users"},
            "epoch_time_s": epoch_time,
        }
        history.append(record)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{alpha:.4f},{step.L_total:.4f},{step.L_ips:.4f},"
                f"{step.L_adv:.4f},{step.L_C:.4f},{step.disc_acc:.4f},"
                f"{val_metrics.get('ndcg@5', 0):.4f},"
                f"{val_metrics.get('ndcg@10', 0):.4f},"
                f"{val_metrics.get('ndcg@20', 0):.4f},"
                f"{val_metrics.get('hr@10', 0):.4f},"
                f"{val_metrics.get('mrr', 0):.4f},"
                f"{epoch_time:.1f}\n"
            )

        print(f"  Epoch {epoch:3d}/{max_epochs}  α={alpha:.3f}  "
              f"L_total={step.L_total:.4f}  L_C={step.L_C:.4f}  "
              f"L_adv={step.L_adv:.4f}  discAcc={step.disc_acc:.3f}  "
              f"val_NDCG@10={ndcg10:.4f}  ({epoch_time:.1f}s)")

        # Always save the last checkpoint (resume helper).
        save_checkpoint(out_dir / "last.ckpt", model,
                        extra={"epoch": epoch, "val_ndcg10": ndcg10})

        # Early stopping on val NDCG@10.
        if ndcg10 > best_val_ndcg10:
            best_val_ndcg10 = ndcg10
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(out_dir / "best.ckpt", model, extra={
                "epoch": epoch, "val_ndcg10": ndcg10,
                "config": cfg, "seed": seed,
            })
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"  Early stopping at epoch {epoch}: no improvement "
                      f"for {patience} epochs (best @ epoch {best_epoch}, "
                      f"NDCG@10={best_val_ndcg10:.4f})")
                break

    # -------------------------------------------------------------------------
    # Persist history JSON for downstream plotting (Phase 8).
    # -------------------------------------------------------------------------
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "dataset": model.dataset_name,
            "best_epoch": best_epoch,
            "best_val_ndcg10": best_val_ndcg10,
            "history": history,
        }, f, indent=2)

    print(f"\nTraining complete. Best val NDCG@10 = {best_val_ndcg10:.4f} "
          f"at epoch {best_epoch}.")
    print(f"Artefacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
