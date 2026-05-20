#!/usr/bin/env python
"""
CEINN — Phase 6 training script (Plan §6).

Implements:
  * §6.1.1 Joint-loss back-propagation via Gradient Reversal Layer (single
    backward suffices because GRL flips the encoder-side gradient sign while
    the discriminator-side flows normally — i.e. the "two-step" prescription
    in the plan is realised in one step, which is the canonical DANN trick).
  * §6.1.2 GRL alpha schedule per epoch (sigmoid warm-up from `alpha_schedule`).
  * §6.1.3 Training monitoring (L_IPS, L_adv, L_C, L_total, Val NDCG@10,
    discriminator macro-OvR AUC) written to a CSV log under `output_dir/`.
  * §6.1.4 Early stopping on Val NDCG@10 (patience from YAML) + best-state
    checkpoint to `checkpoints/`.

Loss strategy (matches the YAML comment "we merge L_IPS into L_C")
-----------------------------------------------------------------
The four ablation modes selectable via `training.ablation.loss_mode`:
  ips_choice    — IPS-weighted choice loss with CLIPPED weights (default).
                  L = mean_over_batch( w_i * CE_per_sample(U, target) )
                  where w_i = clip(1 / p_i, tau) and U(u, i, t) = V - lambda_u * C.
  snips_choice  — Self-Normalised IPS (A4): weights w_i normalised so their
                  mean is 1 over the batch.
  seq_choice    — Plain CE over utilities (A2 w/o IPS) — uniform weights.

The discriminator is trained jointly via the SAME backward pass: forward
passes h_t through `grad_reverse(h_t, alpha)`, so its CE gradient enters
the encoder negated and scaled by `alpha`. No separate optimiser step.

Propensity warm-up
------------------
We pre-train the propensity estimator with a BCE objective for
`propensity_warmup_epochs` (default 1). Positives are the Z buckets of
training-set targets (each interaction contributes one positive); negatives
are Z buckets of uniformly-random items. This trains an exposure-vs-uniform
discriminator whose sigmoid output we then read as p_i during the main loop.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# -----------------------------------------------------------------------------
# Make local imports work when train.py is run from the repo root.
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_loaders.amazon_beauty_loader import AmazonBeautyLoader  # noqa: E402
from data_loaders.movieslens_10M_loader import MovieLens10MLoader  # noqa: E402
from models.ceinn import CEINNModel, build_ceinn_amazon, build_ceinn_movielens
from models.causal_deconfounder import alpha_schedule
from utils.math_utils import batch_jaccard  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_full_ranking,
    ndcg_at_k,
)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Config helpers
# =============================================================================

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# §6.0  Sample construction for training
# =============================================================================

def _build_supervised_index_amazon(loader: AmazonBeautyLoader) -> List[Tuple[int, int]]:
    """
    Return a list of (user_idx, supervised_position) for every user with at
    least two training items. `supervised_position` is 0-indexed.

    For a user with sequence [i_0, i_1, ..., i_{N-1}], we admit positions
    1..N-1 (so that there is at least one history item to consume).
    """
    out: List[Tuple[int, int]] = []
    for u, seq in loader.train_seqs.items():
        if len(seq) < 2:
            continue
        for pos in range(1, len(seq)):
            out.append((u, pos))
    return out


def _build_supervised_index_movielens(loader: MovieLens10MLoader) -> List[Tuple[int, int]]:
    """Same as Amazon variant — interface is identical for both datasets."""
    out: List[Tuple[int, int]] = []
    for u, seq in loader.train_seqs.items():
        if len(seq) < 2:
            continue
        for pos in range(1, len(seq)):
            out.append((u, pos))
    return out


def _sample_one_position_per_user(
    loader, rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    SASRec-style epoch sample: one random supervised position per user
    (uniformly over 1..N-1). Returns a shuffled list of (u, pos) pairs.
    This keeps memory per epoch ≈ n_users supervised positions, which is
    well within reach.
    """
    pairs: List[Tuple[int, int]] = []
    for u, seq in loader.train_seqs.items():
        if len(seq) < 2:
            continue
        pos = int(rng.integers(1, len(seq)))
        pairs.append((u, pos))
    rng.shuffle(pairs)
    return pairs


# =============================================================================
# §6.0  Static-Z bucketisation for MovieLens
# =============================================================================

def build_movielens_static_z(
    loader: MovieLens10MLoader,
    n_z_buckets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a static Z bucket per item from final training-set counts.

    Returns
    -------
    item_z_bucket : (n_items + 1,) int64 — bucket per item (index 0 = PAD = 0).
    bucket_edges  : (n_z_buckets + 1,) float64 — quantile edges used.
    """
    n_items = int(loader.vocab["n_items"])
    counts = np.zeros(n_items + 1, dtype=np.int64)
    for _u, seq in loader.train_seqs.items():
        for (iid, _r, _d) in seq:
            if 0 < iid <= n_items:
                counts[iid] += 1
    real_counts = counts[1:].astype(np.float64)
    edges = np.quantile(real_counts, np.linspace(0.0, 1.0, n_z_buckets + 1))
    # Tighten ties to keep digitize monotonic.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    buckets = np.zeros(n_items + 1, dtype=np.int64)
    if real_counts.size > 0:
        b = np.digitize(real_counts, edges[1:-1], right=True)
        buckets[1:] = np.clip(b, 0, n_z_buckets - 1)
    return buckets, edges


def build_movielens_pop_press(
    item_counts: np.ndarray,
) -> np.ndarray:
    """
    Min-max normalise the per-item training counts to [0, 1] — this is the
    PopPress signal broadcast across the catalogue per (u, t).
    """
    real = item_counts[1:].astype(np.float64)
    if real.size == 0:
        return np.zeros_like(item_counts, dtype=np.float32)
    lo, hi = float(real.min()), float(real.max())
    out = np.zeros_like(item_counts, dtype=np.float32)
    if hi > lo:
        out[1:] = ((real - lo) / (hi - lo)).astype(np.float32)
    return out


# =============================================================================
# §6.0  Batching: turn (u, pos) pairs into padded tensors
# =============================================================================

def make_batch_amazon(
    pairs: List[Tuple[int, int]],
    loader: AmazonBeautyLoader,
    max_seq_len: int,
    pad_index: int,
    device: torch.device,
    *,
    use_rating_emb: bool = True,
    use_temporal_emb: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a training batch from a list of supervised (u, pos) pairs.

    Inputs are the prefix [i_0, ..., i_{pos-1}] right-padded to `max_seq_len`
    with PAD at the END (NB: SequentialBackbone takes the LAST non-PAD
    position as h_t; right-padding keeps the natural left-to-right reading).

    Wait — actually the backbone takes h at last NON-pad position which for
    right-padded input is the actual last token of the prefix, i.e. position
    (pos - 1). That is precisely the state we want to condition on. So
    right-padding is correct.
    """
    B = len(pairs)
    T = max_seq_len
    item_ids   = torch.full((B, T), pad_index, dtype=torch.long)
    rating_ids = torch.full((B, T), pad_index, dtype=torch.long)
    dt_ids     = torch.full((B, T), pad_index, dtype=torch.long)
    target     = torch.empty(B, dtype=torch.long)
    z_target   = torch.empty(B, dtype=torch.long)
    z_cands    = torch.zeros(B, loader.vocab["n_items"] + 1, dtype=torch.long)
    # All candidates share the same per-item Z bucket vector (static for Amazon).
    item_z = loader.bulk_meta_arrays()["Z"]      # (n_items + 1,)

    for b, (u, pos) in enumerate(pairs):
        seq = loader.train_seqs[u]
        # Use up to the last `max_seq_len` history tokens BEFORE pos.
        prefix = seq[max(0, pos - T):pos]
        L = len(prefix)
        for k, (iid, rb, db) in enumerate(prefix):
            item_ids[b, k]   = iid
            rating_ids[b, k] = rb if use_rating_emb else pad_index
            dt_ids[b, k]     = db if use_temporal_emb else pad_index
        # Padding is on the RIGHT (indices L..T-1 stay at pad_index).
        tgt_iid, _tgt_rb, _tgt_db = seq[pos]
        target[b]   = tgt_iid
        z_target[b] = int(item_z[tgt_iid])
        z_cands[b]  = torch.from_numpy(item_z)

    return {
        "item_ids":   item_ids.to(device),
        "rating_ids": rating_ids.to(device),
        "dt_ids":     dt_ids.to(device),
        "target":     target.to(device),
        "z_buckets_target":     z_target.to(device),
        "z_buckets_candidates": z_cands.to(device),
    }


def make_batch_movielens(
    pairs: List[Tuple[int, int]],
    loader: MovieLens10MLoader,
    max_seq_len: int,
    pad_index: int,
    device: torch.device,
    *,
    item_z_bucket: np.ndarray,        # (n_items + 1,) static Z for ML
    pop_press_vec: np.ndarray,        # (n_items + 1,) [0, 1]
    item_genre_mat: np.ndarray,       # (n_items + 1, n_genres)
    use_rating_emb: bool = True,
    use_temporal_emb: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a MovieLens training batch.

    Extra fields needed by `forward_movielens`:
      - genre_red    : (B, V_cat) — Jaccard(cand_i, history-union) computed
                       on the fly from the user's history.
      - recency_press: (B, V_cat) — normalised history length, broadcast.
      - pop_press    : (B, V_cat) — per-item static PopPress, broadcast.
    """
    B = len(pairs)
    T = max_seq_len
    V = int(loader.vocab["n_items"]) + 1
    G = item_genre_mat.shape[1]
    pp_index = pad_index

    item_ids   = torch.full((B, T), pp_index, dtype=torch.long)
    rating_ids = torch.full((B, T), pp_index, dtype=torch.long)
    dt_ids     = torch.full((B, T), pp_index, dtype=torch.long)
    target     = torch.empty(B, dtype=torch.long)
    z_target   = torch.empty(B, dtype=torch.long)
    z_cands    = torch.empty(B, V, dtype=torch.long)

    # On-the-fly per-batch GenreRed computation. We assemble the user's
    # history-union genre vector once per (u, pos), then call batch_jaccard
    # against the full catalogue. This is O(B * V * G) ≈ 32 * 10197 * 19 =
    # ~6M ops per batch — well under a millisecond on CPU.
    history_union = np.zeros((B, G), dtype=np.uint8)
    history_len   = np.zeros(B, dtype=np.float32)

    for b, (u, pos) in enumerate(pairs):
        seq = loader.train_seqs[u]
        prefix = seq[max(0, pos - T):pos]
        L = len(prefix)
        history_len[b] = float(L)
        # Build history-union vector for GenreRed.
        for k, (iid, rb, db) in enumerate(prefix):
            item_ids[b, k]   = iid
            rating_ids[b, k] = rb if use_rating_emb else pp_index
            dt_ids[b, k]     = db if use_temporal_emb else pp_index
            if 0 < iid < V:
                history_union[b] |= (item_genre_mat[iid] > 0).astype(np.uint8)
        tgt_iid, _tgt_rb, _tgt_db = seq[pos]
        target[b]   = tgt_iid
        z_target[b] = int(item_z_bucket[tgt_iid])
        z_cands[b]  = torch.from_numpy(item_z_bucket)

    # (B, V) genre_red via batched Jaccard with all-pairs branch.
    genre_red_np = batch_jaccard(
        history_union, (item_genre_mat > 0).astype(np.uint8)
    )  # (B, V) float64
    genre_red = torch.from_numpy(genre_red_np.astype(np.float32))

    # RecencyPress: normalised history length, broadcast over V.
    rp = (history_len / max(1.0, float(T))).astype(np.float32)
    recency_press = torch.from_numpy(rp).unsqueeze(-1).expand(-1, V).contiguous()

    # PopPress: per-item static, broadcast over batch.
    pp = torch.from_numpy(pop_press_vec.astype(np.float32)).unsqueeze(0).expand(B, -1).contiguous()

    return {
        "item_ids":   item_ids.to(device),
        "rating_ids": rating_ids.to(device),
        "dt_ids":     dt_ids.to(device),
        "target":     target.to(device),
        "z_buckets_target":     z_target.to(device),
        "z_buckets_candidates": z_cands.to(device),
        "genre_red":     genre_red.to(device),
        "recency_press": recency_press.to(device),
        "pop_press":     pp.to(device),
    }


# =============================================================================
# §6.0  Propensity warm-up
# =============================================================================

def warmup_propensity(
    model: nn.Module,
    *,
    pos_z_pool: np.ndarray,           # (n_pos,) Z buckets of train interactions
    item_z_table: np.ndarray,         # (n_items + 1,) Z bucket per item
    n_items: int,
    n_z_buckets: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """
    BCE warm-up of `model.deconfounder.propensity_estimator`.

    Positives: Z buckets from `pos_z_pool` (label 1).
    Negatives: Z buckets of uniform-random items in [1, n_items] (label 0).
    """
    if epochs <= 0 or pos_z_pool.size == 0:
        return
    propensity = model.deconfounder.propensity_estimator
    opt = torch.optim.Adam(propensity.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        perm = rng.permutation(pos_z_pool.size)
        running_loss = 0.0
        n_steps = 0
        for start in range(0, perm.size, batch_size):
            idx = perm[start:start + batch_size]
            z_pos = torch.from_numpy(pos_z_pool[idx]).long().to(device)
            neg_items = rng.integers(1, n_items + 1, size=idx.size)
            z_neg = torch.from_numpy(item_z_table[neg_items]).long().to(device)
            z_all = torch.cat([z_pos, z_neg], dim=0)
            y_all = torch.cat([
                torch.ones(z_pos.size(0), device=device),
                torch.zeros(z_neg.size(0), device=device),
            ], dim=0)

            opt.zero_grad(set_to_none=True)
            logits = propensity(z_all)
            loss = bce(logits, y_all)
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            n_steps += 1
        avg = running_loss / max(1, n_steps)
        print(f"  [warmup] propensity epoch {ep + 1}/{epochs}  bce={avg:.4f}")


# =============================================================================
# §6.1.3  IPS-weighted choice loss (the "merged" L_IPS ⊕ L_C)
# =============================================================================

def ips_weighted_choice_loss(
    U: torch.Tensor,
    target: torch.Tensor,
    propensity_target: torch.Tensor,
    *,
    pad_index: int,
    mode: str,                    # ips_choice | snips_choice | seq_choice
    clip_tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L = mean_over_batch( w_i * CE_i )
        where CE_i = -log softmax(U_i)[target_i]
        and w_i depends on `mode`:
          - ips_choice    : w = min(1 / p_i, clip_tau)
          - snips_choice  : w = (1/p_i) / mean_batch(1/p_i)   (mean 1.0)
          - seq_choice    : w = 1
    PAD candidate (index `pad_index`) is masked from the softmax via -inf.
    """
    # Mask PAD candidate.
    U_masked = U.clone()
    U_masked[:, pad_index] = float("-inf")
    ce_per_sample = F.cross_entropy(
        U_masked, target, reduction="none", ignore_index=pad_index
    )
    valid_mask = (target != pad_index).to(U.dtype)

    if mode == "seq_choice":
        w = torch.ones_like(ce_per_sample)
    else:
        p_safe = torch.clamp(propensity_target, min=eps).to(U.dtype)
        raw_w = 1.0 / p_safe
        if mode == "ips_choice":
            w = torch.clamp(raw_w, max=float(clip_tau))
        elif mode == "snips_choice":
            mw = raw_w * valid_mask
            denom = mw.sum().clamp_min(eps)
            valid_count = valid_mask.sum().clamp_min(1.0)
            w = (raw_w / denom) * valid_count
        else:
            raise ValueError(f"unknown loss mode {mode!r}")
        w = w * valid_mask

    loss = (ce_per_sample * w).sum() / valid_mask.sum().clamp_min(1.0)
    return loss


# =============================================================================
# §6.2  Quick Val NDCG@10 evaluation (used inside the training loop)
# =============================================================================

def quick_validate(
    model: nn.Module,
    loader,
    *,
    dataset_name: str,
    max_seq_len: int,
    pad_index: int,
    device: torch.device,
    item_z_bucket: Optional[np.ndarray] = None,
    pop_press_vec: Optional[np.ndarray] = None,
    item_genre_mat: Optional[np.ndarray] = None,
    eval_batch_size: int = 64,
    max_users: Optional[int] = None,
    ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute Val NDCG@10 (and discriminator macro-OvR AUC) on the validation
    set. Excludes train-seen items from the candidate pool.

    For efficiency this is the only function called per-epoch — we leave
    the full HR/MRR/group/AUC report to `evaluate.py`.
    """
    model.eval()
    ablation = ablation or {}
    use_economics = ablation.get("use_economics", True)
    use_cost      = ablation.get("use_cost", True)
    fixed_lambda  = ablation.get("fixed_lambda_u", False)
    fixed_value   = float(ablation.get("fixed_lambda_u_value", 0.5))
    use_rating    = ablation.get("use_rating_emb", True)
    use_temporal  = ablation.get("use_temporal_emb", True)

    val_users = [u for u in loader.val_seqs if loader.train_seqs.get(u)]
    if max_users is not None:
        val_users = val_users[:int(max_users)]
    if not val_users:
        return {"ndcg@10": float("nan"), "disc_auc": float("nan")}

    all_ranks: List[int] = []
    all_z_logits: List[np.ndarray] = []
    all_z_targets: List[int] = []
    pad = pad_index
    V = int(loader.vocab["n_items"]) + 1

    with torch.no_grad():
        for start in range(0, len(val_users), eval_batch_size):
            batch_users = val_users[start:start + eval_batch_size]
            # Build val batch: input = full train_seqs[u]; target = val_seqs[u].
            B = len(batch_users)
            item_ids   = torch.full((B, max_seq_len), pad, dtype=torch.long)
            rating_ids = torch.full((B, max_seq_len), pad, dtype=torch.long)
            dt_ids     = torch.full((B, max_seq_len), pad, dtype=torch.long)
            targets    = torch.empty(B, dtype=torch.long)
            z_targets  = torch.empty(B, dtype=torch.long)
            history_union = np.zeros((B, item_genre_mat.shape[1] if item_genre_mat is not None else 0), dtype=np.uint8)
            history_len = np.zeros(B, dtype=np.float32)
            seen_lists: List[set] = []

            for b, u in enumerate(batch_users):
                seq = loader.train_seqs[u][-max_seq_len:]
                history_len[b] = float(len(seq))
                seen = set()
                for k, (iid, rb, db) in enumerate(seq):
                    item_ids[b, k]   = iid
                    rating_ids[b, k] = rb if use_rating else pad
                    dt_ids[b, k]     = db if use_temporal else pad
                    if 0 < iid < V:
                        seen.add(iid)
                        if item_genre_mat is not None:
                            history_union[b] |= (item_genre_mat[iid] > 0).astype(np.uint8)
                t_iid, _rb, _db = loader.val_seqs[u]
                targets[b] = int(t_iid)
                if dataset_name == "amazon_beauty":
                    z_targets[b] = int(loader.item_meta[int(t_iid)]["Z"])
                else:
                    z_targets[b] = int(item_z_bucket[int(t_iid)])
                seen_lists.append(seen)

            item_ids   = item_ids.to(device)
            rating_ids = rating_ids.to(device)
            dt_ids     = dt_ids.to(device)
            z_t = z_targets.to(device)

            if dataset_name == "amazon_beauty":
                z_c = torch.from_numpy(loader.bulk_meta_arrays()["Z"]).long().unsqueeze(0).expand(B, -1).to(device)
                out = model.forward_amazon(
                    item_ids, rating_ids, dt_ids,
                    z_buckets_target=z_t,
                    z_buckets_candidates=z_c,
                    grl_alpha=0.0,
                )
            else:
                rp = (history_len / max(1.0, float(max_seq_len))).astype(np.float32)
                recency_press = torch.from_numpy(rp).unsqueeze(-1).expand(-1, V).contiguous().to(device)
                pp = torch.from_numpy(pop_press_vec.astype(np.float32)).unsqueeze(0).expand(B, -1).contiguous().to(device)
                genre_red_np = batch_jaccard(
                    history_union, (item_genre_mat > 0).astype(np.uint8)
                ).astype(np.float32)
                gr = torch.from_numpy(genre_red_np).to(device)
                z_c = torch.from_numpy(item_z_bucket).long().unsqueeze(0).expand(B, -1).to(device)
                out = model.forward_movielens(
                    item_ids, rating_ids, dt_ids,
                    z_buckets_target=z_t,
                    genre_red=gr,
                    recency_press=recency_press,
                    pop_press=pp,
                    z_buckets_candidates=z_c,
                    grl_alpha=0.0,
                )

            # Choose the scoring tensor based on ablation flags.
            V_scores = out["V"]
            C        = out["C"]
            lam      = out["lambda_u"]
            if not use_economics:
                # A5: raw dot-product (bypass W_V).
                E = model.economics.value_head.item_embedding.weight  # (V, d)
                h = out["h_t"]
                scores = h @ E.t()
            elif not use_cost:
                scores = V_scores
            else:
                if fixed_lambda:
                    lam = torch.full_like(lam, float(fixed_value))
                if C.dim() == 1:
                    scores = V_scores - lam.unsqueeze(-1) * C.unsqueeze(0)
                else:
                    scores = V_scores - lam.unsqueeze(-1) * C

            # Discriminator outputs (for AUC monitoring).
            all_z_logits.append(out["z_logits"].detach().cpu().numpy())
            all_z_targets.extend(z_targets.tolist())

            # Full-ranking on CPU (lightweight).
            scores_np = scores.detach().cpu().numpy()
            for b in range(B):
                rank = compute_full_ranking(
                    scores_np[b],
                    target_item=int(targets[b]),
                    seen_items=seen_lists[b],
                    pad_index=pad,
                )
                all_ranks.append(rank)

    ndcg10 = float(ndcg_at_k(all_ranks, k=10).mean())

    # Discriminator macro-OvR AUC (light import to avoid forcing sklearn at module load).
    disc_auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        z_logits = np.concatenate(all_z_logits, axis=0)
        z_target = np.asarray(all_z_targets, dtype=np.int64)
        # Drop classes that don't appear in the val set.
        present = np.unique(z_target)
        if present.size >= 2:
            probs = torch.softmax(torch.from_numpy(z_logits), dim=-1).numpy()
            disc_auc = float(roc_auc_score(
                z_target, probs[:, present], multi_class="ovr",
                average="macro", labels=present,
            ))
    except Exception:
        pass

    model.train()
    return {"ndcg@10": ndcg10, "disc_auc": disc_auc}


# =============================================================================
# §6.0  Build the model + companion arrays
# =============================================================================

def build_model_and_aux(
    cfg: Dict[str, Any], device: torch.device,
):
    """
    Returns (model, loader, aux) where aux is a dict of dataset-specific
    arrays needed at training time. For Amazon, aux is mostly empty; for
    MovieLens it contains the static Z bucketisation and PopPress vector.
    """
    dataset_name = cfg["dataset"]["name"]
    arch = cfg["training"]["architecture"]
    d = int(arch["d"])
    n_heads = int(arch["n_heads"])
    n_layers = int(arch["n_layers"])
    dropout = float(arch["dropout"])
    prop_hidden = int(arch.get("propensity_hidden", 64))
    disc_hidden = arch.get("discriminator_hidden", None)
    if disc_hidden is not None:
        disc_hidden = int(disc_hidden)

    if dataset_name == "amazon_beauty":
        proc_dir = Path(cfg["preprocess"]["output_dir"])
        loader = AmazonBeautyLoader.from_directory(proc_dir)
        model = build_ceinn_amazon(
            loader,
            d=d, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            propensity_hidden=prop_hidden,
            discriminator_hidden=disc_hidden,
        )
        item_z_table = loader.bulk_meta_arrays()["Z"]
        aux = {
            "item_z_bucket": item_z_table,
            "pop_press_vec": None,
            "item_genre_mat": None,
        }
    elif dataset_name == "movielens_10M":
        proc_dir = Path(cfg["preprocess"]["output_dir"])
        loader = MovieLens10MLoader.from_directory(proc_dir)
        n_z_buckets = int(cfg["training"].get("n_z_buckets", 10))
        item_z_bucket, _edges = build_movielens_static_z(loader, n_z_buckets=n_z_buckets)

        # Per-item count + PopPress.
        n_items = int(loader.vocab["n_items"])
        counts = np.zeros(n_items + 1, dtype=np.int64)
        for _u, seq in loader.train_seqs.items():
            for (iid, _r, _d) in seq:
                if 0 < iid <= n_items:
                    counts[iid] += 1
        pop_press_vec = build_movielens_pop_press(counts)
        item_genre_mat = loader.bulk_genre_matrix()  # (n_items + 1, n_genres)

        model = build_ceinn_movielens(
            loader,
            n_z_buckets=n_z_buckets,
            d=d, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            propensity_hidden=prop_hidden,
            discriminator_hidden=disc_hidden,
        )
        aux = {
            "item_z_bucket": item_z_bucket,
            "pop_press_vec": pop_press_vec,
            "item_genre_mat": item_genre_mat,
        }
    else:
        raise ValueError(f"Unknown dataset name {dataset_name!r}")

    model.to(device)
    return model, loader, aux


# =============================================================================
# §6.4  Main training driver
# =============================================================================

def train(cfg: Dict[str, Any], cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    # ---- Bookkeeping ---------------------------------------------------------
    tcfg = cfg["training"]
    seed = int(cli_overrides.get("seed", tcfg.get("seed", 0)))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(cli_overrides.get("device", tcfg.get("device", "auto")))

    output_dir = Path(cli_overrides.get("output_dir", tcfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_log_seed{seed}.csv"

    # ---- Model + data --------------------------------------------------------
    print(f"[setup] device={device}  seed={seed}  output_dir={output_dir}")
    model, loader, aux = build_model_and_aux(cfg, device)
    n_items = int(loader.vocab["n_items"])
    pad_index = int(loader.vocab["pad_index"])
    max_seq_len = int(loader.vocab["max_seq_len"])
    dataset_name = cfg["dataset"]["name"]

    ablation = tcfg.get("ablation", {}) or {}
    loss_mode = str(ablation.get("loss_mode", "ips_choice"))
    use_grl       = bool(ablation.get("use_grl", True))
    use_economics = bool(ablation.get("use_economics", True))
    use_cost      = bool(ablation.get("use_cost", True))
    fixed_lambda  = bool(ablation.get("fixed_lambda_u", False))
    fixed_value   = float(ablation.get("fixed_lambda_u_value", 0.5))
    use_rating    = bool(ablation.get("use_rating_emb", True))
    use_temporal  = bool(ablation.get("use_temporal_emb", True))

    # ---- Optimisers ----------------------------------------------------------
    lr = float(tcfg["lr"])
    lambda_reg = float(tcfg.get("lambda_reg", 0.0))
    lambda_adv = float(tcfg.get("lambda_adv", 0.1))
    clip_tau = float(tcfg.get("ips_clip_tau", 30.0))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    batch_size = int(tcfg["batch_size"])
    max_epochs = int(tcfg["max_epochs"])
    patience = int(tcfg.get("early_stop_patience", 10))

    # Separate parameter groups: propensity gets its own lr (often 0 after warmup).
    prop_params = list(model.deconfounder.propensity_estimator.parameters())
    prop_param_ids = {id(p) for p in prop_params}
    other_params = [p for p in model.parameters() if id(p) not in prop_param_ids]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": lr, "weight_decay": lambda_reg},
        {"params": prop_params, "lr": float(tcfg.get("propensity_lr", 1e-3)), "weight_decay": 0.0},
    ])

    # ---- Propensity warm-up --------------------------------------------------
    print("[warmup] training propensity estimator …")
    pos_z = []
    item_z_table = aux["item_z_bucket"]
    for u, seq in loader.train_seqs.items():
        for (iid, _r, _d) in seq:
            if 0 < iid <= n_items:
                pos_z.append(int(item_z_table[iid]))
    pos_z_arr = np.asarray(pos_z, dtype=np.int64)
    warmup_propensity(
        model,
        pos_z_pool=pos_z_arr,
        item_z_table=item_z_table,
        n_items=n_items,
        n_z_buckets=int(model.n_z_buckets),
        epochs=int(tcfg.get("propensity_warmup_epochs", 1)),
        lr=float(tcfg.get("propensity_lr", 1e-3)),
        batch_size=2048,
        device=device,
        rng=rng,
    )

    # ---- Discriminator CE ----------------------------------------------------
    disc_ce = nn.CrossEntropyLoss()

    # ---- Logging -------------------------------------------------------------
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow([
        "epoch", "alpha", "L_total", "L_choice", "L_adv",
        "val_ndcg@10", "disc_auc", "lr", "elapsed_sec",
    ])
    log_f.flush()

    # ---- Main loop -----------------------------------------------------------
    best_val = -float("inf")
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = -1
    patience_counter = 0
    t0 = time.time()

    eval_batch_size = int(tcfg.get("val_batch_size", 64))
    eval_max_users = tcfg.get("val_max_users", None)

    for epoch in range(1, max_epochs + 1):
        # GRL alpha schedule.
        p = (epoch - 1) / max(1, max_epochs - 1)
        alpha = alpha_schedule(p) if use_grl else 0.0

        # Sample one supervised position per user this epoch.
        pairs_all = _sample_one_position_per_user(loader, rng)
        n_pairs = len(pairs_all)
        if n_pairs == 0:
            print("[warn] no supervised pairs available; aborting training.")
            break

        model.train()
        running_total = 0.0
        running_choice = 0.0
        running_adv = 0.0
        n_steps = 0

        for start in range(0, n_pairs, batch_size):
            pairs = pairs_all[start:start + batch_size]
            if dataset_name == "amazon_beauty":
                batch = make_batch_amazon(
                    pairs, loader, max_seq_len, pad_index, device,
                    use_rating_emb=use_rating, use_temporal_emb=use_temporal,
                )
                out = model.forward_amazon(
                    batch["item_ids"], batch["rating_ids"], batch["dt_ids"],
                    z_buckets_target=batch["z_buckets_target"],
                    z_buckets_candidates=batch["z_buckets_candidates"],
                    grl_alpha=alpha,
                )
            else:
                batch = make_batch_movielens(
                    pairs, loader, max_seq_len, pad_index, device,
                    item_z_bucket=aux["item_z_bucket"],
                    pop_press_vec=aux["pop_press_vec"],
                    item_genre_mat=aux["item_genre_mat"],
                    use_rating_emb=use_rating, use_temporal_emb=use_temporal,
                )
                out = model.forward_movielens(
                    batch["item_ids"], batch["rating_ids"], batch["dt_ids"],
                    z_buckets_target=batch["z_buckets_target"],
                    genre_red=batch["genre_red"],
                    recency_press=batch["recency_press"],
                    pop_press=batch["pop_press"],
                    z_buckets_candidates=batch["z_buckets_candidates"],
                    grl_alpha=alpha,
                )

            # ---- Build the scoring tensor based on ablation flags ----------
            V_scores = out["V"]
            C        = out["C"]
            lam      = out["lambda_u"]
            if not use_economics:
                # A5: raw dot-product (bypass W_V).
                E = model.economics.value_head.item_embedding.weight  # (V, d)
                h = out["h_t"]
                U = h @ E.t()
            elif not use_cost:
                U = V_scores
            else:
                if fixed_lambda:
                    lam = torch.full_like(lam, float(fixed_value))
                if C.dim() == 1:
                    U = V_scores - lam.unsqueeze(-1) * C.unsqueeze(0)
                else:
                    U = V_scores - lam.unsqueeze(-1) * C

            # ---- Choice loss (IPS-weighted) --------------------------------
            L_choice = ips_weighted_choice_loss(
                U,
                target=batch["target"],
                propensity_target=out["propensity_target"],
                pad_index=pad_index,
                mode=loss_mode,
                clip_tau=clip_tau,
            )

            # ---- Adversarial loss (GRL handles backward sign flip) ---------
            if use_grl:
                L_adv = disc_ce(out["z_logits"], batch["z_buckets_target"])
            else:
                L_adv = torch.zeros((), device=device)

            L_total = L_choice + lambda_adv * L_adv

            # ---- Backward + step --------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_total  += float(L_total.item())
            running_choice += float(L_choice.item())
            running_adv    += float(L_adv.item())
            n_steps += 1

        avg_total  = running_total  / max(1, n_steps)
        avg_choice = running_choice / max(1, n_steps)
        avg_adv    = running_adv    / max(1, n_steps)

        # ---- Validation ------------------------------------------------------
        val = quick_validate(
            model, loader,
            dataset_name=dataset_name,
            max_seq_len=max_seq_len,
            pad_index=pad_index,
            device=device,
            item_z_bucket=aux["item_z_bucket"],
            pop_press_vec=aux["pop_press_vec"],
            item_genre_mat=aux["item_genre_mat"],
            eval_batch_size=eval_batch_size,
            max_users=eval_max_users,
            ablation=ablation,
        )
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:3d}/{max_epochs}]  "
            f"α={alpha:.3f}  "
            f"L_total={avg_total:.4f}  L_C={avg_choice:.4f}  L_adv={avg_adv:.4f}  "
            f"val_ndcg@10={val['ndcg@10']:.4f}  disc_auc={val['disc_auc']:.4f}  "
            f"t={elapsed:.0f}s"
        )
        log_writer.writerow([
            epoch, f"{alpha:.4f}",
            f"{avg_total:.6f}", f"{avg_choice:.6f}", f"{avg_adv:.6f}",
            f"{val['ndcg@10']:.6f}", f"{val['disc_auc']:.6f}",
            f"{cur_lr:.2e}", f"{elapsed:.1f}",
        ])
        log_f.flush()

        # ---- Early stopping & checkpoint -------------------------------------
        if val["ndcg@10"] > best_val:
            best_val = float(val["ndcg@10"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            ckpt = {
                "model_state_dict": best_state,
                "epoch": epoch,
                "val_ndcg@10": best_val,
                "config": cfg,
                "seed": seed,
            }
            ckpt_path = ckpt_dir / f"ceinn_seed{seed}_best.pt"
            torch.save(ckpt, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early-stop] no improvement for {patience} epochs (best @ {best_epoch}).")
                break

    log_f.close()
    print(f"[done] best Val NDCG@10 = {best_val:.4f} at epoch {best_epoch}.")
    return {
        "best_val_ndcg10": best_val,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_dir / f"ceinn_seed{seed}_best.pt"),
        "log_path": str(log_path),
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEINN Phase 6 training driver",
    )
    parser.add_argument("--config", required=True, type=str,
                        help="Path to dataset YAML (configs/amazon_beauty.yaml or configs/movielens_10M.yaml).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=[None, "auto", "cpu", "cuda"])
    parser.add_argument("--ablation", type=str, default=None,
                        help="Optional ablation tag: A1..A8. Overrides config ablation flags.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    overrides: Dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.device is not None:
        overrides["device"] = args.device

    # Ablation tag → config patch.
    if args.ablation:
        a = args.ablation.upper()
        abl = cfg["training"].setdefault("ablation", {})
        if a == "A1":
            abl["use_rating_emb"] = False
        elif a == "A2":
            abl["loss_mode"] = "seq_choice"
        elif a == "A3":
            abl["use_grl"] = False
        elif a == "A4":
            abl["loss_mode"] = "snips_choice"
        elif a == "A5":
            abl["use_economics"] = False
        elif a == "A6":
            abl["use_cost"] = False
        elif a == "A7":
            abl["fixed_lambda_u"] = True
        elif a == "A8":
            abl["use_temporal_emb"] = False
        else:
            raise ValueError(f"Unknown ablation tag {args.ablation!r}")
        print(f"[ablation] applying patch for {a}: ablation now = {abl}")

    result = train(cfg, overrides)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
