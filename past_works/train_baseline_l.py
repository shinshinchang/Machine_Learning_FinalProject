"""
Unified Phase-5 baseline training script.

Trains a single baseline model on a single dataset with a single seed.
Run multiple times (different seeds / baselines / datasets) and let
`scripts/aggregate_baselines.py` roll up the results into the summary CSV.

Supported baselines (Plan §5.1):
    poprec      — non-personalised popularity ranker (fit-once)
    bpr_mf      — Bayesian Personalised Ranking with MF (pairwise BPR loss)
    gru4rec     — GRU session-based next-item predictor (full-vocab CE)

Usage
-----
    # PopRec (Amazon Beauty, seed 0)
    python train_baseline.py \\
        --config configs/amazon_beauty.yaml \\
        --baseline poprec --seed 0

    # GRU4Rec on MovieLens (seed 1, overriding the learning rate)
    python train_baseline.py \\
        --config configs/movielens_10M.yaml \\
        --baseline gru4rec --seed 1 \\
        --override training.lr=0.001

    # Custom run name and explicit baseline-config path
    python train_baseline.py \\
        --config configs/amazon_beauty.yaml --baseline bpr_mf --seed 2 \\
        --baseline-config configs/baselines.yaml \\
        --run-name bpr_mf_seed2

Outputs (under <training.output_dir>/baselines/<baseline>/<run_name>/):
    config.yaml          # effective config (merged dataset + baseline + overrides)
    train_log.csv        # one row per epoch (PopRec: a single row)
    best.ckpt            # state_dict at best Val NDCG@10
    last.ckpt            # state_dict at last epoch
    test_metrics.json    # Test-set Full-Ranking metrics + group metrics
    history.json         # epoch-by-epoch record for downstream plotting

Design notes
------------
* All baselines share the SAME Full-Ranking evaluation function defined
  here (`validate_full_ranking`), which masks seen-in-train items and
  uses the same `compute_full_ranking` primitive as CEINN's evaluator.
  Differences between baselines are confined to (a) how training samples
  are produced and (b) how `score_all_items(...)` is computed; both are
  internal to the model class in `models/baselines.py`.

* PopRec needs no training loop. We compute its popularity buffer once,
  then run validation + test exactly like the parametric baselines, so
  the eval pipeline is exercised identically. This is the Phase-5
  quality-gate sanity check: if PopRec NDCG@10 isn't < parametric
  baselines, something is wrong with eval, not with the baselines.

* BPR-MF flattens the training set into a list of (user_id, pos_item)
  pairs at startup. Each epoch shuffles, samples uniform-random
  negatives on the fly, and runs `bpr_loss`. There is NO sequence
  modelling here — including it would conflate BPR-MF with GRU4Rec.

* GRU4Rec uses the SAME `SequentialNextItemDataset` as CEINN's
  `train.py`, with identical shift-by-one supervision. Rating and
  temporal IDs are dropped (the model only consumes item_ids).

* The val/test "input sequence" for every baseline is the user's full
  training sequence truncated to `max_seq_len`. The held-out target is
  `val_seqs[u][0]` (val) or `test_seqs[u][0]` (test). Seen items are
  `{i for (i, _r, _d) in train_seqs[u]}` PLUS the held-out val target
  when scoring TEST (so we don't conflate val and test ranking).

* Early-stopping criterion: Val NDCG@10, patience defaults to 10 (Plan §5.2).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse Phase-2 loaders verbatim — same data, same indexing convention.
from data_loaders.amazon_beauty_loader import AmazonBeautyLoader   # noqa: E402
from data_loaders.movieslens_10M_loader import MovieLens10MLoader  # noqa: E402

# Reuse the Phase-3 metrics — identical Full-Ranking semantics to CEINN.
from utils.metrics import (                                         # noqa: E402
    compute_full_ranking,
    group_metrics,
    standard_topk_report,
)

from models.baselines import (                                      # noqa: E402
    BPRMFModel, GRU4RecModel, PopRec, build_baseline,
)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Config handling — same conventions as train.py
# =============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply `key.path=value` overrides from CLI; value is parsed by YAML."""
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


def merge_baseline_hparams(
    cfg: Dict[str, Any],
    baselines_cfg: Dict[str, Any],
    baseline_name: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Resolve the `training:` block to use baseline-specific hparams.

    Lookup order:
      1. `baselines_cfg[baseline][dataset]`  — per-baseline, per-dataset
      2. `baselines_cfg[baseline]["_default"]` — per-baseline, shared
      3. `baselines_cfg["_default"]`           — global defaults

    The resolved dict OVERRIDES `cfg["training"]` field-by-field. Any
    keys already present in `cfg["training"]` (e.g. legacy `seed`,
    `output_dir`) are retained unless explicitly overwritten.
    """
    cfg = copy.deepcopy(cfg)
    if "training" not in cfg or cfg["training"] is None:
        cfg["training"] = {}

    # Defaults from each level (later overrides earlier).
    layers: List[Dict[str, Any]] = []
    layers.append(baselines_cfg.get("_default", {}))
    bcfg = baselines_cfg.get(baseline_name, {})
    if isinstance(bcfg, dict):
        layers.append(bcfg.get("_default", {}))
        layers.append(bcfg.get(dataset_name, {}))

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for k, v in layer.items():
            cfg["training"][k] = v
    return cfg


# =============================================================================
# Dataset loading
# =============================================================================

def load_dataset_loader(cfg: Dict[str, Any]):
    """Dispatch to the right Phase-2 loader (identical to train.py)."""
    name = cfg["dataset"]["name"].lower()
    pdir = Path(cfg["preprocess"]["output_dir"])
    if name == "amazon_beauty":
        return "amazon_beauty", AmazonBeautyLoader.from_directory(pdir)
    if name in ("movielens_10m", "movielens"):
        return "movielens_10m", MovieLens10MLoader.from_directory(pdir)
    raise ValueError(f"Unknown dataset name {name!r}")


def build_seen_items(train_seqs: Dict[int, List[Tuple[int, int, int]]]) -> Dict[int, Set[int]]:
    return {u: {iid for (iid, _r, _d) in seq} for u, seq in train_seqs.items()}


def load_popularity_groups(processed_dir: Path) -> Optional[Dict[int, str]]:
    """Return Phase-3 head/torso/tail labels, or None if not built."""
    path = processed_dir / "item_popularity_group.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["labels"]


# =============================================================================
# Datasets — sequence-based (GRU4Rec) and pair-based (BPR-MF)
# =============================================================================

@dataclass
class SeqBatch:
    """Batch for GRU4Rec next-item supervision."""
    user_ids: torch.Tensor       # (B,)
    item_ids: torch.Tensor       # (B, T)
    target_ids: torch.Tensor     # (B, T)


class SequentialNextItemDataset(Dataset):
    """
    Same shift-by-one supervision as `train.py::SequentialNextItemDataset`,
    but stripped down: no rating, no dt, no user-meta. GRU4Rec consumes
    only item IDs.

    For each user with sequence [i_1, ..., i_n], yields:
        input  = pad_left_to_T([i_1, ..., i_{n-1}])    # length T
        target = pad_left_to_T([i_2, ..., i_n])

    Both arrays are right-padded with 0 (PAD).

    We require sequences of length >= 2 so there's at least 1 (input, target).
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[Tuple[int, int, int]]],
        max_seq_len: int,
        pad_index: int = 0,
    ) -> None:
        self.pad_index = int(pad_index)
        self.max_seq_len = int(max_seq_len)
        self.users = sorted(u for u, s in train_seqs.items() if len(s) >= 2)
        self.train_seqs = train_seqs

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        u = self.users[idx]
        seq = self.train_seqs[u]
        # Truncate to most recent (T+1) interactions.
        if len(seq) > self.max_seq_len + 1:
            seq = seq[-(self.max_seq_len + 1):]

        T = self.max_seq_len
        item = np.zeros(T, dtype=np.int64)
        target = np.zeros(T, dtype=np.int64)
        n = len(seq) - 1
        for k in range(n):
            item[k] = seq[k][0]
            target[k] = seq[k + 1][0]
        return {"user_id": u, "item_ids": item, "target_ids": target}


def collate_seq(rows: List[Dict[str, Any]]) -> SeqBatch:
    return SeqBatch(
        user_ids=torch.tensor([r["user_id"] for r in rows], dtype=torch.long),
        item_ids=torch.from_numpy(np.stack([r["item_ids"] for r in rows])),
        target_ids=torch.from_numpy(np.stack([r["target_ids"] for r in rows])),
    )


class BPRPairsDataset(Dataset):
    """
    Flat (user_id, positive_item) list for BPR-MF training.

    We materialise the flat list at construction (cheap: ~10M Python
    ints for ML, ~200k for Amazon) so each epoch only does the shuffle
    and the negative-sampling. Negatives are drawn UNIFORMLY at random
    from [1, n_items] at __getitem__ time; we do NOT exclude items the
    user has positively interacted with — the collision probability is
    ~ |H_u| / n_items which is below 1% for both datasets, and the
    expected BPR loss is unbiased to this level of noise.

    Why not exclude observed positives from the negative pool?
    ----------------------------------------------------------
    BPR's correctness is statistical: any negative drawn from the
    catalogue distribution defines a valid pairwise objective in
    expectation. Excluding the user's positives biases the marginal of
    the negatives toward unobserved items, which is the WHOLE POINT of
    the BPR loss — and is mildly inconsistent with the loss formulation
    in the original paper (Eq. 4 there). Standard BPR-MF implementations
    (recbole, implicit, lightfm) all skip this filtering.
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[Tuple[int, int, int]]],
        n_items: int,
        pad_index: int = 0,
    ) -> None:
        self.n_items = int(n_items)
        self.pad_index = int(pad_index)
        # Flat array of (user, pos_item) pairs.
        users: List[int] = []
        items: List[int] = []
        for u, seq in train_seqs.items():
            for triple in seq:
                iid = int(triple[0])
                if iid == self.pad_index:
                    continue
                users.append(int(u))
                items.append(iid)
        self.user_arr = np.asarray(users, dtype=np.int64)
        self.item_arr = np.asarray(items, dtype=np.int64)

    def __len__(self) -> int:
        return self.user_arr.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, int]:
        u = int(self.user_arr[idx])
        ip = int(self.item_arr[idx])
        # Uniform random negative in [1, n_items]. We don't bother
        # collision-rejecting against `ip` for speed; the chance is
        # 1/n_items ≈ 10⁻⁴.
        in_ = int(np.random.randint(1, self.n_items + 1))
        return {"user_id": u, "pos_item": ip, "neg_item": in_}


def collate_bpr(rows: List[Dict[str, int]]) -> Dict[str, torch.Tensor]:
    return {
        "user_ids": torch.tensor([r["user_id"] for r in rows], dtype=torch.long),
        "pos_items": torch.tensor([r["pos_item"] for r in rows], dtype=torch.long),
        "neg_items": torch.tensor([r["neg_item"] for r in rows], dtype=torch.long),
    }


# =============================================================================
# Full-Ranking evaluator — shared across all three baselines
# =============================================================================

@torch.no_grad()
def validate_full_ranking(
    model: nn.Module,
    target_seqs: Dict[int, Tuple[int, int, int]],
    train_seqs: Dict[int, List[Tuple[int, int, int]]],
    seen_items: Dict[int, Set[int]],
    *,
    max_seq_len: int,
    device: torch.device,
    pad_index: int = 0,
    batch_size: int = 256,
    extra_mask_seqs: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Compute corpus-level Full-Ranking metrics for a baseline.

    Parameters
    ----------
    model         : a baseline exposing `score_all_items(user_ids, item_seq)`.
                    `user_ids` is (B,) long; `item_seq` is (B, T) long.
    target_seqs   : {user_idx: (item_idx, rating_bin, dt_bin)} — usually
                    `loader.val_seqs` or `loader.test_seqs`.
    train_seqs    : Phase-2 train sequences (for building the input).
    seen_items    : {user_idx: set of train-seen item ids}.
    max_seq_len   : pad/truncate input sequences to this length.
    extra_mask_seqs : when evaluating TEST, pass `loader.val_seqs` here so
                    the val target is also masked out (avoid contamination
                    between val and test ranks).

    Returns
    -------
    metrics : dict with `ndcg@5`, `ndcg@10`, `ndcg@20`, `hr@5`, `hr@10`,
              `hr@20`, `mrr`, `n_users`.
    ranks   : per-user int ranks (1-indexed), aligned with `target_items`.
    target_items : per-user target item id (for downstream group metrics).
    """
    model.eval()
    users = [u for u in target_seqs.keys() if u in train_seqs and train_seqs[u]]
    if not users:
        return {"ndcg@10": 0.0, "n_users": 0}, [], []

    ranks: List[int] = []
    target_items: List[int] = []

    for start in range(0, len(users), batch_size):
        ubatch = users[start:start + batch_size]
        B = len(ubatch)
        T = max_seq_len
        # Build padded input sequence (item_ids only).
        item = np.zeros((B, T), dtype=np.int64)
        for b, u in enumerate(ubatch):
            seq = train_seqs[u]
            if len(seq) > T:
                seq = seq[-T:]
            for k, (iid, _r, _d) in enumerate(seq):
                item[b, k] = iid

        item_t = torch.from_numpy(item).to(device)
        user_t = torch.tensor(ubatch, dtype=torch.long, device=device)
        scores = model.score_all_items(user_t, item_t)        # (B, V+1)
        scores_np = scores.detach().cpu().numpy()

        for b, u in enumerate(ubatch):
            target = int(target_seqs[u][0])
            if target == pad_index:
                continue
            # Mask train-seen items + (optional) val-seen items.
            mask = set(seen_items.get(u, set()))
            if extra_mask_seqs is not None and u in extra_mask_seqs:
                mask.add(int(extra_mask_seqs[u][0]))
            mask.discard(target)  # never mask the target itself
            r = compute_full_ranking(
                scores_np[b], target_item=target,
                seen_items=mask, pad_index=pad_index,
            )
            ranks.append(r)
            target_items.append(target)

    rep = standard_topk_report(ranks, ks=(5, 10, 20))
    rep["n_users"] = len(ranks)
    return rep, ranks, target_items


# =============================================================================
# Training procedures
# =============================================================================

def train_poprec(
    model: PopRec,
    train_seqs: Dict[int, List[Tuple[int, int, int]]],
    *,
    n_items: int,
    pad_index: int,
    device: torch.device,
) -> PopRec:
    """
    "Train" PopRec by recomputing its popularity buffer. The class method
    already does this — we just call it and move the result to `device`.
    Returns the (now-fitted) model.
    """
    fitted = PopRec.fit_from_train_seqs(
        train_seqs, n_items=n_items, pad_index=pad_index,
    ).to(device)
    return fitted


@dataclass
class EpochStats:
    loss: float
    n_steps: int


def train_bpr_mf_one_epoch(
    model: BPRMFModel,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: float = 0.0,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    n_steps = 0
    for batch in loader:
        u = batch["user_ids"].to(device)
        ip = batch["pos_items"].to(device)
        in_ = batch["neg_items"].to(device)

        loss = model.bpr_loss(u, ip, in_)
        optim.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        total_loss += float(loss.item())
        n_steps += 1
    return EpochStats(loss=total_loss / max(1, n_steps), n_steps=n_steps)


def train_gru4rec_one_epoch(
    model: GRU4RecModel,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    *,
    device: torch.device,
    pad_index: int = 0,
    grad_clip: float = 1.0,
) -> EpochStats:
    """
    Standard sequential next-item CE loss with PAD masking. Identical to
    `utils.losses.SequentialCrossEntropy(pad_index=0)` applied on the
    full-sequence logits.
    """
    model.train()
    total_loss = 0.0
    n_steps = 0
    for batch in loader:
        item_ids = batch.item_ids.to(device)         # (B, T)
        target_ids = batch.target_ids.to(device)     # (B, T)

        logits = model.forward_full_sequence(item_ids)   # (B, T, V+1)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids.reshape(B * T),
            ignore_index=pad_index,
        )

        optim.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        total_loss += float(loss.item())
        n_steps += 1
    return EpochStats(loss=total_loss / max(1, n_steps), n_steps=n_steps)


# =============================================================================
# Checkpoint helpers (mirror of train.py)
# =============================================================================

def save_checkpoint(path: Path, model: nn.Module, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "extra": extra,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, *, map_location=None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("extra", {})


# =============================================================================
# Per-baseline orchestration — each returns the trained model
# =============================================================================

def run_poprec(
    *,
    loader,
    train_cfg: Dict[str, Any],
    seen_items: Dict[int, Set[int]],
    out_dir: Path,
    device: torch.device,
    n_items: int,
    pad_index: int,
    max_seq_len: int,
    csv_path: Path,
) -> Tuple[nn.Module, float, int]:
    """
    Train PopRec (= fit popularity vector once) and write a single-row
    train_log entry. Returns (model, best_val_ndcg10, best_epoch=1).
    """
    model = train_poprec(
        PopRec(n_items=n_items, pad_index=pad_index),
        train_seqs=loader.train_seqs,
        n_items=n_items, pad_index=pad_index, device=device,
    )

    val_metrics, _ranks, _items = validate_full_ranking(
        model, loader.val_seqs, loader.train_seqs, seen_items,
        max_seq_len=max_seq_len, device=device, pad_index=pad_index,
        batch_size=int(train_cfg.get("val_batch_size", 256)),
    )
    ndcg10 = val_metrics.get("ndcg@10", 0.0)

    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(
            f"1,0,0.0,"
            f"{val_metrics.get('ndcg@5', 0):.4f},"
            f"{val_metrics.get('ndcg@10', 0):.4f},"
            f"{val_metrics.get('ndcg@20', 0):.4f},"
            f"{val_metrics.get('hr@10', 0):.4f},"
            f"{val_metrics.get('mrr', 0):.4f},"
            f"0.0\n"
        )
    print(f"  PopRec (single fit) — val NDCG@10 = {ndcg10:.4f}")
    save_checkpoint(out_dir / "best.ckpt", model, extra={
        "epoch": 1, "val_ndcg10": ndcg10, "baseline": "poprec",
    })
    save_checkpoint(out_dir / "last.ckpt", model, extra={
        "epoch": 1, "val_ndcg10": ndcg10, "baseline": "poprec",
    })
    return model, ndcg10, 1


def run_bpr_mf(
    *,
    loader,
    train_cfg: Dict[str, Any],
    seen_items: Dict[int, Set[int]],
    out_dir: Path,
    device: torch.device,
    n_users: int,
    n_items: int,
    pad_index: int,
    max_seq_len: int,
    csv_path: Path,
) -> Tuple[nn.Module, float, int]:
    """
    Train BPR-MF with pairwise loss + uniform-random negative sampling.
    Returns (best_model_loaded, best_val_ndcg10, best_epoch).
    """
    d = int(train_cfg.get("d", 64))
    lr = float(train_cfg.get("lr", 5e-4))
    batch_size = int(train_cfg.get("batch_size", 256))
    max_epochs = int(train_cfg.get("max_epochs", 200))
    patience = int(train_cfg.get("early_stop_patience", 10))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    val_batch_size = int(train_cfg.get("val_batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))

    model = BPRMFModel(
        n_users=n_users, n_items=n_items, d=d, pad_index=pad_index,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model: BPR-MF, n_users={n_users}, n_items={n_items}, d={d}")
    print(f"    Trainable params: {n_params:,}")

    pairs = BPRPairsDataset(loader.train_seqs, n_items=n_items, pad_index=pad_index)
    pair_loader = DataLoader(
        pairs, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_bpr, drop_last=False,
    )
    print(f"    Train pairs: {len(pairs):,}; batch_size={batch_size}; "
          f"steps/epoch ≈ {len(pairs)//max(1,batch_size):,}")

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_ndcg10 = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        stats = train_bpr_mf_one_epoch(
            model, pair_loader, optim, device=device, grad_clip=grad_clip,
        )
        val_metrics, _ranks, _items = validate_full_ranking(
            model, loader.val_seqs, loader.train_seqs, seen_items,
            max_seq_len=max_seq_len, device=device, pad_index=pad_index,
            batch_size=val_batch_size,
        )
        ndcg10 = val_metrics.get("ndcg@10", 0.0)
        epoch_time = time.time() - t0

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{stats.n_steps},{stats.loss:.4f},"
                f"{val_metrics.get('ndcg@5', 0):.4f},"
                f"{val_metrics.get('ndcg@10', 0):.4f},"
                f"{val_metrics.get('ndcg@20', 0):.4f},"
                f"{val_metrics.get('hr@10', 0):.4f},"
                f"{val_metrics.get('mrr', 0):.4f},"
                f"{epoch_time:.1f}\n"
            )
        print(f"  Epoch {epoch:3d}/{max_epochs}  L_BPR={stats.loss:.4f}  "
              f"val_NDCG@10={ndcg10:.4f}  ({epoch_time:.1f}s)")

        save_checkpoint(out_dir / "last.ckpt", model, extra={
            "epoch": epoch, "val_ndcg10": ndcg10, "baseline": "bpr_mf",
        })
        if ndcg10 > best_val_ndcg10:
            best_val_ndcg10 = ndcg10
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(out_dir / "best.ckpt", model, extra={
                "epoch": epoch, "val_ndcg10": ndcg10, "baseline": "bpr_mf",
            })
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"  Early stopping at epoch {epoch}: no improvement "
                      f"for {patience} epochs (best @ epoch {best_epoch}, "
                      f"NDCG@10={best_val_ndcg10:.4f})")
                break

    # Reload best checkpoint for the downstream test evaluation.
    load_checkpoint(out_dir / "best.ckpt", model, map_location=device)
    return model, best_val_ndcg10, best_epoch


def run_gru4rec(
    *,
    loader,
    train_cfg: Dict[str, Any],
    seen_items: Dict[int, Set[int]],
    out_dir: Path,
    device: torch.device,
    n_items: int,
    pad_index: int,
    max_seq_len: int,
    csv_path: Path,
) -> Tuple[nn.Module, float, int]:
    """
    Train GRU4Rec with full-vocab cross-entropy.
    """
    d = int(train_cfg.get("d", 64))
    n_layers = int(train_cfg.get("n_layers", 1))
    dropout = float(train_cfg.get("dropout", 0.1))
    lr = float(train_cfg.get("lr", 5e-4))
    batch_size = int(train_cfg.get("batch_size", 256))
    max_epochs = int(train_cfg.get("max_epochs", 200))
    patience = int(train_cfg.get("early_stop_patience", 10))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    val_batch_size = int(train_cfg.get("val_batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))

    model = GRU4RecModel(
        n_items=n_items, d=d, n_layers=n_layers, dropout=dropout,
        pad_index=pad_index,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model: GRU4Rec, n_items={n_items}, d={d}, layers={n_layers}, "
          f"dropout={dropout}")
    print(f"    Trainable params: {n_params:,}")

    ds = SequentialNextItemDataset(
        loader.train_seqs, max_seq_len=max_seq_len, pad_index=pad_index,
    )
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_seq, drop_last=False,
    )
    print(f"    Train sequences: {len(ds)} users, batch_size={batch_size}")

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_ndcg10 = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        stats = train_gru4rec_one_epoch(
            model, train_loader, optim, device=device,
            pad_index=pad_index, grad_clip=grad_clip,
        )
        val_metrics, _ranks, _items = validate_full_ranking(
            model, loader.val_seqs, loader.train_seqs, seen_items,
            max_seq_len=max_seq_len, device=device, pad_index=pad_index,
            batch_size=val_batch_size,
        )
        ndcg10 = val_metrics.get("ndcg@10", 0.0)
        epoch_time = time.time() - t0

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{stats.n_steps},{stats.loss:.4f},"
                f"{val_metrics.get('ndcg@5', 0):.4f},"
                f"{val_metrics.get('ndcg@10', 0):.4f},"
                f"{val_metrics.get('ndcg@20', 0):.4f},"
                f"{val_metrics.get('hr@10', 0):.4f},"
                f"{val_metrics.get('mrr', 0):.4f},"
                f"{epoch_time:.1f}\n"
            )
        print(f"  Epoch {epoch:3d}/{max_epochs}  L_CE={stats.loss:.4f}  "
              f"val_NDCG@10={ndcg10:.4f}  ({epoch_time:.1f}s)")

        save_checkpoint(out_dir / "last.ckpt", model, extra={
            "epoch": epoch, "val_ndcg10": ndcg10, "baseline": "gru4rec",
        })
        if ndcg10 > best_val_ndcg10:
            best_val_ndcg10 = ndcg10
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(out_dir / "best.ckpt", model, extra={
                "epoch": epoch, "val_ndcg10": ndcg10, "baseline": "gru4rec",
            })
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"  Early stopping at epoch {epoch}: no improvement "
                      f"for {patience} epochs (best @ epoch {best_epoch}, "
                      f"NDCG@10={best_val_ndcg10:.4f})")
                break

    load_checkpoint(out_dir / "best.ckpt", model, map_location=device)
    return model, best_val_ndcg10, best_epoch


# =============================================================================
# Test-set evaluation block (mirrors evaluate.py but standalone)
# =============================================================================

def eval_test_set(
    model: nn.Module,
    loader,
    seen_items: Dict[int, Set[int]],
    *,
    max_seq_len: int,
    device: torch.device,
    pad_index: int,
    val_batch_size: int,
    group_labels: Optional[Dict[int, str]],
) -> Dict[str, Any]:
    """
    Run Full-Ranking eval on the TEST split, with val targets added to
    the seen-mask so val and test ranks aren't conflated.

    Also report Head/Torso/Tail group metrics if `group_labels` is
    provided (Plan §3.3.3 + §5.3).
    """
    rep, ranks, target_items = validate_full_ranking(
        model, loader.test_seqs, loader.train_seqs, seen_items,
        max_seq_len=max_seq_len, device=device, pad_index=pad_index,
        batch_size=val_batch_size,
        extra_mask_seqs=loader.val_seqs,
    )
    result: Dict[str, Any] = {
        "test_metrics": rep,
        "n_test_users": rep.get("n_users", 0),
    }
    if group_labels is not None and ranks:
        gm = group_metrics(ranks, target_items, group_labels, k=10)
        result["group_metrics@10"] = gm
    return result


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="Per-dataset YAML (provides dataset paths + name).")
    ap.add_argument("--baseline", required=True,
                    choices=["poprec", "bpr_mf", "gru4rec"],
                    help="Baseline to train.")
    ap.add_argument("--baseline-config", type=Path,
                    default=Path(__file__).parent / "configs" / "baselines.yaml",
                    help="Baseline hparams YAML.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override seed for this run (default: from config).")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--override", nargs="*", default=[],
                    help="`key.path=value` overrides applied AFTER merge.")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Configuration: merge dataset + baseline + CLI overrides.
    # -------------------------------------------------------------------------
    cfg = load_yaml(args.config)
    baselines_cfg = load_yaml(args.baseline_config) if args.baseline_config.exists() else {}
    dataset_name = cfg["dataset"]["name"].lower()
    cfg = merge_baseline_hparams(cfg, baselines_cfg, args.baseline, dataset_name)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    train_cfg = cfg.setdefault("training", {})

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 0))
    set_seed(seed)

    # -------------------------------------------------------------------------
    # Output layout: runs/<dataset_name>/baselines/<baseline>/<run_name>/
    # -------------------------------------------------------------------------
    base_out = Path(train_cfg.get("output_dir", f"runs/{dataset_name}"))
    run_name = args.run_name or f"seed{seed}_{int(time.time())}"
    out_dir = base_out / "baselines" / args.baseline / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output dir: {out_dir}")

    # Snapshot effective config — Phase-8 reproducibility hook.
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # -------------------------------------------------------------------------
    # Load Phase-2 artefacts.
    # -------------------------------------------------------------------------
    print(f"Loading Phase-2 artefacts for {dataset_name} …")
    _kind, loader = load_dataset_loader(cfg)

    n_users = int(loader.vocab["n_users"])
    n_items = int(loader.vocab["n_items"])
    pad_index = int(loader.vocab.get("pad_index", 0))
    max_seq_len = int(loader.vocab.get("max_seq_len", train_cfg.get("max_seq_len", 50)))
    seen_items = build_seen_items(loader.train_seqs)

    # Popularity groups (optional but should exist after Phase-3 script).
    group_labels = load_popularity_groups(Path(cfg["preprocess"]["output_dir"]))
    if group_labels is None:
        print("  (No item_popularity_group.pkl; group metrics will be skipped.)")

    # -------------------------------------------------------------------------
    # Open the per-epoch CSV. Header columns differ slightly per baseline
    # (PopRec has no train_loss), but we use one canonical schema.
    # -------------------------------------------------------------------------
    csv_path = out_dir / "train_log.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,n_steps,train_loss,"
                "val_ndcg@5,val_ndcg@10,val_ndcg@20,val_hr@10,val_mrr,"
                "epoch_time_s\n")

    # -------------------------------------------------------------------------
    # Dispatch to the chosen baseline.
    # -------------------------------------------------------------------------
    print(f"\n=== Training {args.baseline} on {dataset_name} (seed={seed}) ===")
    t_start = time.time()
    if args.baseline == "poprec":
        model, best_val_ndcg10, best_epoch = run_poprec(
            loader=loader, train_cfg=train_cfg, seen_items=seen_items,
            out_dir=out_dir, device=device, n_items=n_items,
            pad_index=pad_index, max_seq_len=max_seq_len, csv_path=csv_path,
        )
    elif args.baseline == "bpr_mf":
        model, best_val_ndcg10, best_epoch = run_bpr_mf(
            loader=loader, train_cfg=train_cfg, seen_items=seen_items,
            out_dir=out_dir, device=device, n_users=n_users, n_items=n_items,
            pad_index=pad_index, max_seq_len=max_seq_len, csv_path=csv_path,
        )
    else:  # gru4rec
        model, best_val_ndcg10, best_epoch = run_gru4rec(
            loader=loader, train_cfg=train_cfg, seen_items=seen_items,
            out_dir=out_dir, device=device, n_items=n_items,
            pad_index=pad_index, max_seq_len=max_seq_len, csv_path=csv_path,
        )
    train_time = time.time() - t_start

    print(f"\nTraining complete. Best val NDCG@10 = {best_val_ndcg10:.4f} "
          f"at epoch {best_epoch}.")

    # -------------------------------------------------------------------------
    # Test-set evaluation on the best checkpoint.
    # -------------------------------------------------------------------------
    print("\n=== Test-set Full-Ranking evaluation ===")
    val_batch_size = int(train_cfg.get("val_batch_size", 256))
    test_block = eval_test_set(
        model, loader, seen_items,
        max_seq_len=max_seq_len, device=device, pad_index=pad_index,
        val_batch_size=val_batch_size, group_labels=group_labels,
    )

    rep = test_block["test_metrics"]
    print(f"  Test  NDCG@5={rep.get('ndcg@5', 0):.4f}  "
          f"NDCG@10={rep.get('ndcg@10', 0):.4f}  "
          f"NDCG@20={rep.get('ndcg@20', 0):.4f}  "
          f"HR@10={rep.get('hr@10', 0):.4f}  "
          f"MRR={rep.get('mrr', 0):.4f}  "
          f"(n={rep.get('n_users', 0)})")
    if "group_metrics@10" in test_block:
        gm = test_block["group_metrics@10"]
        for group in ("head", "torso", "tail"):
            if group in gm:
                g = gm[group]
                print(f"    {group:>5s}: NDCG@10={g['ndcg@k']:.4f}  "
                      f"HR@10={g['hr@k']:.4f}  n={g['n_users']}")

    # -------------------------------------------------------------------------
    # Persist a compact JSON summary for downstream aggregation.
    # -------------------------------------------------------------------------
    summary = {
        "baseline": args.baseline,
        "dataset": dataset_name,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_ndcg10": float(best_val_ndcg10),
        "train_time_s": float(train_time),
        **test_block,
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # And a flat history.json for plotting.
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline": args.baseline, "dataset": dataset_name,
            "seed": seed, "best_epoch": int(best_epoch),
            "best_val_ndcg10": float(best_val_ndcg10),
        }, f, indent=2)

    print(f"\nArtefacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
