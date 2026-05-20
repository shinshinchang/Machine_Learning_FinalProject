"""
CEINN evaluation script (Phase 6 §6.2 + Phase 8 analysis hooks).

Loads a trained checkpoint and reports on the TEST set:
  * Full Ranking NDCG@{5, 10, 20}, HR@{5, 10, 20}, MRR    (§3.3.1-2)
  * Head / Torso / Tail subgroup metrics                  (§3.3.3)
  * h_t ⊥ Z probe via Logistic Regression macro-OvR AUC   (§3.3.4)
  * λ_u distribution statistics                            (§3.3.5)

Usage
-----
    python evaluate.py --checkpoint <run_dir>/best.ckpt \
        --config configs/amazon_beauty.yaml
    # Optional: drop the per-user CSV
    python evaluate.py --checkpoint <run>/best.ckpt --config <cfg> \
        --dump-ranks <run>/test_ranks.csv

Output
------
JSON to stdout AND to <checkpoint_dir>/test_metrics.json. The JSON
shape matches what `results/` aggregators in Phase 8 expect (see
keys at the bottom of this docstring).

Why a separate script?
----------------------
Per design doc §3.3: the test set is hit ONCE per configuration. Keeping
evaluation separate from training enforces the discipline of running it
exactly once with the chosen best checkpoint. The script must NOT silently
peek at test data during training; the val-vs-test boundary lives at the
file-system level (train.py only ever reads val_seqs.pkl).

Output JSON schema
------------------
{
  "config": <run_name>,
  "seed":   <int>,
  "n_test_users": <int>,
  "metrics": {
    "ndcg@5": float, "ndcg@10": float, "ndcg@20": float,
    "hr@5": float, "hr@10": float, "hr@20": float,
    "mrr": float
  },
  "groups": {
    "head":  {"ndcg@10": ..., "hr@10": ..., "n_users": ...},
    "torso": ...,
    "tail":  ...
  },
  "confounding_auc": float | null,    # null if popularity_groups missing
  "lambda_u": {
    "mean": float, "std": float, "min": float, "max": float,
    "p25": float, "p50": float, "p75": float
  }
}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# Repo-root imports.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.amazon_beauty_loader import AmazonBeautyLoader   # noqa: E402
from data_loaders.movieslens_10M_loader import MovieLens10MLoader  # noqa: E402
from models import (                                                # noqa: E402
    CEINNModel, build_ceinn_amazon, build_ceinn_movielens,
)
from utils.math_utils import apply_bucket_edges, fit_log_quantile_edges  # noqa: E402
from utils.metrics import (                                         # noqa: E402
    compute_full_ranking, confounding_auc, group_metrics, standard_topk_report,
)
from utils.popularity import load_popularity_groups                 # noqa: E402

# Re-use a few train.py helpers without forcing a circular import.
from train import (                                                  # noqa: E402
    build_seen_items, compute_movielens_z_buckets, load_config,
    load_dataset_loader,
)


# =============================================================================
# Model construction + checkpoint loading
# =============================================================================

def build_model_from_config(
    cfg: Dict[str, Any], dataset_kind: str, loader,
    *, n_z_buckets: int,
) -> CEINNModel:
    arch = cfg["training"]["architecture"]
    if dataset_kind == "amazon_beauty":
        return build_ceinn_amazon(
            loader,
            d=arch["d"], n_heads=arch["n_heads"], n_layers=arch["n_layers"],
            dropout=arch["dropout"],
            propensity_hidden=arch.get("propensity_hidden", 64),
            discriminator_hidden=arch.get("discriminator_hidden"),
        )
    else:
        return build_ceinn_movielens(
            loader,
            n_z_buckets=n_z_buckets,
            d=arch["d"], n_heads=arch["n_heads"], n_layers=arch["n_layers"],
            dropout=arch["dropout"],
            propensity_hidden=arch.get("propensity_hidden", 64),
            discriminator_hidden=arch.get("discriminator_hidden"),
        )


def restore_checkpoint(model: CEINNModel, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("extra", {})


# =============================================================================
# Test-set Full Ranking — same engine as the val-time helper in train.py
# but kept here as a standalone copy to (a) decouple from train.py and
# (b) collect side info (h_t, λ_u) that we don't gather during training.
# =============================================================================

@torch.no_grad()
def evaluate_test_set(
    model: CEINNModel,
    test_seqs: Dict[int, Tuple[int, int, int]],
    train_seqs: Dict[int, List[Tuple[int, int, int]]],
    val_seqs: Dict[int, Tuple[int, int, int]],
    seen_items: Dict[int, set],
    *,
    max_seq_len: int,
    device: torch.device,
    item_z_bucket: torch.Tensor,
    ml_loader: Optional[MovieLens10MLoader] = None,
    pad_index: int = 0,
    batch_size: int = 256,
    ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      ranks       : list[int] — per-user 1-indexed test rank
      item_ids    : list[int] — per-user target item id (parallel to ranks)
      h_states    : np.ndarray (N, d) — h_t collected at val-position
                    encoding (used for confounding AUC)
      z_labels    : np.ndarray (N,)   — Z bucket of each test target
      lambda_u    : np.ndarray (N,)

    Important
    ---------
    For test evaluation, the model's INPUT is the user's training
    history + the validation interaction (since at test time the val
    interaction is known). This matches the standard SASRec/BERT4Rec
    evaluation protocol.

    `seen_items` for masking is built from train + val targets (the
    test target is, by construction, not in seen).
    """
    model.eval()
    ablation = ablation or {}
    use_economics = ablation.get("use_economics", True)
    use_cost = ablation.get("use_cost", True)

    users = [u for u in test_seqs.keys() if u in train_seqs and train_seqs[u]]
    ranks: List[int] = []
    item_ids: List[int] = []
    h_states_all: List[np.ndarray] = []
    z_labels_all: List[int] = []
    lambda_all: List[float] = []

    # Precompute the genre matrix once for ML.
    gmat_t = None
    if model.dataset_name == "movielens" and ml_loader is not None:
        gmat_t = torch.from_numpy(ml_loader.bulk_genre_matrix()).to(device).float()

    for start in range(0, len(users), batch_size):
        ubatch = users[start:start + batch_size]
        B = len(ubatch)
        T = max_seq_len
        item = np.zeros((B, T), dtype=np.int64)
        rating = np.zeros((B, T), dtype=np.int64)
        dt = np.zeros((B, T), dtype=np.int64)

        for b, u in enumerate(ubatch):
            # Input = train history APPENDED with the val interaction.
            seq = list(train_seqs[u])
            if u in val_seqs:
                seq = seq + [val_seqs[u]]
            if len(seq) > T:
                seq = seq[-T:]
            for k, (iid, r, d) in enumerate(seq):
                item[b, k]   = iid
                rating[b, k] = r
                dt[b, k]     = d

        if not ablation.get("use_rating_emb", True):
            rating[:] = 0
        if not ablation.get("use_temporal_emb", True):
            dt[:] = 0

        item_t   = torch.from_numpy(item).to(device)
        rating_t = torch.from_numpy(rating).to(device)
        dt_t     = torch.from_numpy(dt).to(device)

        # Encode → h_t.
        h_t = model.encode(item_t, rating_t, dt_t)
        # λ_u — capture for distribution analysis.
        lam = model.economics.lambda_net(item_t)
        lambda_all.extend(lam.detach().cpu().numpy().tolist())

        # V over full V_cat.
        V = model.economics.value_head(h_t)

        # Scoring (same logic as train.py's val helper).
        if use_economics and use_cost:
            if model.dataset_name == "amazon_beauty":
                C_all = model.economics.cost_backend.cost_all_items()
                scores = V - lam.unsqueeze(-1) * C_all.unsqueeze(0)
            else:
                assert gmat_t is not None
                hist_g = gmat_t[item_t]
                hist_union = (hist_g.sum(dim=1) > 0).float()
                g_size = gmat_t.sum(dim=1)
                u_size = hist_union.sum(dim=1)
                inter = hist_union @ gmat_t.t()
                union = (g_size.unsqueeze(0) + u_size.unsqueeze(1) - inter).clamp_min(1e-8)
                genre_red = inter / union
                recency = torch.full_like(V, 1.0)  # test position is end of history
                z_max = max(1.0, float(item_z_bucket.max().item()))
                pop = item_z_bucket.to(device).float() / z_max
                pop = pop.unsqueeze(0).expand(B, -1)
                C = model.economics.cost_backend(genre_red, recency, pop)
                scores = V - lam.unsqueeze(-1) * C
        elif use_economics and not use_cost:
            scores = V
        else:
            scores = V  # A5 fallback

        scores_np = scores.detach().cpu().numpy()
        h_states_np = h_t.detach().cpu().numpy()
        item_z_np = item_z_bucket.detach().cpu().numpy()

        for b, u in enumerate(ubatch):
            target = test_seqs[u][0]
            if target == pad_index:
                continue
            # Seen mask: train items ∪ val item.
            seen = set(seen_items.get(u, set()))
            if u in val_seqs:
                seen.add(val_seqs[u][0])
            r = compute_full_ranking(
                scores_np[b], target_item=int(target),
                seen_items=seen, pad_index=pad_index,
            )
            ranks.append(r)
            item_ids.append(int(target))
            h_states_all.append(h_states_np[b])
            z_labels_all.append(int(item_z_np[target]))

    return {
        "ranks": ranks,
        "item_ids": item_ids,
        "h_states": np.stack(h_states_all) if h_states_all else np.zeros((0, model.d)),
        "z_labels": np.asarray(z_labels_all, dtype=np.int64),
        "lambda_u": np.asarray(lambda_all, dtype=np.float64),
    }


# =============================================================================
# Lambda_u summary statistics
# =============================================================================

def summarise_lambda_u(lambda_u: np.ndarray) -> Dict[str, float]:
    if lambda_u.size == 0:
        return {k: float("nan") for k in
                ("mean", "std", "min", "max", "p25", "p50", "p75")}
    return {
        "mean": float(lambda_u.mean()),
        "std":  float(lambda_u.std(ddof=0)),
        "min":  float(lambda_u.min()),
        "max":  float(lambda_u.max()),
        "p25":  float(np.percentile(lambda_u, 25)),
        "p50":  float(np.percentile(lambda_u, 50)),
        "p75":  float(np.percentile(lambda_u, 75)),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to `best.ckpt` (or `last.ckpt`) saved by train.py.")
    ap.add_argument("--config", required=True, type=Path,
                    help="The same per-dataset YAML used at train time.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dump-ranks", type=Path, default=None,
                    help="If set, write per-user (user_id, target, rank) CSV.")
    ap.add_argument("--skip-auc", action="store_true",
                    help="Skip the LR-based confounding AUC (slow on CPU).")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    cfg = load_config(args.config)

    # If the checkpoint stored its config, prefer it (in case the user
    # passed a generic config that drifted since training).
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    extra = ckpt.get("extra", {})
    if "config" in extra:
        print("    Using config snapshot from checkpoint.")
        cfg = extra["config"]
    seed = int(extra.get("seed", cfg["training"].get("seed", 0)))

    # -------------------------------------------------------------------------
    # Loader + auxiliary tensors.
    # -------------------------------------------------------------------------
    print("Loading Phase-2 artefacts …")
    dataset_kind, loader = load_dataset_loader(cfg)
    pad_index = int(loader.vocab["pad_index"])
    max_seq_len = int(loader.vocab["max_seq_len"])

    if dataset_kind == "amazon_beauty":
        n_z_buckets = int(loader.vocab["n_Z_bins"])
        item_z = torch.from_numpy(loader.bulk_meta_arrays()["Z"]).long().to(device)
        ml_loader = None
    else:
        n_z_buckets = int(cfg["training"].get("n_z_buckets", 10))
        z_per_row, _edges = compute_movielens_z_buckets(loader, n_z_buckets)
        n_items = int(loader.vocab["n_items"])
        last_z = np.zeros(n_items + 1, dtype=np.int64)
        for (u, pos), row_idx in loader._row_index.items():
            iid = loader.train_seqs[u][pos][0]
            last_z[iid] = int(z_per_row[row_idx])
        item_z = torch.from_numpy(last_z).to(device)
        ml_loader = loader

    seen_items = build_seen_items(loader.train_seqs)

    # -------------------------------------------------------------------------
    # Build + restore model.
    # -------------------------------------------------------------------------
    model = build_model_from_config(cfg, dataset_kind, loader,
                                     n_z_buckets=n_z_buckets).to(device)
    restore_checkpoint(model, args.checkpoint, device=device)
    print(f"    Restored from {args.checkpoint}")

    # -------------------------------------------------------------------------
    # Run evaluation.
    # -------------------------------------------------------------------------
    print("Running Full Ranking on the test set …")
    t0 = time.time()
    result = evaluate_test_set(
        model, loader.test_seqs, loader.train_seqs, loader.val_seqs,
        seen_items,
        max_seq_len=max_seq_len, device=device, item_z_bucket=item_z,
        ml_loader=ml_loader, pad_index=pad_index,
        batch_size=int(cfg["training"].get("val_batch_size", 256)),
        ablation=cfg["training"].get("ablation", {}),
    )
    n_users = len(result["ranks"])
    eval_time = time.time() - t0
    print(f"    Evaluated {n_users} users in {eval_time:.1f}s")

    # -------------------------------------------------------------------------
    # Topline metrics.
    # -------------------------------------------------------------------------
    rep = standard_topk_report(result["ranks"], ks=(5, 10, 20))
    print(f"\n  NDCG@10 = {rep['ndcg@10']:.4f}    "
          f"HR@10 = {rep['hr@10']:.4f}    MRR = {rep['mrr']:.4f}")

    # -------------------------------------------------------------------------
    # Subgroup metrics (Head / Torso / Tail).
    # -------------------------------------------------------------------------
    processed_dir = Path(cfg["preprocess"]["output_dir"])
    pop_path = processed_dir / "item_popularity_group.pkl"
    groups_out: Dict[str, Dict[str, float]] = {}
    if pop_path.exists():
        group_labels, _summary = load_popularity_groups(pop_path)
        groups_out = group_metrics(result["ranks"], result["item_ids"],
                                    group_labels, k=10)
        for gn in ("head", "torso", "tail"):
            g = groups_out[gn]
            print(f"    [{gn:>5s}] NDCG@10={g['ndcg@k']:.4f}  "
                  f"HR@10={g['hr@k']:.4f}  (n={g['n_users']})")
    else:
        print(f"    (skipped subgroup metrics: {pop_path} missing — "
              f"run scripts/build_popularity_groups.py first)")

    # -------------------------------------------------------------------------
    # h_t ⊥ Z probe (confounding AUC).
    # -------------------------------------------------------------------------
    if args.skip_auc or result["h_states"].shape[0] == 0:
        auc = None
        print("    (confounding AUC skipped)")
    else:
        print("    Probing h_t ⊥ Z via Logistic Regression …")
        auc = confounding_auc(result["h_states"], result["z_labels"])
        print(f"    Confounding AUC = {auc:.4f}  "
              f"(random baseline = {1.0 / n_z_buckets + 0.5:.4f})")

    # -------------------------------------------------------------------------
    # λ_u distribution.
    # -------------------------------------------------------------------------
    lam_stats = summarise_lambda_u(result["lambda_u"])
    print(f"    λ_u: mean={lam_stats['mean']:.3f}  "
          f"median={lam_stats['p50']:.3f}  "
          f"[{lam_stats['min']:.3f}, {lam_stats['max']:.3f}]")

    # -------------------------------------------------------------------------
    # Assemble output JSON.
    # -------------------------------------------------------------------------
    out_payload: Dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "seed": seed,
        "dataset": model.dataset_name,
        "n_test_users": n_users,
        "eval_time_s": eval_time,
        "metrics": rep,
        "groups": groups_out,
        "confounding_auc": (None if auc is None else float(auc)),
        "lambda_u": lam_stats,
    }
    out_path = args.checkpoint.parent / "test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {out_path}")

    # -------------------------------------------------------------------------
    # Optional per-user ranks dump.
    # -------------------------------------------------------------------------
    if args.dump_ranks is not None:
        with open(args.dump_ranks, "w", encoding="utf-8") as f:
            f.write("user_idx,target_item_idx,rank\n")
            # We need the user_idx parallel to ranks; result didn't store
            # it. Recover by walking test_seqs in the same order
            # evaluate_test_set used.
            users = [u for u in loader.test_seqs.keys()
                     if u in loader.train_seqs and loader.train_seqs[u]]
            for u, r, tgt in zip(users, result["ranks"], result["item_ids"]):
                f.write(f"{u},{tgt},{r}\n")
        print(f"Wrote per-user ranks → {args.dump_ranks}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
