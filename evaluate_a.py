#!/usr/bin/env python
"""
CEINN — Phase 6 evaluation script (Plan §6.2).

Computes the canonical reporting block on the test (or val) split:

  * §6.2.1 Full-Ranking NDCG@{5,10,20}, HR@{5,10,20}, MRR
  * §6.2.2 Analytical metrics:
        - Head / Torso / Tail group NDCG@10 + HR@10
        - Confounding AUC: 5-fold OvR-LR AUC of (h_t) → Z
        - lambda_u distribution: mean, std, quartiles
  * Optional CSV dump of per-user (rank, group, lambda_u) for offline plots.

Output: a JSON dict on stdout AND a CSV row appended to
`results/<dataset>_<tag>_seed<seed>.csv` if --results_csv is provided.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Local imports.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_loaders.amazon_beauty_loader import AmazonBeautyLoader  # noqa: E402
from data_loaders.movieslens_10M_loader import MovieLens10MLoader  # noqa: E402
from models import build_ceinn_amazon, build_ceinn_movielens  # noqa: E402
from utils.math_utils import batch_jaccard  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_full_ranking,
    confounding_auc,
    group_metrics,
    hr_at_k,
    mrr,
    ndcg_at_k,
    standard_topk_report,
)
from utils.popularity import load_popularity_groups  # noqa: E402

# Re-use helpers from train.py to keep behaviour identical between train/eval.
from train import (  # noqa: E402
    build_movielens_pop_press,
    build_movielens_static_z,
    load_yaml,
    pick_device,
    set_seed,
)


# =============================================================================
# Model loading
# =============================================================================

def load_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
):
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
            loader, d=d, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            propensity_hidden=prop_hidden, discriminator_hidden=disc_hidden,
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
        n_items = int(loader.vocab["n_items"])
        counts = np.zeros(n_items + 1, dtype=np.int64)
        for _u, seq in loader.train_seqs.items():
            for (iid, _r, _d) in seq:
                if 0 < iid <= n_items:
                    counts[iid] += 1
        pop_press_vec = build_movielens_pop_press(counts)
        item_genre_mat = loader.bulk_genre_matrix()
        model = build_ceinn_movielens(
            loader, n_z_buckets=n_z_buckets,
            d=d, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            propensity_hidden=prop_hidden, discriminator_hidden=disc_hidden,
        )
        aux = {
            "item_z_bucket": item_z_bucket,
            "pop_press_vec": pop_press_vec,
            "item_genre_mat": item_genre_mat,
        }
    else:
        raise ValueError(f"Unknown dataset name {dataset_name!r}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, loader, aux, ckpt


# =============================================================================
# §6.2.1  Full-ranking sweep
# =============================================================================

def evaluate_split(
    model,
    loader,
    aux: Dict[str, Any],
    *,
    dataset_name: str,
    split: str,                  # "val" or "test"
    device: torch.device,
    max_seq_len: int,
    pad_index: int,
    eval_batch_size: int = 32,
    ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a full-vocabulary ranking sweep over every user in `split` whose
    train sequence is non-empty. Returns:

      ranks       : List[int] — per-user 1-indexed rank of the target item
      target_items: List[int]
      lambda_u    : List[float]
      h_states    : np.ndarray — (n_users, d) latent states for AUC probe
      z_targets   : np.ndarray — (n_users,) Z bucket of each target item
    """
    ablation = ablation or {}
    use_economics = ablation.get("use_economics", True)
    use_cost      = ablation.get("use_cost", True)
    fixed_lambda  = ablation.get("fixed_lambda_u", False)
    fixed_value   = float(ablation.get("fixed_lambda_u_value", 0.5))
    use_rating    = ablation.get("use_rating_emb", True)
    use_temporal  = ablation.get("use_temporal_emb", True)

    if split == "val":
        eval_seqs = loader.val_seqs
        # The history for the val target is the FULL train sequence.
        history_fn = lambda u: loader.train_seqs.get(u, [])
    elif split == "test":
        eval_seqs = loader.test_seqs
        # The history for the test target is train + val (val happened before test).
        def history_fn(u):
            seq = list(loader.train_seqs.get(u, []))
            if u in loader.val_seqs:
                seq.append(loader.val_seqs[u])
            return seq
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    n_items = int(loader.vocab["n_items"])
    V = n_items + 1
    item_genre_mat = aux.get("item_genre_mat")
    item_z_bucket = aux.get("item_z_bucket")
    pop_press_vec = aux.get("pop_press_vec")

    users = [u for u in eval_seqs if history_fn(u)]
    ranks: List[int] = []
    target_items: List[int] = []
    lambdas: List[float] = []
    z_targets: List[int] = []
    h_states_list: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(users), eval_batch_size):
            batch_users = users[start:start + eval_batch_size]
            B = len(batch_users)
            item_ids   = torch.full((B, max_seq_len), pad_index, dtype=torch.long)
            rating_ids = torch.full((B, max_seq_len), pad_index, dtype=torch.long)
            dt_ids     = torch.full((B, max_seq_len), pad_index, dtype=torch.long)
            targets    = torch.empty(B, dtype=torch.long)
            z_t        = torch.empty(B, dtype=torch.long)
            history_union = (
                np.zeros((B, item_genre_mat.shape[1]), dtype=np.uint8)
                if item_genre_mat is not None else None
            )
            history_len = np.zeros(B, dtype=np.float32)
            seen_lists: List[set] = []

            for b, u in enumerate(batch_users):
                hist = history_fn(u)[-max_seq_len:]
                history_len[b] = float(len(hist))
                seen = set()
                for k, (iid, rb, db) in enumerate(hist):
                    item_ids[b, k]   = iid
                    rating_ids[b, k] = rb if use_rating else pad_index
                    dt_ids[b, k]     = db if use_temporal else pad_index
                    if 0 < iid < V:
                        seen.add(iid)
                        if history_union is not None:
                            history_union[b] |= (item_genre_mat[iid] > 0).astype(np.uint8)
                t_iid, _rb, _db = eval_seqs[u]
                targets[b] = int(t_iid)
                if dataset_name == "amazon_beauty":
                    z_t[b] = int(loader.item_meta[int(t_iid)]["Z"])
                else:
                    z_t[b] = int(item_z_bucket[int(t_iid)])
                seen_lists.append(seen)

            item_ids   = item_ids.to(device)
            rating_ids = rating_ids.to(device)
            dt_ids     = dt_ids.to(device)
            z_t_dev    = z_t.to(device)

            if dataset_name == "amazon_beauty":
                z_c = torch.from_numpy(loader.bulk_meta_arrays()["Z"]).long().unsqueeze(0).expand(B, -1).to(device)
                out = model.forward_amazon(
                    item_ids, rating_ids, dt_ids,
                    z_buckets_target=z_t_dev,
                    z_buckets_candidates=z_c,
                    grl_alpha=0.0,
                )
            else:
                rp = (history_len / max(1.0, float(max_seq_len))).astype(np.float32)
                recency_press = torch.from_numpy(rp).unsqueeze(-1).expand(-1, V).contiguous().to(device)
                pp = torch.from_numpy(pop_press_vec.astype(np.float32)).unsqueeze(0).expand(B, -1).contiguous().to(device)
                gr_np = batch_jaccard(
                    history_union, (item_genre_mat > 0).astype(np.uint8)
                ).astype(np.float32)
                gr = torch.from_numpy(gr_np).to(device)
                z_c = torch.from_numpy(item_z_bucket).long().unsqueeze(0).expand(B, -1).to(device)
                out = model.forward_movielens(
                    item_ids, rating_ids, dt_ids,
                    z_buckets_target=z_t_dev,
                    genre_red=gr, recency_press=recency_press, pop_press=pp,
                    z_buckets_candidates=z_c,
                    grl_alpha=0.0,
                )

            V_scores = out["V"]
            C        = out["C"]
            lam      = out["lambda_u"]
            if not use_economics:
                E = model.economics.value_head.item_embedding.weight
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

            scores_np = scores.detach().cpu().numpy()
            h_states_list.append(out["h_t"].detach().cpu().numpy())
            for b in range(B):
                rank = compute_full_ranking(
                    scores_np[b],
                    target_item=int(targets[b]),
                    seen_items=seen_lists[b],
                    pad_index=pad_index,
                )
                ranks.append(rank)
                target_items.append(int(targets[b]))
                lambdas.append(float(out["lambda_u"][b].item()))
                z_targets.append(int(z_t[b]))

    return {
        "ranks":        ranks,
        "target_items": target_items,
        "lambda_u":     lambdas,
        "z_targets":    np.asarray(z_targets, dtype=np.int64),
        "h_states":     np.concatenate(h_states_list, axis=0) if h_states_list else np.zeros((0,)),
    }


# =============================================================================
# §6.2.2  Aggregate report
# =============================================================================

def aggregate_report(
    raw: Dict[str, Any],
    *,
    dataset_name: str,
    proc_dir: Path,
    n_z_buckets: int,
    pop_group_path: Optional[Path],
) -> Dict[str, Any]:
    ranks = raw["ranks"]
    target_items = raw["target_items"]
    lambdas = np.asarray(raw["lambda_u"], dtype=np.float64)
    z_targets = raw["z_targets"]
    h_states = raw["h_states"]

    # §6.2.1 — standard top-K block.
    topk = standard_topk_report(ranks, ks=(5, 10, 20))

    # §6.2.2 (a) — group metrics. If no popularity group file, fall back to NaN.
    group_block: Dict[str, Any] = {}
    if pop_group_path is not None and pop_group_path.exists():
        labels, _summary = load_popularity_groups(pop_group_path)
        group_block = group_metrics(ranks, target_items, labels, k=10)
    else:
        print(f"[warn] popularity group file not found at {pop_group_path}; "
              f"skipping group metrics.")

    # §6.2.2 (b) — confounding AUC over h_t → Z target.
    if h_states.ndim == 2 and h_states.shape[0] >= 50:
        auc = confounding_auc(h_states, z_targets, n_folds=5)
    else:
        auc = float("nan")

    # §6.2.2 (c) — lambda_u distribution.
    if lambdas.size > 0:
        q = np.quantile(lambdas, [0.25, 0.5, 0.75])
        lam_block = {
            "mean": float(lambdas.mean()),
            "std":  float(lambdas.std(ddof=0)),
            "q25":  float(q[0]),
            "q50":  float(q[1]),
            "q75":  float(q[2]),
            "min":  float(lambdas.min()),
            "max":  float(lambdas.max()),
        }
    else:
        lam_block = {"mean": float("nan"), "std": float("nan")}

    return {
        "n_users":  len(ranks),
        "topk":     topk,
        "group@10": group_block,
        "confounding_auc": auc,
        "lambda_u": lam_block,
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CEINN Phase 6 evaluator")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--results_csv", type=str, default=None,
                        help="If given, append a results row to this CSV.")
    parser.add_argument("--tag", type=str, default="ceinn_full",
                        help="Tag for the results row (e.g. ceinn_full, A2, A5).")
    parser.add_argument("--per_user_csv", type=str, default=None,
                        help="If given, dump per-user (uid, target, rank, lambda_u, z_target).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_name = cfg["dataset"]["name"]
    proc_dir = Path(cfg["preprocess"]["output_dir"])
    device = pick_device(args.device)

    # Build model + load checkpoint.
    model, loader, aux, ckpt = load_checkpoint(cfg, args.checkpoint, device)
    print(f"[setup] loaded {args.checkpoint}: epoch={ckpt.get('epoch')} "
          f"val_ndcg@10={ckpt.get('val_ndcg@10')}  seed={ckpt.get('seed')}")

    pad_index = int(loader.vocab["pad_index"])
    max_seq_len = int(loader.vocab["max_seq_len"])
    ablation = cfg["training"].get("ablation", {}) or {}

    t0 = time.time()
    raw = evaluate_split(
        model, loader, aux,
        dataset_name=dataset_name,
        split=args.split,
        device=device,
        max_seq_len=max_seq_len,
        pad_index=pad_index,
        eval_batch_size=args.eval_batch_size,
        ablation=ablation,
    )
    elapsed = time.time() - t0
    print(f"[eval] ranking sweep over {len(raw['ranks'])} users took {elapsed:.1f}s")

    pop_group_path = proc_dir / "item_popularity_group.pkl"
    report = aggregate_report(
        raw,
        dataset_name=dataset_name,
        proc_dir=proc_dir,
        n_z_buckets=int(cfg["training"].get("n_z_buckets", model.n_z_buckets)),
        pop_group_path=pop_group_path,
    )
    report["split"] = args.split
    report["tag"] = args.tag
    report["seed"] = ckpt.get("seed", -1)
    report["checkpoint"] = str(Path(args.checkpoint).name)
    report["best_epoch"] = ckpt.get("epoch", -1)

    print(json.dumps(report, indent=2))

    # ---- Optional CSV outputs -----------------------------------------------
    if args.results_csv:
        out_path = Path(args.results_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not out_path.exists()
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    "tag", "seed", "split", "n_users",
                    "ndcg@5", "ndcg@10", "ndcg@20",
                    "hr@5", "hr@10", "hr@20", "mrr",
                    "head_ndcg@10", "torso_ndcg@10", "tail_ndcg@10",
                    "head_hr@10", "torso_hr@10", "tail_hr@10",
                    "confounding_auc",
                    "lambda_u_mean", "lambda_u_std",
                    "best_epoch", "checkpoint",
                ])
            g = report.get("group@10", {})
            w.writerow([
                report["tag"], report["seed"], report["split"], report["n_users"],
                report["topk"]["ndcg@5"], report["topk"]["ndcg@10"], report["topk"]["ndcg@20"],
                report["topk"]["hr@5"], report["topk"]["hr@10"], report["topk"]["hr@20"],
                report["topk"]["mrr"],
                g.get("head", {}).get("ndcg@k", float("nan")),
                g.get("torso", {}).get("ndcg@k", float("nan")),
                g.get("tail", {}).get("ndcg@k", float("nan")),
                g.get("head", {}).get("hr@k", float("nan")),
                g.get("torso", {}).get("hr@k", float("nan")),
                g.get("tail", {}).get("hr@k", float("nan")),
                report["confounding_auc"],
                report["lambda_u"].get("mean", float("nan")),
                report["lambda_u"].get("std", float("nan")),
                report["best_epoch"], report["checkpoint"],
            ])
        print(f"[results] appended row to {out_path}")

    if args.per_user_csv:
        pu_path = Path(args.per_user_csv)
        pu_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pu_path, "w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["rank", "target_item", "z_target", "lambda_u"])
            for r, tit, zt, lam in zip(
                raw["ranks"], raw["target_items"],
                raw["z_targets"].tolist(), raw["lambda_u"]
            ):
                w.writerow([r, tit, zt, lam])
        print(f"[per-user] wrote {pu_path}")


if __name__ == "__main__":
    main()
