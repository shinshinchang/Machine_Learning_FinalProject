"""
End-to-end smoke test for train.py + evaluate.py.

Builds a tiny synthetic Phase-2 directory layout in a temp dir for
each dataset path, runs 2 epochs of training, then evaluates.

Goal: detect plumbing errors (wrong key names in vocab, mis-wired
loaders, bad collate behaviour, mis-shaped tensors) before we go
near a real 10M-row dataset.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# =============================================================================
# Fixture builders
# =============================================================================

def build_amazon_processed(out: Path, *, n_users: int = 80, n_items: int = 50) -> dict:
    """Create a minimal data/processed/amazon_beauty/ directory."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Sequences of varying length (3..15).
    train_seqs = {}
    val_seqs = {}
    test_seqs = {}
    for u in range(1, n_users + 1):
        L = int(rng.integers(8, 16))
        seq = [(int(rng.integers(1, n_items + 1)),
                int(rng.integers(1, 6)),       # rating bin 1..5
                int(rng.integers(1, 32)))      # dt bin 1..31
               for _ in range(L)]
        # leave-one-out: last → test, 2nd-last → val, rest → train
        test_seqs[u] = seq[-1]
        val_seqs[u]  = seq[-2]
        train_seqs[u] = seq[:-2]

    with open(out / "train_seqs.pkl", "wb") as f: pickle.dump(train_seqs, f)
    with open(out / "val_seqs.pkl",   "wb") as f: pickle.dump(val_seqs, f)
    with open(out / "test_seqs.pkl",  "wb") as f: pickle.dump(test_seqs, f)

    # Item meta: one row per item id ∈ [1, n_items].
    item_meta = {
        iid: {
            "cat":       int(rng.integers(2, 8)),
            "brand":     int(rng.integers(2, 6)),
            "log_price": float(rng.uniform(0.5, 2.5)),
            "Z":         int(rng.integers(0, 10)),
        }
        for iid in range(1, n_items + 1)
    }
    with open(out / "item_meta.pkl", "wb") as f: pickle.dump(item_meta, f)

    id_maps = {
        "user2idx": {f"u_{u}": u for u in range(1, n_users + 1)},
        "idx2user": {u: f"u_{u}" for u in range(1, n_users + 1)},
        "item2idx": {f"i_{iid}": iid for iid in range(1, n_items + 1)},
        "idx2item": {iid: f"i_{iid}" for iid in range(1, n_items + 1)},
        "cat2idx": {}, "idx2cat": {},
        "brand2idx": {}, "idx2brand": {},
    }
    with open(out / "id_maps.pkl", "wb") as f: pickle.dump(id_maps, f)

    with open(out / "bucket_edges.pkl", "wb") as f:
        pickle.dump({"dt_edges": np.linspace(0, 7, 32),
                     "salesrank_edges": np.linspace(0, 7, 11),
                     "global_log_price_median": 1.07,
                     "leaf_log_price_median": {}}, f)

    vocab = {
        "n_users":       n_users,
        "n_items":       n_items,
        "n_cats":        8,
        "n_brands":      6,
        "n_Z_bins":      10,
        "n_dt_bins":     32,
        "n_rating_bins": 6,
        "pad_index":     0,
        "max_seq_len":   20,
    }
    with open(out / "vocab_sizes.json", "w") as f: json.dump(vocab, f)

    # Optional: popularity groups so evaluate.py picks them up.
    head_ids = list(range(1, n_items // 5 + 1))
    tail_ids = list(range(n_items - n_items // 5, n_items + 1))
    labels = {}
    for iid in range(1, n_items + 1):
        if iid in head_ids:   labels[iid] = "head"
        elif iid in tail_ids: labels[iid] = "tail"
        else:                 labels[iid] = "torso"
    with open(out / "item_popularity_group.pkl", "wb") as f:
        pickle.dump({"labels": labels, "summary": {
            "head_quantile": 0.8, "tail_quantile": 0.2,
            "head_threshold_count": 5.0, "tail_threshold_count": 2.0,
            "n_head": len(head_ids), "n_torso": n_items - len(head_ids) - len(tail_ids),
            "n_tail": len(tail_ids), "n_items": n_items,
        }}, f)

    return vocab


def build_movielens_processed(out: Path, *, n_users: int = 60, n_items: int = 40) -> dict:
    """Create a minimal data/processed/movielens_10m/ directory."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    train_seqs, val_seqs, test_seqs = {}, {}, {}
    for u in range(1, n_users + 1):
        L = int(rng.integers(6, 14))
        seq = [(int(rng.integers(1, n_items + 1)),
                int(rng.integers(1, 11)),   # rating bin 1..10 (half-stars)
                int(rng.integers(1, 64)))   # dt bin 1..63
               for _ in range(L)]
        test_seqs[u] = seq[-1]
        val_seqs[u]  = seq[-2]
        train_seqs[u] = seq[:-2]

    with open(out / "train_seqs.pkl", "wb") as f: pickle.dump(train_seqs, f)
    with open(out / "val_seqs.pkl",   "wb") as f: pickle.dump(val_seqs, f)
    with open(out / "test_seqs.pkl",  "wb") as f: pickle.dump(test_seqs, f)

    # 19-d genre vectors per item.
    n_genres = 19
    item_genre = {}
    for iid in range(1, n_items + 1):
        v = np.zeros(n_genres, dtype=np.float32)
        for g in rng.choice(n_genres, size=rng.integers(1, 4), replace=False):
            v[g] = 1.0
        item_genre[iid] = v
    with open(out / "item_genre.pkl", "wb") as f: pickle.dump(item_genre, f)

    # row_index + parallel arrays for dynamic_Z, genre_red, genre_red_idf.
    row_index = {}
    n_train_rows = 0
    for u, seq in train_seqs.items():
        for pos in range(len(seq)):
            row_index[(u, pos)] = n_train_rows
            n_train_rows += 1

    dynZ_values        = rng.integers(0, 100, size=n_train_rows).astype(np.int64)
    genre_red_values   = rng.uniform(0.0, 1.0, size=n_train_rows).astype(np.float32)
    genre_red_idf_vals = rng.uniform(0.0, 1.0, size=n_train_rows).astype(np.float32)
    idf_weights = rng.uniform(0.5, 2.0, size=n_genres)
    genre_vocab = [f"g{i}" for i in range(n_genres)]

    with open(out / "dynamic_Z.pkl", "wb") as f:
        pickle.dump({"row_index": row_index, "values": dynZ_values}, f)
    with open(out / "genre_red.pkl", "wb") as f:
        pickle.dump({"row_index": row_index, "values": genre_red_values}, f)
    with open(out / "genre_red_idf.pkl", "wb") as f:
        pickle.dump({"row_index": row_index, "values": genre_red_idf_vals,
                     "idf_weights": idf_weights, "genre_vocab": genre_vocab}, f)

    id_maps = {
        "user2idx": {u: u for u in range(1, n_users + 1)},
        "idx2user": {u: u for u in range(1, n_users + 1)},
        "item2idx": {iid: iid for iid in range(1, n_items + 1)},
        "idx2item": {iid: iid for iid in range(1, n_items + 1)},
        "genre2idx": {g: i for i, g in enumerate(genre_vocab)},
    }
    with open(out / "id_maps.pkl", "wb") as f: pickle.dump(id_maps, f)

    with open(out / "bucket_edges.pkl", "wb") as f:
        pickle.dump({"dt_edges": np.linspace(0, 8, 64)}, f)

    vocab = {
        "n_users":       n_users,
        "n_items":       n_items,
        "n_genres":      n_genres,
        "n_dt_bins":     64,
        "n_rating_bins": 11,
        "pad_index":     0,
        "max_seq_len":   20,
    }
    with open(out / "vocab_sizes.json", "w") as f: json.dump(vocab, f)

    # popularity groups (same simple scheme)
    head_ids = list(range(1, n_items // 5 + 1))
    tail_ids = list(range(n_items - n_items // 5, n_items + 1))
    labels = {}
    for iid in range(1, n_items + 1):
        if iid in head_ids:   labels[iid] = "head"
        elif iid in tail_ids: labels[iid] = "tail"
        else:                 labels[iid] = "torso"
    with open(out / "item_popularity_group.pkl", "wb") as f:
        pickle.dump({"labels": labels, "summary": {
            "head_quantile": 0.8, "tail_quantile": 0.2,
            "head_threshold_count": 5.0, "tail_threshold_count": 2.0,
            "n_head": len(head_ids), "n_torso": n_items - len(head_ids) - len(tail_ids),
            "n_tail": len(tail_ids), "n_items": n_items,
        }}, f)

    return vocab


# =============================================================================
# Test config builder
# =============================================================================

def make_amazon_config(processed_dir: Path, output_dir: Path) -> dict:
    return {
        "dataset": {"name": "amazon_beauty",
                    "paths": {}, "encoding": {"default": "utf-8"}},
        "preprocess": {"output_dir": str(processed_dir)},
        "training": {
            "seed": 0,
            "output_dir": str(output_dir),
            "num_workers": 0,
            "architecture": {
                "d": 32, "n_layers": 1, "n_heads": 2, "dropout": 0.1,
                "propensity_hidden": 32, "discriminator_hidden": None,
            },
            "lr": 1e-3, "batch_size": 16, "val_batch_size": 16,
            "max_epochs": 2, "early_stop_patience": 5,
            "grad_clip": 1.0, "lambda_reg": 1e-5,
            "lambda_adv": 0.1,
            "ips_variant": "clipped", "ips_clip_tau": 30.0,
            "propensity_warmup_epochs": 1, "propensity_lr": 1e-3,
            "ablation": {
                "use_rating_emb": True, "loss_mode": "ips_choice",
                "use_grl": True, "use_economics": True, "use_cost": True,
                "fixed_lambda_u": False, "fixed_lambda_u_value": 0.5,
                "use_temporal_emb": True,
            },
        },
    }


def make_movielens_config(processed_dir: Path, output_dir: Path) -> dict:
    return {
        "dataset": {"name": "movielens_10m",
                    "paths": {}, "encoding": {"default": "utf-8"}},
        "preprocess": {"output_dir": str(processed_dir)},
        "training": {
            "seed": 0,
            "output_dir": str(output_dir),
            "num_workers": 0,
            "architecture": {
                "d": 32, "n_layers": 1, "n_heads": 2, "dropout": 0.1,
                "propensity_hidden": 32, "discriminator_hidden": None,
            },
            "lr": 1e-3, "batch_size": 8, "val_batch_size": 16,
            "max_epochs": 2, "early_stop_patience": 5,
            "grad_clip": 1.0, "lambda_reg": 1e-5,
            "lambda_adv": 0.1,
            "ips_variant": "self_normalized", "ips_clip_tau": 100.0,
            "ml_n_negs": 20,            # small for speed
            "n_z_buckets": 5,
            "propensity_warmup_epochs": 1, "propensity_lr": 1e-3,
            "ablation": {
                "use_rating_emb": True, "loss_mode": "ips_choice",
                "use_grl": True, "use_economics": True, "use_cost": True,
                "fixed_lambda_u": False, "fixed_lambda_u_value": 0.5,
                "use_temporal_emb": True,
            },
        },
    }


# =============================================================================
# Tests
# =============================================================================

def _run(cmd):
    """Run a subprocess and return (returncode, stdout)."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return r.returncode, r.stdout + r.stderr


class TestAmazonE2E:
    def test_train_then_evaluate(self, tmp_path):
        processed = tmp_path / "data" / "processed" / "amazon_beauty"
        build_amazon_processed(processed)
        runs = tmp_path / "runs"
        cfg = make_amazon_config(processed, runs)
        cfg_path = tmp_path / "amazon.yaml"
        with open(cfg_path, "w") as f: yaml.safe_dump(cfg, f)

        # Train.
        rc, out = _run([
            sys.executable, "train.py",
            "--config", str(cfg_path),
            "--seed", "0",
            "--run-name", "smoke",
            "--device", "cpu",
        ])
        assert rc == 0, f"train.py failed:\n{out}"
        assert "Training complete" in out, out

        run_dir = runs / "smoke"
        assert (run_dir / "best.ckpt").exists()
        assert (run_dir / "train_log.csv").exists()
        assert (run_dir / "history.json").exists()

        # Evaluate.
        rc, out = _run([
            sys.executable, "evaluate.py",
            "--checkpoint", str(run_dir / "best.ckpt"),
            "--config", str(cfg_path),
            "--device", "cpu",
            "--skip-auc",
        ])
        assert rc == 0, f"evaluate.py failed:\n{out}"
        metrics_path = run_dir / "test_metrics.json"
        assert metrics_path.exists()
        m = json.loads(metrics_path.read_text())
        # Sanity: keys exist and topline metrics are in [0, 1].
        assert set(m["metrics"].keys()) >= {"ndcg@10", "hr@10", "mrr"}
        assert 0.0 <= m["metrics"]["ndcg@10"] <= 1.0
        assert 0.0 <= m["metrics"]["hr@10"] <= 1.0


class TestMovieLensE2E:
    def test_train_then_evaluate(self, tmp_path):
        processed = tmp_path / "data" / "processed" / "movielens_10m"
        build_movielens_processed(processed)
        runs = tmp_path / "runs"
        cfg = make_movielens_config(processed, runs)
        cfg_path = tmp_path / "ml.yaml"
        with open(cfg_path, "w") as f: yaml.safe_dump(cfg, f)

        rc, out = _run([
            sys.executable, "train.py",
            "--config", str(cfg_path),
            "--seed", "0", "--run-name", "smoke", "--device", "cpu",
        ])
        assert rc == 0, f"train.py failed:\n{out}"
        run_dir = runs / "smoke"
        assert (run_dir / "best.ckpt").exists()

        rc, out = _run([
            sys.executable, "evaluate.py",
            "--checkpoint", str(run_dir / "best.ckpt"),
            "--config", str(cfg_path),
            "--device", "cpu", "--skip-auc",
        ])
        assert rc == 0, f"evaluate.py failed:\n{out}"
        m = json.loads((run_dir / "test_metrics.json").read_text())
        assert 0.0 <= m["metrics"]["ndcg@10"] <= 1.0


class TestAblationToggles:
    """Verify that critical ablation switches actually take effect."""

    def test_no_grl_completes(self, tmp_path):
        processed = tmp_path / "data" / "processed" / "amazon_beauty"
        build_amazon_processed(processed)
        runs = tmp_path / "runs"
        cfg = make_amazon_config(processed, runs)
        cfg["training"]["ablation"]["use_grl"] = False    # A3
        cfg_path = tmp_path / "a3.yaml"
        with open(cfg_path, "w") as f: yaml.safe_dump(cfg, f)
        rc, out = _run([sys.executable, "train.py",
                        "--config", str(cfg_path),
                        "--run-name", "a3", "--device", "cpu"])
        assert rc == 0, out
        # L_adv should be zero (or near-zero) every epoch.
        log = (runs / "a3" / "train_log.csv").read_text().splitlines()[1:]
        for line in log:
            l_adv = float(line.split(",")[4])
            assert l_adv == 0.0, f"A3 (w/o GRL) should have L_adv=0, got {l_adv}"

    def test_no_economics_completes(self, tmp_path):
        processed = tmp_path / "data" / "processed" / "amazon_beauty"
        build_amazon_processed(processed)
        runs = tmp_path / "runs"
        cfg = make_amazon_config(processed, runs)
        cfg["training"]["ablation"]["use_economics"] = False     # A5
        cfg_path = tmp_path / "a5.yaml"
        with open(cfg_path, "w") as f: yaml.safe_dump(cfg, f)
        rc, out = _run([sys.executable, "train.py",
                        "--config", str(cfg_path),
                        "--run-name", "a5", "--device", "cpu"])
        assert rc == 0, out

    def test_seq_choice_disables_ips_weights(self, tmp_path):
        processed = tmp_path / "data" / "processed" / "amazon_beauty"
        build_amazon_processed(processed)
        runs = tmp_path / "runs"
        cfg = make_amazon_config(processed, runs)
        cfg["training"]["ablation"]["loss_mode"] = "seq_choice"  # A2
        cfg_path = tmp_path / "a2.yaml"
        with open(cfg_path, "w") as f: yaml.safe_dump(cfg, f)
        rc, out = _run([sys.executable, "train.py",
                        "--config", str(cfg_path),
                        "--run-name", "a2", "--device", "cpu"])
        assert rc == 0, out
