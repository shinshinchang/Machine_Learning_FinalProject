"""
EDA for Amazon Beauty — aligned with CEINN modules 3.1, 3.2, 3.3.

Files expected (relative to project root `ceinn/`):
    data/raw/Beauty_5.json          # 5-core review subset (line-delimited JSON, double quotes)
    data/raw/meta_Beauty.json       # metadata (line-delimited Python-dict repr, single quotes)
    data/raw/reviews_Beauty.json    # full review set (line-delimited JSON)

Outputs are written to `eda/outputs/amazon_beauty/`:
    *.png    statistical plots
    *.csv    numeric summaries that train.py / preprocess.py can consume
    summary.json  one-shot machine-readable summary

What this script answers (one block per CEINN section it serves):

  [3.1] Sequential Backbone
    - user sequence length distribution (does 5-core give long-enough sequences?)
    - rating distribution (is r_k truly informative or collapsed at 5?)
    - delta-t distribution on log scale (justifying log bucketization in E_t)

  [3.2] Causal Deconfounding
    - salesRank coverage and distribution (is it a usable Z?)
    - log + quantile bucketization viability (3.2.4)
    - long-tail severity → IPS clipping necessity (3.2.7)
    - propensity simulation: how extreme is 1/p_i at the cold tail?

  [3.3] Economics-Informed Utility
    - price / brand / category coverage (can we even build C(i,t)?)
    - price-by-category boxplot (validates the price × category interaction term)
    - price-salesRank correlation (are α_1 log(price) and α_4 η(salesRank) redundant?)

Usage:
    python -m eda.eda_amazon_beauty          # from project root `ceinn/`
"""

from __future__ import annotations

import ast
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw"
OUT = HERE / "outputs" / "amazon_beauty"
OUT.mkdir(parents=True, exist_ok=True)

CORE_PATH = RAW / "Beauty_5.json"
META_PATH = RAW / "meta_Beauty.json"
REVIEWS_PATH = RAW / "reviews_Beauty.json"

# Plot style: keep neutral and readable. No seaborn dependency.
plt.rcParams.update({
    "figure.figsize": (7, 4.2),
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def safe_log_bins(arr: np.ndarray, n_bins: int = 40, min_val: float = 1.0) -> np.ndarray:
    """Build monotonic log-spaced bins; falls back to linear if range too narrow."""
    arr = np.asarray(arr)
    arr = arr[arr > 0]
    if arr.size == 0:
        return np.linspace(0, 1, n_bins + 1)
    lo = max(float(arr.min()), float(min_val))
    hi = float(arr.max()) + 1.0
    if hi <= lo * 1.01:
        # arr is too narrow to log-bin; widen and use linear bins.
        hi = max(hi, lo + 1.0)
        return np.linspace(lo, hi, n_bins + 1)
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def iter_jsonl(path: Path):
    """Beauty_5.json and reviews_Beauty.json are line-delimited standard JSON."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_metalines(path: Path):
    """meta_Beauty.json is the McAuley-style Python-dict repr (single quotes).

    We cannot use json.loads; ast.literal_eval is the standard reader.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield ast.literal_eval(line)
            except (SyntaxError, ValueError):
                # A handful of malformed lines exist in McAuley's dumps; skip.
                continue


def load_core_as_df() -> pd.DataFrame:
    """The interaction matrix Module 3.1 consumes."""
    rows = []
    for r in iter_jsonl(CORE_PATH):
        rows.append({
            "user": r["reviewerID"],
            "item": r["asin"],
            "rating": float(r["overall"]),
            "ts": int(r["unixReviewTime"]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["user", "ts"]).reset_index(drop=True)
    return df


def load_meta_as_df() -> pd.DataFrame:
    """Extract only the fields CEINN's economic cost function needs.

    For categories we keep the leaf node (last element of the deepest path),
    which matches the one-hot embedding-lookup design in 3.3.3.
    """
    rows = []
    for m in iter_metalines(META_PATH):
        asin = m.get("asin")
        if asin is None:
            continue
        price = m.get("price")
        brand = m.get("brand")

        # salesRank may be missing, or keyed by a category other than 'Beauty'.
        # The 3.2 Z is built on Beauty-rank specifically; we keep that and
        # also a fallback "any-rank" for coverage analysis.
        sr = m.get("salesRank") or {}
        rank_beauty = sr.get("Beauty")
        rank_any = next(iter(sr.values()), None) if sr else None

        cats = m.get("categories") or []
        leaf_cat = None
        top_cat = None
        cat_depth = 0
        if cats:
            # categories is list-of-list: pick the deepest path.
            deepest = max(cats, key=len)
            cat_depth = len(deepest)
            if deepest:
                top_cat = deepest[0] if len(deepest) >= 1 else None
                leaf_cat = deepest[-1]

        rows.append({
            "item": asin,
            "price": price,
            "brand": brand,
            "rank_beauty": rank_beauty,
            "rank_any": rank_any,
            "top_cat": top_cat,
            "leaf_cat": leaf_cat,
            "cat_depth": cat_depth,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module 3.1 — Sequential Backbone EDA
# ---------------------------------------------------------------------------
def eda_module_3_1(core: pd.DataFrame, summary: dict) -> None:
    print("\n[3.1] Sequential Backbone")
    print("-" * 60)

    # --- 3.1.a sequence length per user
    seq_len = core.groupby("user").size()
    summary["seq_len_p50"] = int(seq_len.median())
    summary["seq_len_p90"] = int(seq_len.quantile(0.90))
    summary["seq_len_p99"] = int(seq_len.quantile(0.99))
    summary["seq_len_max"] = int(seq_len.max())
    summary["n_users"] = int(seq_len.size)
    summary["n_items"] = int(core["item"].nunique())
    summary["n_interactions"] = int(len(core))
    summary["density"] = float(len(core) / (summary["n_users"] * summary["n_items"]))
    print(f"  users={summary['n_users']}  items={summary['n_items']}  "
          f"interactions={summary['n_interactions']}  density={summary['density']:.2e}")
    print(f"  seq-length p50={summary['seq_len_p50']}  p90={summary['seq_len_p90']}  "
          f"p99={summary['seq_len_p99']}  max={summary['seq_len_max']}")

    fig, ax = plt.subplots()
    ax.hist(seq_len.values, bins=safe_log_bins(seq_len.values, n_bins=40, min_val=5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length per user (log)")
    ax.set_ylabel("# users (log)")
    ax.set_title("3.1 — user sequence length distribution")
    fig.tight_layout()
    fig.savefig(OUT / "3_1_seq_length.png")
    plt.close(fig)

    # --- 3.1.b rating distribution: is E_r(r_k) actually informative?
    r_counts = core["rating"].value_counts().sort_index()
    summary["rating_share_5"] = float((core["rating"] == 5).mean())
    summary["rating_share_le_2"] = float((core["rating"] <= 2).mean())
    summary["rating_entropy"] = float(
        -np.sum((p := r_counts / r_counts.sum()) * np.log(p + 1e-12))
    )
    print(f"  rating share@5={summary['rating_share_5']:.3f}  "
          f"share≤2={summary['rating_share_le_2']:.3f}  "
          f"H(rating)={summary['rating_entropy']:.3f}")

    fig, ax = plt.subplots()
    ax.bar(r_counts.index.astype(str), r_counts.values, color="#3b6db3")
    ax.set_xlabel("overall rating")
    ax.set_ylabel("# interactions")
    ax.set_title("3.1 — rating distribution (informativeness of E_r)")
    fig.tight_layout()
    fig.savefig(OUT / "3_1_rating_dist.png")
    plt.close(fig)

    # --- 3.1.c Δt distribution: justifying log bucketization in E_t
    # Compute Δt within each user. unixReviewTime is in seconds but Amazon
    # snaps reviewTime to day granularity, so we expect a spike at 0.
    core = core.sort_values(["user", "ts"])
    dt = core.groupby("user")["ts"].diff().dropna().values
    dt_days = dt / 86400.0
    dt_pos = dt_days[dt_days > 0]
    summary["dt_share_same_day"] = float((dt_days == 0).mean())
    summary["dt_log10_range"] = float(
        np.log10(dt_pos.max() + 1) - np.log10(dt_pos.min() + 1)
    ) if dt_pos.size else 0.0
    print(f"  Δt same-day share={summary['dt_share_same_day']:.3f}  "
          f"Δt log10 span={summary['dt_log10_range']:.2f} decades")

    fig, ax = plt.subplots()
    bins = safe_log_bins(dt_pos, n_bins=50, min_val=1.0) if dt_pos.size else np.linspace(0, 1, 10)
    ax.hist(dt_pos, bins=bins, color="#3b6db3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt between consecutive reviews (days, log)")
    ax.set_ylabel("count (log)")
    ax.set_title("3.1 — Δt distribution justifies log bucketization in E_t")
    fig.tight_layout()
    fig.savefig(OUT / "3_1_delta_t.png")
    plt.close(fig)

    # Persist the per-user sequence stats — preprocess.py can sanity-check
    # against this when fixing max_seq_len.
    seq_len.rename("n").to_csv(OUT / "3_1_seq_len_per_user.csv")


# ---------------------------------------------------------------------------
# Module 3.2 — Causal Deconfounding EDA
# ---------------------------------------------------------------------------
def eda_module_3_2(core: pd.DataFrame, meta: pd.DataFrame, summary: dict) -> None:
    print("\n[3.2] Causal Deconfounding")
    print("-" * 60)

    # --- 3.2.a salesRank coverage on the 5-core item universe
    core_items = pd.DataFrame({"item": core["item"].unique()})
    join = core_items.merge(meta, on="item", how="left")
    cov_meta = float(join["price"].notna().mean() + 0)  # placeholder, replaced below
    cov_meta_any = float(join["top_cat"].notna().mean())
    cov_rank_beauty = float(join["rank_beauty"].notna().mean())
    cov_rank_any = float(join["rank_any"].notna().mean())
    cov_price = float(join["price"].notna().mean())
    cov_brand = float(join["brand"].notna().mean())
    summary.update({
        "meta_join_coverage_any_field": cov_meta_any,
        "salesRank_beauty_coverage": cov_rank_beauty,
        "salesRank_any_coverage": cov_rank_any,
        "price_coverage": cov_price,
        "brand_coverage": cov_brand,
    })
    print(f"  meta join coverage (any cat)={cov_meta_any:.3f}  "
          f"salesRank[Beauty]={cov_rank_beauty:.3f}  salesRank[any]={cov_rank_any:.3f}")
    print(f"  price coverage={cov_price:.3f}  brand coverage={cov_brand:.3f}")

    # --- 3.2.b salesRank distribution: raw + log
    rank = join["rank_beauty"].dropna().astype(float).values
    if rank.size:
        log_rank = np.log10(rank + 1)
        summary["log_rank_p50"] = float(np.median(log_rank))
        summary["log_rank_iqr"] = float(np.subtract(*np.percentile(log_rank, [75, 25])))
        print(f"  log10(salesRank) median={summary['log_rank_p50']:.2f}  "
              f"IQR={summary['log_rank_iqr']:.2f}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(rank, bins=60, color="#a44a3f")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("salesRank (raw)")
        axes[0].set_ylabel("# items (log)")
        axes[0].set_title("raw salesRank — heavy long tail")
        axes[1].hist(log_rank, bins=40, color="#a44a3f")
        axes[1].set_xlabel("log10(salesRank + 1)")
        axes[1].set_ylabel("# items")
        axes[1].set_title("log salesRank — bucketization viable")
        fig.suptitle("3.2 — salesRank as confounder Z (Amazon Beauty)")
        fig.tight_layout()
        fig.savefig(OUT / "3_2_salesrank_log.png")
        plt.close(fig)

    # --- 3.2.c popularity long tail on the *interaction* side (not metadata).
    # This is what actually drives the exposure mechanism in 3.2.2.
    item_freq = core.groupby("item").size().sort_values(ascending=False)
    freq_arr = item_freq.values.astype(float)
    cum_share = np.cumsum(freq_arr) / freq_arr.sum()
    head_at_20 = float(cum_share[int(len(freq_arr) * 0.20) - 1]) if len(freq_arr) > 5 else float("nan")
    # Gini on item-frequency
    sorted_freq = np.sort(freq_arr)
    n = len(sorted_freq)
    cum = np.cumsum(sorted_freq)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * sorted_freq) / cum[-1] - (n + 1)) / n)
    summary["pop_head20_share"] = head_at_20
    summary["pop_gini"] = gini
    print(f"  top-20%-items capture {head_at_20*100:.1f}% of interactions  Gini={gini:.3f}")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(freq_arr) + 1) / len(freq_arr), cum_share, color="#a44a3f")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("item rank (fraction)")
    ax.set_ylabel("cumulative interaction share")
    ax.set_title(f"3.2 — popularity Lorenz curve (Gini={gini:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "3_2_popularity_lorenz.png")
    plt.close(fig)

    # --- 3.2.d propensity simulation: how extreme is 1/p_i at the cold tail?
    # We do not need to fit a real propensity estimator here — the point of EDA
    # is to verify *that* IPS clipping (3.2.7) will be necessary. We use the
    # empirical exposure rate p_i ≈ freq_i / max(freq).
    p_emp = freq_arr / freq_arr.max()
    inv_p = 1.0 / np.maximum(p_emp, 1e-6)
    summary["inv_p_p99"] = float(np.percentile(inv_p, 99))
    summary["inv_p_p999"] = float(np.percentile(inv_p, 99.9))
    summary["inv_p_max"] = float(inv_p.max())
    print(f"  empirical 1/p_i  p99={summary['inv_p_p99']:.1f}  "
          f"p99.9={summary['inv_p_p999']:.1f}  max={summary['inv_p_max']:.1f}  "
          f"(→ clipping τ is mandatory)")

    fig, ax = plt.subplots()
    ax.hist(np.log10(inv_p), bins=40, color="#a44a3f")
    ax.set_xlabel("log10(1 / p_i)")
    ax.set_ylabel("# items")
    ax.set_title("3.2 — inverse propensity blow-up at the cold tail")
    fig.tight_layout()
    fig.savefig(OUT / "3_2_ips_blowup.png")
    plt.close(fig)

    # --- 3.2.e bucketization sanity check: log10(rank) → 10 quantile buckets,
    # verify each bucket has a reasonable number of items (otherwise the
    # adversarial discriminator in 3.2.8 will overfit).
    if rank.size:
        buckets = pd.qcut(np.log10(rank + 1), q=10, duplicates="drop")
        per_bucket = buckets.value_counts().sort_index()
        per_bucket.rename("n").to_csv(OUT / "3_2_z_bucket_counts.csv")
        summary["z_min_bucket"] = int(per_bucket.min())
        summary["z_max_bucket"] = int(per_bucket.max())
        print(f"  Z buckets (10-quantile on log salesRank): min={per_bucket.min()}  "
              f"max={per_bucket.max()} items/bucket")


# ---------------------------------------------------------------------------
# Module 3.3 — Economics-Informed Utility EDA
# ---------------------------------------------------------------------------
def eda_module_3_3(core: pd.DataFrame, meta: pd.DataFrame, summary: dict) -> None:
    print("\n[3.3] Economics-Informed Utility")
    print("-" * 60)
    core_items = pd.DataFrame({"item": core["item"].unique()})
    join = core_items.merge(meta, on="item", how="left")

    # --- 3.3.a price distribution: raw + log (justifying log(price) in C)
    price = join["price"].dropna().astype(float).values
    if price.size:
        log_p = np.log10(price + 1e-3)
        summary["price_p50"] = float(np.median(price))
        summary["price_p99"] = float(np.percentile(price, 99))
        summary["price_max"] = float(price.max())
        print(f"  price p50=${summary['price_p50']:.2f}  p99=${summary['price_p99']:.2f}  "
              f"max=${summary['price_max']:.2f}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(price, bins=60, color="#2e8b57")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("price (USD)")
        axes[0].set_ylabel("# items (log)")
        axes[0].set_title("raw price — heavy long tail")
        axes[1].hist(log_p, bins=40, color="#2e8b57")
        axes[1].set_xlabel("log10(price)")
        axes[1].set_ylabel("# items")
        axes[1].set_title("log price — approx. log-normal → α_1·log(price) OK")
        fig.suptitle("3.3 — price distribution")
        fig.tight_layout()
        fig.savefig(OUT / "3_3_price.png")
        plt.close(fig)

    # --- 3.3.b leaf-category distribution: feasible one-hot lookup table size?
    cat_counts = join["leaf_cat"].dropna().value_counts()
    summary["n_leaf_cats"] = int(cat_counts.size)
    summary["leaf_cat_p50_size"] = int(cat_counts.median())
    summary["leaf_cat_top1_share"] = float(cat_counts.iloc[0] / cat_counts.sum()) if cat_counts.size else 0.0
    print(f"  # leaf categories={summary['n_leaf_cats']}  "
          f"top-1 share={summary['leaf_cat_top1_share']:.3f}  "
          f"median items/cat={summary['leaf_cat_p50_size']}")

    top_cats = cat_counts.head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_cats.index[::-1], top_cats.values[::-1], color="#2e8b57")
    ax.set_xlabel("# items")
    ax.set_title("3.3 — top-20 leaf categories (φ lookup table)")
    fig.tight_layout()
    fig.savefig(OUT / "3_3_categories.png")
    plt.close(fig)

    # --- 3.3.c brand distribution: feasibility of ψ(brand) lookup table
    brand_counts = join["brand"].dropna().value_counts()
    summary["n_brands"] = int(brand_counts.size)
    summary["brand_long_tail_share"] = float(
        (brand_counts <= 5).sum() / brand_counts.size
    ) if brand_counts.size else 0.0
    print(f"  # brands={summary['n_brands']}  "
          f"share-of-brands-with-≤5-items={summary['brand_long_tail_share']:.3f}")

    # --- 3.3.d price × category interaction validity (3.3.3 α_5 term)
    # If the same nominal price means different budget pressure in different
    # categories, then price-by-category boxplots should show clearly
    # different distributions — otherwise α_5 is just noise.
    pc = join.dropna(subset=["price", "leaf_cat"]).copy()
    pc["log_price"] = np.log10(pc["price"].astype(float) + 1e-3)
    keep = pc["leaf_cat"].value_counts().head(12).index
    pc_keep = pc[pc["leaf_cat"].isin(keep)]
    if not pc_keep.empty:
        groups = [pc_keep[pc_keep["leaf_cat"] == c]["log_price"].values for c in keep]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(groups, labels=[c[:20] for c in keep], showfliers=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("log10(price)")
        ax.set_title("3.3 — price | leaf-category (justifies α_5 interaction term)")
        fig.tight_layout()
        fig.savefig(OUT / "3_3_price_by_cat.png")
        plt.close(fig)

        # F-statistic-style spread: between-group variance / within-group variance
        means = np.array([g.mean() for g in groups])
        sizes = np.array([len(g) for g in groups])
        overall = np.average(means, weights=sizes)
        between = np.sum(sizes * (means - overall) ** 2) / (len(groups) - 1)
        within = np.sum([
            np.sum((g - g.mean()) ** 2) for g in groups
        ]) / (sum(sizes) - len(groups))
        eta = between / (within + 1e-12)
        summary["price_x_cat_F_proxy"] = float(eta)
        print(f"  F-proxy (between/within) on log(price) | category = {eta:.2f}  "
              f"(>>1 ⇒ interaction term has signal)")

    # --- 3.3.e price ↔ salesRank correlation (are α_1 and α_4 redundant?)
    pr = join.dropna(subset=["price", "rank_beauty"])
    if len(pr) > 100:
        x = np.log10(pr["price"].astype(float).values + 1e-3)
        y = np.log10(pr["rank_beauty"].astype(float).values + 1)
        rho = float(np.corrcoef(x, y)[0, 1])
        summary["corr_logprice_lograng"] = rho
        print(f"  corr(log(price), log(salesRank)) = {rho:+.3f}  "
              f"(near 0 ⇒ α_1 and α_4 capture distinct signal)")

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=4, alpha=0.15, color="#2e8b57")
        ax.set_xlabel("log10(price)")
        ax.set_ylabel("log10(salesRank)")
        ax.set_title(f"3.3 — price ⟂ salesRank  (ρ={rho:+.2f})")
        fig.tight_layout()
        fig.savefig(OUT / "3_3_price_vs_rank.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-cutting checks
# ---------------------------------------------------------------------------
def eda_cross_checks(core: pd.DataFrame, summary: dict) -> None:
    print("\n[X] Cross-cutting")
    print("-" * 60)

    # --- temporal coverage and train/val/test temporal split feasibility
    ts = pd.to_datetime(core["ts"], unit="s")
    span = (ts.max() - ts.min()).days
    summary["time_min"] = str(ts.min().date())
    summary["time_max"] = str(ts.max().date())
    summary["time_span_days"] = int(span)
    print(f"  time span: {summary['time_min']} → {summary['time_max']}  "
          f"({span} days)")

    # Yearly volume — for leave-last-N temporal split, we want healthy mass
    # in the final year (otherwise the test set will be vanishingly small).
    by_year = ts.dt.year.value_counts().sort_index()
    by_year.rename("n").to_csv(OUT / "X_interactions_by_year.csv")

    fig, ax = plt.subplots()
    ax.bar(by_year.index.astype(str), by_year.values, color="#555")
    ax.set_ylabel("# interactions")
    ax.set_xlabel("year")
    ax.set_title("X — interaction volume per year (split feasibility)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "X_volume_by_year.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading Beauty_5.json …")
    core = load_core_as_df()
    print(f"  → {len(core):,} interactions")

    print("Loading meta_Beauty.json …")
    meta = load_meta_as_df()
    print(f"  → {len(meta):,} item metadata records")

    summary: dict = {}
    eda_module_3_1(core, summary)
    eda_module_3_2(core, meta, summary)
    eda_module_3_3(core, meta, summary)
    eda_cross_checks(core, summary)

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    print("\nAll outputs written to:", OUT)
    print("Open summary.json for a machine-readable digest.")


if __name__ == "__main__":
    main()
