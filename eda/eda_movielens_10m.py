"""
EDA for MovieLens 10M — aligned with CEINN modules 3.1, 3.2, 3.3.

Files expected (relative to project root `ceinn/`):
    data/raw/ml-10M100K/movies.dat   movie_id::title::genres
    data/raw/ml-10M100K/ratings.dat  user_id::movie_id::rating::timestamp
    data/raw/ml-10M100K/tags.dat     user_id::movie_id::tag::timestamp

Outputs are written to `eda/outputs/movielens_10m/`.

What this script answers (one block per CEINN section it serves):

  [3.1] Sequential Backbone
    - user sequence length distribution (after 5-core)
    - rating distribution — half-stars: does the 0.5..5.0 grid actually
      get used, or is it bimodal at integers?
    - Δt distribution on log scale (justifying log bucketization in E_t)

  [3.2] Causal Deconfounding (here Z is *dynamic* unlike Amazon)
    - popularity long-tail at the full-corpus level
    - rolling-window popularity dynamics: does the *rank* of an item
      actually move over time? If not, the dynamic Z(t) is no better
      than a static Z, and the avoid-future-leakage argument in 3.2.4
      is moot.
    - sample-trajectory plot for top-K and middle-K films

  [3.3] Economics-Informed Utility (implicit, no price)
    - genre cardinality per movie & per user (lookup-table size)
    - the satiation hypothesis: does the Jaccard between a user's *next*
      film and their recent window of films actually correlate with
      time-to-consumption? This is the empirical pre-test that says
      whether GenreRed in 3.3.4 will carry signal.
    - tag sparsity audit: are tags really too sparse/noisy to use?
      (= justification for tags.dat being excluded from C(i,t))

Usage:
    python -m eda.eda_movielens_10m          # from project root `ceinn/`
"""

from __future__ import annotations

import json
import math
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
RAW = ROOT / "data" / "raw" / "ml-10M100K"
OUT = HERE / "outputs" / "movielens_10m"
OUT.mkdir(parents=True, exist_ok=True)

MOVIES_PATH = RAW / "movies.dat"
RATINGS_PATH = RAW / "ratings.dat"
TAGS_PATH = RAW / "tags.dat"

# 5-core threshold — even though MovieLens 10M is *not* pre-5-cored, the
# CEINN backbone needs decently-long sequences. We apply it here so the
# 3.1 stats reflect what the model will actually see. Configurable.
KCORE = 5

# Movie titles can contain Latin-1 accented characters; the dataset is
# distributed in Latin-1 (per README.html).
ENC = "latin-1"

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
def load_movies() -> pd.DataFrame:
    rows = []
    with MOVIES_PATH.open("r", encoding=ENC) as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            mid, title, genres = parts
            rows.append({
                "item": int(mid),
                "title": title,
                "genres": tuple(g for g in genres.split("|") if g and g != "(no genres listed)"),
            })
    return pd.DataFrame(rows)


def load_ratings() -> pd.DataFrame:
    # ratings.dat is large (~10M lines). Use chunked read for memory safety.
    chunks = pd.read_csv(
        RATINGS_PATH,
        sep="::",
        header=None,
        names=["user", "item", "rating", "ts"],
        engine="python",
        encoding=ENC,
        chunksize=2_000_000,
    )
    df = pd.concat(chunks, ignore_index=True)
    df["user"] = df["user"].astype(np.int32)
    df["item"] = df["item"].astype(np.int32)
    df["rating"] = df["rating"].astype(np.float32)
    df["ts"] = df["ts"].astype(np.int64)
    return df


def load_tags() -> pd.DataFrame | None:
    if not TAGS_PATH.exists():
        return None
    try:
        df = pd.read_csv(
            TAGS_PATH,
            sep="::",
            header=None,
            names=["user", "item", "tag", "ts"],
            engine="python",
            encoding=ENC,
        )
        return df
    except Exception as exc:
        print(f"  tags.dat unreadable: {exc}")
        return None


def kcore_filter(ratings: pd.DataFrame, k: int = KCORE) -> pd.DataFrame:
    """Iterated k-core: drop users/items with <k events until stable."""
    df = ratings
    for _ in range(20):
        before = len(df)
        u_keep = df.groupby("user").size()
        u_keep = u_keep[u_keep >= k].index
        df = df[df["user"].isin(u_keep)]
        i_keep = df.groupby("item").size()
        i_keep = i_keep[i_keep >= k].index
        df = df[df["item"].isin(i_keep)]
        if len(df) == before:
            break
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Module 3.1 — Sequential Backbone EDA
# ---------------------------------------------------------------------------
def eda_module_3_1(ratings: pd.DataFrame, summary: dict) -> None:
    print("\n[3.1] Sequential Backbone")
    print("-" * 60)

    # --- 3.1.a sequence length per user
    seq_len = ratings.groupby("user").size()
    summary["seq_len_p50"] = int(seq_len.median())
    summary["seq_len_p90"] = int(seq_len.quantile(0.90))
    summary["seq_len_p99"] = int(seq_len.quantile(0.99))
    summary["seq_len_max"] = int(seq_len.max())
    summary["n_users"] = int(seq_len.size)
    summary["n_items"] = int(ratings["item"].nunique())
    summary["n_interactions"] = int(len(ratings))
    summary["density"] = float(len(ratings) / (summary["n_users"] * summary["n_items"]))
    print(f"  users={summary['n_users']}  items={summary['n_items']}  "
          f"interactions={summary['n_interactions']}  density={summary['density']:.2e}")
    print(f"  seq-length p50={summary['seq_len_p50']}  p90={summary['seq_len_p90']}  "
          f"p99={summary['seq_len_p99']}  max={summary['seq_len_max']}")

    fig, ax = plt.subplots()
    bins = safe_log_bins(seq_len.values, n_bins=50, min_val=KCORE)
    ax.hist(seq_len.values, bins=bins, color="#3b6db3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length per user (log)")
    ax.set_ylabel("# users (log)")
    ax.set_title("3.1 — user sequence length distribution")
    fig.tight_layout()
    fig.savefig(OUT / "3_1_seq_length.png")
    plt.close(fig)

    # --- 3.1.b rating distribution: MovieLens 10M uses half-stars 0.5..5.0.
    # If half-stars are barely used, E_r needs only ~5 bins, not ~10.
    r_counts = ratings["rating"].value_counts().sort_index()
    half_share = float(((ratings["rating"] * 2) % 2 == 1).mean())
    summary["rating_share_5"] = float((ratings["rating"] == 5.0).mean())
    summary["rating_half_star_share"] = half_share
    summary["rating_entropy"] = float(
    -np.sum((p := r_counts / r_counts.sum()) * np.log2(p + 1e-12))
    )
    # summary["rating_entropy"] = float(
    #     -np.sum((p := r_counts / r_counts.sum()) * np.log(p + 1e-12))
    # )
    print(f"  half-star usage={half_share:.3f}  share@5={summary['rating_share_5']:.3f}  "
          f"H(rating)={summary['rating_entropy']:.3f}")

    fig, ax = plt.subplots()
    ax.bar(r_counts.index.astype(str), r_counts.values, color="#3b6db3")
    ax.set_xlabel("rating")
    ax.set_ylabel("# interactions")
    ax.set_title("3.1 — rating distribution (E_r granularity)")
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "3_1_rating_dist.png")
    plt.close(fig)

    # --- 3.1.c Δt distribution: justifying log bucketization in E_t.
    # Unlike Amazon (day-granular timestamps), MovieLens has true second
    # resolution → we expect a very wide log10 span and meaningful Δt<1day.
    ratings_s = ratings.sort_values(["user", "ts"])
    dt = ratings_s.groupby("user")["ts"].diff().dropna().values
    dt = dt[dt > 0]
    dt_sec = dt
    summary["dt_share_under_1min"] = float((dt_sec < 60).mean())
    summary["dt_share_under_1hr"] = float((dt_sec < 3600).mean())
    summary["dt_log10_range"] = float(np.log10(dt_sec.max()) - np.log10(dt_sec.min()))
    print(f"  Δt <1min share={summary['dt_share_under_1min']:.3f}  "
          f"<1hr share={summary['dt_share_under_1hr']:.3f}  "
          f"log10 span={summary['dt_log10_range']:.2f} decades")

    fig, ax = plt.subplots()
    # subsample so the hist computation finishes fast
    sample = np.random.default_rng(0).choice(dt_sec, size=min(2_000_000, dt_sec.size), replace=False)
    bins = safe_log_bins(sample, n_bins=60, min_val=1.0)
    ax.hist(sample, bins=bins, color="#3b6db3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δt between consecutive ratings (seconds, log)")
    ax.set_ylabel("count (log)")
    ax.set_title("3.1 — Δt distribution: log bucketization is essential")
    fig.tight_layout()
    fig.savefig(OUT / "3_1_delta_t.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Module 3.2 — Causal Deconfounding EDA
# Z(t) here is *dynamic* (rolling interaction frequency). The non-trivial
# question is whether item popularity actually changes over time — if not,
# the static and dynamic Z collapse to the same thing.
# ---------------------------------------------------------------------------
def eda_module_3_2(ratings: pd.DataFrame, summary: dict) -> None:
    print("\n[3.2] Causal Deconfounding")
    print("-" * 60)

    # --- 3.2.a global popularity long tail and Gini
    item_freq = ratings.groupby("item").size().sort_values(ascending=False)
    freq_arr = item_freq.values.astype(float)
    cum_share = np.cumsum(freq_arr) / freq_arr.sum()
    head_at_20 = float(cum_share[max(int(len(freq_arr) * 0.20) - 1, 0)])
    sorted_freq = np.sort(freq_arr)
    n = len(sorted_freq)
    cum = np.cumsum(sorted_freq)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * sorted_freq) / cum[-1] - (n + 1)) / n)
    summary["pop_head20_share"] = head_at_20
    summary["pop_gini"] = gini
    print(f"  top-20%-items capture {head_at_20*100:.1f}% of interactions  Gini={gini:.3f}")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, n + 1) / n, cum_share, color="#a44a3f")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("item rank (fraction)")
    ax.set_ylabel("cumulative interaction share")
    ax.set_title(f"3.2 — popularity Lorenz curve (Gini={gini:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "3_2_popularity_lorenz.png")
    plt.close(fig)

    # --- 3.2.b empirical 1/p_i to justify IPS clipping
    p_emp = freq_arr / freq_arr.max()
    inv_p = 1.0 / np.maximum(p_emp, 1e-6)
    summary["inv_p_p99"] = float(np.percentile(inv_p, 99))
    summary["inv_p_p999"] = float(np.percentile(inv_p, 99.9))
    summary["inv_p_max"] = float(inv_p.max())
    print(f"  empirical 1/p_i  p99={summary['inv_p_p99']:.1f}  "
          f"p99.9={summary['inv_p_p999']:.1f}  max={summary['inv_p_max']:.1f}")

    # --- 3.2.c dynamic popularity — the heart of the 3.2.4 design choice.
    # Bucket the timeline into N windows and compute item-frequency rank
    # *within each window*. If the rank of a top-K item is roughly stable,
    # the dynamic Z(t) collapses to a static one and offers little extra
    # signal. If ranks churn substantially, dynamic Z(t) is justified.
    N_WIN = 20
    edges = np.linspace(ratings["ts"].min(), ratings["ts"].max() + 1, N_WIN + 1)
    ratings_w = ratings.assign(
        win=np.digitize(ratings["ts"].values, edges) - 1
    )
    win_item = ratings_w.groupby(["win", "item"]).size().reset_index(name="n")
    # rank within window (1 = most popular)
    win_item["rank"] = win_item.groupby("win")["n"].rank(method="dense", ascending=False)

    # Top-100 by global popularity — track their per-window rank.
    top100 = item_freq.head(100).index
    traj = win_item[win_item["item"].isin(top100)].pivot(
        index="win", columns="item", values="rank"
    )
    # Fraction of windows where item's rank moves by more than ±50 positions
    # from its median rank: if high, popularity is non-stationary.
    median_rank = traj.median(axis=0)
    shift = (traj.subtract(median_rank, axis=1).abs() > 50).mean(axis=0).mean()
    summary["rank_volatility_top100"] = float(shift)
    print(f"  top-100 items: avg fraction of windows with |rank − median|>50 = {shift:.3f}  "
          f"(higher ⇒ dynamic Z(t) carries more signal than static Z)")

    fig, ax = plt.subplots(figsize=(8, 5))
    # plot a handful of trajectories with alpha to suggest the cloud
    sample_items = np.random.default_rng(1).choice(traj.columns, size=min(30, traj.shape[1]), replace=False)
    for c in sample_items:
        ax.plot(traj.index, traj[c].values, color="#a44a3f", alpha=0.4, linewidth=1)
    ax.invert_yaxis()  # rank 1 should be at top
    ax.set_xlabel(f"time window (1..{N_WIN})")
    ax.set_ylabel("rank within window (1 = most popular)")
    ax.set_title("3.2 — popularity dynamics of top-100 items")
    fig.tight_layout()
    fig.savefig(OUT / "3_2_popularity_dynamics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Module 3.3 — Economics-Informed Utility (implicit cost) EDA
# ---------------------------------------------------------------------------
def eda_module_3_3(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None,
    summary: dict,
) -> None:
    print("\n[3.3] Economics-Informed Utility")
    print("-" * 60)

    # --- 3.3.a genre cardinality per movie + per-user genre breadth
    g_per_movie = movies["genres"].apply(len)
    all_genres = sorted({g for gs in movies["genres"] for g in gs})
    summary["n_genres"] = len(all_genres)
    summary["genres_per_movie_p50"] = int(g_per_movie.median())
    summary["genres_per_movie_max"] = int(g_per_movie.max())
    print(f"  # distinct genres={summary['n_genres']}  "
          f"genres/movie p50={summary['genres_per_movie_p50']}  "
          f"max={summary['genres_per_movie_max']}")

    fig, ax = plt.subplots()
    ax.hist(g_per_movie.values, bins=range(0, g_per_movie.max() + 2),
            color="#2e8b57", align="left")
    ax.set_xlabel("# genres per movie")
    ax.set_ylabel("# movies")
    ax.set_title("3.3 — genre cardinality (φ lookup table)")
    fig.tight_layout()
    fig.savefig(OUT / "3_3_genres_per_movie.png")
    plt.close(fig)

    # --- 3.3.b top genres — used directly as the "category" axis in C(i,t)
    genre_counter = Counter()
    for gs in movies["genres"]:
        genre_counter.update(gs)
    g_series = pd.Series(genre_counter).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g_series.index, g_series.values, color="#2e8b57")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("# movies")
    ax.set_title("3.3 — genre distribution")
    fig.tight_layout()
    fig.savefig(OUT / "3_3_genre_distribution.png")
    plt.close(fig)

    # --- 3.3.c the satiation pre-test (the crucial validation for 3.3.4).
    # The claim: when a user has heavily consumed genre G in their recent
    # window, the next *same-genre* item meets higher behavioral cost,
    # i.e. it takes longer to be consumed (or has lower rating).
    # Operationalization:
    #   for each user, look at consecutive pairs (i_t, i_{t+1}).
    #   compute J(i_{t+1}, recent_K-window-genre-bag) ∈ [0,1].
    #   bin J into quartiles and look at:
    #     - the *next* Δt (do high-Jaccard transitions get pushed further away?)
    #     - the *next* rating (do users rate redundant content lower?)
    # If quartile means trend monotonically with J, satiation is real and
    # GenreRed earns its place in C(i,t).
    print("  running Jaccard-vs-Δt satiation pre-test …")
    K_HIST = 5  # recent K-item bag
    genre_map = dict(zip(movies["item"].values, movies["genres"].values))

    rs = ratings.sort_values(["user", "ts"])
    pairs = []  # (jaccard, next_dt_log, next_rating)

    # Iterate per user; use itertuples for speed.
    for user, sub in rs.groupby("user"):
        items = sub["item"].values
        ts = sub["ts"].values
        ratings_arr = sub["rating"].values
        if len(items) < K_HIST + 2:
            continue
        # rolling genre multiset
        bag = Counter()
        for k in range(K_HIST):
            bag.update(genre_map.get(items[k], ()))
        for k in range(K_HIST, len(items) - 1):
            next_genres = set(genre_map.get(items[k], ()))
            bag_set = set(bag.keys())
            if not next_genres and not bag_set:
                jacc = 0.0
            else:
                inter = len(next_genres & bag_set)
                union = len(next_genres | bag_set)
                jacc = inter / union if union else 0.0
            dt = ts[k] - ts[k - 1]
            if dt <= 0:
                # ties from the day-granular timestamps in some users
                bag.subtract(genre_map.get(items[k - K_HIST], ()))
                bag.update(genre_map.get(items[k], ()))
                continue
            pairs.append((jacc, math.log10(dt), ratings_arr[k]))
            bag.subtract(genre_map.get(items[k - K_HIST], ()))
            # remove zero-counts so set(bag.keys()) stays clean
            bag += Counter()
            bag.update(genre_map.get(items[k], ()))

        if len(pairs) > 1_000_000:
            break  # enough sample for the pre-test; this is EDA, not training

    pdf = pd.DataFrame(pairs, columns=["jacc", "log_dt", "next_rating"])
    if len(pdf) > 1000:
        q = pd.qcut(pdf["jacc"], q=4, duplicates="drop")
        agg = pdf.groupby(q, observed=True).agg(
            n=("jacc", "size"),
            jacc_mean=("jacc", "mean"),
            log_dt_mean=("log_dt", "mean"),
            rating_mean=("next_rating", "mean"),
        )
        agg.to_csv(OUT / "3_3_satiation_pretest.csv")
        # Trend test: if log_dt_mean is monotonic non-decreasing in jacc_mean,
        # satiation hypothesis is supported.
        monotonic_dt = bool(np.all(np.diff(agg["log_dt_mean"].values) >= 0))
        monotonic_rating = bool(np.all(np.diff(agg["rating_mean"].values) <= 0))
        summary["satiation_monotonic_dt"] = monotonic_dt
        summary["satiation_monotonic_rating"] = monotonic_rating
        summary["satiation_log_dt_spread"] = float(
            agg["log_dt_mean"].max() - agg["log_dt_mean"].min()
        )
        print(f"  satiation pre-test: log_dt monotonic in Jaccard? {monotonic_dt}")
        print(f"                      rating monotonically down in Jaccard? {monotonic_rating}")
        print(f"                      log_dt spread across quartiles = "
              f"{summary['satiation_log_dt_spread']:.3f} decades")
        print(agg.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(agg["jacc_mean"], agg["log_dt_mean"], marker="o", color="#2e8b57")
        axes[0].set_xlabel("Jaccard(next_genres, recent_bag)")
        axes[0].set_ylabel("mean log10(Δt to next interaction)")
        axes[0].set_title("does same-genre content get pushed further?")
        axes[1].plot(agg["jacc_mean"], agg["rating_mean"], marker="o", color="#2e8b57")
        axes[1].set_xlabel("Jaccard(next_genres, recent_bag)")
        axes[1].set_ylabel("mean next-item rating")
        axes[1].set_title("does same-genre content get rated lower?")
        fig.suptitle("3.3 — GenreRed satiation pre-test (validates 3.3.4)")
        fig.tight_layout()
        fig.savefig(OUT / "3_3_satiation_pretest.png")
        plt.close(fig)

    # --- 3.3.d tag sparsity audit — the 3.3.4 paper claim: "tags are too
    # sparse and noisy → exclude". This block puts numbers on that claim.
    if tags is not None:
        n_tag_events = len(tags)
        n_tagged_movies = tags["item"].nunique()
        n_tagging_users = tags["user"].nunique()
        rating_items = set(ratings["item"].unique())
        tag_movie_coverage = float(
            len(set(tags["item"].unique()) & rating_items) / len(rating_items)
        )
        rating_users = set(ratings["user"].unique())
        tag_user_coverage = float(
            len(set(tags["user"].unique()) & rating_users) / len(rating_users)
        )
        # vocabulary growth — power-law-y tag distributions usually have
        # >50% of unique tags appearing only once.
        tag_counts = tags["tag"].astype(str).str.lower().value_counts()
        tag_singletons = float((tag_counts == 1).mean())

        summary.update({
            "tag_events": int(n_tag_events),
            "tag_movie_coverage": tag_movie_coverage,
            "tag_user_coverage": tag_user_coverage,
            "tag_vocab_size": int(tag_counts.size),
            "tag_singleton_share": tag_singletons,
        })
        print(f"  tag events={n_tag_events:,}  movies tagged={n_tagged_movies:,}  "
              f"users tagging={n_tagging_users:,}")
        print(f"  rating-side coverage: {tag_movie_coverage:.3f} of movies, "
              f"{tag_user_coverage:.3f} of users")
        print(f"  tag vocab={tag_counts.size:,}  singletons={tag_singletons:.3f}  "
              f"(justifies excluding tags from C)")


# ---------------------------------------------------------------------------
# Cross-cutting checks
# ---------------------------------------------------------------------------
def eda_cross_checks(ratings: pd.DataFrame, summary: dict) -> None:
    print("\n[X] Cross-cutting")
    print("-" * 60)

    ts = pd.to_datetime(ratings["ts"], unit="s")
    span = (ts.max() - ts.min()).days
    summary["time_min"] = str(ts.min().date())
    summary["time_max"] = str(ts.max().date())
    summary["time_span_days"] = int(span)
    print(f"  time span: {summary['time_min']} → {summary['time_max']}  "
          f"({span} days, ≈{span/365.25:.1f} years)")

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
    print("Loading movies.dat …")
    movies = load_movies()
    print(f"  → {len(movies):,} movies")

    print("Loading ratings.dat …")
    ratings = load_ratings()
    print(f"  → {len(ratings):,} ratings before k-core")
    ratings = kcore_filter(ratings, k=KCORE)
    print(f"  → {len(ratings):,} ratings after {KCORE}-core")

    print("Loading tags.dat …")
    tags = load_tags()
    if tags is not None:
        print(f"  → {len(tags):,} tag events")

    summary: dict = {"kcore": KCORE}
    eda_module_3_1(ratings, summary)
    eda_module_3_2(ratings, summary)
    eda_module_3_3(ratings, movies, tags, summary)
    eda_cross_checks(ratings, summary)

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    print("\nAll outputs written to:", OUT)
    print("Open summary.json for a machine-readable digest.")


if __name__ == "__main__":
    main()
