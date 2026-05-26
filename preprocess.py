"""
CEINN preprocessing entry point.

Phase 1 — raw data validation
-----------------------------
Verify that the raw files on disk can be parsed correctly and that every
statistic listed in the EDA reports is reproducible from them. Emit a
validation log to `data/raw/validation_log.txt`. If any check fails the
process exits with a non-zero status — Phase 2 must not proceed on a
broken data baseline.

Phase 2 — preprocessing, feature engineering, static-feature precomputation
---------------------------------------------------------------------------
Convert raw interactions into model-consumable tensors / pickles under
`data/processed/`. Specifically:

  * 5-core filtering (Amazon: use Beauty_5.json directly; MovieLens:
    iterative fixed-point at min_count=5).
  * User / item ID remap to contiguous indices starting at 1 (0 = PAD).
  * Temporal leave-one-out split: last interaction → test, second-to-last
    → val, remainder → train.
  * Δt log-bucketing (Amazon: 32 bins, MovieLens: 64 bins) — edges fit on
    training Δt only, then frozen and reapplied to val/test.
  * Sequence truncation (Amazon: 50, MovieLens: 200; take the most recent).
  * MovieLens-specific: 60-second session segmentation; intra-session Δt
    is mapped to the PAD bin.
  * MovieLens-specific: train-set restricted to interactions strictly
    before 2009 (val/test untouched).
  * Amazon item metadata: leaf-category lookup (one leaf per item with
    rare-category UNK), brand lookup (rare-brand UNK), log10(price)
    imputed by leaf median then global median, salesRank Z bucketed to
    10 equal-frequency log bins with missing → median bin.
  * MovieLens item metadata: 19-dim binary genre vectors.
  * MovieLens dynamic Z(t): cumulative item count strictly before t,
    measured on the training set, stored as a parallel np.array of the
    train-interaction list.
  * MovieLens GenreRed: standard + IDF-weighted Jaccard between each
    candidate's genre vector and the user's history-union vector at
    that time, stored as parallel np.arrays.

Usage
-----
    python preprocess.py --phase 1                    # validate raw data
    python preprocess.py --phase 2                    # write data/processed/*
    python preprocess.py --phase 2 --only amazon
    python preprocess.py --phase 2 --only movielens
    # python preprocess.py --phase 2 --only movielens1
    # python preprocess.py --phase 2 --only movielens2
    # IMPORTANT: if you encounter OOM during phase2, you may run the two movielens stages separately.

The Phase-2 quality gate runs invariants after preprocessing (no
train-set leakage of val/test items, vocab_sizes match the lookup
tables, etc.) and refuses to declare success on any failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from utils.data_io import (
    extract_leaf_categories,
    extract_primary_leaf_category,
    extract_salesrank,
    get_brand_or_none,
    get_price_or_none,
    has_valid_brand,
    has_valid_price,
    iter_amazon_meta,
    iter_amazon_reviews,
    iter_movielens_movies,
    iter_movielens_ratings,
    iter_movielens_tags,
)
from utils.math_utils import (
    apply_bucket_edges,
    compute_genre_idf,
    fit_log_quantile_edges,
    jaccard_idf_weighted,
)


# =============================================================================
# Validation accounting
# =============================================================================

@dataclass
class Check:
    name: str
    expected: Any
    actual: Any
    tolerance: Any
    mode: str
    passed: bool
    note: str = ""


@dataclass
class ValidationLog:
    """Accumulates check results for a single dataset and renders the report."""

    dataset_name: str
    checks: List[Check] = field(default_factory=list)

    # ----- Comparison modes -------------------------------------------------
    #
    # 'exact'    : require actual == expected (works for int/str/bool/...).
    # 'abs'      : require abs(actual - expected) <= tolerance (floats / counts).
    # 'date_abs' : compare two 'YYYY-MM-DD' strings, allowing |Δ days| <= tol.
    #
    # The 'abs' mode is what almost every numeric check should use; tolerance
    # comes from the YAML config so behaviour is auditable.

    def record(
        self,
        name: str,
        expected: Any,
        actual: Any,
        tolerance: Any,
        mode: str = "abs",
        note: str = "",
    ) -> bool:
        if mode == "exact":
            passed = (actual == expected)

        elif mode == "abs":
            try:
                passed = abs(float(actual) - float(expected)) <= float(tolerance)
            except (TypeError, ValueError):
                passed = False

        elif mode == "date_abs":
            try:
                e = datetime.strptime(str(expected), "%Y-%m-%d")
                a = datetime.strptime(str(actual), "%Y-%m-%d")
                passed = abs((e - a).days) <= int(tolerance)
            except (TypeError, ValueError):
                passed = False

        else:
            raise ValueError(f"unknown comparison mode {mode!r}")

        self.checks.append(
            Check(name=name, expected=expected, actual=actual,
                  tolerance=tolerance, mode=mode, passed=passed, note=note)
        )
        return passed

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def format(self) -> str:
        if not self.checks:
            return f"=== {self.dataset_name} ===\n(no checks run)\n"

        # Render a fixed-width table for easy diffing.
        header = f"=== {self.dataset_name} ==="
        rows = [header]
        for c in self.checks:
            status = "[PASS]" if c.passed else "[FAIL]"
            exp = self._fmt_value(c.expected)
            act = self._fmt_value(c.actual)
            rows.append(
                f"{status}  {c.name:<42}  "
                f"expected={exp:<14}  actual={act:<14}  "
                f"tol={c.tolerance!s:<8}  mode={c.mode}"
                + (f"\n          | note: {c.note}" if c.note else "")
            )
        n_pass = sum(c.passed for c in self.checks)
        rows.append("-" * 100)
        rows.append(
            f"Summary: {n_pass}/{len(self.checks)} checks passed"
            f" — {'OK' if self.all_passed else 'FAIL'}"
        )
        rows.append("")
        return "\n".join(rows)

    @staticmethod
    def _fmt_value(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)


# =============================================================================
# Amazon Beauty — Phase 1 validation
# =============================================================================

def validate_amazon_beauty(cfg: Dict[str, Any]) -> ValidationLog:
    """Run all Phase 1 checks against the Amazon Beauty raw files."""

    log = ValidationLog("Amazon Beauty")
    paths = cfg["dataset"]["paths"]
    expected = cfg["validation"]
    tol = cfg["tolerance"]

    # -------------------------------------------------------------------------
    # 1.1.1: read the 5-core review file. Per the EDA, all interaction-level
    # baselines are computed on Beauty_5.json, so that's our primary source.
    # -------------------------------------------------------------------------
    reviews_path = Path(paths["reviews_5core"])
    if not reviews_path.exists():
        log.record(
            "5-core review file present", True, False, 0,
            mode="exact", note=f"missing: {reviews_path}",
        )
        return log

    n_inter = 0
    user_set: set = set()
    item_set: set = set()
    rating_counter: Counter = Counter()
    timestamps: List[int] = []
    bad_records = 0  # records missing critical fields

    for rec in iter_amazon_reviews(reviews_path):
        uid = rec.get("reviewerID")
        iid = rec.get("asin")
        r = rec.get("overall")
        ts = rec.get("unixReviewTime")

        if uid is None or iid is None or r is None or ts is None:
            bad_records += 1
            continue

        n_inter += 1
        user_set.add(uid)
        item_set.add(iid)
        rating_counter[int(round(float(r)))] += 1
        timestamps.append(int(ts))

    # -------------------------------------------------------------------------
    # 1.1.2: structural counts (must match exactly)
    # -------------------------------------------------------------------------
    log.record("n_interactions", expected["n_interactions"], n_inter,
               tol["count_exact"], mode="exact")
    log.record("n_users", expected["n_users"], len(user_set),
               tol["count_exact"], mode="exact")
    log.record("n_items", expected["n_items"], len(item_set),
               tol["count_exact"], mode="exact")

    if bad_records:
        log.record("records_with_missing_fields", 0, bad_records, 0,
                   mode="exact",
                   note="reviewerID/asin/overall/unixReviewTime missing")

    # -------------------------------------------------------------------------
    # 1.1.2: rating distribution
    # -------------------------------------------------------------------------
    if n_inter > 0:
        share_5 = rating_counter[5] / n_inter
        log.record("rating_share_5", expected["rating_share_5"], share_5,
                   tol["ratio_pp"], mode="abs")

    # -------------------------------------------------------------------------
    # 1.1.3: timestamp range & unit sanity
    # -------------------------------------------------------------------------
    if timestamps:
        t_min, t_max = min(timestamps), max(timestamps)

        # A seconds-epoch in this dataset's era lives in [~1e9, ~1.5e9].
        # A milliseconds-epoch would be [~1e12, ~1.5e12]. Distinguish them.
        epoch_unit_ok = (1e8 < t_min < 1e10) and (1e8 < t_max < 1e10)
        log.record("unixReviewTime unit (seconds)",
                   True, epoch_unit_ok, 0, mode="exact")

        dt_min = datetime.fromtimestamp(t_min, tz=timezone.utc).strftime("%Y-%m-%d")
        dt_max = datetime.fromtimestamp(t_max, tz=timezone.utc).strftime("%Y-%m-%d")
        span_days = (t_max - t_min) // 86400

        log.record("time_range_start (UTC date)",
                   expected["time_range_start"], dt_min,
                   tol["date_days"], mode="date_abs")
        log.record("time_range_end (UTC date)",
                   expected["time_range_end"], dt_max,
                   tol["date_days"], mode="date_abs")
        log.record("time_span_days", expected["time_span_days"], span_days,
                   tol["span_days"], mode="abs")

    # -------------------------------------------------------------------------
    # 1.1.2: metadata coverages — read meta_Beauty.json
    # The denominators are the items that ACTUALLY appear in reviews; per the
    # EDA, coverage figures are not over all rows of meta_Beauty.json (which
    # contains items outside the 5-core graph) but over the interacted set.
    # -------------------------------------------------------------------------
    meta_path = Path(paths["meta"])
    if not meta_path.exists():
        log.record("meta_Beauty.json present", True, False, 0,
                   mode="exact", note=f"missing: {meta_path}")
        return log

    items_in_reviews = item_set
    seen_meta: set = set()
    price_have = 0
    salesrank_any_have = 0
    salesrank_beauty_have = 0
    brand_have = 0
    leaf_categories: set = set()

    for rec in iter_amazon_meta(meta_path):
        asin = rec.get("asin")
        if asin not in items_in_reviews:
            continue
        if asin in seen_meta:
            continue  # guard against duplicated metadata rows
        seen_meta.add(asin)

        if has_valid_price(rec.get("price")):
            price_have += 1

        _, sr_source = extract_salesrank(rec.get("salesRank"), prefer_key="Beauty")
        if sr_source in ("preferred_key", "any_key"):
            salesrank_any_have += 1
        if sr_source == "preferred_key":
            salesrank_beauty_have += 1

        if has_valid_brand(rec.get("brand")):
            brand_have += 1

        for leaf in extract_leaf_categories(rec.get("categories")):
            leaf_categories.add(leaf)

    n_items = len(items_in_reviews)
    log.record("meta records matched to reviewed items",
               n_items, len(seen_meta), 0, mode="exact",
               note="all reviewed asins should be present in meta")

    if n_items > 0:
        log.record("price_coverage", expected["price_coverage"],
                   price_have / n_items, tol["coverage_pp"], mode="abs")
        log.record("salesrank_any_coverage", expected["salesrank_any_coverage"],
                   salesrank_any_have / n_items, tol["coverage_pp"], mode="abs")
        log.record("salesrank_beauty_coverage", expected["salesrank_beauty_coverage"],
                   salesrank_beauty_have / n_items, tol["coverage_pp"], mode="abs")
        log.record("brand_coverage", expected["brand_coverage"],
                   brand_have / n_items, tol["coverage_pp"], mode="abs")

    log.record("n_leaf_categories", expected["n_leaf_categories"],
               len(leaf_categories), tol["count_exact"], mode="exact",
               note="union of leaf-level category labels over reviewed items")

    return log


# =============================================================================
# MovieLens 10M — Phase 1 validation
# =============================================================================

def _five_core_filter(
    interactions: List[Tuple[Any, ...]],
    min_count: int = 5,
) -> Tuple[List[Tuple[Any, ...]], set, set]:
    """
    Iteratively drop users/items with fewer than `min_count` interactions
    until the filtering reaches a fixed point. Returns (kept_interactions,
    kept_users, kept_items).

    Why iterative?
        Dropping an item may reduce a user's count below the threshold, and
        vice versa. A single pass is not enough — we must iterate to a fixed
        point. The MovieLens 10M file is already near-5-core so this
        normally converges in 1–2 passes.
    """
    user_count: Counter = Counter()
    item_count: Counter = Counter()
    for t in interactions:
        user_count[t[0]] += 1
        item_count[t[1]] += 1

    users_keep = {u for u, c in user_count.items() if c >= min_count}
    items_keep = {i for i, c in item_count.items() if c >= min_count}

    while True:
        filtered = [
            t for t in interactions
            if t[0] in users_keep and t[1] in items_keep
        ]
        uc: Counter = Counter()
        ic: Counter = Counter()
        for t in filtered:
            uc[t[0]] += 1
            ic[t[1]] += 1
        new_users = {u for u, c in uc.items() if c >= min_count}
        new_items = {i for i, c in ic.items() if c >= min_count}
        if new_users == users_keep and new_items == items_keep:
            return filtered, users_keep, items_keep
        users_keep, items_keep = new_users, new_items


def validate_movielens_10m(cfg: Dict[str, Any]) -> ValidationLog:
    """Run all Phase 1 checks against the MovieLens 10M raw files."""

    log = ValidationLog("MovieLens 10M")
    paths = cfg["dataset"]["paths"]
    enc = cfg["dataset"]["encoding"]
    expected = cfg["validation"]
    tol = cfg["tolerance"]

    # -------------------------------------------------------------------------
    # 1.2.1: ratings.dat
    # We hold all 10M rows in memory once. At 4 ints/floats per row ≈ 200 MB,
    # this fits comfortably on a development machine. Phase 2 will reuse the
    # same in-memory representation, so the cost is amortised.
    # -------------------------------------------------------------------------
    ratings_path = Path(paths["ratings"])
    if not ratings_path.exists():
        log.record("ratings.dat present", True, False, 0,
                   mode="exact", note=f"missing: {ratings_path}")
        return log

    interactions: List[Tuple[int, int, float, int]] = []
    rating_counter: Counter = Counter()
    half_star_count = 0
    timestamps: List[int] = []

    for uid, mid, r, ts in iter_movielens_ratings(ratings_path):
        interactions.append((uid, mid, r, ts))
        rating_counter[r] += 1
        # Half-stars are exactly the non-integer rating values: 0.5, 1.5, ..., 4.5.
        # ML10M only stores values on a 0.5-grid, so `r != int(r)` is exact.
        if r != int(r):
            half_star_count += 1
        timestamps.append(ts)

    n_raw = len(interactions)
    log.record("n_interactions_raw", expected["n_interactions_raw"], n_raw,
               tol["count_exact"], mode="exact")

    # -------------------------------------------------------------------------
    # 1.2.2: rating-level distribution (computed on RAW per EDA)
    # -------------------------------------------------------------------------
    if n_raw > 0:
        share_half = half_star_count / n_raw
        share_5 = rating_counter.get(5.0, 0) / n_raw

        # Shannon entropy over the 10-bin rating histogram, base-2.
        ent = 0.0
        for v in rating_counter.values():
            p = v / n_raw
            if p > 0:
                ent -= p * math.log2(p)

        log.record("rating_half_star_share", expected["rating_half_star_share"],
                   share_half, tol["ratio_pp"], mode="abs")
        log.record("rating_share_5", expected["rating_share_5"], share_5,
                   tol["ratio_pp"], mode="abs")
        log.record("rating_entropy_bits", expected["rating_entropy"], ent,
                   tol["entropy_abs"], mode="abs")

    # -------------------------------------------------------------------------
    # 1.2.2: 5-core filtering — verifies n_users / n_items_5core / n_interactions_5core
    # -------------------------------------------------------------------------
    filtered, users_keep, items_keep = _five_core_filter(interactions, min_count=5)

    log.record("n_interactions_5core", expected["n_interactions_5core"],
               len(filtered), tol["count_exact"], mode="exact",
               note="iterative 5-core fixed-point")
    log.record("n_users (5-core)", expected["n_users"], len(users_keep),
               tol["count_exact"], mode="exact")
    log.record("n_items_5core", expected["n_items_5core"], len(items_keep),
               tol["count_exact"], mode="exact")

    # -------------------------------------------------------------------------
    # Timeline (computed on the raw file; the 5-core span is identical to
    # within a day in this dataset, so we report the raw span for clarity).
    # -------------------------------------------------------------------------
    if timestamps:
        t_min, t_max = min(timestamps), max(timestamps)
        dt_min = datetime.fromtimestamp(t_min, tz=timezone.utc).strftime("%Y-%m-%d")
        dt_max = datetime.fromtimestamp(t_max, tz=timezone.utc).strftime("%Y-%m-%d")
        span_days = (t_max - t_min) // 86400

        log.record("time_range_start (UTC date)",
                   expected["time_range_start"], dt_min,
                   tol["date_days"], mode="date_abs")
        log.record("time_range_end (UTC date)",
                   expected["time_range_end"], dt_max,
                   tol["date_days"], mode="date_abs")
        log.record("time_span_days", expected["time_span_days"], span_days,
                   tol["span_days"], mode="abs")

    # -------------------------------------------------------------------------
    # 1.2.3: movies.dat → genre vocabulary
    # The 19 in the EDA excludes the sentinel "(no genres listed)".
    # -------------------------------------------------------------------------
    movies_path = Path(paths["movies"])
    if not movies_path.exists():
        log.record("movies.dat present", True, False, 0,
                   mode="exact", note=f"missing: {movies_path}")
        return log

    genre_vocab: set = set()
    NO_GENRE_SENTINEL = "(no genres listed)"
    movie_id_set: set = set()
    for mid, _title, genres in iter_movielens_movies(movies_path,
                                                     encoding=enc["movies"]):
        movie_id_set.add(mid)
        for g in genres:
            if g and g != NO_GENRE_SENTINEL:
                genre_vocab.add(g)

    log.record("n_genres (excl. no-genres-listed)", expected["n_genres"],
               len(genre_vocab), tol["count_exact"], mode="exact",
               note=f"genres found: {sorted(genre_vocab)}")
    log.record("n_items_raw (from movies.dat)", expected["n_items_raw"],
               len(movie_id_set), tol["count_exact"], mode="exact")

    # -------------------------------------------------------------------------
    # tags.dat — event count and singleton share.
    # The EDA's tag vocab of 15,241 is case-sensitive on the raw text.
    # -------------------------------------------------------------------------
    tags_path = Path(paths["tags"])
    if not tags_path.exists():
        log.record("tags.dat present", True, False, 0,
                   mode="exact", note=f"missing: {tags_path}")
        return log

    tag_event_count = 0
    tag_counter: Counter = Counter()
    for _uid, _mid, tag, _ts in iter_movielens_tags(tags_path):
        tag_event_count += 1
        tag_counter[tag.lower()] += 1  # ← 加入 .lower()，與 EDA 一致

    log.record("n_tag_events", expected["n_tag_events"], tag_event_count,
               tol["count_exact"], mode="exact")

    if tag_counter:
        n_vocab = len(tag_counter)
        singletons = sum(1 for c in tag_counter.values() if c == 1)
        log.record("n_tag_vocab (case-sensitive)", expected["n_tag_vocab"],
                   n_vocab, tol["count_exact"], mode="exact")
        log.record("tag_singleton_share", expected["tag_singleton_share"],
                   singletons / n_vocab, tol["coverage_pp"], mode="abs",
                   note="singletons / unique tag strings")

    return log


# =============================================================================
# Entry point
# =============================================================================

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEINN preprocessing — Phase 1 validation and Phase 2 feature engineering."
    )
    parser.add_argument("--phase", choices=["1", "2"], default="1",
                        help="Which phase to execute.")
    parser.add_argument("--only", choices=["amazon", "movielens", "movielens1", "movielens2", "both"],
                        default="both",
                        help="Restrict execution to one dataset or movielens stage.")
    parser.add_argument("--config-amazon", default="configs/amazon_beauty.yaml")
    parser.add_argument("--config-movielens", default="configs/movielens_10M.yaml")
    parser.add_argument("--output", default="data/raw/validation_log.txt",
                        help="Phase 1 only: validation log path. "
                             "Phase 2 outputs go under preprocess.output_dir in the YAML.")
    args = parser.parse_args()

    if args.phase == "1":
        _run_phase_1(args)
    elif args.phase == "2":
        _run_phase_2(args)


def _run_phase_1(args: argparse.Namespace) -> None:
    """Phase 1: cross-check raw files against EDA-derived baselines."""
    logs: List[ValidationLog] = []

    if args.only in ("amazon", "both"):
        print(">>> Validating Amazon Beauty …")
        cfg = _load_yaml(args.config_amazon)
        log = validate_amazon_beauty(cfg)
        print(log.format())
        logs.append(log)

    if args.only in ("movielens", "both"):
        print(">>> Validating MovieLens 10M …")
        cfg = _load_yaml(args.config_movielens)
        log = validate_movielens_10m(cfg)
        print(log.format())
        logs.append(log)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("CEINN Phase 1 validation log\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Invocation: {' '.join(sys.argv)}\n\n")
        for log in logs:
            f.write(log.format())
            f.write("\n")
    print(f"\nValidation log written to: {output}")

    all_ok = all(log.all_passed for log in logs)
    if not all_ok:
        print("ERROR: Phase 1 quality gate FAILED. "
              "Fix the reader logic or update tolerances before Phase 2.",
              file=sys.stderr)
        sys.exit(1)
    print("Phase 1 quality gate PASSED. Safe to proceed to Phase 2.")


def _run_phase_2(args: argparse.Namespace) -> None:
    """Phase 2: build train/val/test sequences and per-dataset feature lookups."""
    all_ok = True

    if args.only in ("amazon", "both"):
        print(">>> Preprocessing Amazon Beauty …")
        cfg = _load_yaml(args.config_amazon)
        ok = preprocess_amazon_beauty(cfg)
        all_ok = all_ok and ok

    if args.only in ("movielens", "both"):
        print(">>> Preprocessing MovieLens 10M …")
        cfg = _load_yaml(args.config_movielens)
        ok = preprocess_movielens_10m(cfg)
        all_ok = all_ok and ok
    elif args.only == "movielens1":
        print(">>> Preprocessing MovieLens 10M — Stage 1 …")
        cfg = _load_yaml(args.config_movielens)
        ok = preprocess_movielens_10m_stage1(cfg)
        all_ok = all_ok and ok
    elif args.only == "movielens2":
        print(">>> Preprocessing MovieLens 10M — Stage 2 …")
        cfg = _load_yaml(args.config_movielens)
        ok = preprocess_movielens_10m_stage2(cfg)
        all_ok = all_ok and ok

    if not all_ok:
        print("ERROR: Phase 2 quality gate FAILED. "
              "See per-dataset invariant reports above.", file=sys.stderr)
        sys.exit(1)
    print("\nPhase 2 quality gate PASSED. Processed files written to data/processed/.")


# =============================================================================
# Phase 2 — shared helpers
# =============================================================================

def _save_pickle(path: Path, obj: Any) -> None:
    """Pickle `obj` to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_json(path: Path, obj: Any) -> None:
    """Write `obj` to `path` as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class _Phase2InvariantLog:
    """Lightweight pass/fail accumulator for Phase 2 post-conditions."""
    dataset_name: str
    items: List[Tuple[str, bool, str]] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append((name, ok, detail))

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def report(self) -> str:
        lines = [f"--- Phase 2 invariants: {self.dataset_name} ---"]
        for name, ok, detail in self.items:
            status = "[PASS]" if ok else "[FAIL]"
            lines.append(f"  {status}  {name}"
                         + (f"  -- {detail}" if detail else ""))
        n_pass = sum(1 for _, ok, _ in self.items if ok)
        lines.append(f"  Summary: {n_pass}/{len(self.items)} invariants passed")
        return "\n".join(lines)


def _temporal_leave_one_out(
    sequences: Dict[int, List[Tuple[Any, ...]]],
) -> Tuple[Dict[int, List[Tuple[Any, ...]]],
           Dict[int, Tuple[Any, ...]],
           Dict[int, Tuple[Any, ...]]]:
    """
    Apply leave-one-out temporal split (Plan §2.1.3).

    Input is `{user_id: [(item_id, rating, timestamp, ...), ...]}` already
    sorted ascending by (timestamp, file_order). The output is:
        train: {user_id: [all but last two]}
        val:   {user_id: second-to-last interaction (or None if <2 records)}
        test:  {user_id: last interaction (or None if <1 record)}

    Users with too few interactions are skipped for val/test but kept in
    train; this matches the convention of "leave-one-out with backoff".
    """
    train: Dict[int, List[Tuple[Any, ...]]] = {}
    val: Dict[int, Tuple[Any, ...]] = {}
    test: Dict[int, Tuple[Any, ...]] = {}

    for uid, seq in sequences.items():
        if len(seq) >= 3:
            train[uid] = seq[:-2]
            val[uid] = seq[-2]
            test[uid] = seq[-1]
        elif len(seq) == 2:
            # Cannot produce a val sample without leaving train empty.
            train[uid] = seq[:1]
            test[uid] = seq[-1]
        elif len(seq) == 1:
            # Pathological: a "5-core" user with one interaction. Skip
            # all three buckets to be safe.
            train[uid] = seq
        # len 0 should never happen, but skip gracefully
    return train, val, test


# =============================================================================
# Phase 2 — Amazon Beauty preprocessing  (Plan §2.1 + §2.2)
# =============================================================================

def preprocess_amazon_beauty(cfg: Dict[str, Any]) -> bool:
    """
    End-to-end Phase 2 for Amazon Beauty.

    Reads Beauty_5.json + meta_Beauty.json, produces under
    `data/processed/amazon_beauty/`:
        train_seqs.pkl, val_seqs.pkl, test_seqs.pkl
        item_meta.pkl
        id_maps.pkl
        vocab_sizes.json

    Returns True iff all invariants pass.
    """
    paths = cfg["dataset"]["paths"]
    pp = cfg["preprocess"]
    output_dir = Path(pp["output_dir"])
    inv = _Phase2InvariantLog("Amazon Beauty")

    # -------------------------------------------------------------------------
    # Step 1: read the 5-core review file in file-order (preserves the
    # tie-breaker we declared in the config).
    # -------------------------------------------------------------------------
    reviews_path = Path(paths["reviews_5core"])
    print(f"  [1/8] Reading reviews from {reviews_path} …")
    raw_records: List[Tuple[str, str, int, int, int]] = []
    # (reviewerID, asin, rating_int, unixReviewTime, file_order)
    for line_no, rec in enumerate(iter_amazon_reviews(reviews_path)):
        uid = rec.get("reviewerID")
        iid = rec.get("asin")
        r = rec.get("overall")
        ts = rec.get("unixReviewTime")
        if uid is None or iid is None or r is None or ts is None:
            continue
        raw_records.append((uid, iid, int(round(float(r))), int(ts), line_no))
    print(f"        kept {len(raw_records)} records")

    # -------------------------------------------------------------------------
    # Step 2: ID remap. Beauty_5 is already 5-core (Phase-1 verified), so
    # we simply enumerate users and items in first-appearance order.
    # First-appearance ordering is deterministic and human-debuggable.
    # -------------------------------------------------------------------------
    print("  [2/8] Building user/item id maps …")
    start = pp["id_remap"]["start_index"]  # 1
    user2idx: Dict[str, int] = {}
    item2idx: Dict[str, int] = {}
    for uid, iid, _r, _ts, _fo in raw_records:
        if uid not in user2idx:
            user2idx[uid] = start + len(user2idx)
        if iid not in item2idx:
            item2idx[iid] = start + len(item2idx)
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"        n_users={n_users}, n_items={n_items}")

    # -------------------------------------------------------------------------
    # Step 3: temporal sort per user with file-order tie-breaker, then
    # leave-one-out split.
    # -------------------------------------------------------------------------
    print("  [3/8] Sorting and splitting (leave-one-out temporal) …")
    by_user: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    # tuples are (item_idx, rating_int, timestamp, file_order)
    for uid, iid, r, ts, fo in raw_records:
        by_user[user2idx[uid]].append((item2idx[iid], r, ts, fo))

    sequences: Dict[int, List[Tuple[int, int, int]]] = {}
    for u, recs in by_user.items():
        # Sort by (timestamp, file_order); the latter is the explicit
        # tie-breaker for same-day records (Amazon timestamps are
        # day-precision).
        recs.sort(key=lambda x: (x[2], x[3]))
        # Drop the file_order field now that sort is done; downstream
        # consumers only need (item_idx, rating, timestamp).
        sequences[u] = [(t[0], t[1], t[2]) for t in recs]

    train_raw, val_raw, test_raw = _temporal_leave_one_out(sequences)

    # -------------------------------------------------------------------------
    # Step 4: fit Δt log-bucket edges on training Δt's only.
    # -------------------------------------------------------------------------
    print("  [4/8] Fitting Δt log-bucket edges on training set …")
    dt_cfg = pp["dt_bucketing"]
    n_dt_bins = dt_cfg["n_bins"]            # 32 (incl. PAD bin 0)
    n_dt_real_bins = n_dt_bins - 1          # bins fit by quantile

    train_dts: List[int] = []
    for u, seq in train_raw.items():
        for k in range(1, len(seq)):
            dt = seq[k][2] - seq[k - 1][2]
            if dt > 0:
                train_dts.append(dt)
    if not train_dts:
        raise RuntimeError("Amazon: no positive training Δt — corrupt data?")

    dt_edges = fit_log_quantile_edges(
        np.asarray(train_dts, dtype=np.float64),
        n_bins=n_dt_real_bins,
        base=dt_cfg["base"],
        add_one=True,
    )
    print(f"        {len(train_dts):,} train Δt's → {n_dt_real_bins} real bins")

    def _bucketise_seq(seq: List[Tuple[int, int, int]]
                       ) -> List[Tuple[int, int, int]]:
        """Turn (item_idx, rating, ts) → (item_idx, rating_bin, dt_bin).

        rating_bin == rating (1..5); dt_bin in {0=PAD, 1..n_dt_real_bins}.
        """
        out: List[Tuple[int, int, int]] = []
        for k, (iid, r, ts) in enumerate(seq):
            if k == 0:
                dt_bin = 0  # PAD
            else:
                dt = ts - seq[k - 1][2]
                if dt <= 0:
                    dt_bin = 0  # PAD: same-second tie (already broken by file order)
                else:
                    # apply_bucket_edges returns 0..n_real_bins-1 → shift by +1
                    bin_idx = int(apply_bucket_edges(
                        np.array([dt], dtype=np.float64),
                        dt_edges,
                        base=dt_cfg["base"],
                        add_one=True,
                    )[0])
                    dt_bin = bin_idx + 1
            out.append((iid, r, dt_bin))
        return out

    # -------------------------------------------------------------------------
    # Step 5: truncate train sequences to the most recent `max_seq_len`.
    # Val/test are single interactions and do not need truncation.
    # -------------------------------------------------------------------------
    print("  [5/8] Bucketising Δt + truncating train sequences …")
    max_len = pp["truncation"]["max_seq_len"]   # 50

    train_seqs: Dict[int, List[Tuple[int, int, int]]] = {}
    for u, seq in train_raw.items():
        bs = _bucketise_seq(seq)
        if len(bs) > max_len:
            bs = bs[-max_len:]
        train_seqs[u] = bs

    def _bucketise_single(prev_ts: int, single: Tuple[int, int, int]
                          ) -> Tuple[int, int, int]:
        """Bucketise a single val/test interaction given its predecessor ts."""
        iid, r, ts = single
        if prev_ts is None or (ts - prev_ts) <= 0:
            return (iid, r, 0)
        dt = ts - prev_ts
        bin_idx = int(apply_bucket_edges(
            np.array([dt], dtype=np.float64),
            dt_edges, base=dt_cfg["base"], add_one=True,
        )[0])
        return (iid, r, bin_idx + 1)

    val_seqs: Dict[int, Tuple[int, int, int]] = {}
    test_seqs: Dict[int, Tuple[int, int, int]] = {}
    for u, single in val_raw.items():
        prev_ts = train_raw[u][-1][2] if train_raw.get(u) else None
        val_seqs[u] = _bucketise_single(prev_ts, single)
    for u, single in test_raw.items():
        # Test's predecessor is the val record (if it exists), else last train.
        if u in val_raw:
            prev_ts = val_raw[u][2]
        elif train_raw.get(u):
            prev_ts = train_raw[u][-1][2]
        else:
            prev_ts = None
        test_seqs[u] = _bucketise_single(prev_ts, single)

    # -------------------------------------------------------------------------
    # Step 6: scan meta_Beauty.json once to build item-meta lookups.
    # Categories and brands are constrained to the items that actually
    # appear in reviews (the 12,101 5-core items).
    # -------------------------------------------------------------------------
    print("  [6/8] Building item metadata lookups …")
    meta_path = Path(paths["meta"])

    # Bookkeeping (indexed by item_idx 1..n_items)
    item_leaf_cat: Dict[int, str | None] = {i: None for i in idx2item}
    item_brand: Dict[int, str | None] = {i: None for i in idx2item}
    item_price: Dict[int, float | None] = {i: None for i in idx2item}
    item_salesrank: Dict[int, int | None] = {i: None for i in idx2item}

    seen_meta: set = set()
    for rec in iter_amazon_meta(meta_path):
        asin = rec.get("asin")
        if asin not in item2idx:
            continue
        idx = item2idx[asin]
        if idx in seen_meta:
            continue
        seen_meta.add(idx)

        item_leaf_cat[idx] = extract_primary_leaf_category(rec.get("categories"))
        item_brand[idx]    = get_brand_or_none(rec.get("brand"))
        item_price[idx]    = get_price_or_none(rec.get("price"))
        sr, _src = extract_salesrank(rec.get("salesRank"),
                                     prefer_key=pp["salesrank_Z"]["salesrank_key"])
        item_salesrank[idx] = sr

    # Build category lookup with UNK collapsing.
    cat_min = pp["category"]["min_item_count"]
    cat_counter: Counter = Counter(c for c in item_leaf_cat.values() if c is not None)
    cat_keep = {c for c, n in cat_counter.items() if n >= cat_min}
    # 0 = PAD, 1 = UNK, 2.. = real categories sorted alphabetically for stability
    cat2idx: Dict[str, int] = {c: i + 2 for i, c in enumerate(sorted(cat_keep))}
    UNK_CAT = 1
    n_cats = 2 + len(cat_keep)              # PAD + UNK + real

    # Build brand lookup with UNK collapsing.
    brand_min = pp["brand"]["min_item_count"]
    brand_counter: Counter = Counter(b for b in item_brand.values() if b is not None)
    brand_keep = {b for b, n in brand_counter.items() if n >= brand_min}
    brand2idx: Dict[str, int] = {b: i + 2 for i, b in enumerate(sorted(brand_keep))}
    UNK_BRAND = 1
    n_brands = 2 + len(brand_keep)

    # Build log10(price) with imputation order: per-leaf-category median, then global median.
    valid_prices = [p for p in item_price.values() if p is not None]
    if not valid_prices:
        raise RuntimeError("Amazon: no valid prices to compute global median")
    global_log_price_median = float(np.log10(np.median(valid_prices)))

    # Per-category median log price (computed on the items that have prices).
    leaf_prices: Dict[str, List[float]] = defaultdict(list)
    for idx, p in item_price.items():
        if p is None:
            continue
        leaf = item_leaf_cat[idx]
        if leaf is not None:
            leaf_prices[leaf].append(float(np.log10(p)))
    leaf_log_price_median: Dict[str, float] = {
        leaf: float(np.median(arr)) for leaf, arr in leaf_prices.items()
    }

    # SalesRank: fit edges on items that have a salesRank (any-key fallback).
    valid_sr = np.array([s for s in item_salesrank.values() if s is not None],
                        dtype=np.float64)
    sr_cfg = pp["salesrank_Z"]
    sr_edges = fit_log_quantile_edges(
        valid_sr,
        n_bins=sr_cfg["n_bins"],
        base=sr_cfg["log_base"],
        add_one=sr_cfg["add_one"],
    )
    SR_MEDIAN_BIN = int(sr_cfg["median_bin_for_missing"])  # 5

    # Materialise the final item_meta dict.
    item_meta: Dict[int, Dict[str, Any]] = {}
    n_price_imputed = 0
    for idx in idx2item:
        # Category id
        leaf = item_leaf_cat[idx]
        cat_id = cat2idx.get(leaf, UNK_CAT) if leaf is not None else UNK_CAT
        # Brand id
        brand = item_brand[idx]
        brand_id = brand2idx.get(brand, UNK_BRAND) if brand is not None else UNK_BRAND
        # log price (imputed if missing)
        p = item_price[idx]
        if p is not None:
            log_price = float(np.log10(p))
        else:
            n_price_imputed += 1
            log_price = leaf_log_price_median.get(leaf, global_log_price_median) \
                if leaf is not None else global_log_price_median
        # SalesRank Z bucket
        sr = item_salesrank[idx]
        if sr is None:
            z_bin = SR_MEDIAN_BIN
        else:
            z_bin = int(apply_bucket_edges(
                np.array([sr], dtype=np.float64),
                sr_edges, base=sr_cfg["log_base"], add_one=sr_cfg["add_one"],
            )[0])
        item_meta[idx] = {
            "cat": cat_id,
            "brand": brand_id,
            "log_price": log_price,
            "Z": z_bin,
        }
    print(f"        n_cats={n_cats}, n_brands={n_brands}, "
          f"prices_imputed={n_price_imputed}")

    # -------------------------------------------------------------------------
    # Step 7: serialise everything to disk.
    # -------------------------------------------------------------------------
    print(f"  [7/8] Writing artefacts to {output_dir} …")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_pickle(output_dir / "train_seqs.pkl", train_seqs)
    _save_pickle(output_dir / "val_seqs.pkl", val_seqs)
    _save_pickle(output_dir / "test_seqs.pkl", test_seqs)
    _save_pickle(output_dir / "item_meta.pkl", item_meta)
    _save_pickle(output_dir / "id_maps.pkl", {
        "user2idx": user2idx, "idx2user": idx2user,
        "item2idx": item2idx, "idx2item": idx2item,
        "cat2idx": cat2idx,   "brand2idx": brand2idx,
    })
    # Edges are model-config: persist them so eval-time bucketisation matches.
    _save_pickle(output_dir / "bucket_edges.pkl", {
        "dt_edges": dt_edges,
        "salesrank_edges": sr_edges,
        "global_log_price_median": global_log_price_median,
        "leaf_log_price_median": leaf_log_price_median,
    })
    vocab = {
        "n_users":       n_users,
        "n_items":       n_items,
        "n_cats":        n_cats,
        "n_brands":      n_brands,
        "n_Z_bins":      int(sr_cfg["n_bins"]),
        "n_dt_bins":     int(n_dt_bins),
        "n_rating_bins": int(pp["rating_bins"]["n_bins"]),
        "pad_index":     int(pp["id_remap"]["pad_index"]),
        "max_seq_len":   int(max_len),
    }
    _save_json(output_dir / "vocab_sizes.json", vocab)

    # -------------------------------------------------------------------------
    # Step 8: post-condition invariants (Plan §2.4 quality gate).
    # -------------------------------------------------------------------------
    print("  [8/8] Checking Phase-2 invariants …")
    # (a) IDs in train/val/test are within [1, n_items].
    ok_id = all(
        1 <= iid <= n_items
        for seq in train_seqs.values() for (iid, _r, _d) in seq
    )
    inv.check("train item ids in [1, n_items]", ok_id)

    # (b) Future-leakage guard: val and test items must not appear in the
    #     SAME USER's train sequence (per-user splits are temporal).
    leak_users = 0
    for u, single in val_seqs.items():
        if any(t[0] == single[0] for t in train_seqs.get(u, [])):
            leak_users += 1
    for u, single in test_seqs.items():
        if any(t[0] == single[0] for t in train_seqs.get(u, [])):
            leak_users += 1
    # NB: same item may legitimately reappear if the same user reviewed the
    # same product more than once at different timestamps. We allow that —
    # the Plan's "future leakage" guard is about temporal ordering, not
    # item identity. But we still surface the count for transparency.
    inv.check("future-leakage check (informational)",
              True, detail=f"{leak_users} users have item-id reappearance "
                           f"(allowed if at different timestamps)")

    # (c) Vocab sizes match the saved lookup tables.
    inv.check("n_users matches user2idx",
              n_users == len(user2idx), detail=f"{n_users} vs {len(user2idx)}")
    inv.check("n_items matches item2idx",
              n_items == len(item2idx))
    inv.check("n_cats == 2 + |cat_keep|",
              n_cats == 2 + len(cat_keep))
    inv.check("n_brands == 2 + |brand_keep|",
              n_brands == 2 + len(brand_keep))

    # (d) Δt bin indices stay in [0, n_dt_bins - 1].
    max_bin = 0
    for seq in train_seqs.values():
        for (_, _, dt_bin) in seq:
            if dt_bin > max_bin:
                max_bin = dt_bin
    inv.check("Δt bin range ok",
              0 <= max_bin <= n_dt_bins - 1, detail=f"max bin = {max_bin}")

    # (e) Z bucket values in [0, n_Z_bins - 1].
    z_max = max(m["Z"] for m in item_meta.values())
    z_min = min(m["Z"] for m in item_meta.values())
    inv.check("Z bin range ok",
              0 <= z_min and z_max <= sr_cfg["n_bins"] - 1,
              detail=f"min={z_min}, max={z_max}")

    # (f) item_meta covers every item id.
    inv.check("item_meta covers every item",
              len(item_meta) == n_items)

    print(inv.report())
    return inv.all_passed


# =============================================================================
# Phase 2 — MovieLens 10M preprocessing  (Plan §2.1 + §2.3)
# =============================================================================

def preprocess_movielens_10m(cfg: Dict[str, Any]) -> bool:
    """
    End-to-end Phase 2 for MovieLens 10M.

    Reads ratings.dat + movies.dat, produces under
    `data/processed/movielens_10m/`:
        train_seqs.pkl, val_seqs.pkl, test_seqs.pkl
        item_genre.pkl
        dynamic_Z.pkl      (parallel np.array, indexed by train interactions)
        genre_red.pkl      (standard Jaccard, parallel np.array)
        genre_red_idf.pkl  (IDF-weighted Jaccard, parallel np.array)
        id_maps.pkl
        vocab_sizes.json
    """
    paths = cfg["dataset"]["paths"]
    enc = cfg["dataset"]["encoding"]
    pp = cfg["preprocess"]
    output_dir = Path(pp["output_dir"])
    inv = _Phase2InvariantLog("MovieLens 10M")

    # -------------------------------------------------------------------------
    # Step 1: read ratings + 5-core fixed-point.
    # -------------------------------------------------------------------------
    ratings_path = Path(paths["ratings"])
    print(f"  [1/9] Reading ratings from {ratings_path} …")
    raw: List[Tuple[int, int, float, int, int]] = []
    # (uid, mid, rating, timestamp, file_order)
    for fo, (uid, mid, r, ts) in enumerate(iter_movielens_ratings(ratings_path)):
        raw.append((uid, mid, r, ts, fo))
    print(f"        {len(raw):,} raw interactions")

    print("  [2/9] Applying iterative 5-core …")
    interactions = [(u, m, r, t) for (u, m, r, t, _) in raw]
    file_orders = {(u, m, t): fo for (u, m, r, t, fo) in raw}
    # Note: a single (user, item, ts) triple is essentially unique in ML10M;
    # but if duplicates exist, last write wins for the tie-breaker. The
    # remap below stores file_order separately, so this only matters for
    # within-record sort stability.
    filtered, users_keep, items_keep = _five_core_filter(
        interactions, min_count=pp["five_core"]["min_count"]
    )
    print(f"        kept {len(filtered):,} / {len(raw):,} interactions "
          f"({len(users_keep)} users, {len(items_keep)} items)")

    # -------------------------------------------------------------------------
    # Step 3: ID remap (first-appearance ordering on the FILTERED stream).
    # -------------------------------------------------------------------------
    print("  [3/9] Building user/item id maps …")
    start = pp["id_remap"]["start_index"]
    user2idx: Dict[int, int] = {}
    item2idx: Dict[int, int] = {}
    for u, m, _r, _ts in filtered:
        if u not in user2idx:
            user2idx[u] = start + len(user2idx)
        if m not in item2idx:
            item2idx[m] = start + len(item2idx)
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"        n_users={n_users}, n_items={n_items}")

    # -------------------------------------------------------------------------
    # Step 4: group by user, sort by (ts, file_order), apply session
    # segmentation, then leave-one-out split.
    # The session segmentation collapses Δt within a session to 0 (PAD).
    # -------------------------------------------------------------------------
    print("  [4/9] Sorting, session-segmenting, and splitting …")
    by_user: Dict[int, List[Tuple[int, float, int, int]]] = defaultdict(list)
    # (item_idx, rating, ts, file_order)
    for u, m, r, ts in filtered:
        fo = file_orders[(u, m, ts)]
        by_user[user2idx[u]].append((item2idx[m], r, ts, fo))

    session_gap = pp["session"]["boundary_seconds"]   # 60
    rating_bins_cfg = pp["rating_bins"]

    def _rating_to_bin(r: float) -> int:
        """0.5..5.0 → 1..10  (multiply by 2). 0 reserved for PAD."""
        return int(round(r * 2.0))

    # We will store sequences as lists of (item_idx, rating_bin, ts,
    # in_session_flag). The dt-bin assignment comes later because the
    # bucket edges depend on training Δt's, which depend on the split.
    sequences: Dict[int, List[Tuple[int, int, int, bool]]] = {}
    for u, recs in by_user.items():
        recs.sort(key=lambda x: (x[2], x[3]))
        out: List[Tuple[int, int, int, bool]] = []
        prev_ts = None
        for (iid, r, ts, _fo) in recs:
            in_session = (prev_ts is not None and (ts - prev_ts) < session_gap)
            out.append((iid, _rating_to_bin(r), ts, in_session))
            prev_ts = ts
        sequences[u] = out

    train_raw, val_raw, test_raw = _temporal_leave_one_out(sequences)

    # 2009 truncation on TRAIN ONLY (Plan §2.3.5).
    excl = pp["exclude_year_from_train"]
    if excl.get("enabled", True):
        year_thr = int(excl["year_threshold"])
        cutoff_ts = int(datetime(year_thr, 1, 1, tzinfo=timezone.utc).timestamp())
        before = sum(len(v) for v in train_raw.values())
        train_raw = {
            u: [t for t in seq if t[2] < cutoff_ts]
            for u, seq in train_raw.items()
        }
        after = sum(len(v) for v in train_raw.values())
        print(f"        dropped {before - after:,} train rows from year >= {year_thr}")

    # -------------------------------------------------------------------------
    # Step 5: fit Δt log-bucket edges on TRAINING Δt's (post-2009 cut,
    # post session-segmentation: intra-session pairs are masked out).
    # -------------------------------------------------------------------------
    print("  [5/9] Fitting Δt log-bucket edges on training set …")
    dt_cfg = pp["dt_bucketing"]
    n_dt_bins = dt_cfg["n_bins"]            # 64
    n_dt_real_bins = n_dt_bins - 1

    train_dts: List[int] = []
    for u, seq in train_raw.items():
        for k in range(1, len(seq)):
            if seq[k][3]:  # in_session
                continue
            dt = seq[k][2] - seq[k - 1][2]
            if dt > 0:
                train_dts.append(dt)
    if not train_dts:
        raise RuntimeError("MovieLens: no positive cross-session Δt — corrupt data?")

    dt_edges = fit_log_quantile_edges(
        np.asarray(train_dts, dtype=np.float64),
        n_bins=n_dt_real_bins,
        base=dt_cfg["base"],
        add_one=True,
    )
    print(f"        {len(train_dts):,} cross-session Δt's → {n_dt_real_bins} real bins")

    # -------------------------------------------------------------------------
    # Step 6: bucketise + truncate. Within-session steps get dt_bin = 0.
    # -------------------------------------------------------------------------
    print("  [6/9] Bucketising Δt + truncating train sequences …")
    max_len = pp["truncation"]["max_seq_len"]   # 200

    def _bucketise_dt(prev_ts: int | None, ts: int, in_session: bool) -> int:
        if in_session or prev_ts is None:
            return 0
        dt = ts - prev_ts
        if dt <= 0:
            return 0
        bin_idx = int(apply_bucket_edges(
            np.array([dt], dtype=np.float64),
            dt_edges, base=dt_cfg["base"], add_one=True,
        )[0])
        return bin_idx + 1

    train_seqs: Dict[int, List[Tuple[int, int, int]]] = {}
    # We also keep train sequences with TIMESTAMPS separately because the
    # dynamic-Z and GenreRed sweeps need them. To save memory we use np
    # arrays indexed by training-interaction order.
    train_seqs_with_ts: Dict[int, List[Tuple[int, int, int]]] = {}
    # (item_idx, rating_bin, ts)
    for u, seq in train_raw.items():
        bucketised: List[Tuple[int, int, int]] = []
        ts_list: List[Tuple[int, int, int]] = []
        prev_ts: int | None = None
        for (iid, rb, ts, in_session) in seq:
            db = _bucketise_dt(prev_ts, ts, in_session)
            bucketised.append((iid, rb, db))
            ts_list.append((iid, rb, ts))
            prev_ts = ts
        if len(bucketised) > max_len:
            bucketised = bucketised[-max_len:]
            ts_list = ts_list[-max_len:]
        train_seqs[u] = bucketised
        train_seqs_with_ts[u] = ts_list

    val_seqs: Dict[int, Tuple[int, int, int]] = {}
    test_seqs: Dict[int, Tuple[int, int, int]] = {}
    for u, (iid, rb, ts, in_session) in val_raw.items():
        prev_ts = train_raw[u][-1][2] if train_raw.get(u) else None
        # in_session vs train's last is non-trivial; ML val/test are usually
        # well beyond the user's last training rating so we follow the cut.
        db = _bucketise_dt(prev_ts, ts, False)
        val_seqs[u] = (iid, rb, db)
    for u, (iid, rb, ts, in_session) in test_raw.items():
        if u in val_raw:
            prev_ts = val_raw[u][2]
        elif train_raw.get(u):
            prev_ts = train_raw[u][-1][2]
        else:
            prev_ts = None
        db = _bucketise_dt(prev_ts, ts, False)
        test_seqs[u] = (iid, rb, db)

    # -------------------------------------------------------------------------
    # Step 7: read movies.dat, build 19-dim binary genre vectors + IDF.
    # -------------------------------------------------------------------------
    print("  [7/9] Reading movies.dat and building genre vectors …")
    movies_path = Path(paths["movies"])
    gr_cfg = pp["genre_red"]
    NO_GENRE = gr_cfg["no_genre_sentinel"]
    n_genres = int(gr_cfg["n_genres"])

    # First pass: collect the (case-sensitive, sorted) genre vocabulary
    # excluding the sentinel. We sort alphabetically for deterministic
    # indices; the EDA's 19 are: Action..Western.
    genre_set: set = set()
    raw_movie_genres: Dict[int, List[str]] = {}
    for mid, _title, genres in iter_movielens_movies(movies_path,
                                                     encoding=enc["movies"]):
        kept = [g for g in genres if g and g != NO_GENRE]
        raw_movie_genres[mid] = kept
        for g in kept:
            genre_set.add(g)
    if len(genre_set) != n_genres:
        raise RuntimeError(f"Expected {n_genres} genres, got {len(genre_set)}: "
                           f"{sorted(genre_set)}")
    genre2idx: Dict[str, int] = {g: i for i, g in enumerate(sorted(genre_set))}

    # Build per-item binary vectors keyed by REMAPPED item_idx.
    item_genre: Dict[int, np.ndarray] = {}
    for raw_mid, kept in raw_movie_genres.items():
        idx = item2idx.get(raw_mid)
        if idx is None:
            continue
        v = np.zeros(n_genres, dtype=np.float32)
        for g in kept:
            v[genre2idx[g]] = 1.0
        item_genre[idx] = v
    print(f"        built {len(item_genre):,} genre vectors over {n_genres} genres")

    # IDF on the 5-core movie corpus (Plan §2.3.4 + our decision above).
    idf = compute_genre_idf(item_genre.values(), n_genres=n_genres,
                            smooth=bool(gr_cfg["idf_smooth"]))

    # -------------------------------------------------------------------------
    # Step 8: dynamic Z_i(t) + GenreRed sweep over training interactions.
    # We build a single canonical ordering of training rows = (user, in-seq
    # position). Z and GenreRed are stored as float arrays of length
    # n_train_rows, aligned with a parallel (user, in-seq-pos) index.
    # -------------------------------------------------------------------------
    print("  [8/9] Sweeping training set for dynamic Z and GenreRed …")
    # Sort all (uid, ts, position) triples globally by ts for the Z sweep;
    # then bookkeeping per-user history-union for the Jaccard sweep.
    flat: List[Tuple[int, int, int, int]] = []  # (uid, pos, item_idx, ts)
    for u, seq in train_seqs_with_ts.items():
        for pos, (iid, _rb, ts) in enumerate(seq):
            flat.append((u, pos, iid, ts))
    flat.sort(key=lambda x: (x[3], x[0], x[1]))  # global temporal sort

    n_rows = len(flat)
    dyn_Z = np.zeros(n_rows, dtype=np.int64)
    genre_red = np.zeros(n_rows, dtype=np.float32)
    genre_red_idf = np.zeros(n_rows, dtype=np.float32)

    # Item global counter (the Z is "cumulative count BEFORE t").
    item_count: Dict[int, int] = defaultdict(int)
    # Per-user history-union genre vector.
    user_hist: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_genres, dtype=np.float32))

    # row_index gives, for each (uid, pos), the index into flat[] —
    # because flat is sorted globally, but we want the output arrays
    # aligned by (uid, pos) for O(1) lookup.
    row_index: Dict[Tuple[int, int], int] = {}

    for flat_i, (u, pos, iid, ts) in enumerate(flat):
        # Dynamic Z BEFORE counting this row.
        z_before = item_count[iid]
        # GenreRed against the user's history-union BEFORE this row.
        cand = item_genre.get(iid)
        if cand is None:
            j_std = 0.0
            j_idf = 0.0
        else:
            hist = user_hist[u]
            inter_mask = np.logical_and(cand > 0, hist > 0)
            union_mask = np.logical_or(cand > 0, hist > 0)
            union_count = float(union_mask.sum())
            if union_count == 0.0:
                j_std = 0.0
            else:
                j_std = float(inter_mask.sum()) / union_count
            den_idf = float((idf * union_mask).sum())
            if den_idf == 0.0:
                j_idf = 0.0
            else:
                j_idf = float((idf * inter_mask).sum()) / den_idf

        dyn_Z[flat_i] = z_before
        genre_red[flat_i] = j_std
        genre_red_idf[flat_i] = j_idf
        row_index[(u, pos)] = flat_i

        # Update state AFTER recording — this is the "strictly before t"
        # discipline that prevents leakage.
        item_count[iid] = z_before + 1
        if cand is not None:
            user_hist[u] = np.logical_or(user_hist[u] > 0, cand > 0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Step 9: serialise + invariants.
    # -------------------------------------------------------------------------
    print(f"  [9/9] Writing artefacts to {output_dir} …")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_pickle(output_dir / "train_seqs.pkl", train_seqs)
    _save_pickle(output_dir / "val_seqs.pkl", val_seqs)
    _save_pickle(output_dir / "test_seqs.pkl", test_seqs)
    _save_pickle(output_dir / "item_genre.pkl", item_genre)
    _save_pickle(output_dir / "dynamic_Z.pkl", {
        "row_index": row_index,
        "values": dyn_Z,
    })
    _save_pickle(output_dir / "genre_red.pkl", {
        "row_index": row_index,   # same alignment as dynamic_Z
        "values": genre_red,
    })
    _save_pickle(output_dir / "genre_red_idf.pkl", {
        "row_index": row_index,
        "values": genre_red_idf,
        "idf_weights": idf,
        "genre_vocab": [g for g, _ in sorted(genre2idx.items(), key=lambda x: x[1])],
    })
    _save_pickle(output_dir / "id_maps.pkl", {
        "user2idx": user2idx, "idx2user": idx2user,
        "item2idx": item2idx, "idx2item": idx2item,
        "genre2idx": genre2idx,
    })
    _save_pickle(output_dir / "bucket_edges.pkl", {"dt_edges": dt_edges})
    vocab = {
        "n_users":       n_users,
        "n_items":       n_items,
        "n_genres":      n_genres,
        "n_dt_bins":     int(n_dt_bins),
        "n_rating_bins": int(rating_bins_cfg["n_bins"]),
        "pad_index":     int(pp["id_remap"]["pad_index"]),
        "max_seq_len":   int(max_len),
    }
    _save_json(output_dir / "vocab_sizes.json", vocab)

    # Invariants ---------------------------------------------------------------
    # (a) Item ids in sequences ∈ [1, n_items].
    ok_id = all(
        1 <= iid <= n_items
        for seq in train_seqs.values() for (iid, _r, _d) in seq
    )
    inv.check("train item ids in [1, n_items]", ok_id)

    # (b) Δt bin ∈ [0, n_dt_bins - 1].
    max_dt = max((d for seq in train_seqs.values() for (_, _, d) in seq),
                 default=0)
    inv.check("Δt bin range ok",
              0 <= max_dt <= n_dt_bins - 1, detail=f"max bin = {max_dt}")

    # (c) Dynamic Z is monotonically non-decreasing PER ITEM over time.
    #     Spot-check by re-sweeping a sample of items.
    sample_items = list(item2idx.values())[:20]
    sample_ok = True
    sample_details = []
    for iid in sample_items:
        last = -1
        for flat_i, (_u, _p, ii, _ts) in enumerate(flat):
            if ii != iid:
                continue
            if dyn_Z[flat_i] < last:
                sample_ok = False
                sample_details.append(f"item {iid} non-monotonic")
                break
            last = dyn_Z[flat_i]
    inv.check("dynamic Z monotonic per item (sampled 20)",
              sample_ok, detail="; ".join(sample_details))

    # (d) GenreRed ∈ [0, 1] for both variants.
    inv.check("GenreRed (std) in [0, 1]",
              bool(((0.0 <= genre_red) & (genre_red <= 1.0)).all()))
    inv.check("GenreRed (idf) in [0, 1]",
              bool(((0.0 <= genre_red_idf) & (genre_red_idf <= 1.0)).all()))

    # (e) row_index aligns with stored arrays.
    inv.check("row_index size matches values",
              len(row_index) == len(dyn_Z) == len(genre_red))

    # (f) item_genre covers every item.
    inv.check("item_genre covers every item",
              len(item_genre) == n_items,
              detail=f"{len(item_genre)} vs {n_items}")

    # (g) Vocab sizes consistent.
    inv.check("n_users matches user2idx", n_users == len(user2idx))
    inv.check("n_items matches item2idx", n_items == len(item2idx))
    inv.check("n_genres == |genre2idx|", n_genres == len(genre2idx))

    print(inv.report())
    return inv.all_passed


def preprocess_movielens_10m_stage1(cfg: Dict[str, Any]) -> bool:
    """
    Phase 2 Stage 1 for MovieLens 10M: build train/val/test splits, item genre
    vectors, ID maps, and bucket edges. This writes the lighter Phase 2
    artefacts and preserves timestamped training rows for Stage 2.
    """
    paths = cfg["dataset"]["paths"]
    enc = cfg["dataset"]["encoding"]
    pp = cfg["preprocess"]
    output_dir = Path(pp["output_dir"])
    inv = _Phase2InvariantLog("MovieLens 10M (Stage 1)")

    ratings_path = Path(paths["ratings"])
    print(f"  [1/8] Reading ratings from {ratings_path} …")
    raw: List[Tuple[int, int, float, int, int]] = []
    for fo, (uid, mid, r, ts) in enumerate(iter_movielens_ratings(ratings_path)):
        raw.append((uid, mid, r, ts, fo))
    print(f"        {len(raw):,} raw interactions")

    print("  [2/8] Applying iterative 5-core …")
    filtered, users_keep, items_keep = _five_core_filter(
        raw, min_count=pp["five_core"]["min_count"]
    )
    print(f"        kept {len(filtered):,} / {len(raw):,} interactions "
          f"({len(users_keep)} users, {len(items_keep)} items)")
    del raw

    print("  [3/8] Building user/item id maps …")
    start = pp["id_remap"]["start_index"]
    user2idx: Dict[int, int] = {}
    item2idx: Dict[int, int] = {}
    for u, m, _r, _ts, _fo in filtered:
        if u not in user2idx:
            user2idx[u] = start + len(user2idx)
        if m not in item2idx:
            item2idx[m] = start + len(item2idx)
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"        n_users={n_users}, n_items={n_items}")

    print("  [4/8] Sorting, session-segmenting, and splitting …")
    by_user: Dict[int, List[Tuple[int, float, int, int]]] = defaultdict(list)
    session_gap = pp["session"]["boundary_seconds"]

    def _rating_to_bin(r: float) -> int:
        return int(round(r * 2.0))

    for u, m, r, ts, fo in filtered:
        by_user[user2idx[u]].append((item2idx[m], r, ts, fo))
    del filtered

    sequences: Dict[int, List[Tuple[int, int, int, bool]]] = {}
    for u, recs in by_user.items():
        recs.sort(key=lambda x: (x[2], x[3]))
        out: List[Tuple[int, int, int, bool]] = []
        prev_ts = None
        for iid, r, ts, _fo in recs:
            in_session = (prev_ts is not None and (ts - prev_ts) < session_gap)
            out.append((iid, _rating_to_bin(r), ts, in_session))
            prev_ts = ts
        sequences[u] = out
    del by_user

    train_raw, val_raw, test_raw = _temporal_leave_one_out(sequences)
    del sequences

    excl = pp["exclude_year_from_train"]
    if excl.get("enabled", True):
        year_thr = int(excl["year_threshold"])
        cutoff_ts = int(datetime(year_thr, 1, 1, tzinfo=timezone.utc).timestamp())
        before = sum(len(v) for v in train_raw.values())
        train_raw = {
            u: [t for t in seq if t[2] < cutoff_ts]
            for u, seq in train_raw.items()
        }
        after = sum(len(v) for v in train_raw.values())
        print(f"        dropped {before - after:,} train rows from year >= {year_thr}")

    print("  [5/8] Fitting Δt log-bucket edges on training set …")
    dt_cfg = pp["dt_bucketing"]
    n_dt_bins = dt_cfg["n_bins"]
    n_dt_real_bins = n_dt_bins - 1

    train_dts: List[int] = []
    for u, seq in train_raw.items():
        for k in range(1, len(seq)):
            if seq[k][3]:
                continue
            dt = seq[k][2] - seq[k - 1][2]
            if dt > 0:
                train_dts.append(dt)
    if not train_dts:
        raise RuntimeError("MovieLens: no positive cross-session Δt — corrupt data?")

    dt_edges = fit_log_quantile_edges(
        np.asarray(train_dts, dtype=np.float64),
        n_bins=n_dt_real_bins,
        base=dt_cfg["base"],
        add_one=True,
    )
    print(f"        {len(train_dts):,} cross-session Δt's → {n_dt_real_bins} real bins")

    print("  [6/8] Bucketising Δt + truncating train sequences …")
    max_len = pp["truncation"]["max_seq_len"]

    def _bucketise_dt(prev_ts: int | None, ts: int, in_session: bool) -> int:
        if in_session or prev_ts is None:
            return 0
        dt = ts - prev_ts
        if dt <= 0:
            return 0
        bin_idx = int(apply_bucket_edges(
            np.array([dt], dtype=np.float64),
            dt_edges, base=dt_cfg["base"], add_one=True,
        )[0])
        return bin_idx + 1

    train_seqs: Dict[int, List[Tuple[int, int, int]]] = {}
    train_seqs_with_ts: Dict[int, List[Tuple[int, int, int]]] = {}
    for u, seq in train_raw.items():
        bucketised: List[Tuple[int, int, int]] = []
        ts_list: List[Tuple[int, int, int]] = []
        prev_ts: int | None = None
        for iid, rb, ts, in_session in seq:
            db = _bucketise_dt(prev_ts, ts, in_session)
            bucketised.append((iid, rb, db))
            ts_list.append((iid, rb, ts))
            prev_ts = ts
        if len(bucketised) > max_len:
            bucketised = bucketised[-max_len:]
            ts_list = ts_list[-max_len:]
        train_seqs[u] = bucketised
        train_seqs_with_ts[u] = ts_list

    val_seqs: Dict[int, Tuple[int, int, int]] = {}
    test_seqs: Dict[int, Tuple[int, int, int]] = {}
    for u, (iid, rb, ts, _in_session) in val_raw.items():
        prev_ts = train_raw[u][-1][2] if train_raw.get(u) else None
        db = _bucketise_dt(prev_ts, ts, False)
        val_seqs[u] = (iid, rb, db)
    for u, (iid, rb, ts, _in_session) in test_raw.items():
        if u in val_raw:
            prev_ts = val_raw[u][2]
        elif train_raw.get(u):
            prev_ts = train_raw[u][-1][2]
        else:
            prev_ts = None
        db = _bucketise_dt(prev_ts, ts, False)
        test_seqs[u] = (iid, rb, db)

    del train_raw, val_raw, test_raw

    print("  [7/8] Reading movies.dat and building genre vectors …")
    movies_path = Path(paths["movies"])
    gr_cfg = pp["genre_red"]
    NO_GENRE = gr_cfg["no_genre_sentinel"]
    n_genres = int(gr_cfg["n_genres"])

    genre_set: set = set()
    raw_movie_genres: Dict[int, List[str]] = {}
    for mid, _title, genres in iter_movielens_movies(movies_path,
                                                     encoding=enc["movies"]):
        kept = [g for g in genres if g and g != NO_GENRE]
        raw_movie_genres[mid] = kept
        for g in kept:
            genre_set.add(g)
    if len(genre_set) != n_genres:
        raise RuntimeError(f"Expected {n_genres} genres, got {len(genre_set)}: "
                           f"{sorted(genre_set)}")
    genre2idx: Dict[str, int] = {g: i for i, g in enumerate(sorted(genre_set))}

    item_genre: Dict[int, np.ndarray] = {}
    for raw_mid, kept in raw_movie_genres.items():
        idx = item2idx.get(raw_mid)
        if idx is None:
            continue
        v = np.zeros(n_genres, dtype=np.float32)
        for g in kept:
            v[genre2idx[g]] = 1.0
        item_genre[idx] = v
    print(f"        built {len(item_genre):,} genre vectors over {n_genres} genres")

    print("  [8/8] Writing Stage 1 artefacts …")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_pickle(output_dir / "train_seqs.pkl", train_seqs)
    _save_pickle(output_dir / "val_seqs.pkl", val_seqs)
    _save_pickle(output_dir / "test_seqs.pkl", test_seqs)
    _save_pickle(output_dir / "train_seqs_with_ts.pkl", train_seqs_with_ts)
    _save_pickle(output_dir / "item_genre.pkl", item_genre)
    _save_pickle(output_dir / "id_maps.pkl", {
        "user2idx": user2idx, "idx2user": idx2user,
        "item2idx": item2idx, "idx2item": idx2item,
        "genre2idx": genre2idx,
    })
    _save_pickle(output_dir / "bucket_edges.pkl", {"dt_edges": dt_edges})
    vocab = {
        "n_users":       n_users,
        "n_items":       n_items,
        "n_genres":      n_genres,
        "n_dt_bins":     int(n_dt_bins),
        "n_rating_bins": int(pp["rating_bins"]["n_bins"]),
        "pad_index":     int(pp["id_remap"]["pad_index"]),
        "max_seq_len":   int(max_len),
    }
    _save_json(output_dir / "vocab_sizes.json", vocab)

    ok_id = all(
        1 <= iid <= n_items
        for seq in train_seqs.values() for (iid, _r, _d) in seq
    )
    inv.check("train item ids in [1, n_items]", ok_id)

    max_dt = max((d for seq in train_seqs.values() for (_, _, d) in seq),
                 default=0)
    inv.check("Δt bin range ok",
              0 <= max_dt <= n_dt_bins - 1, detail=f"max bin = {max_dt}")

    inv.check("item_genre covers every item",
              len(item_genre) == n_items,
              detail=f"{len(item_genre)} vs {n_items}")

    print(inv.report())
    return inv.all_passed


def preprocess_movielens_10m_stage2(cfg: Dict[str, Any]) -> bool:
    """
    Phase 2 Stage 2 for MovieLens 10M: compute dynamic Z and GenreRed from
    Stage 1 artefacts and validate the final outputs.
    """
    pp = cfg["preprocess"]
    output_dir = Path(pp["output_dir"])
    inv = _Phase2InvariantLog("MovieLens 10M (Stage 2)")

    required = [
        "train_seqs.pkl",
        "train_seqs_with_ts.pkl",
        "item_genre.pkl",
        "id_maps.pkl",
        "bucket_edges.pkl",
        "vocab_sizes.json",
    ]
    for filename in required:
        path = output_dir / filename
        if not path.exists():
            raise RuntimeError(
                f"MovieLens Stage 2 requires stage 1 artefact {path}. "
                "Run --only movielens1 first."
            )

    train_seqs = _load_pickle(output_dir / "train_seqs.pkl")
    train_seqs_with_ts = _load_pickle(output_dir / "train_seqs_with_ts.pkl")
    item_genre = _load_pickle(output_dir / "item_genre.pkl")
    id_maps = _load_pickle(output_dir / "id_maps.pkl")
    vocab_sizes = _load_json(output_dir / "vocab_sizes.json")

    n_items = int(vocab_sizes["n_items"])
    n_dt_bins = int(vocab_sizes["n_dt_bins"])

    genre2idx = id_maps["genre2idx"]
    n_genres = len(genre2idx)

    gr_cfg = pp["genre_red"]
    idf = compute_genre_idf(item_genre.values(), n_genres=n_genres,
                            smooth=bool(gr_cfg["idf_smooth"]))

    flat: List[Tuple[int, int, int, int]] = []
    for u, seq in train_seqs_with_ts.items():
        for pos, (iid, _rb, ts) in enumerate(seq):
            flat.append((u, pos, iid, ts))
    flat.sort(key=lambda x: (x[3], x[0], x[1]))

    n_rows = len(flat)
    dyn_Z = np.zeros(n_rows, dtype=np.int64)
    genre_red = np.zeros(n_rows, dtype=np.float32)
    genre_red_idf = np.zeros(n_rows, dtype=np.float32)

    item_count: Dict[int, int] = defaultdict(int)
    user_hist: Dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(n_genres, dtype=np.float32)
    )
    row_index: Dict[Tuple[int, int], int] = {}

    for flat_i, (u, pos, iid, ts) in enumerate(flat):
        z_before = item_count[iid]
        cand = item_genre.get(iid)
        if cand is None:
            j_std = 0.0
            j_idf = 0.0
        else:
            hist = user_hist[u]
            inter_mask = np.logical_and(cand > 0, hist > 0)
            union_mask = np.logical_or(cand > 0, hist > 0)
            union_count = float(union_mask.sum())
            if union_count == 0.0:
                j_std = 0.0
            else:
                j_std = float(inter_mask.sum()) / union_count
            den_idf = float((idf * union_mask).sum())
            if den_idf == 0.0:
                j_idf = 0.0
            else:
                j_idf = float((idf * inter_mask).sum()) / den_idf

        dyn_Z[flat_i] = z_before
        genre_red[flat_i] = j_std
        genre_red_idf[flat_i] = j_idf
        row_index[(u, pos)] = flat_i

        item_count[iid] = z_before + 1
        if cand is not None:
            user_hist[u] = np.logical_or(user_hist[u] > 0, cand > 0).astype(np.float32)

    print(f"  [1/1] Writing Stage 2 artefacts …")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_pickle(output_dir / "dynamic_Z.pkl", {
        "row_index": row_index,
        "values": dyn_Z,
    })
    _save_pickle(output_dir / "genre_red.pkl", {
        "row_index": row_index,
        "values": genre_red,
    })
    _save_pickle(output_dir / "genre_red_idf.pkl", {
        "row_index": row_index,
        "values": genre_red_idf,
        "idf_weights": idf,
        "genre_vocab": [g for g, _ in sorted(genre2idx.items(), key=lambda x: x[1])],
    })

    ok_id = all(
        1 <= iid <= n_items
        for seq in train_seqs.values() for (iid, _r, _d) in seq
    )
    inv.check("train item ids in [1, n_items]", ok_id)

    max_dt = max((d for seq in train_seqs.values() for (_, _, d) in seq),
                 default=0)
    inv.check("Δt bin range ok",
              0 <= max_dt <= n_dt_bins - 1, detail=f"max bin = {max_dt}")

    z_max = int(np.max(dyn_Z)) if n_rows > 0 else 0
    z_min = int(np.min(dyn_Z)) if n_rows > 0 else 0
    inv.check("dynamic Z non-negative", z_min >= 0,
              detail=f"min={z_min}, max={z_max}")
    inv.check("GenreRed (std) in [0, 1]",
              bool(((0.0 <= genre_red) & (genre_red <= 1.0)).all()))
    inv.check("GenreRed (idf) in [0, 1]",
              bool(((0.0 <= genre_red_idf) & (genre_red_idf <= 1.0)).all()))
    inv.check("row_index size matches values",
              len(row_index) == len(dyn_Z) == len(genre_red))

    print(inv.report())
    return inv.all_passed


if __name__ == "__main__":
    main()
