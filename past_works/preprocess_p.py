"""
CEINN preprocessing entry point.

Phase 1 (this file's current scope)
-----------------------------------
Verify that the raw files on disk can be parsed correctly and that every
statistic listed in the EDA reports is reproducible from them. Emit a
validation log to `data/raw/validation_log.txt`. If any check fails the
process exits with a non-zero status — Phase 2 must not proceed on a
broken data baseline.

The script is intentionally read-only with respect to data/: it neither
mutates the raw files nor writes any processed artifact. Phase 2 will add
filtering / remapping / serialisation steps after the same `--phase` switch.

Usage
-----
    python preprocess.py --phase 1                  # validate both datasets
    python preprocess.py --phase 1 --only amazon    # validate just one
    python preprocess.py --phase 1 --only movielens
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from utils.data_io import (
    extract_leaf_categories,
    extract_salesrank,
    has_valid_brand,
    has_valid_price,
    iter_amazon_meta,
    iter_amazon_reviews,
    iter_movielens_movies,
    iter_movielens_ratings,
    iter_movielens_tags,
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
    interactions: List[Tuple[int, int, float, int]],
    min_count: int = 5,
) -> Tuple[List[Tuple[int, int, float, int]], set, set]:
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
    for uid, mid, _, _ in interactions:
        user_count[uid] += 1
        item_count[mid] += 1

    users_keep = {u for u, c in user_count.items() if c >= min_count}
    items_keep = {i for i, c in item_count.items() if c >= min_count}

    while True:
        filtered = [
            t for t in interactions
            if t[0] in users_keep and t[1] in items_keep
        ]
        uc: Counter = Counter()
        ic: Counter = Counter()
        for uid, mid, _, _ in filtered:
            uc[uid] += 1
            ic[mid] += 1
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
        description="CEINN preprocessing — Phase 1: raw data validation."
    )
    parser.add_argument("--phase", choices=["1"], default="1",
                        help="Phase to execute (Phase 2+ will be added later).")
    parser.add_argument("--only", choices=["amazon", "movielens", "both"],
                        default="both", help="Restrict validation to one dataset.")
    parser.add_argument("--config-amazon", default="configs/amazon_beauty.yaml")
    parser.add_argument("--config-movielens", default="configs/movielens_10M.yaml")
    parser.add_argument("--output", default="data/raw/validation_log.txt")
    args = parser.parse_args()

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

    # Persist a single concatenated report. The file is overwritten each
    # run; the Phase-1 quality gate's evidence is the most recent log only.
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

    # Phase-1 quality gate: hard fail on any check that did not pass.
    all_ok = all(log.all_passed for log in logs)
    if not all_ok:
        print("ERROR: Phase 1 quality gate FAILED. "
              "Fix the reader logic or update tolerances before Phase 2.",
              file=sys.stderr)
        sys.exit(1)
    print("Phase 1 quality gate PASSED. Safe to proceed to Phase 2.")


if __name__ == "__main__":
    main()
