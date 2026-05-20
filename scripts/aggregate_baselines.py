"""
Aggregate Phase-5 baseline test-set metrics across seeds and baselines.

Scans `runs/<dataset>/baselines/<baseline>/<run_name>/test_metrics.json`
files, groups by (baseline, dataset), and writes a summary CSV with
mean ± std over seeds (Plan §5.3).

Usage
-----
    # Default: scan runs/*/baselines/*/* under repo root.
    python scripts/aggregate_baselines.py

    # Custom roots / output path.
    python scripts/aggregate_baselines.py \\
        --runs-root runs \\
        --output results/baselines_summary.csv \\
        --datasets amazon_beauty movielens_10m

Output CSV schema
-----------------
    baseline, dataset, n_seeds,
    ndcg@5_mean, ndcg@5_std, ndcg@10_mean, ndcg@10_std, ndcg@20_mean, ndcg@20_std,
    hr@5_mean, hr@5_std, hr@10_mean, hr@10_std, hr@20_mean, hr@20_std,
    mrr_mean, mrr_std,
    head_ndcg@10_mean, head_ndcg@10_std,
    torso_ndcg@10_mean, torso_ndcg@10_std,
    tail_ndcg@10_mean, tail_ndcg@10_std,
    seeds (comma-separated)

Group columns may be NaN if the popularity-groups pickle wasn't
available at training time for some run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any, Dict, List, Optional


METRIC_KEYS = (
    "ndcg@5", "ndcg@10", "ndcg@20",
    "hr@5", "hr@10", "hr@20",
    "mrr",
)
GROUP_NAMES = ("head", "torso", "tail")


def _safe_std(values: List[float]) -> float:
    """Sample std with N>1, 0 for N=1. Returns NaN for empty input."""
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return stdev(values)


def collect_runs(runs_root: Path, datasets: Optional[List[str]] = None) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Walk `runs/<dataset>/baselines/<baseline>/<run_name>/test_metrics.json`
    and group by (baseline, dataset).

    Returns
    -------
    grouped : {(baseline, dataset): [test_metrics_dict, ...]}
    """
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    if not runs_root.exists():
        return grouped

    dataset_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    if datasets is not None:
        wanted = set(datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in wanted]

    for dset_dir in dataset_dirs:
        bsd_root = dset_dir / "baselines"
        if not bsd_root.is_dir():
            continue
        for bsl_dir in sorted(p for p in bsd_root.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in bsl_dir.iterdir() if p.is_dir()):
                metrics_path = run_dir / "test_metrics.json"
                if not metrics_path.exists():
                    continue
                with open(metrics_path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                key = (bsl_dir.name, dset_dir.name)
                grouped[key].append(rec)
    return grouped


def aggregate_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Mean ± std for the standard metric block plus group metrics.

    Each record is the JSON written by `train_baseline.py`:
        {"baseline", "dataset", "seed", "best_epoch", "best_val_ndcg10",
         "test_metrics": {...}, "group_metrics@10": {...optional...}, ...}
    """
    seeds = [int(r.get("seed", -1)) for r in records]
    out: Dict[str, Any] = {
        "n_seeds": len(records),
        "seeds": ";".join(str(s) for s in seeds),
    }

    # Corpus-level metrics.
    for k in METRIC_KEYS:
        vals = [
            float(r.get("test_metrics", {}).get(k, math.nan))
            for r in records
        ]
        vals_clean = [v for v in vals if not math.isnan(v)]
        out[f"{k}_mean"] = mean(vals_clean) if vals_clean else float("nan")
        out[f"{k}_std"] = _safe_std(vals_clean)

    # Group NDCG@10 / HR@10.
    for group in GROUP_NAMES:
        ndcg_vals: List[float] = []
        hr_vals: List[float] = []
        for r in records:
            gm = r.get("group_metrics@10", {})
            if not gm:
                continue
            g = gm.get(group, {})
            if "ndcg@k" in g and not math.isnan(g["ndcg@k"]):
                ndcg_vals.append(float(g["ndcg@k"]))
            if "hr@k" in g and not math.isnan(g["hr@k"]):
                hr_vals.append(float(g["hr@k"]))
        out[f"{group}_ndcg@10_mean"] = mean(ndcg_vals) if ndcg_vals else float("nan")
        out[f"{group}_ndcg@10_std"] = _safe_std(ndcg_vals)
        out[f"{group}_hr@10_mean"] = mean(hr_vals) if hr_vals else float("nan")
        out[f"{group}_hr@10_std"] = _safe_std(hr_vals)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--runs-root", type=Path,
        default=repo_root / "runs",
        help="Root containing per-dataset run folders. Default: ./runs",
    )
    ap.add_argument(
        "--output", type=Path,
        default=repo_root / "results" / "baselines_summary.csv",
        help="Output CSV path. Default: results/baselines_summary.csv",
    )
    ap.add_argument(
        "--datasets", nargs="*", default=None,
        help="Restrict to these dataset names (default: all).",
    )
    args = ap.parse_args()

    grouped = collect_runs(args.runs_root, datasets=args.datasets)
    if not grouped:
        print(f"No baseline runs found under {args.runs_root}/. "
              f"Did `train_baseline.py` finish at least one run?")
        return 1

    # Build the row list.
    rows: List[Dict[str, Any]] = []
    for (baseline, dataset), records in sorted(grouped.items()):
        agg = aggregate_group(records)
        row = {"baseline": baseline, "dataset": dataset, **agg}
        rows.append(row)
        print(
            f"  {baseline:<10s}  {dataset:<16s}  "
            f"n_seeds={agg['n_seeds']:>2d}  "
            f"NDCG@10={agg['ndcg@10_mean']:.4f}±{agg['ndcg@10_std']:.4f}  "
            f"HR@10={agg['hr@10_mean']:.4f}±{agg['hr@10_std']:.4f}"
        )

    # Write CSV.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["baseline", "dataset", "n_seeds"]
    for k in METRIC_KEYS:
        columns.append(f"{k}_mean")
        columns.append(f"{k}_std")
    for group in GROUP_NAMES:
        columns.append(f"{group}_ndcg@10_mean")
        columns.append(f"{group}_ndcg@10_std")
        columns.append(f"{group}_hr@10_mean")
        columns.append(f"{group}_hr@10_std")
    columns.append("seeds")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Fill missing columns with empty strings for CSV cleanliness.
            cleaned = {c: row.get(c, "") for c in columns}
            writer.writerow(cleaned)

    print(f"\nWritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
