#!/usr/bin/env bash
# =============================================================================
# Phase-5 baseline sweep driver — train all baselines × seeds × datasets.
#
# Plan §5.3: 5 seeds per baseline per dataset. With 3 baselines and 2
# datasets this is 30 runs. PopRec runs are essentially instant (no
# training), BPR-MF and GRU4Rec scale with dataset size.
#
# Usage
# -----
#     # All baselines, all datasets, default 5 seeds
#     bash scripts/run_baselines.sh
#
#     # Single baseline, single dataset, single seed (dry-run style)
#     BASELINES="poprec" DATASETS="amazon_beauty" SEEDS="0" \
#         bash scripts/run_baselines.sh
#
#     # Skip Amazon
#     DATASETS="movielens_10m" bash scripts/run_baselines.sh
#
# Environment variables:
#     BASELINES   space-separated list  (default: "poprec bpr_mf gru4rec")
#     DATASETS    space-separated list  (default: "amazon_beauty movielens_10m")
#     SEEDS       space-separated list  (default: "0 1 2 3 4")
#     DEVICE      torch device          (default: auto)
#     EXTRA_ARGS  extra args to train_baseline.py
#
# The script exits on first failure so a broken setup doesn't accumulate
# half-finished runs. Use `set +e` if you'd rather press on.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASELINES="${BASELINES:-poprec bpr_mf gru4rec}"
DATASETS="${DATASETS:-amazon_beauty movielens_10m}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE_FLAG=""
if [[ -n "${DEVICE:-}" ]]; then
    DEVICE_FLAG="--device ${DEVICE}"
fi

# Map dataset folder name → config YAML.
config_for_dataset() {
    case "$1" in
        amazon_beauty)  echo "configs/amazon_beauty.yaml" ;;
        movielens_10m)  echo "configs/movielens_10M.yaml" ;;
        *) echo "ERROR: unknown dataset '$1'" >&2; exit 1 ;;
    esac
}

echo "=============================================================="
echo "Phase-5 baseline sweep"
echo "  baselines: ${BASELINES}"
echo "  datasets:  ${DATASETS}"
echo "  seeds:     ${SEEDS}"
echo "=============================================================="

start_t=$(date +%s)
n_runs=0
for ds in ${DATASETS}; do
    cfg=$(config_for_dataset "${ds}")
    if [[ ! -f "${cfg}" ]]; then
        echo "Missing dataset config: ${cfg}" >&2
        exit 1
    fi
    for bs in ${BASELINES}; do
        for seed in ${SEEDS}; do
            n_runs=$((n_runs + 1))
            echo ""
            echo ">>> [${n_runs}] baseline=${bs} dataset=${ds} seed=${seed}"
            python train_baseline.py \
                --config "${cfg}" \
                --baseline "${bs}" \
                --seed "${seed}" \
                --run-name "seed${seed}" \
                ${DEVICE_FLAG} \
                ${EXTRA_ARGS:-}
        done
    done
done

end_t=$(date +%s)
echo ""
echo "=============================================================="
echo "Sweep complete: ${n_runs} runs in $((end_t - start_t))s"
echo "=============================================================="

echo ""
echo ">>> Aggregating results …"
python scripts/aggregate_baselines.py \
    --runs-root runs \
    --output results/baselines_summary.csv

echo ""
echo "Done. Summary: results/baselines_summary.csv"
