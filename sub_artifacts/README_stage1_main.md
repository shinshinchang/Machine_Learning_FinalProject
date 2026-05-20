# CEINN — Causal Economics-Informed Neural Networks for Sequential Recommendation

A reference implementation of CEINN: a sequential recommender that combines
a pure-ID Transformer backbone, an IPS/adversarial causal deconfounder, and
a neuralised discrete choice model with explicit economic costs.

> **Current status: Phase 1 (raw data validation).**
> The `preprocess.py` entry point currently implements the raw-data
> validation pass only. Phases 2–8 (preprocessing, model implementation,
> training, ablation, write-up) will land in subsequent commits and slot
> into the same skeleton without restructuring.

---

## Quickstart — Phase 1

```bash
# 1) Create an isolated environment.
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies (Phase 1 only needs PyYAML; the rest is pre-listed
#    for forward reference).
pip install PyYAML

# 3) Drop the raw files into data/raw/ (see data/raw/README.md for layout).

# 4) Run the validator.
python preprocess.py --phase 1                  # both datasets
python preprocess.py --phase 1 --only amazon    # only Amazon Beauty
python preprocess.py --phase 1 --only movielens # only MovieLens 10M
```

A successful run prints a per-dataset summary and writes
`data/raw/validation_log.txt`. The process exits with status 0 iff every
check listed in the EDA reports is reproducible from the files on disk.

---

## Phase 1 — what is being validated

The validator cross-checks the on-disk files against the baselines fixed
by the EDA reports. **No statistic, however obvious, is assumed correct
until the check fires.** A failure means *the reader logic is wrong*, not
that the data has subtly drifted.

### Amazon Beauty (15 checks)

| Category   | Checked statistics                                                                  |
|------------|-------------------------------------------------------------------------------------|
| Structure  | `n_interactions`, `n_users`, `n_items`, `n_leaf_categories`                         |
| Ratings    | `rating_share_5`                                                                    |
| Coverages  | `price`, `salesrank_any`, `salesrank_beauty`, `brand`                               |
| Timeline   | epoch-unit sanity, UTC start / end dates, span-in-days                              |

### MovieLens 10M (15 checks)

| Category   | Checked statistics                                                                  |
|------------|-------------------------------------------------------------------------------------|
| Structure  | `n_interactions_raw`, `n_interactions_5core`, `n_users`, `n_items_5core`, `n_items_raw` |
| Ratings    | `rating_half_star_share`, `rating_share_5`, `rating_entropy_bits`                   |
| Genres     | `n_genres` (excluding the "(no genres listed)" sentinel)                             |
| Tags       | `n_tag_events`, `n_tag_vocab`, `tag_singleton_share`                                |
| Timeline   | UTC start / end dates, span-in-days                                                  |

Tolerances are *not* hard-coded — every numeric bound lives in the
corresponding `configs/<dataset>.yaml` under the `tolerance:` key. If a
check fails but you believe the data is correct, loosen the tolerance
there, never inside the comparison call.

---

## Repository layout

```
ceinn/
├── configs/
│   ├── amazon_beauty.yaml      # Phase 1: raw paths + EDA baselines + tolerances
│   └── movielens_10M.yaml
├── data/
│   ├── raw/                    # source files; not in VCS
│   └── processed/              # tensorised outputs (populated in Phase 2)
├── utils/
│   ├── __init__.py
│   └── data_io.py              # low-level, pure-function raw readers
├── preprocess.py               # Phase 1 entry point (validation only, for now)
├── requirements.txt
├── .gitignore
└── README.md
```

A small departure from the original project plan: `utils/data_io.py` was
added so that `preprocess.py` can stay focused on orchestration, and so
Phase 2's preprocessing reuses the *same* parsers Phase 1 validated.
Touching the parsers in only one place eliminates the risk of two
near-duplicate readers silently disagreeing on edge cases such as the
`{'asin': ...}` Python-literal format in older Amazon dumps.

---

## Phase 1 quality gate

The plan document specifies: *no preprocessing step shall proceed while a
known data inconsistency remains.* This is enforced operationally:

* `preprocess.py` exits with a non-zero status as soon as any check fails.
* `data/raw/validation_log.txt` records every check's expected value,
  observed value, tolerance, and pass/fail flag — both for human reading
  and as evidence in the paper's appendix on reproducibility.

If a check fails, the recovery procedure is:

1. **Do not loosen the tolerance silently.** Open the relevant section of
   `EDA_report_AmazonBeauty.md` or `EDA_report_Movielens10M.md` and
   recompute the statistic from first principles.
2. If the EDA and our reader disagree, fix the reader in
   `utils/data_io.py` (or the orchestrator in `preprocess.py`). Add a
   regression test under a future `tests/` directory.
3. Only after step 2, if the difference is genuinely an artefact of a
   tolerable interpretation choice (UTC vs. local-tz dates, case folding
   of tags, etc.), document the choice in the YAML config's comments and
   loosen the bound there.

---

## Roadmap

| Phase | Status | Module entry                                            |
|-------|--------|---------------------------------------------------------|
| 1 — raw data validation                  | **done**     | `preprocess.py --phase 1`            |
| 2 — preprocessing & feature engineering  | pending      | `preprocess.py --phase 2`            |
| 3 — utility library & evaluation infra   | pending      | `utils/`, `evaluate.py`              |
| 4 — CEINN model implementation           | pending      | `models/ceinn.py`                    |
| 5 — baseline implementations             | pending      | `models/baselines/`                  |
| 6 — CEINN training & hyperparameter sweep | pending     | `train.py`                           |
| 7 — ablation studies                     | pending      | `train.py --config <ablation>.yaml`  |
| 8 — result aggregation & paper writing   | pending      | `results/`, `README.md` updates      |
