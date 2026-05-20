# CEINN — Causal Economics-Informed Neural Networks for Sequential Recommendation

A reference implementation of CEINN: a sequential recommender that combines
a pure-ID Transformer backbone, an IPS/adversarial causal deconfounder, and
a neuralised discrete choice model with explicit economic costs.

> **Current status: Phase 1 + Phase 2 complete.**
> Raw-data validation and feature engineering both run end-to-end.
> Phases 3–8 will land in subsequent commits; the existing entry points
> and directory layout are forward-compatible with those phases.

---

## Quickstart

```bash
# 1) Create an isolated environment.
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies. Phase 1 only needs PyYAML; Phase 2 adds numpy.
pip install -r requirements.txt   # or: pip install pyyaml numpy

# 3) Drop the raw files into data/raw/ (see data/raw/README.md for layout).

# 4) Phase 1: validate the raw files against EDA baselines.
python preprocess.py --phase 1

# 5) Phase 2: build the processed tensors.
python preprocess.py --phase 2
# Restrict to a single dataset if iterating:
python preprocess.py --phase 2 --only amazon
python preprocess.py --phase 2 --only movielens
```

A successful Phase 2 run writes pickles and a `vocab_sizes.json` under
`data/processed/<dataset>/`. The process exits with status 0 iff every
invariant defined in the per-dataset checker passes.

---

## Phase 2 — what is being produced

### Amazon Beauty (`data/processed/amazon_beauty/`)

| File                 | Format        | Contents                                             |
|----------------------|---------------|------------------------------------------------------|
| `train_seqs.pkl`     | dict          | `{user_idx: [(item_idx, rating_bin, dt_bin), ...]}`  |
| `val_seqs.pkl`       | dict          | `{user_idx: (item_idx, rating_bin, dt_bin)}`         |
| `test_seqs.pkl`      | dict          | `{user_idx: (item_idx, rating_bin, dt_bin)}`         |
| `item_meta.pkl`      | dict          | `{item_idx: {cat, brand, log_price, Z}}`             |
| `id_maps.pkl`        | dict-of-dicts | user2idx / item2idx / cat2idx / brand2idx + inverses |
| `bucket_edges.pkl`   | dict          | Δt and salesRank edges (for reproducible inference)  |
| `vocab_sizes.json`   | JSON          | flat dict of all vocabulary sizes and PAD/`max_seq_len` |

### MovieLens 10M (`data/processed/movielens_10m/`)

| File                  | Format        | Contents                                             |
|-----------------------|---------------|------------------------------------------------------|
| `train_seqs.pkl`      | dict          | `{user_idx: [(item_idx, rating_bin, dt_bin), ...]}`  |
| `val_seqs.pkl`        | dict          | `{user_idx: (item_idx, rating_bin, dt_bin)}`         |
| `test_seqs.pkl`       | dict          | `{user_idx: (item_idx, rating_bin, dt_bin)}`         |
| `item_genre.pkl`      | dict          | `{item_idx: 19-dim float32 binary vector}`           |
| `dynamic_Z.pkl`       | dict          | `{"row_index": {(u, pos): k}, "values": int64 array}` |
| `genre_red.pkl`       | dict          | standard Jaccard, parallel array aligned to `row_index` |
| `genre_red_idf.pkl`   | dict          | IDF-weighted Jaccard variant + IDF weights + vocab   |
| `id_maps.pkl`         | dict-of-dicts | user2idx / item2idx / genre2idx + inverses           |
| `bucket_edges.pkl`    | dict          | Δt edges                                              |
| `vocab_sizes.json`    | JSON          | flat dict of all vocabulary sizes                    |

> **Why "parallel arrays" instead of a dict for dynamic_Z / genre_red?**
> ML10M has ~10 M training rows. A Python `{(uid, iid, ts): float}` dict
> would carry over 600 MB of pure object-header overhead. A `np.ndarray`
> indexed via a small `row_index` keeps the same O(1) lookup at <100 MB.

### Phase 2 design decisions (worth re-reading before changing anything)

| Decision                                                    | Where to find it                                  |
|-------------------------------------------------------------|---------------------------------------------------|
| Amazon: use `Beauty_5.json` directly (no re-filtering)      | `configs/amazon_beauty.yaml` § `preprocess.five_core` |
| Amazon: same-day timestamp ties → file-order tie-breaker    | `configs/amazon_beauty.yaml` § `preprocess.split` |
| Amazon: one primary leaf per item; rare leaves → UNK        | `configs/amazon_beauty.yaml` § `preprocess.category` |
| Δt bucketing: fit on TRAIN Δt only; bin 0 = PAD             | both YAMLs § `preprocess.dt_bucketing`            |
| MovieLens: 60-second session collapse → intra-session Δt = PAD | `configs/movielens_10M.yaml` § `preprocess.session` |
| MovieLens: train restricted to year < 2009                  | `configs/movielens_10M.yaml` § `preprocess.exclude_year_from_train` |
| Dynamic Z & GenreRed: counted strictly BEFORE t (no leakage)| see the sweep in `preprocess.py:preprocess_movielens_10m` |
| IDF on the full 5-core movie corpus (not just train)        | `compute_genre_idf` in `utils/math_utils.py`      |

---

## Phase 1 — what is being validated

The validator cross-checks the on-disk files against the baselines fixed
by the EDA reports. **No statistic, however obvious, is assumed correct
until the check fires.** A failure means *the reader logic is wrong*, not
that the data has subtly drifted.

### Amazon Beauty (14 checks)

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

Tolerances live in `configs/<dataset>.yaml` under the `tolerance:` key.

---

## Repository layout

```
ceinn/
├── configs/
│   ├── amazon_beauty.yaml      # raw paths + Phase-1 baselines + Phase-2 hyper-params
│   └── movielens_10M.yaml
├── data/
│   ├── raw/                    # source files; not in VCS
│   └── processed/              # tensorised outputs from Phase 2
├── data_loaders/
│   ├── __init__.py
│   ├── amazon_beauty_loader.py # eager loader for Phase-2 artefacts
│   └── movieslens_10M_loader.py
├── utils/
│   ├── __init__.py
│   ├── data_io.py              # low-level, pure-function raw readers
│   └── math_utils.py           # log-quantile bucketing, Jaccard, IDF
├── preprocess.py               # Phase-1 validation + Phase-2 feature engineering
├── requirements.txt
├── .gitignore
└── README.md
```

`utils/data_io.py` and `utils/math_utils.py` are pure modules (no I/O
beyond what they declare); `preprocess.py` owns ALL filesystem writes.
The `data_loaders/` package is framework-agnostic — Phase 6 will wrap
the dataclasses in `torch.utils.data.Dataset` without touching the
loader code.

---

## Quality gates

* **Phase 1**: every numeric statistic from the EDA report must be
  reproducible from `data/raw/` within tolerance, or the process exits
  with status 1.
* **Phase 2**: per-dataset invariants (id range, Δt bin range, Z bin
  range, dynamic-Z per-item monotonicity, GenreRed ∈ [0, 1], vocabulary
  size consistency, item-meta coverage) must all pass.

If any invariant fails, the recovery procedure is:

1. **Do not loosen the check silently.** Re-derive the statistic from
   the plan / EDA / first principles.
2. If the cause is a reader or preprocessing bug, fix it in
   `utils/data_io.py`, `utils/math_utils.py`, or `preprocess.py`.
3. If the difference is genuinely a benign interpretation choice,
   document the choice in the YAML config and adjust the tolerance
   field there — never the comparison call.

---

## Roadmap

| Phase | Status | Module entry                                            |
|-------|--------|---------------------------------------------------------|
| 1 — raw data validation                  | **done**     | `preprocess.py --phase 1`            |
| 2 — preprocessing & feature engineering  | **done**     | `preprocess.py --phase 2`            |
| 3 — utility library & evaluation infra   | pending      | `utils/losses.py`, `utils/metrics.py`, `evaluate.py` |
| 4 — CEINN model implementation           | pending      | `models/ceinn.py`                    |
| 5 — baseline implementations             | pending      | `models/baselines/`                  |
| 6 — CEINN training & hyperparameter sweep | pending     | `train.py`                           |
| 7 — ablation studies                     | pending      | `train.py --config <ablation>.yaml`  |
| 8 — result aggregation & paper writing   | pending      | `results/`, `README.md` updates      |
