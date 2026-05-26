# `eda/` — Pre-modeling Empirical Audits for CEINN

This directory contains two stand-alone EDA scripts whose **sole purpose is
to empirically validate the design assumptions of the CEINN framework**
before any model code is written. Every block of code maps to a numbered
claim in the research paper.

```
eda/
├── eda_amazon_beauty.py
├── eda_movielens_10m.py
├── outputs/
│   ├── amazon_beauty/
│   └── movielens_10m/
└── README.md   (this file)
```

## Why these EDAs are not generic

A textbook EDA reports density, sparsity, popularity Lorenz, and stops.
That tells us nothing useful about whether the **causal deconfounder** or
the **economic utility function** in CEINN will work. The questions this
directory is designed to answer are sharper:

| Module                          | Design assumption                                    | EDA pre-test                                                      |
|---------------------------------|------------------------------------------------------|-------------------------------------------------------------------|
| 3.1 Sequential Backbone         | `E_r(r_k)` is informative                            | rating entropy; share-at-5; collapse risk                         |
| 3.1 Sequential Backbone         | `E_t(Δt)` needs log bucketization                    | log10 span of Δt; mass at <1min / same-day                        |
| 3.2 Causal Deconfounder         | Z is constructible from data                         | salesRank coverage (Amazon); rolling-popularity dynamics (ML10M)  |
| 3.2 Causal Deconfounder         | IPS clipping (3.2.7) is *required*                   | empirical 1/p_i tail quantiles                                    |
| 3.2 Causal Deconfounder         | Z-bucket count balance for adversarial D             | min/max items per 10-quantile bucket                              |
| 3.3 Explicit Cost (Amazon)      | α_5 (price × category) carries signal                | F-proxy (between/within) of log(price) by leaf category           |
| 3.3 Explicit Cost (Amazon)      | α_1 and α_4 are not redundant                        | corr(log price, log salesRank)                                    |
| 3.3 Implicit Cost (ML10M)       | GenreRed → satiation is a real signal                | Jaccard-quartile trend test on Δt and next-rating                 |
| 3.3 Implicit Cost (ML10M)       | Tags are unusable                                    | tag coverage on rating-side; singleton-share of tag vocab         |

If any of these pre-tests fails, the corresponding term in `losses.py` /
`economics_utility.py` will not work as advertised; better to find that
out here than after a 12-hour training run.

## Running

From the project root:

```bash
python -m eda.eda_amazon_beauty
python -m eda.eda_movielens_10m
```

Each script reads from `data/raw/...` and writes plots + CSVs + a single
machine-readable `summary.json` into `eda/outputs/<dataset>/`. The
`summary.json` is the canonical artefact that downstream `preprocess.py`
and `train.py` may consume to set thresholds (e.g. IPS clip τ).

Runtime expectations on a 16 GB laptop:
- Amazon Beauty: ~30 seconds (Beauty_5 is small; meta is the only big file)
- MovieLens 10M: ~3 to 6 minutes (10M ratings + satiation pre-test;
  the latter is bounded by a 1M-pair cap)

## Reading the outputs

### Amazon Beauty

`3_1_rating_dist.png` — if share-at-5 exceeds ~70% and entropy drops below
~1.0 nats, `E_r` is largely collapsed and the rating embedding should
either be removed or replaced by a positive/negative-binary signal.

`3_1_delta_t.png` — Amazon timestamps are day-granular, so a left-edge
spike at 1 day is expected. The log span across multiple decades is the
direct justification for log bucketization in `E_t`.

`3_2_salesrank_log.png` — left panel: raw rank is unusable (extreme tail).
Right panel: log(rank+1) should be roughly bell-shaped, so 10-quantile
bucketization yields balanced Z. Check `3_2_z_bucket_counts.csv` — if any
bucket has <100 items, the adversarial discriminator in 3.2.8 will be
unreliable on that bucket.

`3_2_ips_blowup.png` — distribution of log10(1/p_i). The p99 value in
`summary.json` (`inv_p_p99`) is the empirical answer to the question
"what value of τ in 3.2.7 actually clips off the heavy tail?". A common
starting point is `τ = inv_p_p99`.

`3_3_price.png` — same story as salesRank: raw is unusable, log is
amenable to a single coefficient α_1.

`3_3_price_by_cat.png` — boxplots of log(price) by leaf category. The
**F-proxy** in `summary.json` (`price_x_cat_F_proxy`) quantifies whether
the same nominal price means different things in different categories.
A value much greater than 1 is the green light to include α_5.

`3_3_price_vs_rank.png` — a low |correlation| between log(price) and
log(salesRank) means α_1 and α_4 capture distinct economic signals.
A high correlation (>0.6 in magnitude) is a warning that the two terms
will fight each other and one should be removed.

### MovieLens 10M

`3_1_rating_dist.png` — the half-star usage share tells you whether
`E_r` needs 10 bins (0.5..5.0) or can be coarsened to 5. If half-star
share <5%, coarsening is safe.

`3_2_popularity_dynamics.png` — sample trajectories of the top-100
items' within-window ranks. A flat cloud of horizontal lines means
popularity is essentially static and `Z_i(t)` adds nothing over a static
Z. A churning cloud justifies the dynamic-Z design in 3.2.4. The summary
field `rank_volatility_top100` is the headline number.

`3_3_satiation_pretest.png` — **this is the most important plot in the
ML10M EDA**. Left panel: as the next item's genre Jaccard with the
recent window rises, the time-to-consumption Δt should rise too. Right
panel: ratings on high-Jaccard items should drop. Both monotonic trends
are required for the GenreRed term in 3.3.4 to carry behavioral signal.
`summary.json` exposes two boolean fields (`satiation_monotonic_dt`,
`satiation_monotonic_rating`); both `true` is the green light.

The satiation pre-test is intentionally cheap — it uses a fixed K=5
recent-window bag and computes Jaccard over set-union, exactly the
operationalization in 3.3.4. The intent is *not* to estimate the size
of the effect but to confirm its sign.

Tag block: if `tag_movie_coverage < 0.3` and `tag_singleton_share > 0.5`,
the conclusion in 3.3.4 ("tags too sparse and noisy → exclude") is
empirically justified.

## What this EDA does *not* do

- **No model training.** No SASRec, no propensity estimator, no
  adversarial discriminator. All numbers here are purely descriptive.
- **No future-leakage handling.** The popularity statistics in 3.2 are
  global; the actual `Z_i(t)` construction (rolling popularity up to
  time t) lives in `data_loaders/movieslens_10m_loader.py`. The EDA's
  rolling-window plot in 3.2.c is a separate diagnostic, not the
  training-time confounder.
- **No multimodal text features.** Per the paper's 3.3.3 design choice,
  brand and category are treated as one-hot lookup keys. The EDA reports
  cardinalities so you can size the lookup tables, nothing more.

## Dependencies

```
numpy
pandas
matplotlib
```

No seaborn, no scikit-learn. Both scripts run on Python 3.9+.
