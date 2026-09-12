# Data and ML Pipeline

Status: **Phase 4 deliverable — complete**
Last updated: 2026-09-12

How raw behaviour becomes a trained, evaluated, registered model.

```bash
python ml/pipelines/train.py --mlflow          # full run, ~4 minutes
python ml/pipelines/retrain.py --check-drift   # drift assessment only
python ml/pipelines/retrain.py                 # drift-triggered retrain + gate
```

---

## 1. The pipeline

```
extract      load events, catalogue, orders          0.7 s
   ▼
validate     data contracts, fail loud and early     0.1 s
   ▼
split        temporal 70/15/15, leakage-guarded      0.2 s
   ▼
calibrate    derive implicit weights from train      0.3 s
   ▼
features     user / product / interaction            7.4 s
   ▼
train        6 models + 3 rankers                   44   s
   ▼
evaluate     13 model variants on the test fold    177   s
   ▼
compare      paired bootstrap vs the baseline        2   s
   ▼
register     MLflow run + model version              3   s
   ▼
gate         promote only if it beats the incumbent
```

Each stage is a method on `TrainingPipeline`, so the pipeline can be resumed,
tested piecewise, and called from the retraining job without duplicating the
wiring.

---

## 2. Extract

Reads through an abstract `DataSource`, never from Postgres directly:

| Implementation | Used for |
| --- | --- |
| `ParquetSource` | The generated dataset — training, offline evaluation, CI |
| `PostgresSource` | The live database — batch jobs against production data |

Column names are aliased so the two produce identical shapes. Where they
disagree, the fix belongs in the loader, never in a model.

**This split is why the whole ML stack trains, evaluates and regression-tests
with no database at all** — which keeps CI fast and makes the package reusable
from a notebook.

---

## 3. Validate

Cheap contract checks before anything is trained:

| Check | Fails when |
| --- | --- |
| Row count | Fewer than 1,000 interactions |
| Null identifiers | Any null `user_id` or `product_id` |
| Referential integrity | An interaction references an unknown product |

This converts the most common silent failure — an upstream change that empties
a column — into a loud one at the *top* of the run, instead of a mysterious
metric regression at the bottom.

---

## 4. Temporal split (ADR-007)

```
[────── train 70% ──────][── validation 15% ──][── test 15% ──]
   238,153 rows              51,033 rows          51,034 rows
   → learn                   → tune               → report once
```

Boundaries are quantiles of the **interaction timeline**, not the calendar.
Calendar thirds would put very different volumes in each fold whenever traffic
is seasonal, making fold-to-fold comparisons partly an artefact of volume.

`assert_no_leakage` runs on every split and raises `LeakageError` if the folds
overlap. It is tested — it actually fires.

**Why not a random split.** It lets the model see a user's future while
predicting their past: a recommender that has already seen you buy the shoes
trivially "predicts" you will view them. It also leaks globally, because item
popularity computed over the full period encodes the test window.

---

## 5. Feature engineering

### The `as_of` contract

Every builder takes an `as_of` timestamp and reads **only** events strictly
before it. The filter happens once at the top, and `assert_features_respect_cutoff`
verifies it — so a future edit that reaches around the filter fails a test
rather than silently inflating every offline metric.

This is the mechanical defence against the leakage that produces beautiful
numbers and no error.

### User features (27)

| Group | Features |
| --- | --- |
| Volume | total events, views, clicks, carts, purchases, distinct products/categories/brands, sessions, active days |
| Monetary | total spending, average order value, average and median purchase price, mean price percentile |
| Rates | view→cart, cart→purchase, view→purchase, purchase frequency in days |
| Recency | days since last event, last purchase, first seen; events per active day |
| Behaviour | price sensitivity, category concentration, brand concentration |
| Value | customer lifetime value |

Two worth explaining:

**`price_sensitivity`** is defined on the *realised percentile within a
category*, not on absolute price. A user whose purchases sit at the 20th
percentile is price-sensitive; one at the 80th is not. Absolute price would make
this an artefact of which department someone shops in — a £30 book and a £30
laptop accessory mean opposite things.

**Concentration** uses the Herfindahl index (Σ share²) rather than entropy: it
is bounded, needs no normalisation by the number of categories, and reads
directly as "the probability two random interactions fall in the same category".

**Nothing here reads `users.segment`.** That is ground truth for the Phase 2
diagnostic; training on it would be training on the answer key.

### Product features (25)

Volume, rates and derived scores, all bounded by `as_of`. Popularity features
are the most dangerous kind of leakage in a recommender — "how popular is this?"
computed over the full period encodes the test window into every training row,
and the resulting model looks excellent offline and mediocre online.

**Bayesian shrinkage** on every rate:

```
smoothed = (observed_count + prior × global_rate) / (observed + prior)     prior = 20
```

Without it, a product viewed twice and bought once shows a 50% conversion rate
and tops every ranking that uses the feature. The classic small-sample trap.

A real finding from building this: `quality_score` initially had almost no
variance (std 0.011) because products carried under one rating each and the
prior swamped everything. The fix was in the *data*, not the feature — real
stores accumulate ratings before the observation window, so the generator now
seeds pre-window rating history (mean 15 per product). Variance is now 0.106.

### Interaction features (15)

Per (user, product): view/cart/purchase counts, days since last interaction,
category and brand affinity, price distance, content similarity, collaborative
score.

Assembled **vectorised over the whole candidate set**, not per candidate. With
300 candidates and a p99 budget of 150 ms, a Python loop doing 300 dictionary
lookups is the difference between comfortably inside the budget and comfortably
outside it.

### Affinity maps

`{user_id: {category_id: share}}`, built from the **weighted** interaction frame
rather than raw counts — so a purchase moves affinity far more than a view,
which is the whole point of calibrating the weights.

Stored as JSONB maps rather than separate rows: a user has a handful of
meaningful affinities out of hundreds of categories, so a sparse map is smaller
and read in a single fetch on the hot path.

---

## 6. Training/serving skew (NFR-06)

The failure being designed against is the common one: an offline notebook
computes `avg_order_value` one way, the API computes it slightly differently,
and the model degrades in production with no error anywhere.

The defence is structural. `ml/recsys` is an **installable package imported by
both** the training pipelines and the FastAPI service:

```
ml/recsys/features/  ──┬──> ml/pipelines/train.py   (offline)
                       └──> backend/app/...         (online)
```

Feature engineering exists **once**. An integration test asserts that the
ranker's `feature_columns` match the dataset builder's exactly — including
order, because the model indexes positionally and a silent reordering would feed
it scrambled inputs with no error.

`recsys` must never import FastAPI, SQLAlchemy models, or anything web-shaped.
It takes data in and returns data out, so it is testable with plain fixtures and
reusable from a batch job, a notebook or the API.

---

## 7. Feature storage

| Store | Holds | Read by |
| --- | --- | --- |
| Postgres `user_features` / `product_features` | Current materialised snapshot | Serving (fallback) |
| Redis `feat:` | Same values, hot | Serving (primary) |
| Parquet in `ml/artifacts/` | Training snapshot | The engine at start-up |

**Read this before using the Postgres tables in training.** They hold the
*current* snapshot. They are not a time-travel store, and training must never
read them for a historical label — that would attach features computed today to
an interaction from three months ago. Training recomputes with `as_of`.

`as_of_ts` is recorded on every row so a serving-time vector can be reconciled
against the offline one.

---

## 8. Model training

| Model | Time | Notes |
| --- | --- | --- |
| Popularity | 0.1 s | Global + per category |
| Trending | 0.1 s | Time-decayed |
| Content | 1.4 s | TF-IDF → SVD, 491 dims, 85.4% variance |
| ALS | 0.7 s | 7,272 × 4,773, 64 factors, 0.35% density |
| BPR | 0.7 s | Comparison only |
| Co-visitation | 1.5 s | 26,877 sessions → 269,469 pairs |
| Frequently bought | 0.1 s | 7,898 baskets |
| **LambdaRank** | **40 s** | 67,215 rows, 1,596 queries, 83 features |

Everything is seeded. Same commit + same data snapshot → same metrics.

---

## 9. Evaluation

Covered in full in [`evaluation.md`](evaluation.md). The pipeline evaluates
**13 model variants** on the test fold, reports ten metrics each plus
segmentation by user history depth, and runs a paired bootstrap against the
strongest baseline.

The headline: the two-stage ranker wins at NDCG@10 0.0582, and the +4% lift over
category-popularity is **not statistically significant** (p = 0.297). That is
reported rather than buried.

---

## 10. MLflow (ADR-010)

Every run logs:

- All configuration as parameters (`RecsysConfig.to_dict()`)
- Every metric for every model and segment — **1,255 metrics** in the last run
- The calibrated weight table, evaluation CSVs, config JSON as artefacts
- A **dataset fingerprint** tag: a hash of the split boundaries and row counts

That fingerprint answers the question that otherwise becomes archaeology: two
runs with identical parameters and different metrics are explainable ("the data
changed") rather than mysterious.

Backend is SQLite (`ml/mlflow.db`). MLflow 3 put the plain file store into
maintenance mode and raises on it; SQLite is the documented local replacement
and additionally supports the model registry, which the file store never did.

**MLflow is deliberately not on the serving path.** The API resolves models from
a local artefact directory and the `model_versions` table. MLflow being down can
prevent a model *change*, never a recommendation.

---

## 11. The promotion gate

A retrained model does not become production because it finished training.

| Criterion | Requirement | Why |
| --- | --- | --- |
| `ndcg@10` | > incumbent + 0.001 | A coin-flip difference must not trigger a deployment, or the production model random-walks with every retrain |
| `hit_rate@10` | ≥ incumbent | Closest metric to what a user experiences |
| `coverage` | ≥ incumbent − 0.05, floor 0.05 | **The degenerate solution** — recommend the same popular items to everyone — wins on NDCG and collapses coverage |
| `personalisation` | advisory | Reported, does not block |

A rejected candidate is **kept** under `ml/artifacts.rejected/` with the full
decision recorded. Discarding it would make "why was this rejected?"
unanswerable.

Artefacts are swapped atomically — write to staging, move the old aside, move
staging in — so a crash mid-write cannot leave the serving path pointing at a
half-written model.

---

## 12. Drift detection (FR-18)

Two complementary statistics:

| Method | Answers | Thresholds |
| --- | --- | --- |
| **PSI** | How much has the distribution moved? | < 0.10 noise · 0.10–0.25 warning · > 0.25 alert |
| **KS** | Is the shape different? | p < 0.01 |

PSI drives alerting, KS corroborates. Relying on the p-value alone would be a
mistake: with hundreds of thousands of rows a KS test rejects the null for
effects far too small to matter — significance and importance come apart. PSI's
magnitude thresholds keep alerting tied to impact.

**Bin edges come from the reference, not the combined data.** Recomputing edges
each run would move the goalposts with the data and could report zero drift for
a distribution that had shifted wholesale.

**The reference is the training snapshot, not last week.** Comparing against a
rolling recent window would let slow drift pass unnoticed: each week looks like
the last while the cumulative distance from training grows without bound.

**Retraining needs two alerting features**, or one alert plus three warnings.
A single feature can move for a benign reason — a merchandising change, a new
category launch — and retraining on every such event means retraining
constantly.

Verified working: identical data → 0 alerts; injected shift → 3 alerts and
"retrain recommended: yes".

---

## 13. Reproducibility (NFR-05)

- One seed (`RecsysConfig.seed = 42`) flows to every random draw
- The dataset regenerates byte-identically from its own seed in ~20 s
- ALS is deterministic; BPR is not, and is comparison-only for that reason
- CI regenerates a small dataset, runs the diagnostic gate, trains, and asserts
  the best model beats the popularity baseline

A real bug this caught: session keys were generated with `uuid.uuid4()`, which
is unseeded. Two runs with the same seed produced different session keys, and
since the key appears in the event log the dataset was not reproducible. Now
derived from the seeded generator.
