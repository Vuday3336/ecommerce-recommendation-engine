# E-commerce Recommendation & Personalization Engine

A production-grade recommendation platform: event ingestion, feature
engineering, five recommendation strategies, two-stage retrieve-then-rank, a
React storefront and admin dashboard, MLflow model management, A/B testing,
drift detection and an automated retraining loop.

**~21,000 lines of Python · 253 passing tests · 13 model variants evaluated with
significance testing**

---

## What makes this different from a recommendation notebook

| Notebook | This |
| --- | --- |
| Random train/test split | Temporal split with an automated leakage guard that fires |
| One algorithm | 5 strategies + hybrid + learned ranker, all compared |
| `purchase = 10, view = 1` | Weights **derived** from conversion odds |
| "Accuracy" | 10 metrics, segmented by user history depth |
| "Model A wins" | Paired bootstrap — and the 4% lift is **not significant**, reported as such |
| `model.pkl` | MLflow registry with a promotion gate that rejects worse models |
| Scores in a loop | Two-stage pipeline, p95 **19.3 ms** |
| "It works" | Degradation ladder, drift detection, auto-retrain |

Three findings worth reading the docs for:

- **`alpha = 24` was a real bug.** ALS confidence must be tuned against the
  scale of the weights it multiplies, not copied from a paper. Fixing it lifted
  validation NDCG@10 by 35%. ([recommendation-engine.md §2.4](docs/recommendation-engine.md))
- **Hard-negative sampling halved accuracy.** The textbook trick broke the
  train/serve rank distribution. The investigation is more useful than a success
  would have been. ([evaluation.md §9](docs/evaluation.md))
- **Category popularity is a genuinely hard baseline.** It beats ALS and ties
  the learned ranker. Choosing a weak baseline is the most common way to make a
  mediocre model look excellent. ([evaluation.md §5.1](docs/evaluation.md))

---

## Results

Offline evaluation on a held-out future window, 2,000 users:

| Model | NDCG@10 | Recall@10 | HitRate@10 | Coverage |
| --- | --- | --- | --- | --- |
| **two-stage-ranker** | **0.0582** | 0.0736 | 0.2320 | 0.284 |
| two-stage-logistic | 0.0577 | 0.0724 | 0.2255 | 0.311 |
| popularity-category | 0.0560 | 0.0683 | 0.2100 | 0.232 |
| hybrid | 0.0552 | 0.0714 | 0.2210 | 0.316 |
| collaborative-als | 0.0497 | 0.0614 | 0.1970 | 0.225 |
| content | 0.0216 | 0.0299 | 0.0960 | 0.829 |
| popularity-global | 0.0133 | 0.0204 | 0.0670 | 0.004 |

The expected ordering holds — popularity < content < collaborative < hybrid <
learned ranker.

**The top five are statistically indistinguishable at this sample size**
(paired bootstrap, p = 0.297 for the winner). Which is exactly why the A/B
testing framework exists, and why the ranker ships as a treatment arm rather
than an unconditional replacement.

Stage-1 recall@300 is **0.343** — two thirds of relevant items never enter the
candidate pool, making retrieval the highest-leverage place to invest next.

---

## Quick start

No database or Docker required — the ML stack reads Parquet, so everything below
works on a bare Python install.

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r backend/requirements.txt -r ml/requirements.txt -r requirements-dev.txt
```

On macOS or Linux the interpreter is `.venv/bin/python`; everything else below
is identical.

Copy `.env.example` to `.env` and set `JWT_SECRET_KEY`:

```bash
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))"
```

Then generate the dataset and train:

```bash
python data-generation/generate.py && python data-generation/diagnostics.py
```

```bash
python ml/pipelines/train.py --mlflow
```

Two terminals:

```bash
PYTHONPATH="backend;ml" python -m uvicorn app.main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**. Change the user id in the top-right corner to
see the recommendations change; toggle **diagnostics** to see the score, source
and latency behind every item.

### Running it in VS Code

Open the folder and press **F5**. The Run and Debug panel (`Ctrl+Shift+D`)
lists the entry points in the order they need to happen, so "which file do I
run first" is answerable without reading this file:

| Configuration | Runs |
| --- | --- |
| **1 - Generate the dataset** | `data-generation/generate.py` |
| **2 - Check the data is learnable** | `data-generation/diagnostics.py` |
| **3 - Train all models** | `ml/pipelines/train.py --mlflow` |
| **4 - Run the API (with breakpoints)** | `uvicorn app.main:app` |
| **API + storefront** | the API under the debugger, with the Vite dev server beside it |

The API configuration runs **without** `--reload` on purpose: uvicorn's
reloader spawns the application in a subprocess the debugger is not attached
to, so breakpoints silently never hit and it looks like the debugger itself is
broken. A separate auto-reload configuration exists for when you are not
debugging.

`Ctrl+Shift+P` then **Tasks: Run Task** covers the rest - installing
dependencies, the test suite, ruff, migrations and the MLflow UI.

---

Without a database, events are accepted and counted but held in memory, so they
are lost on restart. The API reports that honestly at `/health/ready`
(`"database": false`). Everything else — every recommendation surface, training,
evaluation, MLflow, drift detection — works.

To add a database (still no Docker, and no administrator rights):

```bash
pip install -r requirements-local-db.txt
```

```bash
python scripts/local_postgres.py start && cd backend && alembic upgrade head && cd .. && python scripts/seed_database.py --truncate && python scripts/verify_database.py
```

That takes about a minute and unlocks event persistence, the analytics
endpoints and the database test suite. `stop`, `status` and `destroy` do what
they say. Then measure the serving path:

```bash
python scripts/benchmark_serving.py
```

With Docker instead (see [deployment.md §7](docs/deployment.md) for the caveat):

```bash
docker compose up -d
```

---

## Architecture

```
React storefront + admin dashboard
        │
FastAPI  ·  auth · events · recommendations · feedback · admin
        │
   ┌────┴──────────────────────────────────────────┐
   │  RECOMMENDATION ENGINE                        │
   │  stage 1: 10 candidate sources → ~300         │
   │  stage 2: 83 features → LightGBM LambdaRank   │
   │  stage 3: MMR diversity → explanations        │
   │  + a 7-rung degradation ladder                │
   └────┬──────────────────────────────────────────┘
        │
   Redis (cache, counters)   PostgreSQL + pgvector   MLflow
        │
   Offline: extract → validate → split → features → train
            → evaluate → compare → gate → promote
        │
   Prometheus → Grafana · PSI/KS drift → retraining
```

Full diagram and component responsibilities in
[architecture.md](docs/architecture.md).

---

## Documentation

| Document | Contents |
| --- | --- |
| [requirements.md](docs/requirements.md) | Business problem, 23 FRs, 11 NFRs, non-goals, definition of done |
| [architecture.md](docs/architecture.md) | System design, degradation ladder, caching, cold start, 100 → 100M scaling |
| [decisions.md](docs/decisions.md) | 15 architecture decision records with the alternatives that lost |
| [database.md](docs/database.md) | 20 tables, index design, partitioning, the synthetic dataset |
| [ml-pipeline.md](docs/ml-pipeline.md) | Feature engineering, `as_of` correctness, MLflow, promotion gate, drift |
| [recommendation-engine.md](docs/recommendation-engine.md) | Every algorithm, weight calibration, blending, ranking, explanations |
| [evaluation.md](docs/evaluation.md) | Metrics, leakage audit, model comparison, significance, known biases |
| [api.md](docs/api.md) | Endpoint reference, contracts, performance |
| [deployment.md](docs/deployment.md) | Docker, CI/CD, runbook, security checklist, known gaps |
| [interview-prep.md](docs/interview-prep.md) | Every question from brief §33, answered against code |
| [roadmap.md](docs/roadmap.md) | 20 phases with exit criteria, 12-item risk register |

---

## The dataset

Real recommendation data is not public, so the dataset is generated from an
explicit latent-preference model — and then **proved learnable** before any
model trains on it.

| | |
| --- | --- |
| Users / products / events | 10,000 · 5,000 · 457,272 |
| Sessions / orders | 51,313 · 11,722 |
| Interaction matrix density | 0.33% |
| Generation time | ~21 s, reproducible from a seed |

`data-generation/diagnostics.py` is the Phase 2 exit gate and runs in CI. All
twelve checks pass:

| Hypothesis | Measured |
| --- | --- |
| Users concentrate on a few departments | **0.71** top-department share (uniform = 0.125) |
| Preferred categories convert better | **3.7×** lift |
| Brand affinity shows in behaviour | **4.9×** over catalogue-share baseline |
| Complementary pairs co-purchase | **18.9×** lift (random pairs: 0.00) |
| Consecutive views stay related | **27×** over independence |
| Popularity is unequal | Gini **0.49** |
| Matrix is sparse | **0.33%** dense |
| Cold start is exercised | 2,185 users with no events |

Generating rows proves nothing. If the behaviour were noise dressed up as a
taxonomy, every model would score at chance and the whole evaluation would be
fiction. That is risk R-02, and this gate is its mitigation.

---

## Project layout

```
backend/          FastAPI service: api / services / repositories / models
ml/recsys/        installable ML package, shared by training and serving
ml/pipelines/     train, retrain, MLflow tracking
data-generation/  synthetic behaviour simulator + diagnostic gate
frontend/         React storefront + admin dashboard
monitoring/       Prometheus rules, 3 Grafana dashboards
docker/           per-service Dockerfiles
scripts/          environment check, DDL render, seed, verify
tests/            cross-service integration and end-to-end
docs/             the documentation set above
```

`recsys` is imported by **both** the training pipelines and the API, so feature
engineering exists exactly once. That is the structural defence against
training/serving skew, and a test asserts the feature vectors match.

---

## Testing

```bash
pytest -q
```

| Suite | Tests | Covers |
| --- | --- | --- |
| `ml/tests` | 51 | Metrics, leakage guards, weight calibration, drift, promotion gate |
| `backend/tests` | 136 | Schema invariants, API contracts, auth, event sink, experiments, analytics |
| `tests/` | 67 | Data generation, end-to-end flow, degradation ladder, latency, **live database** |
| **total** | **254** | 253 run; one asserts the API works *without* Postgres and skips when it is present |

Tests that earned their place by catching real bugs:

- **FK index check** — found three missing indexes on first run
- **Reproducibility check** — found unseeded `uuid4()` breaking seed determinism
- **Weight calibration** — proves the ordering is *discovered*, and a companion
  test proves no ordering is *invented* when the data has none
- **Orphan-reference check** — found that the live ingestion path never created
  the `user_sessions` row its events pointed at, which would have left session
  based cold-start retrieval with nothing to read in production
- **CI on a small dataset** — found that three admin endpoints returned 404
  before the first training run, so a fresh deployment's dashboard would have
  shown a routing error on the one screen meant to say "nothing trained yet"
- **Degradation ladder** — removes the ranker and asserts the API still answers
- **Feature-order check** — a silent reordering between training and serving
  would feed the model scrambled inputs with no error anywhere

---

## Status

**All 20 phases complete.** Every exit criterion has been executed and passes.

Measured against a live PostgreSQL 18.6 with pgvector:

| | |
| --- | --- |
| Migration | Applied clean on the first attempt |
| Seeding | 561,173 rows in 17.7 s |
| Database verification | 24/24 checks pass |
| Schema drift (ADR-013) | None |
| Cache hit ratio (NFR-03) | **81.6%** |
| Latency (NFR-01) | cached p95 **6.3 ms**, cold p99 **26.1 ms** |
| Event ingest (NFR-02) | p99 **5.15 ms**, 150/150 rows persisted |
| Promotion gate | A forced retrain was **correctly rejected**; the incumbent kept serving |

**One thing remains unverified: Docker container execution.** Docker Desktop's
engine cannot start without WSL2, which needs Windows features that require
administrator rights. `docker compose config` validates, but the images have
never been built and the stack has never run. The database path uses embedded
PostgreSQL instead ([ADR-016](docs/decisions.md)) — the same PostgreSQL, the
same pgvector, the same migration, the same connection string.

---

## Future improvements

Tracked as the "revisit if" clause on each ADR rather than as a wishlist:

1. **Retrieval before ranking** — recall@300 of 0.343 is the ceiling on
   everything downstream
2. **Logged impressions** to replace bootstrap labels and enable
   inverse-propensity weighting
3. **A neural two-tower retriever** behind the existing `CandidateSource`
   interface
4. **Kafka ingestion** past ~2k events/s, behind the existing `EventSink`
5. **FAISS or a vector database** past ~500k products, behind `VectorIndex`
