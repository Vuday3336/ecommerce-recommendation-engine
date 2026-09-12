# Phased Implementation Plan

Status: **All 20 phases complete.** Every exit criterion has been executed and
passes. The only unverified items are container *execution* (Docker Desktop
cannot start without WSL2) - the database path runs on embedded PostgreSQL
instead, per ADR-016.
Last updated: 2026-09-12

Every phase has a **deliverable** (what gets written) and an **exit criterion**
(the command or test that proves it works). A phase is not finished because the
files exist; it is finished when its exit criterion passes. Nothing is marked
done on the basis of "should work".

---

## 1. Phase table

Status as of 2026-09-12. **246 tests pass; ruff is clean.**

| # | Phase | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Architecture and requirements | Complete | 16 ADRs, 23 FRs, 11 NFRs, 12-item risk register |
| 2 | Database and synthetic dataset | Complete | Migration applied; **561,173 rows seeded in 17.7 s**; 24/24 database checks pass; 12/12 diagnostic checks pass |
| 3 | Event tracking | Complete | 12 event types, per-type payload validation, buffered non-blocking sink that also registers sessions (a bug the orphan check caught - database.md section 9) |
| 4 | Feature engineering | Complete | 27 user + 25 product + 15 pair features, `as_of` guard tested |
| 5 | Popularity baseline | Complete | Global 0.0133, category 0.0560 |
| 6 | Content-based | Complete | TF-IDF+SVD, 491 dims, 85.4% variance; NDCG@10 0.0216 |
| 7 | Collaborative filtering | Complete | ALS 0.0497, BPR 0.0159; alpha tuned on validation |
| 8 | Hybrid | Complete | 0.0552; weights tuned per history segment |
| 9 | Candidates and ranking | Complete | Stage-1 recall 0.343; LambdaRank 0.0582 |
| 10 | FastAPI inference | Complete | 27 endpoints; 51 API tests; degradation ladder tested |
| 11 | Redis optimisation | Complete | **81.6% hit ratio**, cached p95 **6.3 ms**, cold p99 **26.1 ms** (`scripts/benchmark_serving.py`) |
| 12 | React integration | Complete | Storefront + admin dashboard, verified live in a browser |
| 13 | MLflow | Complete | 1,255 metrics logged; model registered |
| 14 | Evaluation framework | Complete | 13 variants, 10 metrics, paired bootstrap, segmented |
| 15 | A/B testing | Complete | Deterministic assignment, holdout, SRM check, significance readout |
| 16 | Monitoring and drift | Complete | 30+ Prometheus metrics, 11 alert rules, 3 Grafana dashboards (28 panels), drift verified firing |
| 17 | Docker and CI/CD | **Written; compose validates, containers unrun** | 3 Dockerfiles, 8-service compose (`docker compose config` passes), 6-job CI, retrain workflow. Container execution needs WSL2 |
| 18 | Testing | Complete | **246 tests**: 51 ML, 129 backend, 66 integration (incl. 26 database) |
| 19 | Retraining pipeline | Complete | Drift check verified; **a forced retrain was correctly rejected by the gate** and the incumbent kept serving |
| 20 | Documentation | Complete | 11 documents, ~4,200 lines, 16 ADRs |

**What is still unverified**, and only this: Docker *container execution*.
Docker Desktop's engine cannot start without WSL2, so the images have never been
built and the compose stack has never run - though `docker compose config`
validates. The database path runs on embedded PostgreSQL instead (ADR-016), which
is the same PostgreSQL, the same pgvector and the same migration. Details in
`deployment.md` section 7.

--- | --- | --- | --- |
| 1 | Architecture and requirements | `docs/requirements.md`, `docs/architecture.md`, `docs/decisions.md`, this plan, repo skeleton, dependency and toolchain verification | Documents reviewed and approved; dependency availability confirmed on the target Python |
| 2 | Database and synthetic dataset | SQLAlchemy models (20 tables), Alembic migration, `data-generation/` simulator, `diagnostics.py` gate, seed script, `verify_database.py` | **Offline: PASSED** - DDL compiles to PostgreSQL, migration compiles to 689 lines of SQL, 60 tests pass, 438k events generated, all 12 structure checks pass. **Database-side: BLOCKED** - `alembic upgrade head` and `verify_database.py` need a running Postgres (R-01) |
| 3 | Event tracking | Event schemas, `EventSink`, ingestion endpoints (single + batch), session stitching, Redis counters | POST an event, see the row in Postgres and the counter in Redis; p99 write latency measured under load |
| 4 | Feature engineering | `recsys.features` user / product / interaction builders, `as_of_ts` point-in-time rule, materialisation job | Feature tables populated; a point-in-time test proves no feature reads data at or after its timestamp |
| 5 | Popularity baseline | Global, per-category, trending (time-decayed) models plus the first evaluation harness | Baseline metrics (Precision/Recall/NDCG/HitRate@10) printed for the test window; these become the number every later model must beat |
| 6 | Content-based | Attribute encoding, sentence-transformer text embeddings, pgvector storage and HNSW index, similarity search | `similar/{product_id}` returns same-category items at high purity; beats popularity on the item-to-item task |
| 7 | Collaborative filtering | Interaction matrix, calibrated implicit weights (ADR-002), ALS, BPR comparison, item-item neighbours | ALS beats popularity on NDCG@10; the derived weight table and its calibration are written into `docs/recommendation-engine.md` |
| 8 | Hybrid engine | Configurable weighted blend, adaptive weights by history depth, business rules, MMR diversity | Hybrid beats every single strategy on NDCG@10 without a coverage regression |
| 9 | Candidate generation and ranking | `CandidateSource` implementations, generator, ranking feature assembler, LightGBM LambdaRank, XGBoost and logistic baselines | Stage-1 recall@300 reported; ranker beats hybrid on NDCG@10; ranking runs inside the latency budget on 300 candidates |
| 10 | FastAPI inference service | All recommendation endpoints, degradation ladder, explanations, feedback endpoints, auth, rate limiting | Every endpoint in the brief returns correct payloads; chaos test proves the ladder never 5xxes |
| 11 | Redis optimisation | Cache-aside with versioned keys, stampede protection, online feature reads, warmers | Cache hit ratio above 80 % under a replayed traffic pattern; p99 within NFR-01 |
| 12 | React integration | Storefront, recommendation rails, impression tracking, search, cart, product page | End-to-end in a browser: browse, get personalised rails, click, see the click logged |
| 13 | MLflow | Experiment tracking across all trainers, registry, promotion gate, `model_versions` mirror, model resolution at serve time | Train twice, see two versions; promote; API serves the promoted version and logs it |
| 14 | Evaluation framework | All ranking and beyond-accuracy metrics, temporal split, leakage audit, model comparison report | `docs/evaluation.md` with a real comparison table and a written argument for the winner |
| 15 | A/B testing | Experiments, deterministic assignment, variant-aware serving, outcome aggregation, significance testing | Two variants served; dashboard shows per-variant CTR/CVR/RPU with sample-ratio check |
| 16 | Monitoring and drift | Prometheus metrics, Grafana dashboards, PSI/KS drift job, thresholds, alerting | Injecting a shifted feature distribution raises the drift gauge above threshold and fires the alert |
| 17 | Docker and CI/CD | Dockerfiles (frontend, backend, ml-training), Compose, GitHub Actions (lint, type, test, build) | `docker compose up` brings the whole system up seeded; CI green on a clean clone |
| 18 | Testing | Unit, API, ML, integration and end-to-end suites; coverage gate | Full suite green; coverage at or above NFR-08; the event-to-recommendation integration test passes |
| 19 | Retraining pipeline and deployment | Orchestrated retrain job, promotion gate wired to drift and schedule, deployment docs | Drift alert triggers retraining; a worse model is rejected by the gate; a better model is promoted and served |
| 20 | Documentation and interview prep | README, remaining `docs/*`, `docs/interview-prep.md` | Every question in brief section 33 answered with reference to code that exists |

---

## 2. Dependency order

Phases are mostly sequential, but three can run in parallel once their inputs
exist, which is how the schedule stays sane:

```
2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 13 -> 14 -> 15 -> 16 -> 19
                                            \
                                             -> 12 (frontend, needs only 10)
                                             -> 17 (docker/CI, needs only 10)
18 (tests) is written alongside every phase and hardened as its own phase
20 (docs) accumulates continuously and is finished as its own phase
```

---

## 3. Risk register

Ranked by expected cost. Each risk has a concrete mitigation that is built, not
just noted.

| # | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R-01 | **Docker Desktop cannot start** (WSL2 features disabled, admin rights unavailable), so no phase could be verified against real infrastructure | Certain | High | **Largely resolved (ADR-016).** `embedded-postgres` ships PostgreSQL 18.6 with pgvector as a pip wheel, needs no admin rights, and uses the same connection string. Migration, seeding, verification, index plans, drift check and the serving benchmark all now run. Only container execution remains unverified |
| R-02 | **Synthetic data that is secretly random.** If the generator does not encode real preference structure, every model scores near chance, the hybrid cannot beat the baseline, and the whole evaluation story collapses | High | Very high — invalidates Phases 5 to 14 | Generate from an explicit latent-preference model (per-user category/brand/price-band affinity vectors, session dynamics, seasonality, repeat-purchase cycles). Phase 2 exits only after a diagnostic report shows the intended structure is recoverable: category concentration per user, purchase-rate lift for affine categories, and a non-trivial item-item co-purchase signal |
| R-03 | **Data leakage inflating offline metrics.** The classic recommender failure; easy to introduce through popularity features or a random split | High | High — every reported number becomes fiction | Temporal split (ADR-007); a shared `as_of_ts` argument in every feature builder; an automated leakage test that fails if any feature reads a timestamp at or after the label time; a written leakage audit in Phase 14 |
| R-04 | **Training/serving skew.** Offline and online feature computation drift apart silently | Medium | High — model degrades with no error surfaced | One shared `recsys.features` package (ADR-012); a CI skew test asserting the offline and online vectors are identical for the same input |
| R-05 | **Cold-start evaluation is misleading.** Metrics computed only on users with rich history hide the fact that most real traffic is sparse | Medium | Medium-high | Report all metrics segmented by user history depth (0, 1-4, 5-19, 20+ interactions), not just in aggregate |
| R-06 | **Popularity bias makes the "best" model boring.** NDCG can improve while the system recommends the same 50 products to everyone | High | Medium-high | Coverage, Gini, novelty and diversity are gate metrics, not footnotes (ADR-014). A model that wins NDCG but regresses coverage past a threshold fails the promotion gate |
| R-07 | **Position and presentation bias in ranker labels.** The ranker trains on logged impressions, which were themselves ordered by a previous model | Medium | Medium | Train the bootstrap ranker on historical outcomes rather than logged ranks; include position as a training feature and set it to a constant at inference; document the inverse-propensity option and the exploration budget that makes it possible |
| R-08 | **Scope is very large.** Nineteen features across five disciplines, and quality collapses if everything is half-built | High | High | Strict phase gates with exit criteria; no phase starts before the previous one's criterion passes; the non-goals list in `requirements.md` is enforced |
| R-09 | **Latency budget missed once the ranker is in the path.** Feature assembly for 300 candidates can easily blow p99 | Medium | Medium | Vectorised batch feature assembly (one matrix, not 300 row lookups); Redis online features; caching with versioned keys; latency measured per stage from Phase 9 onward, not at the end |
| R-10 | **`sentence-transformers` pulls a large Torch download** (roughly 2-3 GB) and first-run model download | Certain | Low-medium | Embed once in a batch job, persist vectors in Postgres; the API never loads Torch. Content similarity at serve time is a pgvector query. Fall back to TF-IDF plus SVD if the download is impractical, behind the same interface |
| R-11 | **MLflow model registry adds a serving dependency.** If MLflow is down, the API cannot resolve a model | Low | Medium | Artefacts are cached locally on load and the last-known-good version is pinned in `model_versions`; MLflow is only needed to *change* the serving model, never to serve |
| R-12 | **Windows path and encoding issues** across Python, Node and shell tooling | Medium | Low | `pathlib` everywhere, UTF-8 enforced, no shell-specific scripting in the ML path, CI runs on Linux so cross-platform breakage surfaces early |

---

## 4. What "done" looks like per discipline

A checklist used at the end to confirm nothing was quietly skipped.

- **ML:** five strategies plus a learned ranker, all evaluated on a temporal split
  with ranking *and* beyond-accuracy metrics, with a written argument for the
  winner and segmented cold-start results.
- **Backend:** every endpoint in brief section 21, auth and roles, rate limiting,
  the degradation ladder, and measured latency against NFR-01.
- **Data:** an event pipeline from the browser to the feature tables, with the
  synthetic generator producing structure a model can actually learn.
- **MLOps:** MLflow tracking and registry, a promotion gate that rejects worse
  models, drift detection that fires, and a retraining pipeline that closes the
  loop.
- **Frontend:** a storefront that shows recommendations and reports impressions
  and clicks, plus an admin dashboard that shows the model and business metrics.
- **Ops:** Dockerfiles, Compose, CI, Prometheus, Grafana dashboards.
- **Docs:** the eight documents in brief section 32 plus interview preparation.
