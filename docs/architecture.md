# System Architecture

Status: **Phase 1 proposal**
Last updated: 2026-09-08

This document is the technical contract for the build. Every later phase either
implements something described here or amends this document with a new ADR.

---

## 1. Architecture at a glance

```
                                 BROWSER
   +--------------------------------------------------------------+
   |  React 18 + Vite + TS                                         |
   |   Storefront (catalogue, PDP, cart, search)                   |
   |   Admin ML dashboard (Recharts)                               |
   |   trackEvent() -> batched, non-blocking                       |
   +---------------------------+----------------------------------+
                               | HTTPS / JSON
                               v
   +--------------------------------------------------------------+
   |  FastAPI application (uvicorn workers)                        |
   |   api/v1  routers  -> auth, catalogue, events, recs, admin    |
   |   middleware: JWT, CORS, rate limit, request-id, metrics      |
   +------+-----------------+-----------------+-------------------+
          |                 |                 |
          v                 v                 v
   +-------------+  +-----------------+  +------------------+
   | EventService|  | RecommendationS.|  | ExperimentService|
   | validate    |  | orchestrator    |  | deterministic    |
   | enrich      |  |                 |  | bucketing        |
   | sink        |  |                 |  |                  |
   +------+------+  +--------+--------+  +---------+--------+
          |                  |                     |
          |                  v                     |
          |   +--------------------------------------------------+
          |   |          RECOMMENDATION ENGINE (recsys)          |
          |   |                                                  |
          |   |  STAGE 1  candidate generation (parallel)        |
          |   |    ALS collaborative      ~150                   |
          |   |    content / pgvector kNN ~100                   |
          |   |    co-visitation + FBT    ~80                    |
          |   |    category & brand affinity ~60                 |
          |   |    trending + popularity  ~60                    |
          |   |    recently viewed        ~20                    |
          |   |         -> dedupe, business-rule filter -> ~300  |
          |   |                                                  |
          |   |  STAGE 2  ranking                                |
          |   |    feature assembly (offline + online features)  |
          |   |    LightGBM LambdaRank -> relevance score        |
          |   |    re-rank: MMR diversity, category caps,        |
          |   |             popularity discount, business rules  |
          |   |                                                  |
          |   |  STAGE 3  explanation (rule + SHAP)              |
          |   |                                                  |
          |   |  Degradation ladder if anything is missing       |
          |   +---------------+-----------------+----------------+
          |                   |                 |
          v                   v                 v
   +-------------+   +----------------+  +---------------------+
   |  Redis      |   |  PostgreSQL 16 |  |  MLflow             |
   |  cache      |   |  + pgvector    |  |  tracking +         |
   |  online     |   |  OLTP + events |  |  model registry     |
   |  features   |   |  + features    |  |  artefact store     |
   |  counters   |   |  + experiments |  |                     |
   +-------------+   +--------+-------+  +----------+----------+
                              |                     ^
                              v                     |
   +--------------------------------------------------------------+
   |  OFFLINE / BATCH (ml/pipelines, scheduled)                    |
   |   extract -> validate -> features -> train -> evaluate ->     |
   |   compare with production -> register -> promote if better    |
   +--------------------------------------------------------------+

   Cross-cutting: Prometheus scrapes API + engine metrics; Grafana dashboards;
   drift job compares live feature distributions with the training snapshot.
```

---

## 2. Component responsibilities

### 2.1 Frontend (`frontend/`)

Two applications behind one Vite build and one React Router tree:

- **Storefront** — catalogue, product detail page, cart, search, and the
  recommendation surfaces (`Recommended for you`, `Because you viewed X`,
  `Frequently bought together`, `Trending now`, `Recently viewed`,
  `You may also like`).
- **Admin ML dashboard** — model performance, business metrics, model
  monitoring, recommendation distribution, A/B experiment readouts.

State: Redux Toolkit for auth/session/cart (genuinely global, cross-route), RTK
Query or plain Axios services for server state. Recommendation responses are
*not* put in Redux — they are per-surface, short-lived and cached server-side.

Every recommendation strip is rendered by one `<RecommendationRail>` component
that takes a `section` payload. Impressions fire through an `IntersectionObserver`
so the impression is real (the strip was actually on screen), not merely fetched.
Clicks fire `RECOMMENDATION_CLICK` carrying the `recommendation_id` that the API
returned, which is what makes attribution joinable in SQL.

### 2.2 API layer (`backend/app/api`)

Thin. Routers validate input with Pydantic, resolve the current principal,
delegate to a service, and shape the response. No SQL and no ML in routers.

Router groups:

| Prefix | Purpose |
| --- | --- |
| `/api/v1/auth` | register, login, refresh, me |
| `/api/v1/catalogue` | products, categories, search |
| `/api/v1/events` | single and batch event ingestion |
| `/api/v1/recommendations` | all recommendation surfaces |
| `/api/v1/feedback` | impressions, clicks, conversions |
| `/api/v1/admin` | metrics, models, experiments, drift (role-gated) |
| `/health`, `/metrics` | liveness/readiness, Prometheus exposition |

### 2.3 Service layer (`backend/app/services`)

Business logic and orchestration. Depends on repositories (data access) and on
`recsys` (ML), never the reverse. This is the seam that keeps the ML package
framework-agnostic and unit-testable without a running FastAPI app.

### 2.4 Repository layer (`backend/app/repositories`)

All SQLAlchemy access. One repository per aggregate. Keeps query tuning and index
usage in one place, and lets service tests use in-memory fakes.

### 2.5 `recsys` package (`ml/recsys`)

An installable Python package, imported by **both** the training pipelines and
the FastAPI service. This is the single most important structural decision in the
project: feature engineering exists exactly once, so an offline-trained model
cannot be fed differently shaped features online (NFR-06, ADR-012).

```
recsys/
  config/        typed settings, weight profiles, thresholds
  data/          loaders, snapshot readers, schema contracts
  preprocessing/ cleaning, encoders, temporal splitting
  features/      user / product / interaction feature builders + transforms
  models/        popularity, content, ALS, BPR, hybrid, FBT, covisit
  candidates/    CandidateSource implementations + the generator
  ranking/       LightGBM LambdaRank trainer, feature assembler, re-rankers
  evaluation/    metric implementations, model comparison, report builder
  inference/     runtime engine used by the API, artefact loading, caching
  explain/       rule-based reasons + SHAP for ranker attributions
  drift/         PSI, KS, distribution snapshots, thresholds
  registry/      MLflow wrappers, promotion gate, model resolution
```

---

## 3. Data architecture

### 3.1 Storage roles

| Store | Holds | Why |
| --- | --- | --- |
| PostgreSQL | catalogue, users, orders, raw events, materialised features, recommendation logs, experiments, model registry mirror, product embeddings (pgvector) | one consistent, queryable source of truth; analytics are SQL, not a second pipeline |
| Redis | rendered recommendation payloads, online user features, real-time counters (trending, recently viewed), rate-limit buckets, idempotency keys | sub-millisecond reads on the hot path; TTL-based freshness |
| MLflow artefact store | model binaries, embeddings matrices, encoders, evaluation reports, feature-snapshot statistics | model lineage and rollback |

### 3.2 Schema map (detail lands in Phase 2, `docs/database.md`)

Domain groups and their key relationships:

```
users ──< orders ──< order_items >── product_variants >── products >── categories
  │                                                            │
  │                                                            └─< product_features
  ├──< user_events (partitioned by month) >── products         └─< product_embeddings (vector)
  ├──< user_product_interactions  (rolled-up affinity)
  ├──< user_features
  ├──< experiment_assignments >── experiments
  └──< recommendations ──< recommendation_impressions
                        ├─< recommendation_clicks
                        └─< recommendation_conversions
model_versions  (serving contract mirror of the MLflow registry)
```

Design rules applied throughout:

- Surrogate `BIGINT` primary keys; a separate public `uuid` on user-facing rows so
  IDs are not enumerable.
- Every table gets `created_at`, `updated_at`; soft deletion (`deleted_at`) on
  `products`, `product_variants`, `users`, `experiments` — anything a
  recommendation may historically reference and that must stay joinable.
- Foreign keys everywhere, with `ON DELETE RESTRICT` on catalogue references so a
  delete cannot orphan a logged recommendation.
- `user_events` is **append-only and partitioned by month**. It is the single
  largest table and the only one whose growth is unbounded; partitioning gives
  cheap retention drops and keeps the training extract a partition scan.
- `user_product_interactions` is the **aggregate** of events per (user, product):
  view count, cart count, purchase count, last-seen, implicit weight. This is
  what collaborative filtering reads, so training never scans raw events.
- Indexes are defined for the actual access paths, not by reflex: composite
  `(user_id, created_at DESC)` for recency reads, `(product_id, event_type)` for
  product analytics, a partial index on active products, and an HNSW index on the
  embedding column.

### 3.3 Event model

One `user_events` table, an enum-backed `event_type`, and a JSONB `metadata`
column. New event types require a new enum value and nothing else (FR-21). Typed
per-event payload validation is enforced at the API boundary in Pydantic, so
JSONB is not a dumping ground: each event type has a declared payload model.

Anonymous traffic is tracked by `session_id`; on login the session is stitched to
the user by an update on the recent partition (FR-22).

### 3.4 Implicit feedback weighting

Interactions are not equal. The weight applied to `(user, item)` in the
collaborative model comes from the observed strength of the signal:

```
PURCHASE > ADD_TO_CART > WISHLIST > PRODUCT_RATING(high) > PRODUCT_VIEW > IMPRESSION
```

The weights are **derived, not guessed** — see ADR-002. Starting priors are set
from the empirical odds that an event leads to a purchase of that product within
the next 30 days, then normalised. The scheme, the calibration procedure and the
sensitivity analysis are written up in `docs/recommendation-engine.md` in Phase 7.

---

## 4. The recommendation request path

### 4.1 Two-stage retrieval and ranking

Scoring every user against every product with a rich model is O(users x items x
features). At 5k products that is survivable; at 500k it is not, and the whole
point is to build the architecture that survives. So:

**Stage 1 — candidate generation.** Cheap, recall-oriented, parallel. Each source
implements one interface:

```python
class CandidateSource(Protocol):
    name: str
    def generate(self, ctx: RequestContext, limit: int) -> list[Candidate]: ...
```

Sources: ALS neighbours, pgvector content kNN over the user's affinity centroid
and recent items, co-visitation, frequently-bought-together, category and brand
affinity, trending, popularity, recently viewed. Target: 300 candidates with high
recall of anything the user might plausibly want.

**Stage 2 — ranking.** Expensive, precision-oriented, applied to 300 rows instead
of 5000. A LightGBM LambdaRank model scores each candidate using collaborative
score, content similarity, user-product affinity, popularity, rating, price
distance from the user's typical price band, category and brand match, recency,
historical conversion rate, and the source that produced the candidate.

**Stage 3 — re-rank and explain.** MMR for diversity, per-category caps,
popularity discounting, business rules (no out-of-stock, no already-purchased
non-consumables, honour merchandising pins/blocks), then attach an explanation.

### 4.2 Degradation ladder (NFR-04)

A recommendation surface must never fail. Each level falls to the next:

```
1  ranker + full candidate set        (normal)
2  hybrid weighted score, no ranker   (ranker artefact missing or slow)
3  CF + content only                  (feature store unavailable)
4  content + trending                 (CF model missing / cold user)
5  category-popular                   (no user context at all)
6  global trending from Redis         (database degraded)
7  cached static fallback list        (everything degraded)
```

The response always states which level served it (`strategy`, `model_version`),
so the dashboard can show fallback rate as a first-class health metric.

### 4.3 Caching strategy

| Key | TTL | Invalidation |
| --- | --- | --- |
| `rec:home:{user}:{model_ver}:{variant}` | 15 min | version bump, or high-signal event (purchase, cart) for that user |
| `rec:similar:{product}:{model_ver}` | 6 h | model version bump |
| `rec:fbt:{product}:{model_ver}` | 12 h | batch refresh |
| `rec:trending:{scope}` | 5 min | rolling recompute |
| `feat:user:{user}` | 30 min | event-driven partial update |
| `recent:{user}` | list, 30 days | push on view |

The model version is part of every cache key, so promoting a model invalidates
the cache implicitly rather than requiring a flush. Cache stampedes are handled
with a short lock plus stale-while-revalidate.

---

## 5. Cold start

| Case | Strategy |
| --- | --- |
| New user, no session yet | trending plus category-popular, diversified across top categories; optional onboarding category picker seeds an affinity prior |
| New user, mid-session | session-based recommendations from the co-visitation matrix using the items viewed so far — this works from the very first view and is the highest-value cold-start lever |
| Returning user, sparse history | content-based from viewed items, blended with popularity; hybrid weights shift toward content as a function of interaction count |
| New product | metadata plus text embedding places it in the content space immediately, so it can be retrieved as a similar item before it has any interactions; a small exploration budget guarantees impressions so it can earn collaborative signal |
| New category or seasonal spike | trending source with a short half-life picks this up without retraining |

The hybrid weights are a **function of user history depth**, not constants. A user
with 3 events should not be scored mostly by a collaborative model that has
nearly no signal for them. This is specified in ADR-014.

---

## 6. Offline pipeline

```
extract (partitioned event scan + catalogue snapshot)
   -> validate (schema, null rates, row counts, referential integrity)
   -> temporal split (train / validation / test by time, not randomly)
   -> feature engineering (point-in-time correct, as_of_ts enforced)
   -> train (popularity, content, ALS, BPR, hybrid, ranker)
   -> evaluate (ranking + beyond-accuracy metrics on the future window)
   -> compare with the current production model
   -> register in MLflow
   -> promote only if the gate passes
```

Point-in-time correctness is enforced by a single rule that the feature builders
share: **a feature computed for a training row with timestamp `t` may only read
events strictly before `t`.** Violating it is the classic recommender leak; the
leakage analysis is in `docs/evaluation.md` in Phase 14.

---

## 7. Scale story

| Stage | Shape | What the architecture does |
| --- | --- | --- |
| 100 users | anything works | single Postgres, single API process, models retrained on demand |
| 10k users, 5k products (this build) | ~1M interactions | as designed: batch nightly training, Redis cache, ALS on a sparse matrix in memory, pgvector kNN |
| 1M users, 500k products | ~100M interactions | ALS factors no longer fit a single request path: move item-embedding kNN to a dedicated ANN index (FAISS/Milvus), shard the feature store, precompute homepage candidates offline for the long tail and compute online only for active users, read replicas for analytics |
| 100M events/day | ~1.2k events/s average, ~10k/s peak | events go to Kafka rather than a synchronous insert; a stream consumer maintains Redis counters and a compacted interaction store; Postgres keeps only aggregates; training reads from the lake, not the OLTP database |

The seams that make that evolution cheap are built now: `EventSink` (sync insert
today, Kafka producer later), `CandidateSource` (ALS today, two-tower ANN later),
`FeatureStore` (Postgres + Redis today, Feast or equivalent later), and
`ModelResolver` (MLflow alias lookup, so the serving code never hardcodes a
model path).

---

## 8. Security model

- JWT access tokens (short-lived) plus refresh tokens; passwords hashed with
  bcrypt/argon2. Tokens carry `sub` and `role`.
- Roles: `customer`, `analyst`, `admin`. Admin dashboard endpoints require
  `analyst` or above; model promotion requires `admin`.
- All input validated by Pydantic; SQL exclusively through SQLAlchemy bound
  parameters (no string-built SQL anywhere, including analytics queries).
- Rate limiting in Redis: per-IP on auth, per-user on events, per-token on
  recommendations.
- CORS restricted to configured origins; no wildcard with credentials.
- Configuration exclusively from environment variables via a typed settings
  object. `.env.example` documents every key; no real value is ever committed.
- Event ingestion accepts a `user_id` only from the token, never from the body,
  so a client cannot write events attributed to another user.

---

## 9. Observability

Prometheus metrics exported by the API:

- `http_request_duration_seconds` (histogram, by route and status)
- `recommendation_latency_seconds` (by surface and stage: candidates, ranking, total)
- `recommendation_cache_hits_total` / `misses_total`
- `recommendation_fallback_total` (by degradation level) — the key health signal
- `recommendation_candidates_generated` (by source)
- `recommendation_score` (histogram) — score distribution shift shows up here first
- `model_info` (gauge with model name and version labels)
- `events_ingested_total` (by type), `event_ingest_failures_total`
- `drift_score` (gauge per monitored feature)

Grafana dashboards: API health, recommendation engine, model and drift, business
KPIs. Structured JSON logging with a request id propagated from the middleware
through the services into the engine, so one recommendation can be traced.

---

## 10. Repository layout

```
.
├── backend/            FastAPI service (api / services / repositories / models)
├── ml/
│   ├── recsys/         installable ML package, shared by training and serving
│   ├── pipelines/      orchestrated jobs: train, evaluate, drift, retrain
│   ├── artifacts/      local artefact scratch (gitignored)
│   └── tests/
├── data-generation/    synthetic behaviour simulator
├── frontend/           React storefront + admin dashboard
├── monitoring/         prometheus config, grafana dashboards and provisioning
├── docker/             per-service Dockerfiles
├── scripts/            developer entry points (setup, seed, train, verify)
├── tests/              cross-service integration and end-to-end tests
├── docs/               this documentation set
├── notebooks/          experimentation only, never imported by production code
├── docker-compose.yml
└── .github/workflows/  CI
```

`notebooks/` may import `recsys`; `recsys` may never import from `notebooks/`.
That one-way rule is what keeps experimentation out of the production path.
