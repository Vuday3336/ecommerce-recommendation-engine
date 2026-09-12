# Architecture Decision Records

Each ADR records a decision, the alternatives that were actually considered, why
they lost, and what would make us revisit. Status is `Accepted` unless noted.
ADRs are amended, never silently rewritten.

---

## ADR-001 — Two-stage retrieval then ranking

**Status:** Accepted (Phase 1)

**Context.** A homepage request must produce a ranked list from the whole
catalogue. The straightforward approach scores every product for the user with
the best available model.

**Decision.** Split into cheap high-recall candidate generation (target ~300
items from six parallel sources) and expensive high-precision ranking over only
those candidates.

**Why.** Cost scales with `sources x k` rather than with catalogue size, so the
p99 budget in NFR-01 holds as the catalogue grows. It also lets each stage be
optimised for a different objective: stage 1 is measured by **recall@300**
(did the right item survive?), stage 2 by **NDCG@10** (is it near the top?). A
single-stage system has to trade those against each other with one model. Two
stages also make the system pluggable: a neural retriever can be added as one
more `CandidateSource` without touching the ranker, and the ranker can be swapped
without touching retrieval.

**Alternatives.** (a) Single-model full scoring — simpler, but the ranking model
we want uses per-pair features that are expensive to assemble for 5k items and
impossible for 500k. (b) Pure ANN retrieval with no ranker — fast, but cannot use
non-embedding signals such as price fit, stock, conversion rate, or business rules.

**Revisit if.** Catalogue stays under ~2k items forever and latency is not a
concern, in which case stage 1 is pure overhead.

---

## ADR-002 — Weighted implicit feedback, weights derived from conversion odds

**Status:** Accepted (Phase 1, calibrated in Phase 7)

**Context.** We have no reliable explicit ratings at volume. All we have is
behaviour, and behaviours differ enormously in what they say about preference.
Treating a view and a purchase as the same "1" throws away most of the signal;
inventing weights like "purchase = 10" by intuition is unjustifiable in an
interview and often wrong.

**Decision.** Use implicit feedback with a confidence weight per interaction, and
**derive** the weights from data rather than assert them. For each event type `e`,
compute the empirical probability that a user who performs `e` on product `p`
purchases `p` within a 30-day window, and set the base weight proportional to the
lift of that probability over the base rate:

```
w(e)  proportional to  log(1 + P(purchase | e) / P(purchase))
```

The weight is then modulated per interaction by recency (exponential decay with a
half-life fitted on repeat-purchase intervals) and capped to limit the influence
of a single obsessive user. Negative signals (`REMOVE_FROM_CART`) reduce the
weight rather than adding a separate negative class, because ALS with implicit
feedback has no natural place for negatives — everything unobserved is already
weak-negative.

Expected ordering, to be confirmed empirically:
`PURCHASE > ADD_TO_CART > WISHLIST > high RATING > PRODUCT_VIEW > IMPRESSION`.

**Why this framing matters.** The ALS objective minimises
`sum over (u,i) of c_ui * (p_ui - x_u . y_i)^2` where `c_ui = 1 + alpha * r_ui`.
The weight is *confidence*, not preference — it controls how hard the model is
pushed to fit that cell. Framing weights as confidence is what makes deriving
them from conversion odds coherent rather than arbitrary.

**Alternatives.** Fixed hand-picked weights (fast, arbitrary); binarised
interactions (loses the strongest signal we have); explicit-rating matrix
factorisation (we do not have enough ratings, and ratings are missing-not-at-random).

**Revisit if.** Calibration shows the derived ordering is unstable across
categories, in which case weights become per-category.

---

## ADR-003 — ALS as the primary collaborative model, BPR as the comparison

**Status:** Accepted

**Decision.** `implicit.als.AlternatingLeastSquares` is the production
collaborative model. BPR is trained as an evaluation competitor, not shipped by
default.

**Why ALS.** It is built for exactly our data shape: implicit, confidence-weighted,
sparse. It has a closed-form alternating solve, so it is fast and deterministic
given a seed (which NFR-05 requires), it parallelises across 16 cores, and a
10k x 5k matrix factorises in seconds — meaning retraining is cheap enough to run
nightly. It also yields item factors directly usable for item-item similarity,
which serves FR-02 and FR-04 for free.

**Why BPR loses as the default.** BPR optimises a pairwise ranking loss, which is
theoretically better matched to a ranking metric, but it is sensitive to negative
sampling, non-deterministic, slower to converge, and — critically — we already
have a dedicated pairwise/listwise ranking stage (ADR-005). Putting the ranking
objective in stage 2, where the rich features live, is the better division of
labour. BPR is still trained so the evaluation report can state this with numbers
instead of assertion.

**Revisit if.** BPR beats ALS on recall@300 (the stage-1 metric) by a meaningful
margin in Phase 14.

---

## ADR-004 — pgvector for product embeddings, not FAISS (yet)

**Status:** Accepted

**Decision.** Store `sentence-transformers` product embeddings in a Postgres
`vector` column with an HNSW index; query them with SQL. Keep FAISS out of the
initial build.

**Why.** At 5k products, an HNSW index in Postgres answers a kNN query in single-
digit milliseconds — well inside the latency budget. More importantly it avoids an
entire class of production bug: with a separate FAISS index we would have two
stores that must be kept consistent, and a new product would exist in Postgres but
not in the index until a rebuild, breaking FR-10 (new-product cold start)
intermittently and invisibly. With pgvector, insert and index are the same
transaction, and a kNN search can be filtered by stock, price band or category in
the *same query* — a pre-filter FAISS cannot do without a second pass.

**Cost.** Postgres kNN throughput is lower than a tuned FAISS index, and vector
search competes with OLTP for the same connection pool.

**Revisit at.** Roughly 500k+ vectors, or when vector QPS is a measured
bottleneck. The `VectorIndex` interface is defined so a FAISS implementation is a
drop-in; the interface, not the store, is the commitment.

---

## ADR-005 — LightGBM LambdaRank for the stage-2 ranker

**Status:** Accepted

**Decision.** LightGBM with `objective="lambdarank"` and `ndcg` evaluation,
grouped by request (one group = one user's candidate set).

**Why gradient boosting at all.** The ranking features are heterogeneous and
tabular: a mix of scores, counts, ratios, log-scaled prices, recency in days, and
categorical matches. Trees handle mixed scales, non-linearities and interactions
(for example "high content similarity matters much more when the collaborative
score is weak") without feature scaling, and they train in seconds on ~1M rows.
A neural ranker needs far more data and tuning for no expected gain at this size.

**Why LambdaRank rather than binary classification.** Optimising per-item click
probability treats each item independently, but the product is a *list*. LambdaRank
optimises a listwise NDCG surrogate directly, weighting swaps by how much they
would change NDCG — the metric we actually report. Logistic ranking is trained as
a baseline so this claim is measured, not assumed.

**Why LightGBM rather than XGBoost.** Both have ranking objectives. LightGBM's
leaf-wise growth and histogram binning are faster on this shape, native
categorical support avoids one-hot blowup on brand/category, and its `group`
handling for ranking is more ergonomic. XGBoost stays a dependency for the
comparison run so the choice is evidenced.

**Label construction.** Graded relevance from post-impression outcome:
purchase = 3, add-to-cart = 2, click = 1, impression-only = 0. Training rows come
from *logged recommendation impressions* once available, and from a simulated
candidate set with historical outcomes before that (bootstrap phase). The
position-bias problem this introduces is documented in `docs/evaluation.md`.

---

## ADR-006 — Redis as cache, online feature store and real-time counter store

**Status:** Accepted

**Decision.** One Redis instance with three logical namespaces: `rec:` (rendered
payload cache), `feat:` (online user features), `rt:` (real-time counters,
recently viewed, trending windows), plus `rl:` for rate limiting.

**Why.** The hot path needs the user's current feature vector in under a
millisecond; reading it from Postgres would put an OLTP query on every
recommendation request. Redis also gives us sorted sets, which is exactly the
right structure for trending (score = time-decayed velocity) and capped lists,
which is exactly right for recently-viewed.

**Consistency stance.** Redis is a **derived** store. Every key is either
reconstructible from Postgres or is a cache with a TTL. Losing Redis entirely
degrades latency and trending freshness but loses no durable data — that property
is what makes it safe to treat as ephemeral infrastructure.

**Cache key discipline.** Model version is embedded in every recommendation cache
key, so promoting a model rolls the cache over naturally instead of requiring a
flush and a cold-cache latency spike.

---

## ADR-007 — Temporal split, never random

**Status:** Accepted

**Decision.** Split interactions by time: train on `[start, T1)`, validate on
`[T1, T2)`, test on `[T2, end]`. Users are not held out; time is.

**Why.** A random split lets the model see a user's future interactions while
predicting their past, which inflates every metric — a recommender that has seen
you buy the shoes trivially "predicts" you will view them. It also leaks global
information: item popularity computed over the full period encodes the test
window. Temporal splitting reproduces the only situation that matters in
production, where the model always predicts forward.

**Consequences we accept.** Metrics get worse and more honest. Cold users appear
in the test window with no training history, which is correct — that is the real
cold-start rate. Popularity features must be computed strictly from the training
window (a rule enforced by the shared `as_of_ts` parameter in the feature
builders).

---

## ADR-008 — Split feature store: Postgres offline, Redis online, one code path

**Status:** Accepted

**Decision.** Features are computed by one set of builders in `recsys.features`.
Batch materialisation writes to `user_features` / `product_features` in Postgres;
a lightweight online path writes the same computed values to Redis. Serving reads
Redis, falls back to Postgres, and falls back again to on-the-fly computation.

**Why.** This is the training/serving skew defence (NFR-06). The failure mode we
are designing against is the common one: an offline notebook computes
`avg_order_value` one way, the API computes it slightly differently, and the model
degrades silently in production with no error anywhere. Because both call the same
function, a skew test can assert byte-equality of the produced vectors, and CI
fails if they ever diverge.

---

## ADR-009 — Synchronous event ingestion now, Kafka behind an interface

**Status:** Accepted

**Decision.** `EventSink` is an interface. v1 ships `PostgresEventSink`
(batched insert, `COPY` for bulk) with an in-process bounded queue so the request
returns before the write completes. `KafkaEventSink` is specified but not built.

**Why not Kafka now.** At the target scale (10k users, ~1M events total) Kafka
adds a broker, a schema registry, a consumer deployment and a delivery-semantics
problem to solve, in exchange for throughput we do not need. Introducing
infrastructure that carries no load is exactly the "complexity for its own sake"
the brief warns against — and it makes the local one-command bring-up (NFR-11)
materially harder.

**Why the interface still matters.** The scale story in `architecture.md` section
7 depends on being able to move ingestion to a log without rewriting the API. The
interface is the cheap part; buying it now costs one file.

**Trigger to build it.** Sustained ingest above roughly 2k events/s, or the first
requirement for a second independent consumer of the event stream.

---

## ADR-010 — MLflow registry plus a `model_versions` table

**Status:** Accepted

**Decision.** MLflow is the source of truth for experiments, parameters, metrics
and artefacts. A `model_versions` table in Postgres mirrors *which* version is
serving, since when, and with what evaluation numbers.

**Why both.** Every served recommendation is logged with a `model_version`, and
attribution analytics must join those logs to model metadata in SQL. Doing that
against MLflow's API from an analytics query is awkward and couples the dashboard
to MLflow uptime. The table is a serving contract and an audit trail; MLflow
remains the system of record for lineage. The mirror is written by the promotion
step, so it cannot drift from the registry without the promotion failing.

---

## ADR-011 — Deterministic hash bucketing for experiment assignment

**Status:** Accepted

**Decision.** `bucket = sha256(f"{experiment_key}:{user_id}") % 10000`, compared
against the variant's allocation range. Assignment is computed, not looked up;
it is persisted asynchronously to `experiment_assignments` for analysis.

**Why.** Assignment must be sticky (a user must not flip variants between
requests, or the experiment measures nothing), and it must not cost a database
read on the hot path. Hashing gives both for free, plus reproducibility: the same
user in the same experiment always lands in the same bucket, so assignment can be
recomputed for any historical analysis. Salting with the experiment key prevents
correlated assignment across concurrent experiments.

**Guardrails.** Sample-ratio-mismatch check on the observed split, minimum sample
size and minimum runtime before results are declared, and two-proportion z-test
(with a stated multiple-comparison correction) for significance.

---

## ADR-012 — `recsys` as a shared installed package, not duplicated code

**Status:** Accepted

**Decision.** `ml/recsys` is a real installable package. The backend depends on
it (editable install locally, wheel in the image). Training pipelines import the
same package.

**Why.** See ADR-008. This is also why the ML package must not import FastAPI,
SQLAlchemy models, or anything web-shaped: it takes data in and returns data out,
so it can be tested with plain fixtures and reused from a batch job, a notebook,
or the API without a database.

---

## ADR-013 — SQLAlchemy 2.0 typed ORM with Alembic migrations as the schema source of truth

**Status:** Accepted

**Decision.** Declarative models with `Mapped[...]` annotations; Alembic
autogenerate reviewed by hand before commit; migrations checked into version
control and applied on startup in development only.

**Why.** The brief requires that database models match migrations. Generating
migrations from typed models and reviewing the diff makes drift detectable in CI
(a "no pending autogenerate diff" test), rather than discovered at deploy time.

---

## ADR-014 — Adaptive hybrid weights, plus explicit popularity-bias correction

**Status:** Accepted

**Decision.** The hybrid score is a configurable weighted blend, but the weights
are a function of user history depth rather than global constants, and a
popularity discount is applied before final ranking.

```
score = w_cf(n) * cf + w_content(n) * content + w_trend * trend
      + w_affinity * affinity + w_pop(n) * popularity
```

where `n` is the user's interaction count. As `n` grows, `w_cf` rises and
`w_pop` / `w_content` fall. Weights are chosen by grid search on the validation
window against NDCG@10 subject to a diversity floor, then frozen per release and
recorded with the model version — not tuned on the test window.

**Popularity bias.** Left alone, every component of this system amplifies
popularity: popular items get more impressions, so more clicks, so more training
signal. Counter-measures, all measured rather than assumed: an inverse-propensity
discount on the popularity term, MMR re-ranking for intra-list diversity,
per-category caps, and a small exploration budget that guarantees impressions to
cold items. The evaluation report tracks catalogue coverage and Gini concentration
alongside NDCG, so a diversity regression is visible as a regression.

**Why not just fixed weights like 0.45 / 0.25 / 0.15 / 0.10 / 0.05.** Those are a
reasonable starting point and will be the initial configuration, but shipping them
as constants means a user with two clicks is scored mostly by a collaborative
model that knows nothing about them. Making the weights a function of signal
availability is the difference between a demo and a system.

---

## ADR-015 — Docker Desktop on WSL2 as the local development substrate

**Status:** Accepted (Phase 1)

**Context.** The development machine had no Docker, no working WSL (the WSL
service failed with `REGDB_E_CLASS_NOT_REGISTERED`), no PostgreSQL and no Redis.
The deliverable requires Postgres with pgvector, Redis, and a `docker compose up`
bring-up (NFR-11). Something has to host the developer loop so each phase can be
*verified* rather than merely written.

**Decision.** Repair WSL2 and install Docker Desktop. All infrastructure —
Postgres 16 with the `pgvector/pgvector` image, Redis 7, MLflow, Prometheus and
Grafana — runs in Compose from Phase 2 onward. Application code (FastAPI, the
training pipelines, Vite) runs natively on the Windows Python 3.13 and Node 24
already present, pointed at the containerised services; only the packaged images
are built in Docker. This keeps the edit-run cycle fast while making the Compose
topology real rather than aspirational.

**Why.** It is the only option with full fidelity to the target architecture:
pgvector, JSONB, native partitioning, real index behaviour, real Redis semantics
and a genuinely testable one-command bring-up. Every alternative sacrifices at
least one load-bearing part of the design.

**Alternatives considered.** Managed free tiers (Neon plus Upstash) — works
immediately and keeps pgvector, but puts network latency inside every local query
so the latency budget in NFR-01 could not be measured honestly. Native Windows
PostgreSQL plus Memurai — fast, but pgvector on Windows needs a prebuilt binary or
an MSVC build, and Memurai is Redis-compatible rather than Redis. SQLite plus
`fakeredis` — rejected: it removes pgvector, JSONB metadata and partitioning,
which are load-bearing, so several phases would be written but not verified.

**Prerequisite.** Docker Desktop must be running before Phase 2 begins;
`python scripts/check_environment.py` reports readiness.

---

## ADR-016 - Embedded PostgreSQL as the developer loop when Compose cannot run

**Status:** Accepted (supersedes the fallback clause of ADR-015)

**Context.** ADR-015 chose Docker Desktop on WSL2. On the development machine
that proved unavailable: `VirtualMachinePlatform` and
`Microsoft-Windows-Subsystem-Linux` are disabled, enabling them needs
administrator rights, and without them Docker Desktop's Linux engine will not
start.

The consequence was not cosmetic. Every database-dependent claim in the project
was unverifiable - the migration had never been applied, the seed script had
never run, no index had ever been chosen by a real planner, and the cache hit
ratio was an aspiration rather than a measurement.

**Decision.** Use `embedded-postgres`, which ships PostgreSQL 18.6 **with
pgvector 0.8.6** as a pip wheel and runs from a user-writable directory, driven
by `scripts/local_postgres.py`.

**Why this is not a downgrade.** It is the same PostgreSQL, the same pgvector,
the same migration and the same connection string. Only the packaging differs,
and nothing in the application knows or cares which one is running. The
server is started with the same tuning as the Compose service, so the planner
makes the same choices.

The alternative considered was a managed free tier (Neon plus Upstash). It was
rejected for the developer loop because it puts network latency inside every
local query, which would make the NFR-01 measurements dishonest - the whole
point of being able to measure them.

**What this unblocked, immediately.** The migration applied on the first
attempt; 561,173 rows seeded in 17.7 s; all 24 database verification checks
pass; query plans confirm partition pruning and partial-index usage; the
schema-drift check runs and reports no drift; and the serving benchmark
measured an 81.6% cache hit ratio with a cached p95 of 6.3 ms.

It also found four real bugs that only a live database could surface: pandas
promoting a nullable integer column to float and emitting `1.0` for a BIGINT;
`pd.NA` not being caught by the CSV null check; the generator emitting two
order lines for the same variant, violating a unique constraint that was
correct and a generator that was not; and - the serious one - the live
ingestion path never creating the `user_sessions` row its events referred to,
so in production the session table would have held only synthetic rows and
session-scoped cold-start retrieval would have silently had nothing to read.

That last one is the justification for this ADR on its own. It was invisible in
every unit test, because the unit tests use an in-memory sink, and invisible in
the seeded data, because the seeder writes both tables. It took real API
traffic against a real database to surface, which is exactly what a decision to
run without a database would have postponed indefinitely.

**Docker Compose remains the deployment target.** This is the developer loop,
not a replacement for it.

**Revisit if.** WSL2 becomes available, in which case Compose is preferred for
parity with deployment - though the embedded path is worth keeping for CI-free
local work and for contributors without Docker.
