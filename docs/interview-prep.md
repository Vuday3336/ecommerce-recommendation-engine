# Interview Preparation

Status: **Phase 20 deliverable — complete**
Last updated: 2026-09-12

Every answer here points at code or a measured number in this repository. The
value of the project is not that it has a recommender — it is that every
decision has evidence behind it, including the ones that did not work.

**Three things to lead with, because they are unusual:**

1. The implicit-feedback weights were **derived from conversion odds**, not
   asserted.
2. The best model's 4% lift is **not statistically significant**, and the
   report says so.
3. Hard-negative sampling — the textbook trick — **halved accuracy**, and the
   investigation of why is more interesting than if it had worked.

---

## ML

### Why collaborative filtering?

Because it finds patterns no amount of metadata can. Content-based similarity
can tell you two running shoes resemble each other; only behaviour tells you
that people who buy this particular shoe also buy a specific foam roller.

It is not sufficient on its own — measured here, ALS scores **exactly 0.0000**
for users with no history, which is most first-time traffic. That is why it is
one component of a hybrid, weighted by how much history a user actually has.

> `ml/recsys/models/collaborative.py`, ADR-003

### Why implicit feedback, and why weighted?

We have no reliable ratings at volume — only behaviour. But behaviours differ
enormously in meaning, and treating a view and a purchase as the same "1" throws
away most of the signal.

The unusual part is **how** the weights were set. Rather than asserting
`purchase = 10`, each weight is derived from the observed odds:

```
w(e) ∝ log(1 + P(purchase | e) / P(purchase))
```

Result: purchase 5.71, cart 2.79, click 1.59, view 1.00, remove-from-cart −0.83.

The framing that makes this coherent: in ALS, `c_ui = 1 + α·r_ui` is
**confidence**, not preference — "how sure are we this user likes this item?" —
which is exactly what a conversion-odds ratio measures.

**If pushed:** the window is symmetric around the event, not forward-only,
because ratings and reviews are inherently post-purchase. A forward-only window
scores them at zero, implying a five-star rating says nothing about preference.

> `ml/recsys/preprocessing/weighting.py`, ADR-002

### Why hybrid?

Because each component fails somewhere different, and the failures are
complementary: collaborative filtering is blind to new users and new items;
content-based cannot capture cross-category taste; popularity is not
personalised at all.

The part worth defending is that **the blend weights are a function of user
history depth, not constants**. Tuned per segment on validation:

| | cold | sparse | warm | rich |
| --- | --- | --- | --- | --- |
| Collaborative | 0.05 | 0.30 | 0.42 | **0.55** |
| Popularity | **0.75** | 0.30 | 0.18 | 0.05 |

Fixed weights would score a user with two clicks 45% by a model that knows
nothing about them.

> ADR-014, `ml/recsys/models/hybrid.py`

### Why two-stage retrieval and ranking?

Cost. Scoring every user against every product with a rich feature model is
O(users × items × features). At 5,000 products that is survivable; at 500,000 it
is not.

Splitting lets each stage optimise for a different thing: stage 1 for
**recall@300** (did the right item survive?), stage 2 for **NDCG@10** (is it
near the top?). A single-stage model has to trade those against each other.

Measured: retrieval 4 ms, ranking 12 ms, p95 total **19.3 ms** against an 80 ms
budget.

**The number to quote:** stage-1 recall is 0.343. Two thirds of relevant items
never enter the pool, so retrieval — not ranking — is where the next
improvement is.

> ADR-001, `ml/recsys/candidates/`, `ml/recsys/ranking/`

### How do you handle cold start?

Four distinct cases, not one:

| | Approach |
| --- | --- |
| New user, no session | Trending + category popularity, diversified |
| New user, mid-session | Co-visitation from what they have viewed — works from the first page view |
| Sparse user | Content + popularity; hybrid weights shift automatically |
| New product | Text and attribute embedding places it in the content space immediately |

**The measurement that makes this concrete:** ALS scores 0.0000 for cold users.
Not a bug — a matrix factorisation has no vector for someone with no
interactions. Without the popularity and content components, 100% of cold
traffic would get nothing.

For new products there is also a small **exploration budget** in candidate
generation. It contributes almost nothing to recall, by design — its purpose is
to guarantee impressions so cold items can accumulate the signal the
collaborative model needs. Judging it on recall measures the wrong thing.

### How do you prevent data leakage?

Four defences, and only the first is the obvious one:

1. **Temporal split**, never random. A random split lets the model see a user's
   future while predicting their past.
2. **`as_of` on every feature builder**, with `assert_features_respect_cutoff`
   raising if any input row is at or after the cut-off. It is tested — it fires.
3. **Weight calibration on the training fold only.** Conversion odds computed
   over the full period would leak the test window into the weights, which feed
   the matrix, which trains the model.
4. **Hyperparameters selected on validation, test read once.** Selecting on test
   makes every number a best-of-N maximum — the same leak, one level up.

**The one exposure we admit:** the ranker's early-stopping set comes from the
same validation window as its labels. Test is untouched, so the numbers are
honest, but a strict four-way split would be cleaner. It is documented rather
than hidden.

> ADR-007, `ml/recsys/preprocessing/splitting.py`, `docs/evaluation.md` §2

### How do you evaluate recommendations?

Ten metrics in two groups, because accuracy alone has a degenerate solution.

**Accuracy:** Precision@K, Recall@K, MAP@K, NDCG@K, HitRate@K, MRR.
**Beyond accuracy:** coverage, intra-list diversity, novelty, serendipity,
personalisation, Gini exposure.

The easiest way to raise NDCG is to recommend the same 50 popular products to
everyone. Coverage and diversity are therefore **gate metrics** — a model that
wins NDCG while collapsing coverage is rejected by the promotion gate.

Results are also **segmented by user history depth**, because an aggregate hides
the group that matters: most real traffic is sparse.

**Serendipity** is the one worth explaining: relevant hits the popularity
baseline would *not* have produced. A model that only re-finds what popularity
already surfaces adds nothing, however good its NDCG looks.

> `ml/recsys/evaluation/metrics.py`, `docs/evaluation.md`

### How do you handle popularity bias?

It is a feedback loop: popular items get more impressions → more clicks → more
training signal → more impressions. Four counter-measures, all **measured**
rather than assumed:

1. Inverse-popularity discount on the popularity term
2. MMR re-ranking plus hard category and brand caps
3. An exploration budget guaranteeing impressions to cold items
4. Coverage and Gini as gate metrics

The measured trade: diversity re-ranking buys **+47% intra-list diversity** for
**−11.5% NDCG** — and that cost is not statistically significant (p = 0.057).

**The interesting finding:** catalogue coverage *fell* (0.284 → 0.207). The caps
push each individual list toward distinct categories, but push every user's list
toward the same broad popular ones, so the union narrows. Intra-list diversity
and catalogue coverage are different objectives, and this configuration trades
one for the other.

### How do you improve diversity?

MMR with λ = 0.72, plus hard caps of 4 per category and 3 per brand. MMR alone
lets a strong category dominate when its items are also the most relevant; the
caps are what stop a rail being five running shoes — the failure users actually
complain about.

### How do you monitor model degradation?

Three layers:

| Layer | Signal | Detects |
| --- | --- | --- |
| **Serving** | `recommendation_fallback_total` | The model is not running at all |
| **Distribution** | `recommendation_score` histogram | Inputs have shifted — the earliest observable sign |
| **Input** | PSI + KS per feature | The world has moved from the training snapshot |

**The one I would page on is fallback rate.** When the ranker fails, latency
*improves*, errors stay at zero, and the only other symptom is a slow CTR
decline a week later.

Drift uses PSI for alerting and KS for corroboration. Not the p-value alone: at
hundreds of thousands of rows a KS test rejects the null for effects far too
small to matter, so significance and importance come apart.

**Retraining needs two alerting features**, not one. A single feature moves for
benign reasons — a merchandising change, a category launch — and retraining on
every such event means retraining constantly.

---

## Backend

### How does the recommendation API scale?

Four things, in order of impact:

1. **Two-stage retrieval** — cost is O(sources × k), not O(catalogue)
2. **Redis cache with versioned keys** — the model version is part of every key,
   so a promotion rolls the cache over naturally instead of needing a flush and
   the cold-cache latency spike that follows
3. **In-memory catalogue** — 5,000 products is ~2 MB; reading them from Postgres
   per request puts an OLTP query on the hot path for data that changes hourly
4. **Stateless API** — scales horizontally; all state is in Postgres or Redis

At 1M users: move item kNN to a dedicated ANN index, shard the feature store,
precompute homepage candidates for the long tail and compute online only for
active users.

### Why Redis?

Sub-millisecond reads for things touched on **every** page view: the user's
feature vector, trending, recently-viewed. Postgres is 5–50 ms; at p95 < 80 ms
that difference is most of the budget.

Its data structures also fit: sorted sets are exactly right for time-decayed
trending, capped lists for recently-viewed.

**The important property is that Redis is a *derived* store.** Every key is
either reconstructible from Postgres or a cache with a TTL. Losing Redis
entirely costs latency and trending freshness — nothing durable. That is what
makes it safe to treat as ephemeral infrastructure.

### How do you reduce latency?

Measured p95 of 19.3 ms uncached, from:

- **Vectorised feature assembly** — user features broadcast once, product
  features one reindex, pair features one call. A Python loop over 300
  DataFrame rows would cost tens of ms
- Artefacts loaded once at start-up, never per request
- Catalogue in memory
- Cache-aside with stale-while-revalidate
- `pool_pre_ping` so a stale connection is not a user-visible 500

### How do you handle concurrent requests?

Uvicorn workers, a stateless API, and a connection pool with overflow. The one
piece needing care is event ingestion: a **bounded queue** with a background
worker.

Three properties are deliberate:

- **Bounded** — an unbounded queue turns a database outage into an OOM crash of
  the API; the failure spreads instead of staying contained
- **Drops rather than blocks** when full — blocking would push database latency
  straight into the page render
- **Write failures do not kill the worker** — one that dies on the first
  transient error silently stops ingestion for the process lifetime

All three are tested.

### How do you version models?

MLflow is the system of record; a `model_versions` table mirrors which version
is serving.

**Why both.** Every served recommendation is stamped with a `model_version_id`,
and attribution analytics must join those rows to model metadata in SQL. Doing
that against MLflow's API from a dashboard query couples reporting to MLflow
uptime. A partial unique index enforces at most one Production version per model
name, so two concurrent promotions fail loudly rather than leaving two live
models.

MLflow is **not on the serving path**: it can prevent a model *change*, never a
recommendation.

---

## Data engineering

### How are events collected?

Browser → batched queue → `POST /events/batch` → validated → `EventSink`.

Three guarantees:

1. **Never blocks a render.** Queue + timer flush, non-blocking enqueue
2. **Never loses a session tail.** Unload flush uses `sendBeacon`, because a
   normal request is cancelled when the document goes away — and the events
   nearest a conversion are the last ones
3. **Never breaks the page.** Every failure path swallowed

`visibilitychange` rather than `beforeunload`: mobile browsers routinely never
fire the latter when an app is backgrounded.

`user_id` comes from the token, never the body — otherwise any client could
write events attributed to another user.

The sink also upserts the `user_sessions` row for each batch, in the same
transaction as the event `COPY` — one statement per batch, not per event, on the
background worker. That is there because it was missing, and the story is worth
telling (see below).

### How is feature data generated?

Batch builders in `recsys.features`, each taking an `as_of` cut-off. Offline
materialisation writes to Postgres; a lightweight online path writes the same
computed values to Redis.

**The crucial property is that both call the same function.** The failure being
designed against is the common one: an offline notebook computes
`avg_order_value` one way, the API slightly differently, and the model degrades
silently. A test asserts the ranker's feature columns match the builder's
exactly, including order.

### How would Kafka fit?

`EventSink` is an interface. v1 ships `PostgresEventSink` behind
`BufferedEventSink`; `KafkaEventSink` implements the same protocol.

**Why not now:** at ~1M events, Kafka adds a broker, a schema registry, a
consumer deployment and a delivery-semantics problem in exchange for throughput
we do not need — and it makes the one-command local bring-up materially harder.

**Trigger to build it:** sustained ingest above ~2k events/s, or the first
requirement for a second independent consumer.

### How would the system handle millions of events?

| Stage | Change |
| --- | --- |
| Now | Partitioned `user_events`, batched COPY |
| 10k/s | Kafka ingestion; a stream consumer maintains Redis counters |
| 100M/day | Postgres keeps aggregates only; training reads from the lake |

`user_events` is already partitioned monthly. That buys three things: retention
becomes `DROP PARTITION` instead of a multi-hour `DELETE`; the training extract
is a partition scan; and autovacuum works per partition instead of fighting one
enormous table.

---

## MLOps

### How are models tracked?

MLflow, per run: all configuration as parameters, every metric for every model
and segment (**1,255** in the last run), evaluation tables as artefacts, and a
**dataset fingerprint** tag hashing the split boundaries and row counts.

That fingerprint answers what is otherwise archaeology: two runs with identical
parameters and different metrics become explainable ("the data changed") rather
than mysterious.

### How are models promoted?

Through a gate that can say no:

| Criterion | Requirement |
| --- | --- |
| `ndcg@10` | > incumbent + 0.001 — a coin-flip must not trigger a deployment |
| `hit_rate@10` | ≥ incumbent |
| `coverage` | ≥ incumbent − 0.05, floor 0.05 |

The coverage guardrail exists because the degenerate solution — recommend the
same popular items to everyone — wins on NDCG and collapses coverage. Tested.

Rejected candidates are **kept** with the decision recorded; discarding them
would make "why was this rejected?" unanswerable. Artefact swaps are atomic.

### How do you detect drift?

PSI and KS per monitored feature, against the **training** snapshot rather than
a rolling recent window — comparing to last week lets slow drift pass unnoticed
while the cumulative distance grows without bound.

Bin edges come from the reference, not the combined data; recomputing them each
run moves the goalposts with the data.

Verified: identical data → 0 alerts; injected shift → 3 alerts and "retrain
recommended".

### How does retraining work?

```
drift check → retrain → evaluate → gate → promote or reject
```

Drift-triggered, not unconditional. Nightly at 03:00 UTC. Exit code 2 means
"retrained and rejected" — a real signal, not a failure.

### How do you safely deploy a new model?

1. Gate decides on evidence
2. Artefacts written to staging, then swapped atomically
3. Previous version kept for rollback
4. Cache keys embed the model version, so promotion rolls the cache over — and
   so does a rollback
5. `model_info` gauge lets any metric be joined to the version that produced it
6. **Next step, not yet built:** an A/B test rather than a wholesale swap

---

## System design: 100 → 100M

| Scale | Shape | Architecture |
| --- | --- | --- |
| 100 users | anything works | Single Postgres, on-demand training |
| **10k users** (this build) | ~450k events | Nightly batch training, Redis cache, ALS in memory, pgvector kNN |
| 1M users, 500k products | ~100M interactions | ANN index (FAISS/Milvus), sharded feature store, precomputed candidates for the long tail, read replicas |
| 100M events/day | ~10k/s peak | Kafka ingestion, stream consumer for counters, Postgres aggregates only, training from the lake |

**The seams that make this cheap exist now:** `EventSink`, `CandidateSource`,
`VectorIndex`, `ModelResolver`. Buying the interface costs one file; buying the
infrastructure before it carries load costs a deployment nobody needs.

---

## Questions to expect back

### "Your model is only 4% better than popularity. Is it worth it?"

**Honest answer: not proven yet, and the report says so.** The paired bootstrap
gives p = 0.297 — that 4% could be luck at n = 2,000.

What it *is* worth: the infrastructure is in place to find out. A 4% CTR lift is
easily detectable online with a few tens of thousands of sessions, where it is
not detectable offline with 2,000 users. That is why the ranker ships as the
treatment arm of an A/B test rather than as an unconditional replacement.

The wrong answer would be to quote the +337% improvement over *global*
popularity — technically true, and dishonest by baseline selection.

### "Why is category popularity so strong?"

Because a large part of what "personalisation" means in e-commerce is knowing
which aisle someone shops in. Users concentrate 71% of their interactions in one
department. A model that gets the department right has already captured most of
the available signal.

It also explains the tuned hybrid weights: popularity carries 0.75 for cold
users and 0.05 for rich ones — the same fact expressed as a blending policy.

### "Your LambdaRank barely beat a logistic regression. Doesn't that undermine ADR-005?"

Yes, partly, and the ADR was **amended rather than defended**. The measured gap
is +0.9% relative — indistinguishable.

Two readings the evidence does not separate: either the features already encode
most of the ranking signal, or 1,596 queries is too small for a listwise loss to
show its advantage. LambdaRank stays because it wins on the point estimate and
on hit-rate@10, but the *claim* "listwise beats pointwise" is not supported by
this dataset and should not be asserted without the caveat.

### "Tell me about a bug you found."

`user_events` has no foreign keys — deliberately, because a partitioned,
append-only log ingesting thousands of rows a second cannot pay a referential
check per row. That trade is only defensible if something else validates the
references, so `scripts/verify_database.py` counts orphans and the nightly
pipeline repeats the check.

It read zero for months. Then the first real API traffic hit a real database —
the integration tests and the serving benchmark — and it read 394.

Nothing had written a bad row. Something had failed to write a row at all: the
ingestion path wrote `user_events` and nothing anywhere created the
`user_sessions` record those events pointed at. The seeder writes both tables,
so the seeded data looked perfect; the unit tests use an in-memory sink, so they
could not see it either.

The consequence was not a crash. In production `user_sessions` would have held
only synthetic rows, and session-scoped retrieval — which is the highest-value
cold-start lever, the thing that turns three page views into a usable signal for
a visitor with no history — would have had nothing to read. No error, no alert,
no failing test. Just recommendations that were quietly worse for exactly the
users they were designed to rescue.

Two things about it are worth saying out loud. First, it is the strongest
argument for the compensating control: dropping foreign keys was defensible only
*because* something else checked, and the check earned its place the first time
it had real data to check. Second, it is the strongest argument for ADR-016 —
running a real PostgreSQL locally instead of deferring the database. A design
that looks correct on paper and passes every in-memory test can still be wrong
in the one way that matters.

The fix is an upsert per batch that accumulates `event_count`, widens the
session window with `LEAST`/`GREATEST` so out-of-order batches stay correct, and
coalesces `user_id` so the anonymous-to-signed-in stitch is one-way. Two
integration tests now hold it in place.

### "What would you do differently?"

1. **Invest in retrieval, not ranking.** Recall@300 is 0.343 — two thirds is
   unreachable regardless of the ranker. `category_affinity` returns the most
   recall per slot (0.186 from a budget of 60 versus 0.245 from 150); rebalancing
   is the obvious next experiment.
2. **Get real impression logs.** Bootstrap labels are missing-not-at-random:
   items the user never saw are labelled 0 regardless of whether they would have
   liked them.
3. **Validate the Docker path.** It is written and unrun, which is the largest
   honest gap in the project.

### "What's the biggest risk you identified?"

That the synthetic data would be **secretly random** — in which case every model
scores at chance, the hybrid cannot beat the baseline, and the whole evaluation
is fiction.

So the dataset does not pass on row count. It passes a twelve-check diagnostic
proving the injected structure is *recoverable from behaviour alone*: users
concentrate 0.71 on one department, preferred categories convert 3.7× better,
declared complementary pairs show 18.9× co-purchase lift against 0.00 for random
pairs. That gate runs in CI on every commit.
