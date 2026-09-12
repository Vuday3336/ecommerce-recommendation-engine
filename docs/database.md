# Database Design

Status: **Phase 2 deliverable — complete and verified against a live database**
Last updated: 2026-09-12

20 tables, 271 columns, 65 indexes, 85 constraints, 12 native enum types, one
partitioned table. The schema is defined by the SQLAlchemy models in
`backend/app/models/` and applied by `backend/alembic/versions/0001_initial_schema.py`.

Render the full DDL at any time without a database:

```bash
python scripts/render_ddl.py --summary
```

---

## 1. Domain map

```
categories ──┐                                brands ──┐
             │  (self-referencing tree)                │
             ▼                                         ▼
           products ─────────────────────────< product_variants
             │  │  │                                   │
             │  │  └──< product_features               │
             │  └─────< product_embeddings (vector)    │
             │                                         │
users ──< orders ──< order_items >─────────────────────┘
  │                      │
  │                      └──> recommendations  (revenue attribution)
  ├──< user_sessions
  ├──< user_features
  ├──< user_product_interactions >── products
  ├──< experiment_assignments >── experiments
  └──< recommendations ──< recommendation_impressions
                        ├──< recommendation_clicks
                        └──< recommendation_conversions

user_events   (partitioned by month, no foreign keys — see section 4)
model_versions (serving mirror of the MLflow registry)
```

Four groups, each with a different job:

| Group | Tables | Access pattern |
| --- | --- | --- |
| **Catalogue** | `categories`, `brands`, `products`, `product_variants` | read-heavy, small, cached |
| **Behaviour** | `user_events`, `user_sessions`, `user_product_interactions` | write-heavy, unbounded growth, bulk-read for training |
| **Features & models** | `user_features`, `product_features`, `product_embeddings`, `model_versions` | batch-written, point-read on the hot path |
| **Serving & experiments** | `recommendations`, impressions/clicks/conversions, `experiments`, `experiment_assignments` | append-only fact tables, aggregated by analytics |

---

## 2. Conventions applied everywhere

- **`BIGINT` surrogate primary keys.** `INT` would overflow on `user_events` and
  the recommendation log at the scale in `architecture.md` section 7, and
  widening a primary key later means rewriting every referencing table.
- **A separate public `UUID`** on `users`, `products` and `orders`. Internal
  joins use the integer; anything exposed over the API uses the UUID, so
  external parties cannot enumerate rows or infer volumes from ids.
- **Timezone-aware timestamps only.** A test asserts no `TIMESTAMP WITHOUT TIME
  ZONE` column exists. Mixing naive and aware timestamps produces feature
  windows that are silently wrong by an hour rather than an error.
- **Business time is separate from row time.** `orders.placed_at` and
  `user_events.occurred_at` are the business timestamps; `created_at` /
  `ingested_at` record when the row was written. The synthetic generator
  backfills 180 days in one batch, so conflating the two would collapse the
  entire temporal split.
- **Money is `NUMERIC(12,2)`, never `FLOAT`.** Enforced by a test. Binary
  floating point cannot represent `0.10`, so summing float prices drifts and
  attributed revenue stops reconciling with the orders table.
- **Named constraints via a metadata naming convention.** Without it Alembic
  emits unnamed constraints that cannot later be dropped by name, and the
  "no pending autogenerate diff" CI check would be permanently red.
- **Native enums, not text.** 4 bytes instead of a varying-length string, and an
  invalid value becomes a database error rather than a data-quality bug found
  three phases later.

### Soft deletion

`users`, `products`, `product_variants`, `categories` and `brands` carry
`is_active` and `deleted_at`, and every foreign key pointing at them is
`ON DELETE RESTRICT`. The reason is specific rather than stylistic: a
recommendation logged six months ago must stay joinable to the product it
recommended. Hard deletion would make historical attribution silently lose rows
— the analytics would still run, just quietly wrong.

`is_active` is the flag the hot path filters on because it is cheap to index;
`deleted_at` records when the state changed.

---

## 3. Extensibility of the event model

One table, one enum-backed `event_type`, one JSONB `event_metadata`.

Adding an event type is `ALTER TYPE event_type ADD VALUE` plus a member on the
Python enum. No table is rewritten, no column changes shape (FR-21).

JSONB is not a dumping ground: everything the feature pipeline filters or groups
on is a real, indexed column (`user_id`, `product_id`, `session_key`,
`event_type`, `occurred_at`, `source`, `device_type`). `event_metadata` holds
only the per-type payload — search terms, rating values, cart quantities — and
each event type has a declared Pydantic payload model validated at the API
boundary in Phase 3.

`product_id` is nullable because `SEARCH`, `SESSION_START` and `SESSION_END`
have no product. A `CHECK` constraint enforces that the nine product-scoped
event types *do* carry one, so the nullable column cannot hide a bug.

---

## 4. `user_events`: partitioning and the missing foreign keys

### Partitioning

```sql
CREATE TABLE user_events (...)  PARTITION BY RANGE (occurred_at);
```

Monthly range partitions, created by the migration for 13 months back and 3
forward, plus a `DEFAULT` catch-all. Three concrete payoffs:

1. **Retention is `DROP TABLE`**, not a multi-hour `DELETE` followed by a bloat
   problem.
2. **The training extract is a partition scan.** Reading "March through July"
   touches five partitions instead of index-scanning the whole log.
3. **Autovacuum works per partition** instead of fighting one enormous table.

The composite primary key `(id, occurred_at)` is not a modelling preference —
PostgreSQL requires the partition key in every unique constraint on a
partitioned table.

**The `DEFAULT` partition is a deliberate trade with a real cost.** It means an
event outside every declared range is stored rather than rejected. But once a
row for month *M* lands there, PostgreSQL refuses to attach a partition for *M*
until that row is moved. So the migration also creates:

```sql
ensure_user_events_partition(target date)
```

which scheduled maintenance calls to keep declared ranges ahead of ingestion,
and `scripts/verify_database.py` asserts the default partition is empty.

### Why there are no foreign keys on this table

A partitioned, append-only log ingesting thousands of rows per second cannot pay
a referential check per row, and an event must be accepted for a product that is
mid-deletion rather than rejected at the API edge. This is the standard trade for
event logs.

It is only defensible because something else validates it:
`scripts/verify_database.py` checks for orphaned product, user and session
references, and the nightly pipeline validation step repeats the check. A trade
made without the compensating control would just be a missing constraint.

---

## 5. Index design

Indexes are chosen per access path, not by reflex. Every one below serves a
query the system actually runs.

| Index | Serves |
| --- | --- |
| `ix_user_events_user_id_occurred_at (user_id, occurred_at DESC)` | "this user's recent behaviour" — the single most frequent read in feature building |
| `ix_user_events_product_id_event_type` | per-product funnel counts |
| `ix_products_active_category (category_id, price) WHERE is_active AND stock_quantity > 0` | category candidate generation |
| `ix_products_active_brand ... WHERE is_active AND stock_quantity > 0` | brand-affinity candidates |
| `ix_products_search_tsv` GIN on `to_tsvector(name ‖ description)` | keyword stage of personalised search |
| `ix_products_attributes` GIN on JSONB | attribute filters |
| `ix_product_embeddings_hnsw_cosine` HNSW `vector_cosine_ops` | content kNN |
| `ix_product_features_trending / _popularity` | trending and popularity baselines |
| `ix_user_product_interactions_user_weight (user_id, implicit_weight DESC)` | the user's strongest affinities |
| `ix_user_product_interactions_product_id` | item-major direction for item-item neighbours |
| `ix_recommendations_*` (user, product, surface, model version, experiment) | attribution analytics |
| `uq_model_versions_one_production_per_name (name) WHERE stage='production'` | at most one live model per name |

Two patterns worth calling out:

**Partial indexes on the catalogue.** Every candidate source filters active,
in-stock products. Indexing only those rows keeps the index roughly an order of
magnitude smaller once soft-deleted and out-of-stock rows accumulate.

**A partial *unique* index as a business rule.** `uq_model_versions_one_production_per_name`
makes "which model is live?" unambiguous at the database level. Two concurrent
promotions fail loudly instead of leaving two production models — enforcing it in
application code would leave that race open.

**Every foreign key leads an index.** This is asserted by
`backend/tests/test_models.py::test_every_foreign_key_column_is_indexed`. An
unindexed foreign key turns every parent delete and join into a sequential scan;
it is the most common schema performance bug, so it is tested rather than
reviewed. That test found three real misses (`orders.session_id`,
`order_items.variant_id`, `recommendation_conversions.order_item_id`) on its
first run.

### HNSW rather than IVFFlat

HNSW needs no training step and no periodic rebuild as rows are added. IVFFlat
requires a training pass over existing vectors and degrades as the catalogue
grows until it is retrained — which would make new-product cold start
intermittently worse in a way nothing would alert on. Cosine distance is used
because sentence-transformer output is directionally meaningful but not
normalised in a way that makes L2 sensible.

---

## 6. The recommendation log is a fact table

`recommendations` holds **one row per served item**, carrying the request
context (strategy, model version, experiment variant, latency, cache hit) on
every row rather than in a separate request dimension.

This is deliberate denormalisation. These rows are written once and read by
analytics forever, and every dashboard query — CTR by surface, conversion by
candidate source, revenue by model version, position-bias curves — would
otherwise pay a join to a request table it never filters on independently.
Item-level grain makes "most recommended products" and "conversion rate by
source" single-table `GROUP BY`s instead of unnesting a JSONB array.

Impressions are a **separate table from serving** because an API response is not
an impression. A rail below the fold is served but never seen, and counting it
would depress CTR for exactly the surfaces that work. The frontend reports
impressions from an `IntersectionObserver`, so CTR's denominator is "items a
human could see".

`recommendation_conversions` stores `attribution_window_hours` and
`attribution_model` **per row**, because attribution is a policy, not a fact.
Changing the window from 24 hours to 7 days changes every historical number, and
without recording the policy in force at write time nobody can explain why last
quarter's revenue moved.

---

## 7. Feature tables are a snapshot, not a time machine

`user_features` and `product_features` hold the **current** materialisation for
online serving. They are not a historical store, and training must never read
them for a past label — that would attach features computed today to an
interaction from three months ago, which is precisely the leakage ADR-007
forbids.

Training features are computed by the same `recsys.features` builders with an
explicit `as_of_ts` cut-off. `as_of_ts` is recorded on every row here so a
serving-time vector can be reconciled against the offline one, which is what the
skew test (NFR-06) compares.

`products.rating_average` / `view_count` / `purchase_count` are a *third* thing:
denormalised counters for the catalogue API, so a shopper sees a star rating
without a join. They are intentionally not the same numbers a model trains on.

---

## 8. The synthetic dataset

Generated by `data-generation/`, seeded by `scripts/seed_database.py`.

```bash
python data-generation/generate.py        # ~28s
python data-generation/diagnostics.py     # the Phase 2 gate
python scripts/seed_database.py --truncate
python scripts/verify_database.py
```

| | |
| --- | --- |
| Users | 10,000 |
| Products | 5,000 across 97 categories and 33 brands |
| Product variants | 13,646 |
| Events | 438,443 over 180 days |
| Sessions | 51,510 |
| Orders / order items | 10,364 / 15,087 |
| Interaction matrix density | 0.33 % |

### What makes it learnable

Structure was injected deliberately and is verified as *recoverable from
behaviour alone* by `diagnostics.py`, which is the Phase 2 exit gate. All twelve
checks pass on the committed dataset:

| Hypothesis | Measured |
| --- | --- |
| Users concentrate on a few departments | median top-department share **0.71** (uniform would be 0.125) |
| Preferred categories convert better | **3.70×** view-to-purchase lift |
| Brand affinity shows in behaviour | **4.86×** over catalogue-share baseline |
| Price behaviour tracks latent target | correlation **0.34** |
| Complementary pairs co-purchase | median lift **18.9×** (unrelated pairs: 0.00 median, 0.70 at p90) |
| Consecutive views stay related | **27×** over independence |
| Segments differ in spend | high-value spends **41×** window-shoppers |
| Popularity is unequal | Gini **0.49**; top 1 % of products take 12.8 % of views |
| Matrix is sparse | **0.33 %** dense |
| Repeat purchase exists | **9.9 %** of user-product purchase pairs repeat |
| Cold start is exercised | 2,185 users with no events, 595 with ≤3 |
| Temporal coverage is complete | 181/181 active days |

The latent user taste vectors used to *verify* these are written to
`latent_user_profiles.parquet` and are **never loaded into the database** — a
test asserts they are absent from the seed plan. Training on them would be
training on the answer key.

### An honest deviation

The generated funnel converts more than a real storefront: **8.2 % of product
views** and **20 % of sessions** end in a purchase, against a realistic 2–3 % of
sessions.

This is a deliberate trade, stated rather than hidden. At a realistic 2 %
conversion, 10,000 users over 180 days produce roughly 1,500 purchases across
5,000 products — under one purchase per product. The co-purchase matrix would be
too sparse for frequently-bought-together to be evaluated at all, and Phase 8
would be comparing noise. The elevated rate buys 15,087 purchase lines and a
measurable FBT signal.

Two consequences follow, and both are handled:

- **Absolute conversion numbers from this dataset are not benchmarks.** Only
  *relative* comparisons between models are meaningful, which is all the
  evaluation report claims.
- **Getting realistic rates means more traffic, not different behaviour.** Raise
  `SEGMENT_BEHAVIOUR` session rates and lower `FunnelConfig` transitions
  together in `data-generation/config/simulation.py`; the structure is unchanged.

---

## 9. Verified against a live database

Everything below was executed against PostgreSQL 18.6 with pgvector 0.8.6,
run from a pip wheel without Docker or administrator rights (ADR-016).

**Migration.** `alembic upgrade head` applied cleanly on the first attempt:
20 tables, 82 named indexes, 12 native enum types, 18 monthly partitions and
the HNSW vector index.

**Seeding.** 561,173 rows in **17.7 seconds** via `COPY`:

| Table | Rows | Time |
| --- | --- | --- |
| `user_events` | 452,591 | 12.7 s |
| `user_sessions` | 50,970 | 1.6 s |
| `order_items` | 17,316 | 0.7 s |
| `product_variants` | 13,763 | 0.2 s |
| `orders` | 11,403 | 0.4 s |
| `users` | 10,000 | 0.2 s |
| `products` | 5,000 | 0.4 s |

**Verification.** All 24 checks in `scripts/verify_database.py` pass, including
18 partitions attached with **zero rows in the default partition**, no orphaned
event references, and 12 distinct event types present.

**Query plans.** The index design holds up under a real planner:

| Query | Plan | Time |
| --- | --- | --- |
| Recent events for one user | Partition pruning + index scan per partition | **0.26 ms** |
| Active products in a category | `ix_products_active_category` (the partial index) | **0.05 ms** |
| Co-purchase pairs for a product | `ix_order_items_order_id` | **0.14 ms** |

No sequential scans anywhere.

**Schema drift.** `alembic revision --autogenerate` against the migrated
database produces **no operations** — the ORM models and the applied migration
agree exactly, which is ADR-013's guarantee.

One exclusion was needed and is worth recording: PostgreSQL normalises index
expressions when it stores them, so `to_tsvector('english', name || ' ' ||
description)` comes back as `to_tsvector('english'::regconfig, (name::text ||
' '::text) || description)`. Alembic string-compares those, finds them
different, and proposes dropping and recreating an index that is correct and
unchanged — on every run. Left in, that makes the CI drift check permanently
red, which trains everyone to ignore it. Expression indexes are therefore
excluded from autogenerate in `alembic/env.py` and managed by hand.

### Four bugs that only a live database could find

1. **`1.0` for a BIGINT.** pandas has no integer dtype that holds NaN, so
   `categories.parent_id` — which has exactly one null, the root — was silently
   promoted to float64. `COPY` rejected the entire load. Fixed by casting
   integral float columns back to pandas' nullable `Int64`.
2. **`<NA>` reaching PostgreSQL.** After that fix, the CSV writer's null check
   handled `None`, `nan` and `NaT` but not `pd.NA`, so the literal string
   `<NA>` was written. Every pandas null spelling now goes through one helper.
3. **Two order lines for the same variant.** The generator could put a product
   in a basket twice, violating `uq_order_items_order_id_variant_id`. The
   constraint was right — a real order has one line per variant with a quantity
   — and the generator was wrong. Duplicates are now merged into one line.

4. **Every live session orphaned.** `scripts/verify_database.py` counts events
   whose `session_key` matches no row in `user_sessions`. It read zero for
   months, because the only thing writing events was the seeder, which writes
   both tables. The first time real API traffic reached a real database - the
   integration tests and the serving benchmark - the count jumped to 394.

   The cause was not a bad write. It was a missing one: `PostgresEventSink`
   wrote `user_events` and nothing anywhere created the session. `user_events`
   carries no foreign keys by design, so nothing rejected the orphan either.
   In production `user_sessions` would have contained only synthetic rows, and
   session-scoped retrieval - the highest-value cold-start lever in
   architecture.md section 5 - would have had nothing to read for exactly the
   brand-new visitors it exists to serve. No error, no alert, no failing test:
   just recommendations that were quietly worse for the hardest users.

   The sink now upserts one session row per batch in the same transaction as
   the `COPY`, accumulating `event_count`, widening the `started_at`/`ended_at`
   window, and coalescing `user_id` so the anonymous-to-signed-in stitch
   (FR-22) is one-way. It is one statement per batch rather than per event, on
   the background worker, so the page render pays nothing.

   This is the clearest possible argument for section 4: dropping foreign keys
   on the event log is only defensible **because** something else checks the
   references. Here the compensating control earned its place.

None of these is visible in Python. All four would have shipped.

---

## 10. Seeding

`scripts/seed_database.py` uses `COPY`, not ORM inserts — at 438k events that is
seconds instead of minutes, and it exercises the same mechanism a production
backfill would use.

Three details that matter:

- **Load order follows the dependency graph**, so foreign keys hold at every
  step rather than only at the end.
- **Identity sequences are advanced** after loading explicit ids. Skipping this
  makes the first application insert collide with a seeded row — a confusing and
  very common seeding bug.
- **`ANALYZE` runs afterwards.** Freshly bulk-loaded tables have no statistics,
  so the planner assumes tiny row counts and chooses sequential scans over the
  indexes this schema exists to provide. Without it, the first latency
  measurements would be meaningless.

The script refuses to load into a non-empty database without `--truncate`, so a
half-loaded dataset cannot silently accumulate duplicates.

---

## 11. Not yet populated

Written and migrated, filled in later phases:

| Table | Filled in |
| --- | --- |
| `user_product_interactions` | Phase 4 (aggregate) / Phase 7 (calibrated `implicit_weight`, ADR-002) |
| `user_features`, `product_features` | Phase 4 |
| `product_embeddings` | Phase 6 |
| `recommendations` + outcome tables | Phase 10 |
| `model_versions` | Phase 13 |
| `experiments`, `experiment_assignments` | Phase 15 |

They exist now because their shape constrains earlier decisions — the ranking
feature list, the attribution join, the promotion gate — and discovering a
missing column in Phase 13 would mean a migration against a populated database.
