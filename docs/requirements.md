# Requirements Specification

Status: **Phase 1 — approved scope baseline**
Last updated: 2026-09-08

---

## 1. Business problem

A mid-size e-commerce catalogue (~5k SKUs, ~10k active users) converts poorly on
generic merchandising. Every shopper sees the same homepage, the same "related
products" strip, and the same search ordering. The commercial hypothesis:

> Ranking products by *predicted relevance to this specific user* instead of by
> global popularity increases click-through rate, add-to-cart rate and revenue
> per session, without increasing catalogue or traffic spend.

The platform must therefore not only *produce* recommendations but **prove**
they are better — offline (ranking metrics on held-out future behaviour) and
online (A/B test on CTR / ATC / conversion / revenue-per-user).

### Why this is a production ML problem, not a notebook

| Notebook version | Production version (what we build) |
| --- | --- |
| Random train/test split | Temporal split plus a leakage audit |
| One algorithm | 5 strategies + hybrid + learned ranker, compared |
| `model.pkl` on disk | MLflow registry, versioned, promotion gate |
| Accuracy | Precision / Recall / MAP / NDCG / HitRate / MRR + coverage, diversity, novelty, serendipity |
| Score in a loop | Two-stage retrieve-then-rank under a p99 latency budget |
| "It works" | Drift detection, degradation ladder, auto-retrain, alerting |
| Static CSV | Live event stream to feature pipeline to next model |

---

## 2. Functional requirements

Each requirement has an ID used throughout the code (`# FR-07`) and in the phase
plan. "Verified by" is the artefact that proves it works.

### 2.1 Recommendation surfaces

| ID | Requirement | Verified by |
| --- | --- | --- |
| FR-01 | Personalised recommendations for a known user | `GET /api/v1/recommendations/home/{user_id}` returns different lists for two users with different histories (integration test) |
| FR-02 | Similar-product recommendations | `GET /recommendations/similar/{product_id}`; offline category-purity check |
| FR-03 | Frequently bought together | `GET /recommendations/frequently-bought/{product_id}`; co-purchase lift above 1 |
| FR-04 | Customers who viewed this also viewed | co-view matrix endpoint |
| FR-05 | Trending products | time-decayed velocity; list changes when events change |
| FR-06 | Recently viewed | per-user Redis list, session-aware |
| FR-07 | Personalised homepage (multi-section) | single call returns all sections plus metadata |
| FR-08 | Personalised search / product ranking | same query, two users, different order; NDCG uplift over keyword-only |
| FR-09 | Cold start: new user | user with zero events still receives a full, non-empty homepage |
| FR-10 | Cold start: new product | product inserted with metadata only is retrievable as a similar-product candidate within one embedding refresh |
| FR-11 | Recommendation explanations | every item carries a human-readable reason plus machine-readable evidence |
| FR-12 | Feedback tracking | impression, click and conversion recorded and joinable to the serving decision |

### 2.2 Platform

| ID | Requirement | Verified by |
| --- | --- | --- |
| FR-13 | Offline training pipeline | one command trains all models and logs to MLflow |
| FR-14 | Online inference service | FastAPI, cached, meets the latency budget (NFR-01) |
| FR-15 | Recommendation performance analytics | admin dashboard: CTR, CVR, revenue attribution |
| FR-16 | A/B testing framework | deterministic assignment, per-variant metrics, significance test |
| FR-17 | Model monitoring | latency, score distribution, coverage, error rate in Prometheus and Grafana |
| FR-18 | Data-drift monitoring | PSI and KS on tracked features, thresholds, alert |
| FR-19 | Automatic retraining | drift or schedule triggers the pipeline; promotion only if the evaluation gate passes |

### 2.3 Event tracking

| ID | Requirement |
| --- | --- |
| FR-20 | Ingest the 12 baseline event types (`PRODUCT_VIEW` through `SESSION_END`) |
| FR-21 | New event types addable without a schema migration (typed enum plus JSONB metadata) |
| FR-22 | Events accepted for anonymous sessions and later stitched to a user on login |
| FR-23 | Event ingestion must never block or fail a page render (bounded latency, batch endpoint) |

---

## 3. Non-functional requirements

| ID | Requirement | Target | How measured |
| --- | --- | --- | --- |
| NFR-01 | Recommendation read latency | p50 < 25 ms, p95 < 80 ms, p99 < 150 ms on cache hit; p99 < 400 ms cold | Prometheus histogram per endpoint |
| NFR-02 | Event write latency | p99 < 30 ms at the API boundary | histogram |
| NFR-03 | Cache hit ratio on home and similar | above 80 % in steady state | Redis counters exported to Prometheus |
| NFR-04 | Availability under model failure | never 5xx a recommendation surface, degrade instead | chaos test: remove the ranker artefact, endpoint still returns 200 with `strategy=fallback` |
| NFR-05 | Reproducibility | same commit plus same data snapshot gives the same metrics (seeded) | CI re-runs training on a small fixture |
| NFR-06 | No training/serving skew | feature transforms defined once, imported by both paths | single `recsys.features` package; a skew test asserts the offline vector equals the online vector |
| NFR-07 | Type safety | backend and ML fully annotated, frontend strict TypeScript | `mypy` and `tsc --noEmit` in CI |
| NFR-08 | Test coverage on ML and service layer | at least 80 % lines on `recsys/` and `app/services/` | `pytest --cov` gate in CI |
| NFR-09 | Secrets | zero credentials in source, all config via environment | CI secret scan |
| NFR-10 | Cold-start coverage | 100 % of users and products receive at least N recommendations | evaluation report `coverage` metric |
| NFR-11 | One-command local bring-up | `docker compose up` yields a working system with seeded data | documented plus CI smoke test |

---

## 4. Explicit non-goals for the Phase 1 baseline

Stated so scope creep is a visible decision rather than drift:

- **Not** a full e-commerce checkout or payments system. The storefront exists to
  generate behaviour and to display recommendations.
- **Not** multi-tenant. Single catalogue, single locale, single currency.
- **Not** real-time model *training*. Inference is real-time; training is batch.
  Streaming feature updates are in scope through Redis counters; streaming
  *learning* is not.
- **Not** deep-learning two-tower or transformer recommenders in v1. The two-stage
  architecture is deliberately built so a neural retriever can replace ALS behind
  the same `CandidateSource` interface later, and that seam is itself a deliverable.
- **Not** Kafka in the initial implementation. The `EventSink` abstraction makes it
  a drop-in replacement; see ADR-009 in `decisions.md`.

---

## 5. Success criteria (definition of done for the whole project)

The project is complete when this loop is demonstrable on a running system, not
described in a README:

1. A user views, carts or purchases products in the React storefront.
2. The event lands in `user_events` in Postgres and updates Redis real-time counters.
3. The feature pipeline materialises updated user, product and interaction features.
4. A homepage request triggers candidate generation, ML ranking and explanation.
5. Impressions, clicks and conversions on those recommendations are recorded.
6. The admin dashboard shows CTR, CVR and revenue per recommendation source and per
   A/B variant, with the current model version and prediction latency.
7. The offline evaluation report ranks popularity below content below collaborative
   filtering below hybrid below the learned ranker on NDCG@10, with the reasoning
   written down.
8. Injecting a distribution shift raises PSI above the threshold, fires an alert and
   triggers the retraining pipeline.
9. The retrained model is registered in MLflow and is promoted to Production **only**
   if it beats the incumbent on the gate metrics.

Each numbered item maps to a phase exit criterion in `docs/roadmap.md`.
