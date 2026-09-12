# API Reference

Status: **Phase 10 deliverable — complete**
Last updated: 2026-09-12

Interactive docs, generated from the code, are served at **`/docs`** when the
API is running. This file covers the contracts and the reasoning OpenAPI cannot
express: what a field means, what it guarantees, and why the endpoint behaves
the way it does.

Base URL: `http://localhost:8000`, API prefix `/api/v1`.

---

## 1. Endpoint index

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| GET | `/health` | Liveness | — |
| GET | `/health/ready` | Readiness with per-dependency status | — |
| GET | `/metrics` | Prometheus exposition | — |
| **Recommendations** | | | |
| GET | `/api/v1/recommendations/home` | Homepage for the current session (anonymous allowed) | optional |
| GET | `/api/v1/recommendations/home/{user_id}` | Homepage for a specific user (FR-07) | optional |
| GET | `/api/v1/recommendations/similar/{product_id}` | Similar products (FR-02) | — |
| GET | `/api/v1/recommendations/frequently-bought/{product_id}` | Frequently bought together (FR-03) | — |
| GET | `/api/v1/recommendations/also-viewed/{product_id}` | Customers also viewed (FR-04) | — |
| GET | `/api/v1/recommendations/trending` | Trending (FR-05) | — |
| GET | `/api/v1/recommendations/recent/{user_id}` | Recently viewed (FR-06) | — |
| GET | `/api/v1/recommendations/explain/{user_id}/{product_id}` | Why this product (FR-11) | — |
| GET | `/api/v1/recommendations/health` | Engine health | — |
| **Events** | | | |
| POST | `/api/v1/events` | Ingest one event | optional |
| POST | `/api/v1/events/batch` | Ingest up to 100 events | optional |
| POST | `/api/v1/feedback/impression` | An item became visible (FR-12) | optional |
| POST | `/api/v1/feedback/click` | An item was clicked (FR-12) | optional |
| **Auth** | | | |
| POST | `/api/v1/auth/register` | Create an account | — |
| POST | `/api/v1/auth/login` | Exchange credentials for tokens | — |
| POST | `/api/v1/auth/refresh` | Exchange a refresh token | — |
| GET | `/api/v1/auth/me` | Current principal | required |
| **Admin** | | | |
| GET | `/api/v1/admin/models` | Offline evaluation from the last run | analyst |
| GET | `/api/v1/admin/monitoring` | Serving model and engine health | analyst |
| GET | `/api/v1/admin/drift` | Feature drift (FR-18) | analyst |
| GET | `/api/v1/admin/distribution` | What the engine recommends | analyst |
| GET | `/api/v1/admin/analytics` | Business metrics (FR-15) | analyst |
| GET | `/api/v1/admin/analytics/products` | Most recommended / converted | analyst |
| GET | `/api/v1/admin/experiments` | Running experiments | analyst |
| GET | `/api/v1/admin/experiments/{key}/results` | A/B readout (FR-16) | analyst |

---

## 2. Conventions

### Headers

| Header | Direction | Meaning |
| --- | --- | --- |
| `Authorization: Bearer <token>` | in | JWT access token. Optional on most routes |
| `X-Session-Id` | in | Browsing session, 8–64 chars. **Required for meaningful recommendations** |
| `X-Request-Id` | in/out | Trace id. Generated if absent, echoed on the response |
| `X-Response-Time-Ms` | out | Server-side duration |

**`X-Session-Id` matters more than it looks.** Co-visitation and session-based
cold start both depend on the session being stable across requests. Only the
client can guarantee that, so the server does not invent one — it accepts what
it is given and falls back to a shared `anonymous-session` bucket otherwise,
which produces noticeably worse recommendations.

### Status codes

| Code | When |
| --- | --- |
| 200 | Success |
| 202 | Event **accepted for processing** — deliberately not 201 |
| 401 | Missing or invalid credentials on a protected route |
| 403 | Authenticated but insufficient role |
| 422 | Validation failure, with the offending field named |
| 429 | Rate limit exceeded (`Retry-After` header included) |

**Why 202 and not 201 for events.** Ingestion is buffered: the request returns
before the database write completes. Returning 201 ("created") would be a claim
the server cannot make and the client might act on.

### Authentication behaviour

Most recommendation routes take an **optional** token. This is deliberate:

- **No token** → anonymous visitor, served session-based cold-start recommendations
- **Valid token** → personalised for that user
- **Invalid or expired token** → treated as anonymous, **not rejected**

That last case matters. A stale token sitting in a browser tab should degrade to
public browsing, not break the storefront.

Admin routes are the opposite: they reject rather than degrade.

### Rate limits

| Scope | Default | Configurable via |
| --- | --- | --- |
| Auth | 10/min | `RATE_LIMIT_AUTH_PER_MINUTE` |
| Events | 600/min | `RATE_LIMIT_EVENTS_PER_MINUTE` |
| Recommendations | 120/min | `RATE_LIMIT_RECOMMENDATIONS_PER_MINUTE` |

Fixed window, backed by Redis, keyed by user id when authenticated and by IP
otherwise. **It fails open**: if Redis is unavailable the request is allowed. A
rate limiter that takes down the storefront during a cache outage is worse than
the abuse it prevents.

---

## 3. Recommendation endpoints

### `GET /api/v1/recommendations/home/{user_id}`

All homepage rails in **one call**. One call rather than five because the rails
share a user context and a candidate pass, which is the expensive part of the
request.

**Query:** `limit` (1–50, default 12)

**Response:**

```json
{
  "user_id": 42,
  "for_you": [ /* RecommendationItem[] */ ],
  "because_you_viewed": [],
  "trending": [],
  "frequently_bought_together": [],
  "continue_shopping": [],
  "sections": {
    "for_you": {
      "section": "for_you",
      "surface": "home_for_you",
      "items": [],
      "strategy": "ranker",
      "model_version": "v1",
      "request_id": "3f2a...",
      "variant": "treatment",
      "experiment_key": "ranker_v1",
      "candidate_pool_size": 300,
      "cache_hit": false,
      "latency_ms": 13.16
    }
  },
  "model_version": "v1",
  "request_id": "3f2a..."
}
```

The top-level arrays are for simple clients. `sections` carries the same items
plus the metadata a dashboard or a debugging session needs.

**A section may be absent.** A user with no browsing history has no
`because_you_viewed`. Clients must handle a missing key rather than assuming all
five are present.

#### `RecommendationItem`

```json
{
  "product": {
    "id": 2593,
    "name": "Daily Glow Compact Serum 500",
    "price": 42.78,
    "category_id": 51,
    "brand_id": 24,
    "rating_average": 3.71,
    "image_url": null
  },
  "score": 1.0,
  "recommendation_type": "ranker",
  "explanation": "Popular with shoppers who like what you like",
  "explanation_evidence": {
    "sources": ["collaborative_als", "category_affinity", "trending"],
    "components": { "ranker": 1.0, "collaborative_als": 0.87 },
    "anchor_product_id": null,
    "category_id": 51,
    "brand_id": 24
  },
  "score_components": { "ranker": 1.0 },
  "position": 0,
  "recommendation_id": "3f2a...:0"
}
```

| Field | Contract |
| --- | --- |
| `score` | Normalised to [0, 1] **within this response**. Not comparable across responses or model versions |
| `recommendation_type` | The candidate source that produced the item |
| `explanation` | Shopper-facing text. Never empty |
| `explanation_evidence` | Machine-readable support, for dashboards and debugging |
| `position` | Zero-based slot. Needed for position-bias analysis |
| **`recommendation_id`** | **Echo this back on impression and click.** Without it, an outcome cannot be joined to the decision that produced it, and attribution becomes guesswork |

#### `strategy` — the degradation ladder

Every section reports which rung served it:

| Value | Meaning |
| --- | --- |
| `ranker` | Normal: full candidate set, learned ranking |
| `hybrid` | Ranker unavailable; weighted blend |
| `collaborative_content` | Feature store unavailable |
| `content_trending` | Collaborative model missing, or a cold user |
| `category_popular` | No user context |
| `global_trending` | Database degraded |
| `static_fallback` | Everything degraded |

**A recommendation surface never returns 5xx.** It degrades and says so. That is
what makes *fallback rate* a measurable health metric rather than an invisible
failure — see `monitoring/prometheus/rules/alerts.yml`.

### `GET /api/v1/recommendations/home`

Identical, but resolves the user from the token. This is what a real storefront
calls. Anonymous visitors get cold-start recommendations (FR-09) rather than an
error.

### `GET /api/v1/recommendations/similar/{product_id}`

Content-based neighbours. **Substitutes** — products a shopper would consider
instead of this one.

### `GET /api/v1/recommendations/frequently-bought/{product_id}`

Basket co-occurrence. **Complements** — products bought *alongside* this one.

Deliberately a different model from `similar`. Two running shoes are substitutes
and should never be sold as a pair; shoes and socks are complements and should.
Serving one list under both headings produces the classic broken product page.

**May legitimately return zero items.** A product with no co-purchase evidence
has no complements, and inventing some would be worse than showing nothing.

### `GET /api/v1/recommendations/also-viewed/{product_id}`

Session co-visitation — the **comparison set** a shopper looked at in the same
session. Works from the very first page view, which makes it the strongest
signal an anonymous visitor generates.

### `GET /api/v1/recommendations/trending`

Time-decayed velocity, 6-hour half-life. **Query:** `limit`, `category_id`.

The short half-life is the design: a burst is visible within the hour and mostly
gone by the next day. A long half-life turns this into a slow popularity chart,
which the popularity model already is.

### `GET /api/v1/recommendations/explain/{user_id}/{product_id}`

Answers *"would we recommend this, and why or why not?"*.

```json
{
  "product_id": 100,
  "would_recommend": false,
  "reason": "This product is not in the user's top 100 candidates. It may be out of stock, already purchased, or outside their price and category profile.",
  "model_version": "v1"
}
```

The **negative** case is usually the more useful debugging answer, which is why
this endpoint covers it rather than only explaining items already shown.

---

## 4. Event endpoints

### `POST /api/v1/events` · `POST /api/v1/events/batch`

```json
{
  "event_type": "PRODUCT_VIEW",
  "session_id": "a1b2c3d4e5f6",
  "product_id": 100,
  "occurred_at": "2026-09-12T10:30:00Z",
  "source": "pdp",
  "device_type": "desktop",
  "metadata": { "dwell_ms": 4200 }
}
```

**`user_id` is deliberately absent from the schema.** It is resolved from the
token. Accepting it from the body would let any client write events attributed
to another user, poisoning both that user's recommendations and the training set.

#### Event types and their payloads

| `event_type` | Requires `product_id` | `metadata` fields |
| --- | --- | --- |
| `PRODUCT_VIEW` | yes | `dwell_ms`, `position` |
| `PRODUCT_CLICK` | yes | `position` |
| `ADD_TO_CART` / `REMOVE_FROM_CART` / `PURCHASE` | yes | `quantity`, `variant_id` |
| `WISHLIST` / `PRODUCT_SHARE` | yes | `channel` (share only) |
| `PRODUCT_RATING` | yes | `rating` (1–5) |
| `PRODUCT_REVIEW` | yes | `rating`, `length` |
| `SEARCH` | **no** | `query`, `results`, `filters` |
| `SESSION_START` / `SESSION_END` | **no** | `entry`, `referrer` |

`metadata` is JSONB in the database but **not** unvalidated. Each event type has
a declared payload model, and unknown keys are rejected with 422. That split is
what makes FR-21 — new event types with no migration — safe rather than a
schemaless dumping ground.

#### Timestamp rules

- `occurred_at` is optional; the server stamps it if absent
- Must include a timezone offset — naive timestamps are rejected
- More than **5 minutes in the future** → silently clamped to now. A wrong
  device clock is not a reason to discard real behaviour
- More than **7 days in the past** → rejected. Unbounded backdating would let a
  client write into a partition already used for training

### `POST /api/v1/feedback/impression` · `/click`

```json
{
  "recommendation_id": "3f2a...:0",
  "product_id": 2593,
  "session_id": "a1b2c3d4e5f6",
  "position": 0,
  "visible_ms": 1500
}
```

**Impressions come from an `IntersectionObserver`, not from the API response.**
A rail below the fold was served but never seen; counting it would depress CTR
for exactly the surfaces that are working. The client reports an impression only
once an item has been genuinely visible for ~800ms.

---

## 5. Admin endpoints

All require the `analyst` role or above.

| Endpoint | Returns |
| --- | --- |
| `/admin/models` | The offline comparison table, per-segment results, calibrated weights, stage-1 recall, source contribution, feature importance, **and significance tests** |
| `/admin/monitoring` | Serving model version, loaded models, engine health, event-sink stats, dependency availability |
| `/admin/drift` | PSI and KS per monitored feature, severity, and whether a retrain is recommended |
| `/admin/distribution` | Most-recommended products, source mix and coverage over a live sample |
| `/admin/analytics` | CTR, conversion, revenue by surface / source / model version |
| `/admin/experiments/{key}/results` | Per-arm funnel, significance tests, sample-ratio check, readiness |

### A note on `available: false`

Analytics endpoints return `{"available": false, "reason": "..."}` when the
database is unreachable, rather than zeros.

This is not defensiveness for its own sake. A dashboard of zeros reads as
*"nothing is converting"* — an alarming and wrong claim. *"No data yet"* is the
truth, and the two should never be confused.

### Experiment results

```json
{
  "experiment_key": "ranker_v1",
  "arms": [ { "label": "control", "impressions": 0, "ctr": 0.0 } ],
  "tests": {
    "treatment": {
      "ctr": { "relative_lift": 0.04, "p_value": 0.03, "significant": true }
    }
  },
  "sample_ratio_p_value": 0.42,
  "sample_ratio_mismatch": false,
  "ready": false,
  "notes": ["Smallest arm has 0 users, below the 1000 minimum."]
}
```

**Read `ready` before reading anything else.** The sample-ratio check runs
*before* the significance tests and can mark a readout unusable on its own: if
traffic did not split as intended, every number below it is invalid — and a
broken experiment usually produces a convincing-looking win.

`revenue_per_user` is reported **without** a significance test. It is continuous
and heavily skewed — a handful of large orders dominate — so a proportion test
does not apply and a t-test would be badly calibrated. A bootstrap over per-user
revenue is the correct tool and is not yet wired in.

---

## 6. Errors

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, PURCHASE requires product_id"
    }
  ]
}
```

Unhandled errors return a generic message plus the `request_id`, which is the
key to finding the full trace in the logs. Internal details are never returned
to the client.

---

## 7. Performance

Measured on the reference machine (16 cores), uncached, full 300-candidate pool:

| Surface | p50 | p95 |
| --- | --- | --- |
| Homepage (`for_you`) | 17.1 ms | 19.3 ms |

Comfortably inside the NFR-01 budget (p95 < 80 ms) **before** caching. Stage
breakdown: ~4 ms candidate generation, ~12 ms ranking.

Cache TTLs:

| Surface | TTL |
| --- | --- |
| Homepage | 15 min |
| Similar / also-viewed | 6 h |
| Frequently bought | 12 h |
| Trending | 5 min |

Cache keys embed the **model version** and the **experiment variant**. The first
means promoting a model rolls the cache over naturally instead of needing a
flush; the second means two experiment arms can never share a cached payload,
which would silently null out every experiment result.
