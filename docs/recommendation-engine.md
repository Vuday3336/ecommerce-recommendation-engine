# Recommendation Engine

Status: **Phase 8 deliverable — complete**
Last updated: 2026-09-12

How a recommendation is produced, from a user id to a ranked list with reasons.

---

## 1. The request path

```
RequestContext (user, recent products, affinities, price band)
        │
        ▼
STAGE 1 — CANDIDATE GENERATION            ~4 ms
   10 sources run in parallel, each with a budget
   → union by reciprocal-rank fusion
   → business rules filter
   → ~300 candidates with provenance
        │
        ▼
STAGE 2 — RANKING                         ~12 ms
   assemble 83 features × 300 rows (vectorised)
   → LightGBM LambdaRank
   → normalised relevance score
        │
        ▼
STAGE 3 — RE-RANK AND EXPLAIN
   MMR diversity + category/brand caps
   → rule-based explanation per item
        │
        ▼
12 items, each with score, source, reason and evidence
```

Why two stages (ADR-001): cost scales with `sources × k`, not catalogue size.
Each stage is optimised for a different thing — stage 1 for **recall@300**
("did the right item survive?"), stage 2 for **NDCG@10** ("is it near the top?").
A single-stage system has to trade those against each other with one model.

---

## 2. The five strategies

### 2.1 Popularity

Log-scaled accumulation of weighted events, globally and per category.

```
score(item) = normalise( log(1 + Σ event_weight) )
```

Log-scaled because raw counts are power-law distributed: a linear score is
decided entirely by the head, and everything outside the top hundred scores
indistinguishably.

**This is a much stronger baseline than it looks.** Category-popularity — the
most-engaged products *within the categories a user actually shops* — scores
NDCG@10 **0.0560** and is statistically tied with the learned ranker. A large
part of what "personalisation" means in e-commerce is knowing which aisle
someone shops in.

Choosing a weak baseline is the most common way to make a mediocre model look
excellent: against *global* popularity (0.0133), our ranker would show a
**+337%** improvement. Against the honest baseline it shows +4%.

### 2.2 Trending

Time-decayed velocity, 6-hour half-life:

```
score(item) = Σ event_weight × 0.5 ^ (age_hours / 6)
```

Computed two ways that must agree: a batch version in `features/product.py` and
an online version in `app/cache/counters.py` using Redis sorted sets bucketed by
hour. Same weights, same half-life — otherwise the homepage and the batch job
would hold two different opinions about what is trending.

The 6-hour half-life is a product decision. Longer, and this becomes a slow
popularity chart, which the popularity model already is.

### 2.3 Content-based

Products become vectors, then cosine similarity.

```
vector = 0.6 × normalise(TF-IDF → SVD of name + category + brand + attributes + description)
       + 0.4 × normalise(one-hot category ⊕ brand ⊕ price band ⊕ scaled log price)
```

Each block is L2-normalised *before* combining, so the 60/40 split is real
rather than decided by whichever block happens to have larger raw magnitude.

**Why TF-IDF + SVD rather than a sentence transformer by default.** It is
deterministic, trains in ~1.4 s, needs no 2.5 GB Torch download, and on this
catalogue captures essentially the same structure. `SentenceTransformerEncoder`
implements the same interface for catalogues with genuinely free-form text —
the choice is measured in the evaluation report, not asserted.

Result: 491 dimensions, SVD explaining **85.4%** of variance.

**This is the new-product cold-start path.** A product with zero interactions
has no collaborative representation at all, but it has a category, a brand and a
description — so it lands in this space the moment it is created.

### 2.4 Collaborative filtering (ALS)

Factorises the user × item matrix. The objective:

```
minimise  Σ over all (u,i)  c_ui · (p_ui − xᵤ·yᵢ)²  +  λ(‖x‖² + ‖y‖²)
where     c_ui = 1 + α · r_ui
```

Three things make this the right choice here (ADR-003):

1. **The sum runs over every cell**, not just observed ones. Unobserved pairs
   are weak negatives with confidence 1. That matters because we never see
   "user dislikes item" — only absence, which is weak evidence rather than none.
2. `c_ui` is **confidence**, not preference. This is what makes deriving the
   weights from conversion odds coherent (§3) rather than arbitrary.
3. It is deterministic given a seed, fits a 7k × 5k matrix in 0.7 s, and yields
   item factors directly usable for item-item similarity.

**Tuned hyperparameters** (validation fold, not test):

| Parameter | Value | Note |
| --- | --- | --- |
| `factors` | 64 | 128 overfits at this density |
| `regularization` | 0.10 | |
| `alpha` | **2.0** | see below |
| `iterations` | 20 | |

`alpha` deserves attention. The textbook value is 40, and using it here was a
real bug: our calibrated weights already average ~25, so `alpha=24` gave
confidences near 1000 and drove the factorisation to fit heavy users almost
exclusively. Moving 24 → 2 lifted validation NDCG@10 from 0.038 to **0.051**, a
35% relative gain from one number. **Alpha must be tuned against the scale of
the weights it multiplies**, not copied from a paper that used binary
interactions.

BPR is trained as a measured competitor and loses decisively (0.0159 vs 0.0497).

### 2.5 Co-visitation and frequently-bought-together

Two item-item models from co-occurrence, answering different questions:

| | Scope | Captures | Surface |
| --- | --- | --- | --- |
| Co-visitation | session | **substitutes** — things compared against each other | "Customers also viewed" |
| Frequently bought | order | **complements** — things bought together | "Frequently bought together" |

Conflating them produces the classic broken rail: *"customers who bought this
phone also bought these four other phones."*

Both score by **lift**, not raw count:

```
lift(a,b) = observed co-occurrence / (P(a) × P(b) × N)
```

Raw counts are dominated by popularity — the bestseller co-occurs with
everything. Lift plus a support floor requires both a real association and
enough evidence.

Single-item orders still count toward the denominator. Dropping them would
shrink the base rate and inflate every lift value.

---

## 3. Implicit-feedback weighting (ADR-002)

A view and a purchase are not equally meaningful. Asserting `purchase = 10,
view = 1` is arbitrary. We derive them:

```
w(e) ∝ log(1 + P(purchase | e) / P(purchase))
```

**Calibrated on the training fold only** — computing conversion odds over the
full period would leak the test window into the weights, which feed the matrix,
which trains the model.

Derived weights on the current dataset:

| Event | Support | P(purchase \| event) | Lift | **Weight** |
| --- | --- | --- | --- | --- |
| PURCHASE | 11,676 | 1.000 | 11.9× | **5.71** |
| PRODUCT_REVIEW | 1,065 | 0.899 | 10.7× | 3.57 |
| PRODUCT_RATING | 2,962 | 0.883 | 10.5× | 3.55 |
| ADD_TO_CART | 19,659 | 0.492 | 5.9× | 2.79 |
| PRODUCT_CLICK | 61,157 | 0.167 | 2.0× | 1.59 |
| WISHLIST | 3,617 | 0.089 | 1.06× | 1.05 |
| **PRODUCT_VIEW** | 133,773 | 0.083 | 0.99× | **1.00** |
| PRODUCT_SHARE | 539 | 0.011 | 0.13× | 0.18 |
| REMOVE_FROM_CART | 3,343 | 0.065 | 0.77× | **−0.83** |

Four design points worth defending:

**The window is symmetric around the event, not forward-only.** The quantity
being estimated is *confidence that this pair represents a real preference*, not
a forecast. Ratings and reviews are inherently post-purchase, so a forward-only
window scores them at exactly zero — telling us a five-star rating carries no
information about preference, which is obviously false.

**Normalised so a view = 1.0.** Anchoring on the *weakest* signal makes the
scale hostage to the noisiest event type: when `PRODUCT_SHARE` has near-zero
lift it becomes the divisor and inflates every other weight into the hundreds.
A view is the highest-volume, most stable event, so anchoring there keeps
weights interpretable — "a purchase is worth about six views".

**`REMOVE_FROM_CART` is negative.** ALS has no place for a negative class —
everything unobserved is already weak-negative — so this *reduces* the pair's
confidence rather than forming its own signal.

**Weights decay and are capped.** Exponential recency decay with a 45-day
half-life (roughly the median repeat-purchase interval), and a cap of 60 per
pair so one obsessive user cannot dominate an item's factor vector.

**Honest observation:** WISHLIST calibrates to only 1.05× a view on this
dataset. The ordering matches the brief, but the magnitude is weak, and that is
a property of the simulator rather than of real wishlisting behaviour. The
method is what matters — the weights are *read from the data*, and reporting
what the data says rather than what we assumed is the point.

---

## 4. Hybrid blending (ADR-014)

The brief suggests fixed weights: `0.45 CF + 0.25 content + 0.15 trending +
0.10 affinity + 0.05 popularity`. Those are a reasonable starting point and are
close to our `rich` profile. **Shipping them as constants would be the mistake.**

A user with two clicks would be scored 45% by a collaborative model that has
essentially no information about them, while the signals that *do* work for them
— content similarity to what they just viewed, and what is popular in their
category — get 30% between them.

So the weights are a function of history depth. Selected by randomised simplex
search on the validation fold, scored per segment:

| Regime | Interactions | CF | Content | Trending | Affinity | Popularity |
| --- | --- | --- | --- | --- | --- | --- |
| cold | 0 | 0.05 | 0.10 | 0.05 | 0.05 | **0.75** |
| sparse | 1–4 | 0.30 | 0.14 | 0.04 | 0.22 | 0.30 |
| warm | 5–19 | 0.42 | 0.18 | 0.06 | 0.16 | 0.18 |
| rich | 20+ | **0.55** | 0.18 | 0.08 | 0.14 | 0.05 |

The raw per-segment argmax confirmed the hypothesis at the extremes —
popularity 0.81 → 0.05 and collaborative 0.07 → 0.55 as history deepens — but
was noisy in the middle, where each segment holds only a few hundred evaluation
users. The shipped values are **constrained to be monotone** in history depth: a
deliberate bias-variance trade that costs a little validation NDCG in the warm
bucket and avoids fitting the search to segment sampling noise.

**Score normalisation is not optional.** An ALS dot product, a cosine similarity
and a log-popularity value live on completely different scales. Summing them
with weights without normalising first makes the weights decorative — the
largest-magnitude component wins regardless of its weight.

---

## 5. Candidate generation (stage 1)

Ten sources, each with a budget:

| Source | Budget | Recall contribution |
| --- | --- | --- |
| `collaborative_als` | 150 | 0.245 |
| `content` | 100 | 0.163 |
| `covisitation` | 80 | 0.115 |
| `category_affinity` | 60 | **0.186** |
| `brand_affinity` | 40 | 0.091 |
| `trending` | 40 | 0.037 |
| `popularity` | 40 | 0.041 |
| `frequently_bought_together` | 40 | 0.013 |
| `recently_viewed` | 20 | — |
| `exploration` | 15 | 0.001 |

**Stage-1 recall@300 = 0.343.** Of everything a user engaged with in the test
window, 34% entered the pool. Two thirds is unreachable no matter how good the
ranker becomes — which makes retrieval, not ranking, the highest-leverage place
to invest next.

`category_affinity` returns the most recall **per slot**: 0.186 from a budget of
60, against 0.245 from 150 for collaborative filtering. Reallocating budget
toward it is the obvious next experiment.

`exploration` contributes ~nothing to recall, **by design**. Its purpose is to
guarantee impressions to cold items so they can accumulate the interactions the
collaborative model needs. Judging it on recall measures the wrong thing; its
payoff is a catalogue that does not ossify.

### Merging

Sources are unioned by **reciprocal-rank fusion**:

```
fused(item) = Σ over sources  1 / (60 + rank_in_that_source)
```

Raw scores are not comparable across sources — a cosine similarity and a lift
value live on different scales — but *ranks* always are. Fusing on rank is the
one merge that does not require every source to be calibrated against every
other.

**Provenance survives the merge.** A product found by three sources keeps all
three names and scores. The ranker uses `source_agreement` as a feature (and
independently learned it is worth 2.7% of gain), and the serving log records it
so a source can earn or lose its budget on evidence.

### Business rules

| Rule | Why |
| --- | --- |
| In stock | Never recommend something that cannot be bought |
| Active product | Soft-deleted items stay joinable but unsellable |
| Not recently purchased **unless consumable** | Someone who bought a laptop does not want another; someone who bought coffee does. A uniform rule would delete the entire replenishment use case |
| Price band (loose, ±0.55 percentile) | Removes only the extremes. A tight band would trap users in whatever they bought first and eliminate any upsell |

---

## 6. Ranking (stage 2)

**LightGBM LambdaRank**, 83 features per (user, candidate) pair:

| Group | Count | Examples |
| --- | --- | --- |
| User | 27 | total events, purchase rate, price sensitivity, recency, CLV, category concentration |
| Product | 25 | price, popularity, trending, conversion rate, quality, days since release, is_cold |
| Pair | 15 | view/cart/purchase counts, days since interaction, category & brand affinity, price distance, content similarity, collaborative score |
| Source | 16 | per-source score and rank, source agreement, fused candidate rank |

**Labels** come from the validation window: purchase = 3, cart = 2, click = 1,
otherwise 0. LambdaRank uses gains of `2^label − 1`, so the spacing (7 / 3 / 1 /
0) tells the model a purchase is more than twice as valuable as a cart addition.

### Negative downsampling — a measured decision

With 293 candidates and typically two positives, the positive rate is ~0.7%, and
early stopping halted after **9 of 400 trees**: nearly every pairwise comparison
sampled was negative-vs-negative and carried no gradient.

We tested four strategies:

| Strategy | NDCG@10 | Trees |
| --- | --- | --- |
| Full pool, no sampling | 0.0502 | 9 |
| Uniform, 60 negatives | 0.0511 | 17 |
| **Uniform, 120 negatives** | **0.0530** | 34 |
| Top-biased ("hard"), 60 negatives | **0.0215** | 9 |

**Hard-negative sampling — the textbook trick — was actively harmful**, halving
accuracy. The cause is a train/serve distribution mismatch, not anything wrong
with hard negatives in principle: `candidate_rank` is the strongest single
feature, biased sampling left the model with almost no examples above rank ~60,
and at inference every deep-pool item fell into one under-determined leaf and
got surfaced arbitrarily. Coverage doubling from 0.27 to 0.52 while accuracy
halved is that failure made visible.

### What it learned

| Feature | Gain share |
| --- | --- |
| `candidate_rank` | 16.5% |
| `category_affinity` | 8.8% |
| `src_category_affinity_score` | 8.6% |
| `product_view_count_total` | 6.0% |
| `src_collaborative_als_score` | 4.0% |
| `brand_affinity` | 3.7% |
| `source_agreement` | 2.7% |

`candidate_rank` is the **fused retrieval rank**, computed identically at train
and serve time — it is *not* a display position. That distinction matters:
display position is bias-prone and would have to be held constant at inference.
This one never is.

### The honest caveat on ADR-005

ADR-005 predicted a listwise objective would beat a pointwise one. The measured
gap against the logistic control is **+0.9% relative** — indistinguishable.

Either the feature set already encodes most of the ranking signal, or 1,596
training queries is too small for a listwise loss to show its advantage. The
evidence does not separate them. LambdaRank stays in production because it wins
on the point estimate and dominates on hit-rate@10, but the *claimed reason*
should not be asserted without this caveat.

---

## 7. Diversity re-ranking

Greedy MMR with hard category and brand caps:

```
value(candidate) = λ · relevance − (1−λ) · redundancy     λ = 0.72
redundancy = (items already selected from this category) / (items selected)
```

MMR alone lets a strong category dominate when its items are also the most
relevant, so hard caps (max 4 per category, 3 per brand) sit on top. The caps
are what stop a rail from being five running shoes — the failure users actually
complain about.

Measured cost:

| | NDCG@10 | Diversity | Coverage |
| --- | --- | --- | --- |
| Ranker | 0.0582 | 0.578 | 0.284 |
| Ranker + diversity | 0.0515 | **0.848** | 0.207 |
| **Change** | **−11.5%** | **+47%** | **−27%** |

The accuracy cost is **not statistically significant** (p = 0.057).

Coverage *falling* is the surprise, and worth understanding rather than
explaining away: the caps push each individual list toward distinct categories,
but push *every* user's list toward the same broad popular categories, so the
union across users narrows. Intra-list diversity and catalogue coverage are
different objectives, and this configuration trades one for the other.

**Recommendation: ship it on.** A rail of five near-identical shoes is a visible
product failure and the accuracy cost is within noise. Then measure it online.

---

## 8. Cold start

| Case | Strategy |
| --- | --- |
| **New user, no session** | Trending + category-popular, diversified. Optional onboarding category picker seeds an affinity prior |
| **New user, mid-session** | Co-visitation from items viewed so far. Works from the very first view — the highest-value cold-start lever |
| **Sparse user** | Content from viewed items blended with popularity; hybrid weights shift toward content and popularity automatically |
| **New product** | Text + attribute embedding places it in the content space immediately, retrievable before it has any interactions |
| **New category / seasonal spike** | Trending picks it up within hours, no retraining |

Measured: ALS scores **exactly 0.0000** for cold users. Not a bug — a matrix
factorisation has no vector for a user with no interactions. This is the entire
justification for adaptive hybrid weights; without the popularity and content
components, 100% of cold traffic would get nothing.

The engine serves a cold user a full 12-item page in the integration tests.

---

## 9. Explanations (FR-11)

Two layers, deliberately separate:

**Rule-based, shopper-facing.** Derived from which candidate source retrieved
the item, with a priority ordering so the strongest *true* statement wins:

| Source | Reason | Priority |
| --- | --- | --- |
| `frequently_bought_together` | "Frequently bought with X" | 100 |
| `covisitation` | "Customers who viewed X also viewed this" | 90 |
| `content` | "Because you viewed X" | 80 |
| `collaborative_als` | "Popular with shoppers who like what you like" | 70 |
| `brand_affinity` | "You like {brand}" | 60 |
| `category_affinity` | "You often shop {category}" | 50 |
| `trending` | "Trending in {category} right now" | 30 |

On a product page the surface fixes the reason — the anchor is unambiguous, so
the source ordering does not apply.

Each explanation carries machine-readable `evidence` (sources, component scores,
anchor product) which becomes `recommendations.explanation_evidence` in the
serving log. That is what makes a ranking decision reconstructible months later,
after the model that made it has been replaced.

**SHAP, analyst-facing.** Per-prediction feature attribution for the admin
dashboard and for debugging a specific bad recommendation. Expensive, so never
on the shopper hot path. Falls back to global gain importance when `shap` is not
installed — a poorer substitute, but honest about what it is.

---

## 10. Performance

Measured, uncached, full pool:

| Stage | Time |
| --- | --- |
| Candidate generation | ~4 ms |
| Ranking (83 features × 300 rows) | ~12 ms |
| **Total p50 / p95** | **17.1 ms / 19.3 ms** |

Against a budget of p95 < 80 ms. The margin comes from vectorising feature
assembly: user features broadcast once, product features a single reindex, pair
features one call. A Python loop over 300 DataFrame rows would cost tens of
milliseconds per request and put the budget out of reach.
