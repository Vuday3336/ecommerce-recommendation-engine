# Evaluation

Status: **Phase 14 deliverable — complete**
Last updated: 2026-09-08

Reproduce everything here with:

```bash
python ml/pipelines/train.py --eval-users 2000 --ranker-users 3500 --mlflow
```

Runtime ~4 minutes. Artefacts and the full metric tables land in `ml/artifacts/`.

---

## 1. Protocol

**Temporal split (ADR-007).** Interactions are ordered by time and cut at the
70th and 85th percentiles of the interaction timeline:

| Fold | Rows | Window |
| --- | --- | --- |
| Train | 238,153 | start → 2026-07-11 |
| Validation | 51,033 | 2026-07-11 → 2026-08-07 |
| Test | 51,034 | 2026-08-07 → end |

Boundaries are quantiles of the *interaction* timeline, not the calendar.
Calendar thirds would put very different volumes in each fold whenever traffic
is seasonal, which makes fold-to-fold comparisons partly an artefact of volume.

**Roles are strict.** Features and models come from train only. Every
hyperparameter — ALS `alpha` and `factors`, hybrid blend weights, negative
sampling rate — was selected on **validation**. Test was read once, to produce
the table below. Selecting on test would make every number a best-of-N maximum
rather than an estimate of future performance: the same leak as a random split,
one level up.

**Ground truth.** A test-window `PURCHASE`, `ADD_TO_CART` or `PRODUCT_CLICK` on
a product counts as relevant. Plain views are excluded: "did the user look at
this?" is a question popularity already answers well, and it is not the business
question.

**Already-seen items are excluded from retrieval.** Re-recommending something
the user interacted with during training is trivially correct for
repeat-purchase items and would flatter every model equally.

---

## 2. Data-leakage audit

Leakage is the failure that produces beautiful numbers and no error, so it is
defended mechanically rather than by review:

| Risk | Defence | Where |
| --- | --- | --- |
| Random split lets the model see a user's future | Temporal split with an ordering assertion on every fold boundary | `assert_no_leakage`, run inside `split_temporal` |
| A feature reads events at or after its own timestamp | Every builder takes `as_of` and filters first; a guard raises if any input row is at or after it | `assert_features_respect_cutoff` |
| Popularity computed over the full period encodes the test window | All popularity, trending and conversion-rate features are bounded by `as_of` | `features/product.py` |
| Implicit weights calibrated on all data | `calibrate_weights` is called on `split.train` only | `pipelines/train.py` |
| Serving feature tables reused as training features | `user_features` / `product_features` hold the *current* snapshot and are documented as not time-travel stores; training recomputes with `as_of` | `app/models/features.py` module docstring |
| Ranker labels drawn from the test window | Labels come from **validation**; test is never read during training | `ranking/dataset.py` |

**One known, deliberate exposure.** The ranker's early-stopping set is a slice
of the same validation window its training labels come from. Test is still
untouched, so the reported numbers are honest, but the ranker's stopping point
is chosen with slightly more information than a strict three-way split allows.
The alternative — a fourth fold — would cost roughly a quarter of the labelling
data on a dataset that is already thin on positives. The trade is recorded here
rather than hidden.

---

## 3. Model comparison (test fold, 2,000 users)

| Model | P@10 | R@10 | **NDCG@10** | MAP@10 | HitRate@10 | MRR | Coverage | Diversity | Novelty | Personalisation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **two-stage-ranker** | 0.0290 | 0.0736 | **0.0582** | 0.0313 | 0.2320 | 0.1046 | 0.284 | 0.578 | 10.27 | 0.975 |
| two-stage-logistic | 0.0287 | 0.0724 | 0.0577 | 0.0311 | 0.2255 | 0.1042 | 0.311 | 0.551 | 10.32 | 0.976 |
| popularity-category | 0.0273 | 0.0683 | 0.0560 | 0.0308 | 0.2100 | 0.1023 | 0.232 | 0.259 | 10.57 | 0.978 |
| two-stage-xgboost | 0.0285 | 0.0702 | 0.0558 | 0.0298 | 0.2235 | 0.1012 | 0.314 | 0.608 | 10.45 | 0.975 |
| hybrid | 0.0276 | 0.0714 | 0.0552 | 0.0289 | 0.2210 | 0.0999 | 0.316 | 0.464 | 10.44 | 0.979 |
| two-stage-ranker + diversity | 0.0254 | 0.0606 | 0.0515 | 0.0279 | 0.2080 | 0.0995 | 0.207 | 0.848 | 9.88 | 0.962 |
| collaborative-als | 0.0251 | 0.0614 | 0.0497 | 0.0263 | 0.1970 | 0.0928 | 0.225 | 0.499 | 9.42 | 0.989 |
| hybrid + diversity | 0.0235 | 0.0596 | 0.0489 | 0.0258 | 0.1990 | 0.0952 | 0.299 | 0.812 | 10.14 | 0.973 |
| content | 0.0122 | 0.0299 | 0.0216 | 0.0104 | 0.0960 | 0.0383 | 0.829 | 0.117 | 10.86 | 0.997 |
| collaborative-bpr | 0.0087 | 0.0212 | 0.0159 | 0.0077 | 0.0820 | 0.0315 | 0.382 | 0.745 | 10.33 | 0.992 |
| popularity-global | 0.0069 | 0.0204 | 0.0133 | 0.0062 | 0.0670 | 0.0240 | 0.004 | 0.992 | 7.93 | 0.144 |
| trending | 0.0061 | 0.0171 | 0.0120 | 0.0059 | 0.0585 | 0.0221 | 0.003 | 0.958 | 8.18 | 0.134 |
| covisitation | 0.0065 | 0.0173 | 0.0117 | 0.0056 | 0.0550 | 0.0214 | 0.409 | 0.542 | 10.67 | 0.993 |

The expected ordering holds — **popularity < content < collaborative < hybrid <
learned ranker** — with two instructive exceptions covered below.

Every row is measured without the diversity re-ranker so the comparison is
like-for-like; the two `+ diversity` rows isolate its cost.

---

## 4. Which model wins, and how confident we are

**The two-stage ranker wins on NDCG@10, and the win is not statistically
significant.**

Paired bootstrap, 5,000 resamples, against the strongest simple baseline:

| Candidate vs popularity-category | Δ NDCG@10 | Relative | 95 % CI | p | Verdict |
| --- | --- | --- | --- | --- | --- |
| two-stage-ranker | +0.00223 | **+4.0 %** | [−0.0019, +0.0063] | 0.297 | not significant |
| two-stage-logistic | +0.00172 | +3.1 % | [−0.0023, +0.0056] | 0.399 | not significant |
| two-stage-xgboost | −0.00016 | −0.3 % | [−0.0053, +0.0048] | 0.941 | not significant |
| hybrid | −0.00075 | −1.3 % | [−0.0046, +0.0029] | 0.682 | not significant |
| two-stage-ranker + diversity | −0.00443 | −7.9 % | [−0.0092, +0.0001] | 0.057 | not significant |
| collaborative-als | −0.00624 | −11.2 % | [−0.0108, −0.0019] | 0.005 | **worse** |
| hybrid + diversity | −0.00704 | −12.6 % | [−0.0114, −0.0029] | 0.001 | **worse** |
| content | −0.03437 | −61.4 % | [−0.0405, −0.0282] | 0.0004 | **worse** |
| collaborative-bpr | −0.04010 | −71.7 % | [−0.0462, −0.0345] | 0.0004 | **worse** |
| popularity-global | −0.04264 | −76.2 % | [−0.0486, −0.0369] | 0.0004 | **worse** |
| trending | −0.04396 | −78.6 % | [−0.0502, −0.0381] | 0.0004 | **worse** |
| covisitation | −0.04430 | −79.2 % | [−0.0506, −0.0384] | 0.0004 | **worse** |

The bootstrap is *paired* — both models are scored on the same users, so
user-to-user variance, which dwarfs the model-to-model difference, cancels. An
unpaired test would need roughly an order of magnitude more users to detect the
same effect.

**What to conclude.** The large gaps are real: every model below `collaborative-als`
is decisively worse than category popularity. The top five are statistically
indistinguishable at n = 2,000. Declaring the ranker the winner on a +4 % point
estimate would be exactly the mistake that leads to shipping a model that is no
better than its predecessor and then spending a quarter explaining why the
online metrics did not move.

**So the ranker ships as the treatment arm of an A/B test, not as an
unconditional replacement.** That is not a hedge — it is the correct next step,
and it is why the experimentation framework (Phase 15) exists. A 4 % CTR lift is
easily detectable online with a few tens of thousands of sessions, where it is
not detectable offline with 2,000 users.

---

## 5. The two surprises

### 5.1 Category popularity is a genuinely hard baseline

`popularity-category` — the most-engaged products within the categories a user
actually shops — beats ALS, beats content, and is statistically tied with the
learned ranker. This is not an artefact. A large part of what "personalisation"
means in e-commerce is *knowing which aisle someone shops in*, and this baseline
captures exactly that with no model at all.

It also explains the tuned hybrid weights (ADR-014): the popularity term carries
0.75 weight for cold users and only 0.05 for users with deep history, which is
the same fact expressed as a blending policy.

The practical lesson is about baselines. Had `popularity-global` (NDCG 0.0133)
been used as *the* baseline, the ranker would show a **+337 %** improvement and
the report would be dishonest by omission. Choosing a weak baseline is the most
common way to make a mediocre model look excellent.

### 5.2 The pointwise logistic control nearly matches LambdaRank

ADR-005 predicted that a listwise objective would beat a pointwise one, because
the product is a list. The measured gap is +0.9 % relative — indistinguishable.

Two honest readings, and the evidence does not separate them. Either the feature
set already encodes most of the ranking signal, leaving little for the objective
to add; or the training set (1,596 queries after negative sampling) is too small
for a listwise loss to show its advantage, since LambdaRank's benefit grows with
the number of comparable pairs per query.

**ADR-005 is amended rather than defended.** LambdaRank stays the production
ranker because it wins on the point estimate and dominates on hit-rate@10
(0.2320 vs 0.2255), and because the logistic control needs feature scaling that
the tree models do not. But the claimed reason — "listwise beats pointwise" — is
not supported by this dataset and should not be asserted in an interview without
this caveat.

---

## 6. Segmented results (NDCG@10 by user history depth)

Aggregate metrics hide the group that matters most. Most real traffic is sparse,
and a model that looks excellent overall can be useless for it.

| Model | cold (0) | sparse (1–4) | warm (5–19) | rich (20+) |
| --- | --- | --- | --- | --- |
| two-stage-ranker | 0.0167 | 0.0389 | **0.0697** | 0.0623 |
| two-stage-logistic | 0.0300 | 0.0374 | 0.0632 | 0.0616 |
| two-stage-xgboost | 0.0114 | 0.0440 | 0.0620 | 0.0612 |
| popularity-category | 0.0286 | 0.0306 | 0.0536 | 0.0625 |
| collaborative-als | **0.0000** | 0.0324 | 0.0520 | 0.0575 |
| content | 0.0000 | 0.0132 | 0.0191 | 0.0260 |
| trending | 0.0234 | 0.0227 | 0.0214 | 0.0068 |

Three things this table says that the aggregate does not:

1. **ALS scores exactly zero for cold users.** Not a bug — a matrix
   factorisation has no vector for a user with no interactions. This is the
   entire justification for the adaptive hybrid weights and the cold-start path;
   without the popularity and content components, 100 % of cold traffic would
   get nothing.
2. **Trending is the *only* model whose performance falls as history grows**
   (0.0234 cold → 0.0068 rich). It is a cold-start instrument, and using it as a
   general recommender would be a mistake.
3. **The ranker's advantage is concentrated in the warm segment** (+30 % over
   category popularity), where there is enough signal to learn from but not
   enough for ALS to dominate. That is a precise, actionable statement about
   where the model earns its complexity.

---

## 7. Stage-1 retrieval

The ranker can only reorder what retrieval found, so recall@300 is the ceiling
on the entire system.

**Stage-1 recall@300 = 0.343.** Of everything a user engaged with in the test
window, 34 % entered the candidate pool. Roughly two thirds of the theoretical
maximum is unreachable no matter how good the ranker becomes — which makes
retrieval, not ranking, the highest-leverage place to invest next.

Recall contribution per source:

| Source | Contribution | Budget |
| --- | --- | --- |
| collaborative_als | 0.245 | 150 |
| category_affinity | 0.186 | 60 |
| content | 0.163 | 100 |
| covisitation | 0.115 | 80 |
| brand_affinity | 0.091 | 40 |
| popularity | 0.041 | 40 |
| trending | 0.037 | 40 |
| frequently_bought_together | 0.013 | 40 |
| exploration | 0.001 | 15 |

`category_affinity` returns the most recall per slot of any source — 0.186 from
a budget of 60, against 0.245 from 150 for collaborative filtering. Reallocating
budget toward it is the obvious next experiment.

`exploration` contributes almost nothing to recall, and that is *by design*. Its
purpose is to guarantee impressions to cold items so they can accumulate the
interactions the collaborative model needs. Judging it on recall would be
measuring the wrong thing; its payoff is a catalogue that does not ossify
(R-06), and it shows up in coverage, not accuracy.

---

## 8. What the ranker learned

Top features by gain:

| Feature | Gain share |
| --- | --- |
| `candidate_rank` | 16.5 % |
| `category_affinity` | 8.8 % |
| `src_category_affinity_score` | 8.6 % |
| `product_view_count_total` | 6.0 % |
| `src_collaborative_als_score` | 4.0 % |
| `brand_affinity` | 3.7 % |
| `src_collaborative_als_rank` | 3.7 % |
| `product_view_count_30d` | 3.6 % |
| `user_total_clicks` | 3.0 % |
| `source_agreement` | 2.7 % |

`candidate_rank` dominating is expected and correct — it is the fused retrieval
score, computed identically at train and serve time. It is **not** a display
position, and the distinction matters: display position would be bias-prone and
would have to be held constant at inference. See §9.

`source_agreement` earning 2.7 % confirms the design decision to carry
provenance through the candidate merge: the ranker independently learned that
agreement between unrelated retrieval sources is evidence.

---

## 9. Known biases and limitations

**Bootstrap labels are missing-not-at-random.** Labels describe what users did
while browsing on their own, not what they would have done had we shown them
this list. An item the user never encountered is labelled 0 regardless of
whether they would have liked it. This is the standard bias in any recommender
trained before it has served traffic. The exploration budget exists to generate
the unbiased impressions that a later inverse-propensity-weighted retrain would
need; until that data accumulates, the bias is present and unquantified.

**Negative sampling changes score calibration.** Training keeps all positives
against 120 uniformly sampled negatives, so predicted scores are not calibrated
probabilities. Irrelevant for ranking, where only the ordering is used; it would
matter if these scores were displayed or thresholded, and they are not.

**Hard-negative sampling was tried and rejected on evidence.** Sampling
negatives preferentially from the top of the pool is the textbook trick, and
here it halved accuracy (NDCG@10 0.050 → 0.022) while coverage doubled — a
train/serve distribution mismatch, since `candidate_rank` is the strongest
feature and biased sampling left the model with almost no examples above rank
~60. Full results in the `ranking/dataset.py` docstring.

**Absolute numbers are not benchmarks.** The synthetic dataset converts at
8.2 % of views against a realistic 2–3 % of sessions (`docs/database.md` §8).
Only relative comparisons between models are meaningful.

**Single split, single seed.** Every number here comes from one temporal split.
Repeated splits over rolling windows would give confidence intervals on the
metrics themselves, not just on the differences between models.

---

## 10. The cost of diversity

Diversity is a gate metric, not a footnote (ADR-014), because the easiest way to
raise NDCG is to recommend the same popular items to everyone.

| | NDCG@10 | Intra-list diversity | Coverage |
| --- | --- | --- | --- |
| two-stage-ranker | 0.0582 | 0.578 | 0.284 |
| two-stage-ranker + diversity | 0.0515 | 0.848 | 0.207 |
| **change** | **−11.5 %** | **+47 %** | **−27 %** |

MMR plus per-category caps buys a 47 % increase in intra-list diversity for an
11.5 % NDCG cost — and that cost is *not* statistically significant (p = 0.057).

Coverage falling is the surprise, and it is worth understanding rather than
explaining away: the caps push each individual list toward distinct categories,
but they push *every* user's list toward the same broad, popular categories, so
the union across users narrows. Intra-list diversity and catalogue coverage are
different objectives and this configuration trades one for the other.

**Recommendation.** Ship diversity on, because a rail of five near-identical
running shoes is the failure users actually complain about and the accuracy cost
is within noise. Then measure it online — intra-list diversity is a proxy for
satisfaction, and only a live experiment can say whether the proxy holds.

---

## 11. Reproducibility

- Every run is seeded (`RecsysConfig.seed = 42`); the dataset is regenerable from
  its own seed in ~20 s.
- The full configuration is logged to MLflow as parameters, and the split
  boundaries are hashed into a `dataset_fingerprint` tag — so two runs with
  identical parameters but different metrics are explainable rather than
  mysterious.
- One run: **1,255 metrics** across 13 models and 4 segments, plus artefacts, in
  about 4 minutes on 16 cores.

---

## 12. What would improve results most

In order of expected value:

1. **Retrieval, not ranking.** Recall@300 is 0.343, so two thirds of relevant
   items are unreachable. Rebalancing budget toward `category_affinity`, adding
   a session-based sequential retriever, and widening the pool are all higher
   leverage than any ranker change.
2. **More ranking data.** 1,596 queries is small for a listwise objective, which
   is a plausible explanation for LambdaRank's failure to separate from the
   pointwise control.
3. **Logged impressions.** Replacing bootstrap labels with real impression logs
   removes the largest known bias and makes inverse-propensity weighting
   possible.
4. **An online experiment.** The offline differences at the top of the table are
   within noise. Only a live A/B test can resolve them, which is what Phase 15
   is for.
