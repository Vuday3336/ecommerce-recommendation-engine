"""End-to-end training pipeline.

    python ml/pipelines/train.py
    python ml/pipelines/train.py --users 800 --no-ranker      # fast iteration
    python ml/pipelines/train.py --mlflow                     # with tracking

Runs the full sequence from `architecture.md` section 6:

    extract -> validate -> temporal split -> calibrate weights
      -> feature engineering -> train models -> build candidate generator
      -> build ranking dataset -> train rankers -> evaluate on test
      -> save artefacts (-> register in MLflow)

Every stage is a method so the pipeline can be resumed, tested piecewise, and
called from the retraining job without duplicating the wiring.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ml"))

from recsys.candidates.generator import (  # noqa: E402
    CandidateGenerator,
    active_product_rule,
    in_stock_rule,
    not_recently_purchased_rule,
    recall_at_n,
    source_contribution,
)
from recsys.candidates.sources import (  # noqa: E402
    BrandAffinitySource,
    CategoryAffinitySource,
    ExplorationSource,
    ItemNeighbourSource,
    ModelSource,
    RecentlyViewedSource,
    StaticListSource,
)
from recsys.config.settings import (  # noqa: E402
    DiversityConfig,
    EvaluationConfig,
    RecsysConfig,
)
from recsys.data.loaders import Dataset, load_dataset  # noqa: E402
from recsys.evaluation.evaluator import (  # noqa: E402
    EvaluationResult,
    Evaluator,
    comparison_table,
    segment_table,
)
from recsys.evaluation.significance import paired_bootstrap  # noqa: E402
from recsys.features.interaction import PairFeatureAssembler, build_pair_history  # noqa: E402
from recsys.features.product import build_product_features  # noqa: E402
from recsys.features.user import build_category_affinity, build_user_features  # noqa: E402
from recsys.models.base import RecommendationContext  # noqa: E402
from recsys.models.collaborative import ALSRecommender, BPRRecommender  # noqa: E402
from recsys.models.content import ContentRecommender  # noqa: E402
from recsys.models.covisit import CoVisitationRecommender, FrequentlyBoughtTogether  # noqa: E402
from recsys.models.hybrid import HybridRecommender  # noqa: E402
from recsys.models.popularity import PopularityRecommender, TrendingRecommender  # noqa: E402
from recsys.preprocessing.splitting import TemporalSplit, split_temporal  # noqa: E402
from recsys.preprocessing.weighting import (  # noqa: E402
    WeightCalibration,
    build_interaction_matrix_frame,
    calibrate_weights,
)
from recsys.ranking.dataset import RankingDatasetBuilder, build_labels  # noqa: E402
from recsys.ranking.pipeline import TwoStageRecommender  # noqa: E402
from recsys.ranking.ranker import LambdaRanker, LogisticRanker, XGBoostRanker  # noqa: E402

logger = logging.getLogger("recsys.train")

#: Products whose repeat rate makes re-recommendation sensible after purchase.
CONSUMABLE_THRESHOLD = 0.30


@dataclass
class TrainedArtifacts:
    """Everything one training run produces."""

    config: RecsysConfig
    split: TemporalSplit
    calibration: WeightCalibration
    interaction_weights: pd.DataFrame
    user_features: pd.DataFrame
    product_features: pd.DataFrame
    category_affinity: dict[int, dict[int, float]]
    brand_affinity: dict[int, dict[int, float]]
    models: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluations: list[EvaluationResult] = field(default_factory=list)


class TrainingPipeline:
    """Orchestrates one full training run."""

    def __init__(self, config: RecsysConfig | None = None, *, dataset: Dataset | None = None) -> None:
        self.config = config or RecsysConfig()
        self.dataset = dataset
        self.artifacts: TrainedArtifacts | None = None
        self._timings: dict[str, float] = {}

    # -- stages -----------------------------------------------------------

    def extract(self) -> Dataset:
        if self.dataset is None:
            self.dataset = load_dataset(self.config.data_dir)
        logger.info("dataset: %s", self.dataset.summary())
        return self.dataset

    def validate(self, dataset: Dataset) -> dict[str, Any]:
        """Data contracts checked before anything is trained on them.

        Cheap, and it converts the most common silent failure - an upstream
        change that empties a column - into a loud one at the top of the run
        instead of a mysterious metric regression at the bottom.
        """
        interactions = dataset.interactions
        checks = {
            "events": len(dataset.events),
            "interactions": len(interactions),
            "users_with_interactions": int(interactions["user_id"].nunique()),
            "products_with_interactions": int(interactions["product_id"].nunique()),
            "null_user_ids": int(interactions["user_id"].isna().sum()),
            "null_product_ids": int(interactions["product_id"].isna().sum()),
            "orphan_products": int(
                (~interactions["product_id"].isin(dataset.products["id"])).sum()
            ),
        }
        if checks["interactions"] < 1000:
            raise ValueError(f"only {checks['interactions']} interactions; refusing to train")
        if checks["null_user_ids"] or checks["null_product_ids"]:
            raise ValueError(f"null identifiers in the interaction frame: {checks}")
        if checks["orphan_products"]:
            raise ValueError(
                f"{checks['orphan_products']} interactions reference unknown products"
            )
        logger.info("validation passed: %s", checks)
        return checks

    def split(self, dataset: Dataset) -> TemporalSplit:
        split = split_temporal(dataset.interactions, self.config.split)
        logger.info("split: %s", split.summary())
        return split

    def build_features(
        self, dataset: Dataset, split: TemporalSplit
    ) -> tuple[WeightCalibration, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
        calibration = calibrate_weights(split.train, self.config.weighting)
        logger.info("calibrated weights: %s", {k: round(v, 3) for k, v in calibration.weights.items()})

        weights = build_interaction_matrix_frame(
            split.train, calibration, self.config.weighting, as_of=split.train_end
        )
        user_features = build_user_features(
            split.train,
            as_of=split.train_end,
            products=dataset.products,
            orders=dataset.orders,
        )
        product_features = build_product_features(
            split.train, dataset.products, as_of=split.train_end, config=self.config.popularity
        )
        category_affinity = build_category_affinity(
            split.train, weights, as_of=split.train_end, level="category_id"
        )
        brand_affinity = build_category_affinity(
            split.train, weights, as_of=split.train_end, level="brand_id", top_n=6
        )
        return (
            calibration,
            weights,
            user_features,
            product_features,
            category_affinity,
            brand_affinity,
        )

    def train_models(
        self,
        dataset: Dataset,
        split: TemporalSplit,
        weights: pd.DataFrame,
        product_features: pd.DataFrame,
        *,
        train_bpr: bool = True,
    ) -> dict[str, Any]:
        models: dict[str, Any] = {}

        models["popularity"] = PopularityRecommender(self.config.popularity).fit(
            split.train, dataset.products, as_of=split.train_end
        )
        models["trending"] = TrendingRecommender(self.config.popularity).fit(
            split.train, dataset.products, as_of=split.train_end
        )
        models["content"] = ContentRecommender(self.config.content).fit(
            dataset.products, dataset.categories
        )
        models["collaborative_als"] = ALSRecommender(self.config.collaborative).fit(
            weights, min_item_interactions=self.config.split.min_item_interactions
        )
        if train_bpr:
            models["collaborative_bpr"] = BPRRecommender(self.config.collaborative).fit(
                weights, min_item_interactions=self.config.split.min_item_interactions
            )
        models["covisitation"] = CoVisitationRecommender().fit(split.train)

        train_orders = dataset.orders[
            dataset.orders["placed_at"] < pd.Timestamp(split.train_end)
        ]
        models["frequently_bought_together"] = FrequentlyBoughtTogether().fit(
            dataset.order_items[dataset.order_items["order_id"].isin(train_orders["id"])]
        )

        models["hybrid"] = HybridRecommender(
            collaborative=models["collaborative_als"],
            content=models["content"],
            trending=models["trending"],
            popularity=models["popularity"],
            covisitation=models["covisitation"],
            config=self.config.hybrid,
            diversity=self.config.diversity,
            product_categories=dataset.products.set_index("id")["category_id"].to_dict(),
            product_brands=dataset.products.set_index("id")["brand_id"].to_dict(),
            novelty=product_features["novelty_score"].to_dict(),
        )
        return models

    def build_generator(
        self,
        dataset: Dataset,
        split: TemporalSplit,
        models: dict[str, Any],
        product_features: pd.DataFrame,
    ) -> CandidateGenerator:
        products = dataset.products
        popularity = models["popularity"]

        category_top = _top_by_group(
            products, popularity.global_scores, "category_id", limit=60
        )
        brand_top = _top_by_group(products, popularity.global_scores, "brand_id", limit=40)
        global_ranked = sorted(
            popularity.global_scores.items(), key=lambda item: -item[1]
        )[:500]
        trending_ranked = sorted(models["trending"].scores.items(), key=lambda item: -item[1])[:300]

        cold_products = product_features.index[product_features["is_cold"] > 0].to_numpy()

        sources = {
            "collaborative_als": ModelSource(models["collaborative_als"], "collaborative_als"),
            "content": ModelSource(models["content"], "content"),
            "covisitation": ItemNeighbourSource(
                models["covisitation"].neighbours, "covisitation"
            ),
            "frequently_bought_together": ItemNeighbourSource(
                models["frequently_bought_together"].neighbours,
                "frequently_bought_together",
            ),
            "category_affinity": CategoryAffinitySource(category_top),
            "brand_affinity": BrandAffinitySource(brand_top),
            "trending": StaticListSource(trending_ranked, "trending"),
            "popularity": StaticListSource(global_ranked, "popularity"),
            "recently_viewed": RecentlyViewedSource(),
            "exploration": ExplorationSource(cold_products, seed=self.config.seed),
        }

        purchased = (
            split.train[split.train["event_type"] == "PURCHASE"]
            .groupby("user_id")["product_id"]
            .apply(set)
            .to_dict()
        )
        # `repeat_rate` and `is_active` exist in the database but not in the
        # generated Parquet catalogue, so both are read defensively. Using
        # `DataFrame.get` with a scalar default silently returns that scalar
        # rather than a Series, which then fails inside `.loc` - the column
        # presence has to be tested explicitly.
        if "repeat_rate" in products.columns:
            consumable = set(
                products.loc[products["repeat_rate"] >= CONSUMABLE_THRESHOLD, "id"]
            )
        else:
            consumable = set()
        if "is_active" in products.columns:
            active = set(products.loc[products["is_active"].astype(bool), "id"])
        else:
            active = set(products["id"])
        stock = products.set_index("id")["stock_quantity"].to_dict()

        return CandidateGenerator(
            sources,
            config=self.config.candidates,
            business_rules=[
                active_product_rule(active),
                in_stock_rule(stock),
                not_recently_purchased_rule(purchased, consumable),
            ],
        )

    def build_contexts(
        self,
        split: TemporalSplit,
        user_ids: list[int],
        *,
        category_affinity: dict,
        brand_affinity: dict,
        user_features: pd.DataFrame,
    ) -> dict[int, RecommendationContext]:
        seen = split.train.groupby("user_id")["product_id"].apply(set).to_dict()
        recent = (
            split.train.sort_values("occurred_at")
            .groupby("user_id")["product_id"]
            .apply(lambda s: tuple(reversed(list(dict.fromkeys(s))))[:20])
            .to_dict()
        )
        counts = split.train.groupby("user_id").size().to_dict()
        percentiles = (
            user_features["price_percentile_mean"].to_dict()
            if "price_percentile_mean" in user_features.columns
            else {}
        )

        contexts: dict[int, RecommendationContext] = {}
        for user_id in user_ids:
            user_seen = seen.get(user_id, set())
            contexts[user_id] = RecommendationContext(
                user_id=int(user_id),
                seen_products=tuple(sorted(user_seen)),
                recent_products=recent.get(user_id, ()),
                category_affinity=category_affinity.get(user_id, {}),
                brand_affinity=brand_affinity.get(user_id, {}),
                price_percentile=float(percentiles.get(user_id, 0.5)),
                interaction_count=int(counts.get(user_id, 0)),
                exclude=frozenset(user_seen),
            )
        return contexts

    # -- orchestration ----------------------------------------------------

    def run(
        self,
        *,
        train_ranker: bool = True,
        train_bpr: bool = True,
        eval_users: int | None = 1500,
        ranker_users: int = 2500,
    ) -> TrainedArtifacts:
        started = time.perf_counter()

        dataset = self.extract()
        validation_report = self.validate(dataset)
        split = self.split(dataset)

        step = time.perf_counter()
        (
            calibration,
            weights,
            user_features,
            product_features,
            category_affinity,
            brand_affinity,
        ) = self.build_features(dataset, split)
        self._timings["features"] = time.perf_counter() - step

        step = time.perf_counter()
        models = self.train_models(
            dataset, split, weights, product_features, train_bpr=train_bpr
        )
        self._timings["models"] = time.perf_counter() - step

        generator = self.build_generator(dataset, split, models, product_features)
        models["generator"] = generator

        pair_history = build_pair_history(split.train, as_of=split.train_end)
        pair_assembler = PairFeatureAssembler(
            product_features=product_features,
            pair_history=pair_history,
            interaction_weights=weights,
            category_affinity=category_affinity,
            brand_affinity=brand_affinity,
        )
        dataset_builder = RankingDatasetBuilder(
            generator=generator,
            user_features=user_features,
            product_features=product_features,
            pair_assembler=pair_assembler,
            config=self.config.ranker,
        )
        models["pair_assembler"] = pair_assembler
        models["dataset_builder"] = dataset_builder

        metrics: dict[str, Any] = {
            "validation": validation_report,
            "split": split.summary(),
            "weights": calibration.weights,
        }

        if train_ranker:
            step = time.perf_counter()
            ranker_metrics = self._train_rankers(
                dataset, split, models, dataset_builder,
                category_affinity=category_affinity,
                brand_affinity=brand_affinity,
                user_features=user_features,
                max_users=ranker_users,
            )
            metrics.update(ranker_metrics)
            self._timings["ranker"] = time.perf_counter() - step

        step = time.perf_counter()
        evaluations = self._evaluate(
            dataset, split, models,
            category_affinity=category_affinity,
            brand_affinity=brand_affinity,
            eval_users=eval_users,
        )
        self._timings["evaluation"] = time.perf_counter() - step
        self._timings["total"] = time.perf_counter() - started

        metrics["timings_seconds"] = {k: round(v, 2) for k, v in self._timings.items()}

        self.artifacts = TrainedArtifacts(
            config=self.config,
            split=split,
            calibration=calibration,
            interaction_weights=weights,
            user_features=user_features,
            product_features=product_features,
            category_affinity=category_affinity,
            brand_affinity=brand_affinity,
            models=models,
            metrics=metrics,
            evaluations=evaluations,
        )
        return self.artifacts

    def _train_rankers(
        self,
        dataset: Dataset,
        split: TemporalSplit,
        models: dict[str, Any],
        dataset_builder: RankingDatasetBuilder,
        *,
        category_affinity: dict,
        brand_affinity: dict,
        user_features: pd.DataFrame,
        max_users: int,
    ) -> dict[str, Any]:
        """Train the ranker on candidates labelled by the validation window."""
        labels = build_labels(split.validation, self.config.ranker)
        if not labels:
            logger.warning("no ranking labels available; skipping the ranker")
            return {}

        rng = np.random.default_rng(self.config.seed)
        candidate_users = sorted(labels)
        if len(candidate_users) > max_users:
            chosen = rng.choice(len(candidate_users), size=max_users, replace=False)
            candidate_users = sorted(candidate_users[int(i)] for i in chosen)

        # A time-ordered split of the labelling window: the ranker's own
        # early-stopping set must also come from the future relative to its
        # training rows, or early stopping selects on information the model
        # would not have at serving time.
        cut = int(len(candidate_users) * 0.85)
        train_users, validation_users = candidate_users[:cut], candidate_users[cut:]

        contexts = self.build_contexts(
            split, candidate_users,
            category_affinity=category_affinity,
            brand_affinity=brand_affinity,
            user_features=user_features,
        )

        train_data = dataset_builder.build(
            {u: contexts[u] for u in train_users}, labels
        )
        validation_data = dataset_builder.build(
            {u: contexts[u] for u in validation_users}, labels
        )
        logger.info("ranking dataset: %s", train_data.summary())

        if len(train_data) == 0:
            logger.warning("ranking dataset is empty; skipping the ranker")
            return {}

        ranker = LambdaRanker(self.config.ranker).fit(train_data, validation_data)
        models["ranker"] = ranker

        # Controls for ADR-005, trained on identical data.
        controls: dict[str, Any] = {}
        try:
            controls["ranker_xgboost"] = XGBoostRanker(self.config.ranker).fit(train_data)
        except Exception:
            logger.exception("xgboost control failed to train")
        try:
            controls["ranker_logistic"] = LogisticRanker(self.config.ranker).fit(train_data)
        except Exception:
            logger.exception("logistic control failed to train")
        models.update(controls)

        # Stage-1 recall: the ceiling on everything stage 2 can achieve.
        candidate_sets = {
            user_id: dataset_builder.generator.generate(contexts[user_id])
            for user_id in validation_users[:400]
        }
        targets = {
            user_id: set(labels.get(user_id, {}))
            for user_id in candidate_sets
        }
        return {
            "ranking_dataset": train_data.summary(),
            "stage1_recall": round(recall_at_n(candidate_sets, targets), 5),
            "source_contribution": {
                k: round(v, 5) for k, v in source_contribution(candidate_sets, targets).items()
            },
            "ranker_best_iteration": ranker.best_iteration,
            "feature_importance": ranker.feature_importance(20).to_dict(orient="records"),
        }

    def _evaluate(
        self,
        dataset: Dataset,
        split: TemporalSplit,
        models: dict[str, Any],
        *,
        category_affinity: dict,
        brand_affinity: dict,
        eval_users: int | None,
    ) -> list[EvaluationResult]:
        evaluator = Evaluator(
            split,
            products=dataset.products,
            config=EvaluationConfig(
                max_eval_users=eval_users, random_state=self.config.seed
            ),
        )
        results: list[EvaluationResult] = []

        results.append(
            evaluator.evaluate(models["popularity"], record_baseline=True, name="popularity-global")
        )
        results.append(
            evaluator.evaluate(
                models["popularity"], category_affinity=category_affinity, name="popularity-category"
            )
        )
        results.append(evaluator.evaluate(models["trending"], name="trending"))
        results.append(evaluator.evaluate(models["content"], name="content"))
        results.append(evaluator.evaluate(models["covisitation"], name="covisitation"))
        results.append(
            evaluator.evaluate(models["collaborative_als"], name="collaborative-als")
        )
        if "collaborative_bpr" in models:
            results.append(
                evaluator.evaluate(models["collaborative_bpr"], name="collaborative-bpr")
            )
        # Evaluated without the diversity re-ranker so it is compared on the
        # same footing as every other row; the diversified variant is reported
        # separately below.
        hybrid_plain = copy.copy(models["hybrid"])
        hybrid_plain.diversity = DiversityConfig(enabled=False)
        results.append(
            evaluator.evaluate(
                hybrid_plain,
                category_affinity=category_affinity,
                brand_affinity=brand_affinity,
                name="hybrid",
            )
        )
        results.append(
            evaluator.evaluate(
                models["hybrid"],
                category_affinity=category_affinity,
                brand_affinity=brand_affinity,
                name="hybrid+diversity",
            )
        )

        # The diversity re-ranker is evaluated as a separate variant rather
        # than baked in. Every other model in this table is reported without
        # it, so folding MMR and category caps into the two-stage pipeline
        # would compare a diversified list against undiversified ones and make
        # the ranker look worse than it is. Reporting both isolates the
        # accuracy cost of diversity, which ADR-014 commits to measuring
        # rather than assuming.
        no_diversity = DiversityConfig(enabled=False)

        if "ranker" in models:
            for label, key in (
                ("two-stage-ranker", "ranker"),
                ("two-stage-xgboost", "ranker_xgboost"),
                ("two-stage-logistic", "ranker_logistic"),
            ):
                if key not in models:
                    continue
                pipeline = TwoStageRecommender(
                    generator=models["generator"],
                    dataset_builder=models["dataset_builder"],
                    ranker=models[key],
                    hybrid=models["hybrid"],
                    diversity=no_diversity,
                    content_model=models["content"],
                    collaborative_model=models["collaborative_als"],
                )
                results.append(
                    evaluator.evaluate(
                        pipeline,
                        category_affinity=category_affinity,
                        brand_affinity=brand_affinity,
                        name=label,
                    )
                )

            diversified = TwoStageRecommender(
                generator=models["generator"],
                dataset_builder=models["dataset_builder"],
                ranker=models["ranker"],
                hybrid=models["hybrid"],
                diversity=self.config.diversity,
                content_model=models["content"],
                collaborative_model=models["collaborative_als"],
            )
            results.append(
                evaluator.evaluate(
                    diversified,
                    category_affinity=category_affinity,
                    brand_affinity=brand_affinity,
                    name="two-stage-ranker+diversity",
                )
            )
        return results


def _top_by_group(
    products: pd.DataFrame, scores: dict[int, float], column: str, *, limit: int
) -> dict[int, list[tuple[int, float]]]:
    """Top-scoring products per category or brand."""
    frame = products[["id", column]].copy()
    frame["score"] = frame["id"].map(scores).fillna(0.0)
    out: dict[int, list[tuple[int, float]]] = {}
    for key, group in frame.groupby(column):
        ranked = group.nlargest(limit, "score")
        out[int(key)] = [
            (int(row.id), float(row.score)) for row in ranked.itertuples(index=False)
        ]
    return out


def save_artifacts(artifacts: TrainedArtifacts, directory: Path) -> dict[str, str]:
    """Persist models and tables for the serving path."""
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for name in (
        "popularity",
        "trending",
        "content",
        "collaborative_als",
        "collaborative_bpr",
        "covisitation",
        "frequently_bought_together",
    ):
        model = artifacts.models.get(name)
        if model is None:
            continue
        path = directory / f"{name}.joblib"
        joblib.dump(model, path)
        saved[name] = str(path)

    ranker = artifacts.models.get("ranker")
    if ranker is not None:
        saved["ranker"] = str(ranker.save(directory / "ranker.joblib"))

    artifacts.user_features.to_parquet(directory / "user_features.parquet")
    artifacts.product_features.to_parquet(directory / "product_features.parquet")
    artifacts.interaction_weights.to_parquet(directory / "interaction_weights.parquet")
    artifacts.calibration.as_report().to_csv(directory / "weight_calibration.csv", index=False)

    (directory / "affinity.json").write_text(
        json.dumps(
            {
                "category": {str(k): v for k, v in artifacts.category_affinity.items()},
                "brand": {str(k): v for k, v in artifacts.brand_affinity.items()},
            }
        ),
        encoding="utf-8",
    )

    table = comparison_table(artifacts.evaluations)
    table.to_csv(directory / "evaluation.csv", index=False)
    segment_table(artifacts.evaluations).to_csv(directory / "evaluation_by_segment.csv", index=False)

    (directory / "metrics.json").write_text(
        json.dumps(artifacts.metrics, indent=2, default=str), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps(artifacts.config.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "ml" / "artifacts")
    parser.add_argument("--eval-users", type=int, default=1500)
    parser.add_argument("--ranker-users", type=int, default=2500)
    parser.add_argument("--no-ranker", action="store_true")
    parser.add_argument("--no-bpr", action="store_true")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    config = RecsysConfig(seed=args.seed)
    if args.data:
        config = RecsysConfig(seed=args.seed, data_dir=args.data)

    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run(
        train_ranker=not args.no_ranker,
        train_bpr=not args.no_bpr,
        eval_users=args.eval_users,
        ranker_users=args.ranker_users,
    )

    print("\n" + "=" * 110)
    print("MODEL COMPARISON (test fold)")
    print("=" * 110)
    columns = [
        "model", "precision@10", "recall@10", "ndcg@10", "map@10",
        "hit_rate@10", "mrr", "coverage", "diversity", "novelty",
        "personalisation", "seconds",
    ]
    table = comparison_table(artifacts.evaluations)
    print(table[[c for c in columns if c in table.columns]].to_string(index=False))

    print("\nNDCG@10 by user history depth")
    print(segment_table(artifacts.evaluations).to_string(index=False))

    # Paired significance against the strongest simple baseline. A 0.002 gap on
    # 1,500 users is not a result, and reporting it as one is how a model that
    # is no better than its predecessor gets shipped.
    baseline_name = "popularity-category"
    baseline = next(
        (r for r in artifacts.evaluations if r.model == baseline_name), None
    )
    if baseline is not None and baseline.per_user_scores:
        print(f"\nPaired bootstrap on NDCG@10 against {baseline_name} (5,000 resamples)")
        comparisons = []
        for result in sorted(
            artifacts.evaluations, key=lambda r: -r.metrics.get("ndcg@10", 0.0)
        ):
            if result.model == baseline_name or not result.per_user_scores:
                continue
            comparison = paired_bootstrap(
                baseline.per_user_scores,
                result.per_user_scores,
                baseline_name=baseline_name,
                candidate_name=result.model,
            )
            comparisons.append(comparison)
            print(f"  {comparison.summary()}")
        artifacts.metrics["significance"] = [
            {
                "candidate": c.candidate,
                "baseline": c.baseline,
                "difference": round(c.difference, 6),
                "ci_low": round(c.ci_low, 6),
                "ci_high": round(c.ci_high, 6),
                "p_value": round(c.p_value, 6),
                "significant": c.significant,
            }
            for c in comparisons
        ]

    if "stage1_recall" in artifacts.metrics:
        print(f"\nStage-1 recall@{config.candidates.total}: {artifacts.metrics['stage1_recall']}")
        print("Source contribution to recall:")
        for source, value in artifacts.metrics.get("source_contribution", {}).items():
            print(f"  {source:<28} {value:.4f}")

    if "feature_importance" in artifacts.metrics:
        print("\nTop ranking features by gain:")
        for row in artifacts.metrics["feature_importance"][:12]:
            print(f"  {row['feature']:<44} {row['gain_share']:.3%}")

    saved = save_artifacts(artifacts, args.out)
    print(f"\nArtefacts written to {args.out} ({len(saved)} models)")
    print(f"Timings: {artifacts.metrics['timings_seconds']}")

    if args.mlflow:
        from pipelines.tracking import log_training_run

        run_id = log_training_run(artifacts, args.out)
        print(f"MLflow run: {run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
