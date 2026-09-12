"""The serving engine.

Loads trained artefacts once at start-up and answers recommendation requests.
This is the object the FastAPI service holds; the API layer above it does
transport, auth and caching, and knows nothing about how a recommendation is
produced.

**Why the catalogue lives in memory.** Five thousand products is roughly 2 MB of
Python objects. Reading them from Postgres on every request would put an OLTP
query on the hot path for data that changes hourly at most. The index is built
at start-up from whichever `CatalogueSource` is configured - the database in
production, the Parquet snapshot in development - and refreshed by a background
job. This is also what lets the recommendation endpoints run and be tested with
no database at all.

**Nothing here raises on a missing model.** Every artefact is optional and the
engine degrades through the ladder in `architecture.md` §4.2. An engine with no
artefacts at all still answers, with global trending, and says so.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.candidates.generator import (
    CandidateGenerator,
    active_product_rule,
    in_stock_rule,
    not_recently_purchased_rule,
)
from recsys.candidates.sources import (
    BrandAffinitySource,
    CategoryAffinitySource,
    ExplorationSource,
    ItemNeighbourSource,
    ModelSource,
    RecentlyViewedSource,
    StaticListSource,
)
from recsys.config.settings import RecsysConfig
from recsys.explain.explainer import Explainer, Explanation
from recsys.features.interaction import PairFeatureAssembler
from recsys.models.base import RecommendationContext, Scored
from recsys.models.hybrid import HybridRecommender
from recsys.ranking.dataset import RankingDatasetBuilder
from recsys.ranking.pipeline import RankedResult, TwoStageRecommender

logger = logging.getLogger(__name__)

ARTIFACT_MODELS: tuple[str, ...] = (
    "popularity",
    "trending",
    "content",
    "collaborative_als",
    "covisitation",
    "frequently_bought_together",
)


@dataclass(slots=True)
class ProductRecord:
    """The catalogue fields a recommendation response needs."""

    id: int
    name: str
    category_id: int
    brand_id: int
    price: float
    price_band: str
    rating_average: float
    stock_quantity: int
    image_slug: str = ""


@dataclass(slots=True)
class CatalogueIndex:
    """In-memory catalogue, refreshed periodically."""

    products: dict[int, ProductRecord] = field(default_factory=dict)
    category_names: dict[int, str] = field(default_factory=dict)
    brand_names: dict[int, str] = field(default_factory=dict)
    category_paths: dict[int, str] = field(default_factory=dict)
    refreshed_at: dt.datetime | None = None

    def __len__(self) -> int:
        return len(self.products)

    def get(self, product_id: int) -> ProductRecord | None:
        return self.products.get(int(product_id))

    def many(self, product_ids: list[int]) -> list[ProductRecord]:
        return [p for pid in product_ids if (p := self.products.get(int(pid))) is not None]

    @classmethod
    def from_frames(
        cls, products: pd.DataFrame, categories: pd.DataFrame, brands: pd.DataFrame
    ) -> CatalogueIndex:
        records = {
            int(row.id): ProductRecord(
                id=int(row.id),
                name=str(row.name),
                category_id=int(row.category_id),
                brand_id=int(row.brand_id),
                price=float(row.price),
                price_band=str(row.price_band),
                rating_average=float(row.rating_average),
                stock_quantity=int(row.stock_quantity),
            )
            for row in products.itertuples(index=False)
        }
        return cls(
            products=records,
            category_names=dict(
                zip(categories["id"].astype(int), categories["name"], strict=True)
            ),
            brand_names=dict(zip(brands["id"].astype(int), brands["name"], strict=True)),
            category_paths=dict(
                zip(categories["id"].astype(int), categories["path"], strict=True)
            ),
            refreshed_at=dt.datetime.now(dt.UTC),
        )


@dataclass(slots=True)
class RecommendationItem:
    """One item in a served response."""

    product: ProductRecord
    score: float
    source: str
    explanation: Explanation
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RecommendationResponse:
    """A served rail."""

    items: list[RecommendationItem]
    surface: str
    strategy: str
    model_version: str
    candidate_pool_size: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


class RecommendationEngine:
    """Loads artefacts and serves every recommendation surface."""

    def __init__(
        self,
        *,
        config: RecsysConfig | None = None,
        model_version: str = "unversioned",
    ) -> None:
        self.config = config or RecsysConfig()
        self.model_version = model_version
        self.catalogue = CatalogueIndex()

        self.models: dict[str, Any] = {}
        self.user_features: pd.DataFrame = pd.DataFrame()
        self.product_features: pd.DataFrame = pd.DataFrame()
        self.interaction_weights: pd.DataFrame = pd.DataFrame()
        self.category_affinity: dict[int, dict[int, float]] = {}
        self.brand_affinity: dict[int, dict[int, float]] = {}

        self.generator: CandidateGenerator | None = None
        self.pipeline: TwoStageRecommender | None = None
        self.explainer: Explainer | None = None
        self.loaded = False

    # -- loading ----------------------------------------------------------

    def load(self, artifact_dir: Path, dataset_dir: Path | None = None) -> RecommendationEngine:
        """Load artefacts and build the serving pipeline."""
        import joblib

        artifact_dir = Path(artifact_dir)
        if not artifact_dir.exists():
            logger.warning("no artefacts at %s; the engine will serve fallbacks only", artifact_dir)
            return self

        for name in ARTIFACT_MODELS:
            path = artifact_dir / f"{name}.joblib"
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                except Exception:
                    logger.exception("failed to load %s; continuing without it", name)

        ranker_path = artifact_dir / "ranker.joblib"
        if ranker_path.exists():
            try:
                from recsys.ranking.ranker import LambdaRanker

                self.models["ranker"] = LambdaRanker.load(ranker_path)
            except Exception:
                logger.exception("failed to load the ranker; falling back to the hybrid")

        for name, attribute in (
            ("user_features", "user_features"),
            ("product_features", "product_features"),
            ("interaction_weights", "interaction_weights"),
        ):
            path = artifact_dir / f"{name}.parquet"
            if path.exists():
                setattr(self, attribute, pd.read_parquet(path))

        affinity_path = artifact_dir / "affinity.json"
        if affinity_path.exists():
            import json

            payload = json.loads(affinity_path.read_text(encoding="utf-8"))
            self.category_affinity = {
                int(k): {int(kk): float(vv) for kk, vv in v.items()}
                for k, v in payload.get("category", {}).items()
            }
            self.brand_affinity = {
                int(k): {int(kk): float(vv) for kk, vv in v.items()}
                for k, v in payload.get("brand", {}).items()
            }

        if dataset_dir is not None:
            self.load_catalogue_from_parquet(dataset_dir)

        self._build_pipeline()
        self.loaded = bool(self.models)
        logger.info(
            "engine loaded: %d models, %d products, version=%s",
            len(self.models),
            len(self.catalogue),
            self.model_version,
        )
        return self

    def load_catalogue_from_parquet(self, dataset_dir: Path) -> None:
        dataset_dir = Path(dataset_dir)
        try:
            self.catalogue = CatalogueIndex.from_frames(
                pd.read_parquet(dataset_dir / "products.parquet"),
                pd.read_parquet(dataset_dir / "categories.parquet"),
                pd.read_parquet(dataset_dir / "brands.parquet"),
            )
        except Exception:
            logger.exception("failed to load the catalogue from %s", dataset_dir)

    def set_catalogue(self, index: CatalogueIndex) -> None:
        self.catalogue = index
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Assemble the candidate generator and the two-stage pipeline."""
        if not self.models or not self.catalogue.products:
            return

        popularity = self.models.get("popularity")
        trending = self.models.get("trending")
        global_scores = getattr(popularity, "global_scores", {}) if popularity else {}

        category_top = self._top_by(lambda p: p.category_id, global_scores, limit=60)
        brand_top = self._top_by(lambda p: p.brand_id, global_scores, limit=40)
        global_ranked = sorted(global_scores.items(), key=lambda item: -item[1])[:500]
        trending_ranked = sorted(
            getattr(trending, "scores", {}).items(), key=lambda item: -item[1]
        )[:300]

        cold_products = (
            self.product_features.index[self.product_features["is_cold"] > 0].to_numpy()
            if "is_cold" in self.product_features.columns
            else np.array([], dtype=np.int64)
        )

        sources: dict[str, Any] = {}
        if "collaborative_als" in self.models:
            sources["collaborative_als"] = ModelSource(
                self.models["collaborative_als"], "collaborative_als"
            )
        if "content" in self.models:
            sources["content"] = ModelSource(self.models["content"], "content")
        if "covisitation" in self.models:
            sources["covisitation"] = ItemNeighbourSource(
                self.models["covisitation"].neighbours, "covisitation"
            )
        if "frequently_bought_together" in self.models:
            sources["frequently_bought_together"] = ItemNeighbourSource(
                self.models["frequently_bought_together"].neighbours,
                "frequently_bought_together",
            )
        sources["category_affinity"] = CategoryAffinitySource(category_top)
        sources["brand_affinity"] = BrandAffinitySource(brand_top)
        if trending_ranked:
            sources["trending"] = StaticListSource(trending_ranked, "trending")
        if global_ranked:
            sources["popularity"] = StaticListSource(global_ranked, "popularity")
        sources["recently_viewed"] = RecentlyViewedSource()
        if len(cold_products):
            sources["exploration"] = ExplorationSource(cold_products, seed=self.config.seed)

        stock = {pid: p.stock_quantity for pid, p in self.catalogue.products.items()}
        active = set(self.catalogue.products)

        self.generator = CandidateGenerator(
            sources,
            config=self.config.candidates,
            business_rules=[
                active_product_rule(active),
                in_stock_rule(stock),
                not_recently_purchased_rule({}, set()),
            ],
        )

        hybrid = HybridRecommender(
            collaborative=self.models.get("collaborative_als"),
            content=self.models.get("content"),
            trending=trending,
            popularity=popularity,
            covisitation=self.models.get("covisitation"),
            config=self.config.hybrid,
            diversity=self.config.diversity,
            product_categories={p.id: p.category_id for p in self.catalogue.products.values()},
            product_brands={p.id: p.brand_id for p in self.catalogue.products.values()},
            novelty=(
                self.product_features["novelty_score"].to_dict()
                if "novelty_score" in self.product_features.columns
                else {}
            ),
        )
        self.models["hybrid"] = hybrid

        if not self.product_features.empty:
            assembler = PairFeatureAssembler(
                product_features=self.product_features,
                interaction_weights=self.interaction_weights,
                category_affinity=self.category_affinity,
                brand_affinity=self.brand_affinity,
            )
            builder = RankingDatasetBuilder(
                generator=self.generator,
                user_features=self.user_features,
                product_features=self.product_features,
                pair_assembler=assembler,
                config=self.config.ranker,
            )
            self.pipeline = TwoStageRecommender(
                generator=self.generator,
                dataset_builder=builder,
                ranker=self.models.get("ranker"),
                hybrid=hybrid,
                diversity=self.config.diversity,
                content_model=self.models.get("content"),
                collaborative_model=self.models.get("collaborative_als"),
                fallback=[pid for pid, _ in global_ranked[:100]],
            )

        self.explainer = Explainer(
            catalogue=self.catalogue,
            category_affinity=self.category_affinity,
            brand_affinity=self.brand_affinity,
        )

    def _top_by(self, key, scores: dict[int, float], *, limit: int) -> dict[int, list[tuple[int, float]]]:
        grouped: dict[int, list[tuple[int, float]]] = {}
        for product in self.catalogue.products.values():
            grouped.setdefault(key(product), []).append(
                (product.id, scores.get(product.id, 0.0))
            )
        return {
            group: sorted(items, key=lambda item: -item[1])[:limit]
            for group, items in grouped.items()
        }

    # -- context ----------------------------------------------------------

    def build_context(
        self,
        *,
        user_id: int | None,
        recent_products: tuple[int, ...] = (),
        seen_products: tuple[int, ...] = (),
        exclude: frozenset[int] = frozenset(),
    ) -> RecommendationContext:
        interaction_count = 0
        price_percentile = 0.5
        if user_id is not None and user_id in self.user_features.index:
            row = self.user_features.loc[user_id]
            interaction_count = int(row.get("total_events", 0))
            price_percentile = float(row.get("price_percentile_mean", 0.5)) or 0.5

        return RecommendationContext(
            user_id=user_id,
            recent_products=recent_products,
            seen_products=seen_products,
            category_affinity=self.category_affinity.get(user_id or -1, {}),
            brand_affinity=self.brand_affinity.get(user_id or -1, {}),
            price_percentile=price_percentile,
            interaction_count=interaction_count,
            exclude=exclude or frozenset(seen_products),
        )

    # -- surfaces ---------------------------------------------------------

    def personalised(
        self,
        context: RecommendationContext,
        *,
        k: int = 12,
        surface: str = "home_for_you",
    ) -> RecommendationResponse:
        import time

        started = time.perf_counter()
        if self.pipeline is not None:
            result = self.pipeline.rank(context, k)
        else:
            result = self._degraded(context, k)

        return self._to_response(
            result, surface=surface, latency_ms=(time.perf_counter() - started) * 1000.0
        )

    def _degraded(self, context: RecommendationContext, k: int) -> RankedResult:
        """Rungs 5-7: no pipeline available."""
        hybrid = self.models.get("hybrid")
        if hybrid is not None:
            items = hybrid.recommend(context, k)
            if items:
                return RankedResult(items=items, strategy="hybrid", candidate_pool_size=len(items))

        popularity = self.models.get("popularity")
        if popularity is not None:
            items = popularity.recommend(context, k)
            if items:
                return RankedResult(
                    items=items, strategy="category_popular", candidate_pool_size=len(items)
                )

        ranked = sorted(self.catalogue.products.values(), key=lambda p: -p.rating_average)[:k]
        return RankedResult(
            items=[Scored(p.id, float(p.rating_average), "static_fallback") for p in ranked],
            strategy="static_fallback",
            candidate_pool_size=len(ranked),
        )

    def similar(self, product_id: int, k: int = 10) -> RecommendationResponse:
        """Content-based item-item similarity (FR-02)."""
        content = self.models.get("content")
        items = content.similar_products(product_id, k) if content else []
        if not items and "collaborative_als" in self.models:
            items = self.models["collaborative_als"].similar_products(product_id, k)
        return self._to_response(
            RankedResult(items=items, strategy="content", candidate_pool_size=len(items)),
            surface="pdp_similar",
            anchor_product_id=product_id,
        )

    def frequently_bought(self, product_id: int, k: int = 5) -> RecommendationResponse:
        """Basket complements (FR-03)."""
        model = self.models.get("frequently_bought_together")
        items = model.bought_together(product_id, k) if model else []
        return self._to_response(
            RankedResult(
                items=items, strategy="frequently_bought", candidate_pool_size=len(items)
            ),
            surface="pdp_frequently_bought_together",
            anchor_product_id=product_id,
        )

    def also_viewed(self, product_id: int, k: int = 10) -> RecommendationResponse:
        """Session co-visitation (FR-04)."""
        model = self.models.get("covisitation")
        items = model.also_viewed(product_id, k) if model else []
        return self._to_response(
            RankedResult(items=items, strategy="covisitation", candidate_pool_size=len(items)),
            surface="pdp_also_viewed",
            anchor_product_id=product_id,
        )

    def trending(self, k: int = 12, *, category_id: int | None = None) -> RecommendationResponse:
        """Time-decayed velocity (FR-05)."""
        model = self.models.get("trending")
        if model is None:
            return self._to_response(
                RankedResult(items=[], strategy="static_fallback", candidate_pool_size=0),
                surface="home_trending",
            )
        context = RecommendationContext(
            category_affinity={category_id: 1.0} if category_id else {}
        )
        items = model.recommend(context, k)
        return self._to_response(
            RankedResult(items=items, strategy="trending", candidate_pool_size=len(items)),
            surface="home_trending",
        )

    def because_you_viewed(
        self, product_id: int, context: RecommendationContext, k: int = 12
    ) -> RecommendationResponse:
        """Content neighbours of one anchor, filtered by what the user has seen."""
        content = self.models.get("content")
        if content is None:
            return self.similar(product_id, k)
        items = [
            item
            for item in content.similar_products(product_id, k * 3)
            if item.product_id not in context.exclude
        ][:k]
        return self._to_response(
            RankedResult(items=items, strategy="content", candidate_pool_size=len(items)),
            surface="home_because_you_viewed",
            anchor_product_id=product_id,
        )

    # -- assembly ---------------------------------------------------------

    def _to_response(
        self,
        result: RankedResult,
        *,
        surface: str,
        anchor_product_id: int | None = None,
        latency_ms: float = 0.0,
    ) -> RecommendationResponse:
        items: list[RecommendationItem] = []
        for scored in result.items:
            product = self.catalogue.get(scored.product_id)
            if product is None:
                continue
            components = result.score_components.get(scored.product_id, {})
            explanation = (
                self.explainer.explain(
                    product_id=scored.product_id,
                    surface=surface,
                    sources=result.sources.get(scored.product_id, [scored.source]),
                    components=components,
                    anchor_product_id=anchor_product_id,
                )
                if self.explainer
                else Explanation(text="Recommended for you", reason_code="generic")
            )
            items.append(
                RecommendationItem(
                    product=product,
                    score=scored.score,
                    source=scored.source,
                    explanation=explanation,
                    score_components=components,
                )
            )

        return RecommendationResponse(
            items=items,
            surface=surface,
            strategy=result.strategy,
            model_version=self.model_version,
            candidate_pool_size=result.candidate_pool_size,
            latency_ms=latency_ms,
            stage_timings_ms=result.stage_timings_ms,
        )

    # -- introspection ----------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model_version": self.model_version,
            "models": sorted(self.models),
            "catalogue_size": len(self.catalogue),
            "has_ranker": "ranker" in self.models,
            "has_pipeline": self.pipeline is not None,
            "users_with_features": len(self.user_features),
        }


__all__ = [
    "ARTIFACT_MODELS",
    "CatalogueIndex",
    "ProductRecord",
    "RecommendationEngine",
    "RecommendationItem",
    "RecommendationResponse",
]
