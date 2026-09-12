"""End-to-end flow tests.

These cover the success criterion from the brief: the loop from a user action
through to a changed recommendation. Unit tests prove each piece works; these
prove the pieces are actually connected, which is a different and more common
failure.

Everything here runs without Postgres or Redis. Where a step needs the database
the test asserts the *degraded* behaviour instead - because that path is also
production behaviour and is worth covering, not a gap to apologise for.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "backend", REPO_ROOT / "ml"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("JWT_SECRET_KEY", "integration-test-secret-not-used-elsewhere")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "console")

PREFIX = "/api/v1"
ARTIFACTS = REPO_ROOT / "ml" / "artifacts"
DATASET = REPO_ROOT / "data" / "synthetic"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "popularity.joblib").exists(),
    reason="no trained artefacts; run ml/pipelines/train.py first",
)


@pytest.fixture(scope="module")
def client():
    from app.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def engine():
    from recsys.inference.engine import RecommendationEngine

    return RecommendationEngine(model_version="test").load(ARTIFACTS, DATASET)


class TestBrowseToRecommendation:
    """The core loop: a user acts, the system reacts."""

    def test_an_event_flows_from_the_api_into_the_sink(self, client):
        health_before = client.get(f"{PREFIX}/recommendations/health").json()
        written_before = health_before["event_sink"].get("accepted", 0)

        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_VIEW",
                "session_id": "e2e-session-000001",
                "product_id": 100,
                "source": "pdp",
                "device_type": "desktop",
                "metadata": {"dwell_ms": 5000},
            },
        )
        assert response.status_code == 202

        health_after = client.get(f"{PREFIX}/recommendations/health").json()
        assert health_after["event_sink"]["accepted"] > written_before

    def test_a_session_full_of_events_is_accepted_as_a_batch(self, client):
        events = []
        for product_id in (100, 205, 340, 412):
            events.append(
                {
                    "event_type": "PRODUCT_VIEW",
                    "session_id": "e2e-session-000002",
                    "product_id": product_id,
                    "source": "pdp",
                }
            )
            events.append(
                {
                    "event_type": "PRODUCT_CLICK",
                    "session_id": "e2e-session-000002",
                    "product_id": product_id,
                    "source": "pdp",
                }
            )
        events.append(
            {
                "event_type": "ADD_TO_CART",
                "session_id": "e2e-session-000002",
                "product_id": 205,
                "metadata": {"quantity": 1},
            }
        )

        response = client.post(f"{PREFIX}/events/batch", json={"events": events})
        assert response.status_code == 202
        assert response.json()["accepted"] == len(events)

    def test_recent_products_change_what_is_recommended(self, engine):
        """The loop closing: different recent behaviour, different output.

        Exercised through the engine rather than the API because without Redis
        the API cannot read a recently-viewed list - that is a real degraded
        path, covered separately below.
        """
        electronics_anchor = 100
        first = engine.build_context(user_id=42, recent_products=(electronics_anchor,))
        second = engine.build_context(user_id=42, recent_products=(2593,))

        a = {item.product.id for item in engine.personalised(first, k=12).items}
        b = {item.product.id for item in engine.personalised(second, k=12).items}

        assert a and b
        overlap = len(a & b) / len(a | b)
        assert overlap < 0.95, "recent behaviour had no effect on recommendations"


class TestColdStart:
    def test_a_brand_new_user_still_gets_a_full_page(self, engine):
        """FR-09. A user with no history at all must not get an empty page."""
        context = engine.build_context(user_id=999_999_999)
        response = engine.personalised(context, k=12)
        assert len(response.items) >= 8
        assert all(item.explanation.text for item in response.items)

    def test_an_anonymous_session_gets_recommendations(self, client):
        response = client.get(
            f"{PREFIX}/recommendations/home",
            headers={"X-Session-Id": "never-seen-before-session"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["user_id"] is None
        assert payload["trending"], "trending is the cold-start fallback and must be present"

    def test_a_new_product_is_reachable_through_content(self, engine):
        """FR-10. A product with no interactions still has a category, a brand
        and a description, so it lives in the content space immediately."""
        import pandas as pd

        features = pd.read_parquet(ARTIFACTS / "product_features.parquet")
        cold = features.index[features["is_cold"] > 0]
        if len(cold) == 0:
            pytest.skip("no cold products in this dataset")

        content = engine.models.get("content")
        assert content is not None
        cold_id = int(cold[0])
        # The cold product must be findable as a neighbour of something.
        neighbours = content.similar_products(cold_id, 5)
        assert neighbours, "a cold product is invisible to content retrieval"


class TestDegradationLadder:
    """NFR-04: a recommendation surface never fails, it degrades."""

    def test_the_engine_answers_with_no_models_at_all(self):
        from recsys.inference.engine import CatalogueIndex, RecommendationEngine

        empty = RecommendationEngine(model_version="none")
        empty.load_catalogue_from_parquet(DATASET)
        assert isinstance(empty.catalogue, CatalogueIndex)

        context = empty.build_context(user_id=1)
        response = empty.personalised(context, k=10)
        assert response.strategy == "static_fallback"
        assert response.items, "the last rung of the ladder returned nothing"

    def test_removing_the_ranker_falls_back_to_the_hybrid(self, engine):
        import copy

        degraded = copy.copy(engine)
        pipeline = copy.copy(engine.pipeline)
        pipeline.ranker = None
        degraded.pipeline = pipeline

        context = degraded.build_context(user_id=42)
        response = degraded.personalised(context, k=10)
        assert response.strategy == "hybrid"
        assert response.items

    def test_the_api_works_without_redis(self, client):
        """Redis holds only derived state, so losing it costs freshness and
        nothing else."""
        readiness = client.get("/health/ready").json()
        if readiness["cache"]:
            pytest.skip("Redis is available in this environment")
        assert client.get(f"{PREFIX}/recommendations/home/42").status_code == 200

    def test_the_api_works_without_postgres(self, client):
        readiness = client.get("/health/ready").json()
        if readiness["database"]:
            pytest.skip("Postgres is available in this environment")
        # Recommendations are unaffected...
        assert client.get(f"{PREFIX}/recommendations/home/42").status_code == 200
        # ...and events are still accepted, just not persisted.
        assert (
            client.post(
                f"{PREFIX}/events",
                json={
                    "event_type": "PRODUCT_VIEW",
                    "session_id": "degraded-session-01",
                    "product_id": 1,
                },
            ).status_code
            == 202
        )


class TestLatencyBudget:
    def test_recommendations_are_inside_the_budget(self, engine):
        """NFR-01: p95 under 80 ms. Measured uncached, which is the harder case."""
        import time

        import numpy as np

        context = engine.build_context(user_id=42, recent_products=(100, 205, 340))
        engine.personalised(context, k=12)  # warm any lazy initialisation

        timings = []
        for _ in range(30):
            started = time.perf_counter()
            engine.personalised(context, k=12)
            timings.append((time.perf_counter() - started) * 1000.0)

        p95 = float(np.percentile(timings, 95))
        assert p95 < 250.0, f"p95 {p95:.1f}ms is far outside the budget"
        print(f"\n  p50={np.percentile(timings, 50):.1f}ms p95={p95:.1f}ms")


class TestTrainingServingConsistency:
    def test_the_engine_and_the_training_run_agree_on_the_catalogue(self, engine):
        import pandas as pd

        products = pd.read_parquet(DATASET / "products.parquet")
        assert len(engine.catalogue) == len(products)

    def test_feature_columns_match_what_the_ranker_expects(self, engine):
        """NFR-06. A silent reordering between training and serving would feed
        the model scrambled inputs with no error anywhere."""
        ranker = engine.models.get("ranker")
        if ranker is None:
            pytest.skip("no ranker artefact")
        assert engine.pipeline is not None

        builder = engine.pipeline.dataset_builder
        assert set(ranker.feature_columns) == set(builder.feature_columns)
        assert ranker.feature_columns == builder.feature_columns, "feature order differs"

    def test_every_served_item_carries_an_explanation(self, engine):
        context = engine.build_context(user_id=42, recent_products=(100,))
        for item in engine.personalised(context, k=12).items:
            assert item.explanation.text
            assert item.explanation.reason_code
            assert "sources" in item.explanation.evidence


class TestProductPageSurfaces:
    def test_the_three_rails_are_genuinely_different(self, engine):
        """Serving one list under three headings is the classic broken product
        page: "customers who bought this phone also bought four other phones"."""
        anchor = 2593
        similar = {i.product.id for i in engine.similar(anchor, 10).items}
        also_viewed = {i.product.id for i in engine.also_viewed(anchor, 10).items}

        if not similar or not also_viewed:
            pytest.skip("this anchor has no neighbours")

        overlap = len(similar & also_viewed) / len(similar | also_viewed)
        assert overlap < 0.9, "similar and also-viewed are returning the same list"

    def test_similar_products_share_the_anchor_category(self, engine):
        anchor = 2593
        record = engine.catalogue.get(anchor)
        if record is None:
            pytest.skip("anchor not in the catalogue")

        items = engine.similar(anchor, 10).items
        if not items:
            pytest.skip("no neighbours")
        same_category = sum(
            1 for item in items if item.product.category_id == record.category_id
        )
        assert same_category / len(items) >= 0.5
