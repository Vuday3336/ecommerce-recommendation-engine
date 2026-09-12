"""API tests against the real engine.

These check the contract the frontend depends on and the guarantees the
architecture claims - particularly that a recommendation surface never fails,
which is the one promise the degradation ladder exists to keep.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

PREFIX = "/api/v1"


class TestHealth:
    def test_liveness_does_not_depend_on_anything(self, client):
        """A liveness probe that fails when Redis is down would have the
        orchestrator restart a healthy process, turning a cache outage into an
        application outage."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_reports_each_dependency_separately(self, client):
        payload = client.get("/health/ready").json()
        assert "engine" in payload
        assert "cache" in payload
        assert "database" in payload

    def test_recommendation_health_exposes_the_engine(self, client):
        payload = client.get(f"{PREFIX}/recommendations/health").json()
        assert "engine" in payload
        assert "event_sink" in payload

    def test_metrics_endpoint_serves_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "http_requests_total" in response.text


class TestRecommendationSurfaces:
    @pytest.mark.parametrize(
        "path",
        [
            f"{PREFIX}/recommendations/home/42",
            f"{PREFIX}/recommendations/similar/100",
            f"{PREFIX}/recommendations/frequently-bought/100",
            f"{PREFIX}/recommendations/also-viewed/100",
            f"{PREFIX}/recommendations/trending",
            f"{PREFIX}/recommendations/recent/42",
        ],
    )
    def test_every_surface_returns_200(self, client, path):
        """NFR-04: a recommendation surface must never 5xx.

        An empty list is an acceptable answer; an error is not.
        """
        assert client.get(path).status_code == 200

    def test_home_returns_all_declared_sections(self, client, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")

        payload = client.get(f"{PREFIX}/recommendations/home/42?limit=6").json()
        for key in (
            "for_you",
            "because_you_viewed",
            "trending",
            "frequently_bought_together",
            "continue_shopping",
        ):
            assert key in payload, f"missing section {key}"
        assert payload["user_id"] == 42
        assert payload["model_version"]

    def test_items_carry_everything_the_brief_requires(self, client, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")

        items = client.get(f"{PREFIX}/recommendations/home/42?limit=5").json()["for_you"]
        assert items, "the for-you rail is empty"
        for item in items:
            assert item["product"]["id"] > 0
            assert item["product"]["name"]
            assert item["score"] >= 0
            assert item["recommendation_type"]
            assert item["explanation"], "every recommendation needs a reason (FR-11)"
            # The join key that makes click attribution possible at all.
            assert item["recommendation_id"]

    def test_the_response_reports_which_rung_served_it(self, client, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")
        section = client.get(f"{PREFIX}/recommendations/home/42").json()["sections"]["for_you"]
        assert section["strategy"] in {
            "ranker", "hybrid", "collaborative_content", "content_trending",
            "category_popular", "global_trending", "static_fallback",
        }

    def test_two_users_get_different_recommendations(self, client, artifacts_available):
        """If this fails, the system is not personalising - whatever the
        offline metrics say."""
        if not artifacts_available:
            pytest.skip("no trained artefacts")

        first = client.get(f"{PREFIX}/recommendations/home/42?limit=12").json()["for_you"]
        second = client.get(f"{PREFIX}/recommendations/home/7?limit=12").json()["for_you"]
        if not first or not second:
            pytest.skip("one of the users has no recommendations")

        ids_a = {item["product"]["id"] for item in first}
        ids_b = {item["product"]["id"] for item in second}
        overlap = len(ids_a & ids_b) / max(len(ids_a | ids_b), 1)
        assert overlap < 0.8, f"users see near-identical lists (overlap {overlap:.0%})"

    def test_similar_products_are_related_to_the_anchor(self, client, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")
        items = client.get(f"{PREFIX}/recommendations/similar/100").json()["items"]
        if not items:
            pytest.skip("no similar products for this anchor")
        # Similarity is content-based, so the neighbours should mostly share a
        # category. A rail of unrelated products means the embedding is broken.
        categories = [item["product"]["category_id"] for item in items]
        dominant = max(set(categories), key=categories.count)
        assert categories.count(dominant) / len(categories) >= 0.5

    def test_anonymous_users_still_get_recommendations(self, client):
        """Cold start (FR-09): a first-time visitor must not get an error."""
        response = client.get(
            f"{PREFIX}/recommendations/home", headers={"X-Session-Id": "brand-new-session-1"}
        )
        assert response.status_code == 200
        assert response.json()["user_id"] is None

    def test_limits_are_validated(self, client):
        assert client.get(f"{PREFIX}/recommendations/trending?limit=0").status_code == 422
        assert client.get(f"{PREFIX}/recommendations/trending?limit=999").status_code == 422
        assert client.get(f"{PREFIX}/recommendations/similar/-5").status_code == 422

    def test_explanation_answers_both_directions(self, client, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")
        payload = client.get(f"{PREFIX}/recommendations/explain/42/100").json()
        assert "would_recommend" in payload
        # The negative case must explain itself rather than returning nothing -
        # "why is this *not* recommended" is the more useful debugging answer.
        if not payload["would_recommend"]:
            assert payload["reason"]


class TestEventIngestion:
    def test_a_valid_event_is_accepted(self, client):
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_VIEW",
                "session_id": "test-session-0001",
                "product_id": 100,
                "source": "pdp",
                "device_type": "desktop",
                "metadata": {"dwell_ms": 4200},
            },
        )
        # 202, not 201: accepted for processing, not yet persisted.
        assert response.status_code == 202
        assert response.json()["accepted"] == 1

    def test_product_scoped_events_require_a_product(self, client):
        """Mirrors the database CHECK constraint, caught at the edge so the
        client gets a 422 explaining the problem instead of a 500."""
        response = client.post(
            f"{PREFIX}/events",
            json={"event_type": "PURCHASE", "session_id": "test-session-0002"},
        )
        assert response.status_code == 422
        assert "product_id" in str(response.json())

    def test_unscoped_events_reject_a_product(self, client):
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "SESSION_START",
                "session_id": "test-session-0003",
                "product_id": 100,
            },
        )
        assert response.status_code == 422

    def test_unknown_event_types_are_rejected(self, client):
        response = client.post(
            f"{PREFIX}/events",
            json={"event_type": "TELEPORT", "session_id": "test-session-0004"},
        )
        assert response.status_code == 422

    def test_payloads_are_validated_per_event_type(self, client):
        """FR-21's other half: JSONB metadata is extensible but not unchecked."""
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_RATING",
                "session_id": "test-session-0005",
                "product_id": 100,
                "metadata": {"rating": 47},
            },
        )
        assert response.status_code == 422

    def test_unknown_metadata_keys_are_rejected(self, client):
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_VIEW",
                "session_id": "test-session-0006",
                "product_id": 100,
                "metadata": {"totally_made_up": 1},
            },
        )
        assert response.status_code == 422

    def test_batches_are_accepted(self, client):
        events = [
            {
                "event_type": "PRODUCT_VIEW",
                "session_id": "test-session-batch",
                "product_id": 100 + index,
            }
            for index in range(10)
        ]
        response = client.post(f"{PREFIX}/events/batch", json={"events": events})
        assert response.status_code == 202
        assert response.json()["accepted"] == 10

    def test_oversized_batches_are_rejected(self, client):
        events = [
            {
                "event_type": "PRODUCT_VIEW",
                "session_id": "test-session-big",
                "product_id": 1,
            }
        ] * 500
        assert client.post(f"{PREFIX}/events/batch", json={"events": events}).status_code == 422

    def test_backdating_beyond_the_window_is_rejected(self, client):
        """Unbounded backdating would let a client write into a partition that
        has already been used for training."""
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_VIEW",
                "session_id": "test-session-0007",
                "product_id": 100,
                "occurred_at": "2020-01-01T00:00:00+00:00",
            },
        )
        assert response.status_code == 422

    def test_naive_timestamps_are_rejected(self, client):
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_VIEW",
                "session_id": "test-session-0008",
                "product_id": 100,
                "occurred_at": "2026-09-01T00:00:00",
            },
        )
        assert response.status_code == 422


class TestAuthorisation:
    ADMIN_PATHS: ClassVar[list[str]] = [
        f"{PREFIX}/admin/models",
        f"{PREFIX}/admin/monitoring",
        f"{PREFIX}/admin/drift",
        f"{PREFIX}/admin/experiments",
        f"{PREFIX}/admin/analytics",
    ]

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_admin_endpoints_reject_anonymous_callers(self, client, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_admin_endpoints_reject_customers(self, client, path, customer_token):
        response = client.get(path, headers={"Authorization": f"Bearer {customer_token}"})
        assert response.status_code == 403

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_analysts_are_allowed(self, client, path, analyst_token):
        response = client.get(path, headers={"Authorization": f"Bearer {analyst_token}"})
        assert response.status_code == 200

    def test_a_malformed_token_degrades_to_anonymous_on_public_routes(self, client):
        """A stale token in a browser tab should mean public browsing, not a
        broken storefront."""
        response = client.get(
            f"{PREFIX}/recommendations/trending",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 200

    def test_a_refresh_token_is_not_accepted_as_an_access_token(self, client):
        from app.core.security import create_token
        from app.models.enums import UserRole

        refresh = create_token(
            user_id=1, public_id="x", role=UserRole.ADMIN, token_type="refresh"
        )
        response = client.get(
            f"{PREFIX}/admin/models", headers={"Authorization": f"Bearer {refresh}"}
        )
        # Without the `type` check, a 14-day refresh token would silently
        # become the real session length.
        assert response.status_code == 401


class TestAdminData:
    def test_model_comparison_is_served(self, client, auth, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")
        payload = client.get(f"{PREFIX}/admin/models", headers=auth).json()
        assert len(payload["comparison"]) > 5
        assert payload["stage1_recall"] is not None
        assert payload["significance"], "the comparison must report significance"

    def test_drift_is_computed(self, client, auth, artifacts_available):
        if not artifacts_available:
            pytest.skip("no trained artefacts")
        payload = client.get(f"{PREFIX}/admin/drift", headers=auth).json()
        assert "features" in payload
        assert "should_retrain" in payload

    def test_analytics_declares_itself_unavailable_without_a_database(self, client, auth):
        """Zeros would read as "nothing is converting", which is a very
        different claim from "we have no data yet"."""
        payload = client.get(f"{PREFIX}/admin/analytics", headers=auth).json()
        assert "available" in payload
        if not payload["available"]:
            assert payload["reason"]

    def test_experiment_results_report_readiness(self, client, auth):
        payload = client.get(
            f"{PREFIX}/admin/experiments/ranker_v1/results", headers=auth
        ).json()
        assert "ready" in payload
        assert "sample_ratio_mismatch" in payload


class TestResponseHeaders:
    def test_every_response_carries_a_request_id(self, client):
        response = client.get("/health")
        assert response.headers["X-Request-Id"]
        assert response.headers["X-Response-Time-Ms"]

    def test_a_supplied_request_id_is_propagated(self, client):
        response = client.get("/health", headers={"X-Request-Id": "trace-me-123"})
        assert response.headers["X-Request-Id"] == "trace-me-123"
