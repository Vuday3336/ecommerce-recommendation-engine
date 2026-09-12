"""FastAPI application factory and lifespan.

Everything expensive - loading model artefacts, building the candidate
generator, connecting to Redis, starting the event sink worker - happens once
in the lifespan handler. Doing any of it per request would load joblib files
from disk on every call.

The application starts even when its dependencies do not. A missing artefact
directory, an unreachable Redis or a down database each degrade one capability
and are reported by `/health/ready`; none of them prevent the process from
serving. That is what makes the degradation ladder real rather than
theoretical: it has to hold at start-up, not only at request time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routers import admin as admin_router
from app.api.v1.routers import auth as auth_router
from app.api.v1.routers import events as events_router
from app.api.v1.routers import recommendations as recommendations_router
from app.cache.client import get_redis, reset_redis
from app.cache.counters import RealTimeCounters
from app.core import metrics
from app.core.config import settings
from app.core.logging import configure_logging, request_id_var
from app.services.analytics import AnalyticsService
from app.services.event_sink import (
    BufferedEventSink,
    MemoryEventSink,
    PostgresEventSink,
)
from app.services.events import EventService
from app.services.experiments import ExperimentDefinition, ExperimentService, Variant
from app.services.recommendations import RecommendationService

logger = logging.getLogger(__name__)

#: The default experiment, so the A/B machinery is exercised from the first
#: request rather than lying dormant until someone configures one. Control is
#: the hybrid blend; treatment adds the learned ranker (ADR-005 / Phase 15).
DEFAULT_EXPERIMENTS = [
    ExperimentDefinition(
        key="ranker_v1",
        variants=(
            Variant("control", 0.5, {"enable_ml_ranker": False}),
            Variant("treatment", 0.5, {"enable_ml_ranker": True}),
        ),
    )
]


def _build_engine():
    """Load the recommendation engine, tolerating missing artefacts."""
    import sys

    ml_root = Path(__file__).resolve().parents[2] / "ml"
    if str(ml_root) not in sys.path:
        sys.path.insert(0, str(ml_root))

    from recsys.inference.engine import RecommendationEngine

    engine = RecommendationEngine(model_version=settings.model_version)
    try:
        engine.load(settings.artifact_dir, settings.dataset_dir)
    except Exception:
        logger.exception("failed to load model artefacts; serving fallbacks only")
    return engine


def _build_sink():
    """Postgres sink when the database is reachable, in-memory otherwise."""
    try:
        from app.db.session import check_connection, engine

        if check_connection():
            return BufferedEventSink(PostgresEventSink(engine)), True
    except Exception:
        logger.exception("could not build the Postgres event sink")

    logger.warning(
        "database unavailable: events are buffered in memory and will be lost on "
        "restart. Recommendations still work; ingestion does not persist."
    )
    return BufferedEventSink(MemoryEventSink()), False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info("starting %s (%s)", settings.app_name, settings.app_env)
    settings.require_secrets()

    engine = _build_engine()
    app.state.engine = engine
    metrics.model_loaded.set(1 if engine.loaded else 0)
    metrics.set_model_info(settings.model_name, engine.model_version)

    redis = get_redis()
    app.state.redis = redis
    counters = RealTimeCounters(redis)
    app.state.counters = counters

    sink, database_available = _build_sink()
    app.state.event_sink = sink
    app.state.database_available = database_available

    category_lookup = {
        product_id: record.category_id
        for product_id, record in engine.catalogue.products.items()
    }
    app.state.event_service = EventService(
        sink, counters, category_lookup=category_lookup
    )

    # Analytics reads the serving log. Without a database it reports
    # "unavailable" rather than zeros - a dashboard of zeros reads as "nothing
    # is converting", which is a very different and much more alarming claim
    # than "we have no data yet".
    if database_available:
        from app.db.session import SessionLocal

        app.state.analytics = AnalyticsService(SessionLocal)
    else:
        app.state.analytics = AnalyticsService(None)

    experiments = ExperimentService(DEFAULT_EXPERIMENTS)
    app.state.experiments = experiments
    app.state.recommendation_service = RecommendationService(
        engine, redis=redis, counters=counters, experiments=experiments
    )

    logger.info("startup complete: %s", engine.health())
    try:
        yield
    finally:
        logger.info("shutting down")
        # Flush before exit so buffered events are not silently discarded.
        sink.close()
        reset_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="E-commerce Recommendation & Personalization Engine",
        version="0.1.0",
        description=(
            "Two-stage recommendation platform: candidate generation, learned "
            "ranking, explanations, experiments and monitoring."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Session-Id"],
        max_age=600,
    )

    @app.middleware("http")
    async def observability(request: Request, call_next):
        """Request id propagation plus latency and status metrics."""
        import uuid

        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        # The *route template* is the label, not the raw path. Labelling by
        # path would create a distinct time series per product id and blow up
        # Prometheus cardinality.
        route = request.scope.get("route")
        route_label = getattr(route, "path", request.url.path)

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            logger.exception("unhandled error on %s %s", request.method, route_label)
            response = JSONResponse(
                status_code=500,
                content={"detail": "internal server error", "request_id": request_id},
            )
        finally:
            duration = time.perf_counter() - started
            request_id_var.reset(token)

        route_label = getattr(request.scope.get("route"), "path", route_label)
        metrics.http_request_duration_seconds.labels(
            method=request.method, route=route_label
        ).observe(duration)
        metrics.http_requests_total.labels(
            method=request.method, route=route_label, status=str(status_code)
        ).inc()
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.1f}"
        return response

    prefix = settings.api_v1_prefix
    app.include_router(recommendations_router.router, prefix=prefix)
    app.include_router(events_router.router, prefix=prefix)
    app.include_router(events_router.feedback_router, prefix=prefix)
    app.include_router(admin_router.router, prefix=prefix)
    app.include_router(auth_router.router, prefix=prefix)

    @app.get("/health", tags=["health"], summary="Liveness")
    def health() -> dict[str, str]:
        """Liveness only: is the process up?

        Deliberately does not check dependencies. A liveness probe that fails
        when Redis is down would have the orchestrator restart a perfectly
        healthy process, turning a cache outage into an outage.
        """
        return {"status": "ok", "service": settings.app_name}

    @app.get("/health/ready", tags=["health"], summary="Readiness")
    def readiness(request: Request) -> dict[str, object]:
        """Readiness: can this instance serve useful traffic?"""
        engine = getattr(request.app.state, "engine", None)
        redis = getattr(request.app.state, "redis", None)
        return {
            "status": "ready" if engine is not None else "degraded",
            "engine": engine.health() if engine else {"loaded": False},
            "cache": redis is not None,
            "database": getattr(request.app.state, "database_available", False),
        }

    @app.get(f"{prefix}/recommendations/health", tags=["health"])
    def recommendation_health(request: Request) -> dict[str, object]:
        """Engine health, as required by the brief's endpoint list."""
        engine = getattr(request.app.state, "engine", None)
        sink = getattr(request.app.state, "event_sink", None)
        return {
            "status": "ok" if engine and engine.loaded else "degraded",
            "engine": engine.health() if engine else {"loaded": False},
            "event_sink": sink.stats() if sink else {},
        }

    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

    return app


app = create_app()
