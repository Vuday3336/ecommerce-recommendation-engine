"""Prometheus metrics (FR-17).

The metric set is chosen so the dashboards can answer specific operational
questions rather than to instrument everything that moves:

* Is the API healthy? → request duration, request count, error rate
* Is the recommender fast enough? → per-surface and per-stage latency histograms
* Is the cache working? → hit and miss counters
* **Is the recommender actually working?** → `recommendation_fallback_total`

That last one is the metric this project would page on. Every other signal can
look healthy while the ranker quietly fails and every request is served by the
static fallback: latency *improves*, errors stay at zero, and the only visible
symptom is a slow decline in click-through a week later. Fallback rate turns
that into an immediate alert.

`recommendation_score` exists for the same reason from the model's side: a
shift in the score distribution is the earliest observable sign that inputs
have drifted, well before any business metric moves.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.core import REGISTRY

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: Buckets chosen around the NFR-01 budget (p95 < 80 ms, p99 < 150 ms) rather
#: than Prometheus defaults. Default buckets jump 0.1 -> 0.25 -> 0.5, which
#: puts no boundary anywhere near the numbers we actually commit to, so the
#: p95 estimate would be interpolated across a bucket four times too wide.
LATENCY_BUCKETS = (
    0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100,
    0.150, 0.250, 0.500, 1.000, 2.500,
)

# --- HTTP -------------------------------------------------------------------

http_requests_total = Counter(
    "http_requests_total",
    "HTTP requests handled",
    ["method", "route", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "route"],
    buckets=LATENCY_BUCKETS,
)

# --- Recommendations --------------------------------------------------------

recommendation_latency_seconds = Histogram(
    "recommendation_latency_seconds",
    "End-to-end recommendation latency",
    ["surface"],
    buckets=LATENCY_BUCKETS,
)

recommendation_stage_latency_seconds = Histogram(
    "recommendation_stage_latency_seconds",
    "Latency of one recommendation stage",
    ["stage"],
    buckets=LATENCY_BUCKETS,
)

recommendation_requests_total = Counter(
    "recommendation_requests_total",
    "Recommendation requests served",
    ["surface", "strategy"],
)

recommendation_fallback_total = Counter(
    "recommendation_fallback_total",
    "Recommendations served by a degraded strategy",
    ["surface", "strategy"],
)

recommendation_empty_total = Counter(
    "recommendation_empty_total",
    "Recommendation responses that contained no items",
    ["surface"],
)

recommendation_candidates = Histogram(
    "recommendation_candidate_pool_size",
    "Candidates surviving stage 1",
    buckets=(0, 10, 50, 100, 200, 300, 500, 1000),
)

recommendation_candidates_by_source = Counter(
    "recommendation_candidates_by_source_total",
    "Candidates contributed per source",
    ["source"],
)

recommendation_score = Histogram(
    "recommendation_score",
    "Distribution of served recommendation scores",
    ["surface"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# --- Cache ------------------------------------------------------------------

cache_hits_total = Counter("recommendation_cache_hits_total", "Cache hits", ["surface"])
cache_misses_total = Counter(
    "recommendation_cache_misses_total", "Cache misses", ["surface"]
)
cache_errors_total = Counter("recommendation_cache_errors_total", "Cache errors")

# --- Events -----------------------------------------------------------------

events_ingested_total = Counter(
    "events_ingested_total", "Events accepted", ["event_type"]
)
events_dropped_total = Counter(
    "events_dropped_total", "Events dropped because the ingest buffer was full"
)
event_ingest_duration_seconds = Histogram(
    "event_ingest_duration_seconds",
    "Time to accept an event batch at the API boundary",
    buckets=(0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.250),
)

# --- Model and drift --------------------------------------------------------

model_info = Gauge(
    "model_info",
    "Currently-served model, as labels with a constant value of 1",
    ["model_name", "model_version", "stage"],
)

model_loaded = Gauge("model_loaded", "1 when the engine has usable artefacts")

drift_score = Gauge(
    "feature_drift_score", "Drift statistic per monitored feature", ["feature", "method"]
)

drift_alerts_total = Counter(
    "feature_drift_alerts_total", "Features that crossed the drift alert threshold", ["feature"]
)

# --- Business ---------------------------------------------------------------

recommendation_impressions_total = Counter(
    "recommendation_impressions_total", "Recommendation impressions", ["surface"]
)
recommendation_clicks_total = Counter(
    "recommendation_clicks_total", "Recommendation clicks", ["surface"]
)
recommendation_conversions_total = Counter(
    "recommendation_conversions_total", "Recommendation conversions", ["surface"]
)
recommendation_revenue_total = Counter(
    "recommendation_revenue_total", "Revenue attributed to recommendations", ["surface"]
)

experiment_exposures_total = Counter(
    "experiment_exposures_total", "Users exposed to an experiment arm", ["experiment", "variant"]
)


def set_model_info(name: str, version: str, stage: str = "production") -> None:
    """Record the serving model.

    A gauge with labels and a constant value of 1 is the standard way to expose
    build metadata: it lets a dashboard join any metric against the model
    version that produced it, so a regression can be attributed to a specific
    promotion rather than guessed at.
    """
    model_info.labels(model_name=name, model_version=version, stage=stage).set(1)


def render(registry: CollectorRegistry | None = None) -> bytes:
    """Render the exposition format for the /metrics endpoint."""
    return generate_latest(registry or REGISTRY)


def multiprocess_registry() -> CollectorRegistry:
    """Registry that aggregates across uvicorn workers.

    With several worker processes, each holds its own counters and a scrape
    hits one at random - so every rate is understated by the worker count and
    jumps around. This requires `PROMETHEUS_MULTIPROC_DIR` to be set.
    """
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


__all__ = [
    "CONTENT_TYPE",
    "LATENCY_BUCKETS",
    "cache_errors_total",
    "cache_hits_total",
    "cache_misses_total",
    "drift_alerts_total",
    "drift_score",
    "event_ingest_duration_seconds",
    "events_dropped_total",
    "events_ingested_total",
    "experiment_exposures_total",
    "http_request_duration_seconds",
    "http_requests_total",
    "model_info",
    "model_loaded",
    "multiprocess_registry",
    "recommendation_candidates",
    "recommendation_candidates_by_source",
    "recommendation_clicks_total",
    "recommendation_conversions_total",
    "recommendation_empty_total",
    "recommendation_fallback_total",
    "recommendation_impressions_total",
    "recommendation_latency_seconds",
    "recommendation_requests_total",
    "recommendation_revenue_total",
    "recommendation_score",
    "recommendation_stage_latency_seconds",
    "render",
    "set_model_info",
]
