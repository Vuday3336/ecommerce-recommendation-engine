"""Admin ML dashboard endpoints (FR-15, FR-17, FR-18).

Everything here is role-gated to `analyst` or above. The data is read from the
artefact directory the engine is actually serving from, not from a separate
reporting copy - so the dashboard cannot show metrics for a model that is not
the one answering requests, which is a surprisingly common and very confusing
failure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.deps import get_engine, require_analyst
from app.core.config import settings
from app.core.security import Principal

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_analyst)],
)

AnalystDep = Annotated[Principal, Depends(require_analyst)]


def _artifact_dir() -> Path:
    directory = Path(settings.artifact_dir)
    if not directory.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no model artefacts at {directory}",
        )
    return directory


def _read_json(name: str) -> dict[str, Any]:
    path = _artifact_dir() / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("could not read %s", path)
        return {}


def _read_csv(name: str) -> list[dict[str, Any]]:
    path = _artifact_dir() / name
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
        return frame.where(pd.notna(frame), None).to_dict(orient="records")
    except (OSError, ValueError):
        logger.exception("could not read %s", path)
        return []


@router.get("/models", summary="Model performance from the last training run")
def model_performance() -> dict[str, Any]:
    """The offline comparison table plus segmented results.

    This is the evaluation the promotion decision was made on, served verbatim.
    Recomputing it here would risk showing numbers that differ from the ones the
    gate actually saw.
    """
    metrics = _read_json("metrics.json")
    return {
        "comparison": _read_csv("evaluation.csv"),
        "by_segment": _read_csv("evaluation_by_segment.csv"),
        "weight_calibration": _read_csv("weight_calibration.csv"),
        "stage1_recall": metrics.get("stage1_recall"),
        "source_contribution": metrics.get("source_contribution", {}),
        "feature_importance": metrics.get("feature_importance", [])[:20],
        "significance": metrics.get("significance", []),
        "split": metrics.get("split", {}),
        "training_seconds": metrics.get("timings_seconds", {}),
        "ranker_best_iteration": metrics.get("ranker_best_iteration"),
        "ranking_dataset": metrics.get("ranking_dataset", {}),
    }


@router.get("/monitoring", summary="Serving model and engine health")
def monitoring(request: Request) -> dict[str, Any]:
    engine = get_engine(request)
    sink = getattr(request.app.state, "event_sink", None)
    metrics = _read_json("metrics.json")
    config = _read_json("config.json")

    return {
        "model": {
            "name": settings.model_name,
            "version": engine.model_version,
            "loaded": engine.loaded,
            "models": sorted(engine.models),
            "has_ranker": "ranker" in engine.models,
            "catalogue_size": len(engine.catalogue),
            "users_with_features": len(engine.user_features),
        },
        "training": {
            "split": metrics.get("split", {}),
            "timings_seconds": metrics.get("timings_seconds", {}),
            "seed": config.get("seed"),
        },
        "event_sink": sink.stats() if sink else {},
        "cache_available": getattr(request.app.state, "redis", None) is not None,
        "database_available": getattr(request.app.state, "database_available", False),
    }


@router.get("/drift", summary="Feature drift against the training snapshot (FR-18)")
def drift(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict[str, Any]:
    """Drift for the monitored features.

    Computed on demand rather than read from a cache: this is an admin endpoint
    called a few times an hour, and a stale drift number is worse than a slow
    one.
    """
    import sys

    ml_root = Path(__file__).resolve().parents[5] / "ml"
    if str(ml_root) not in sys.path:
        sys.path.insert(0, str(ml_root))

    from recsys.drift.detector import (
        MONITORED_USER_FEATURES,
        DriftDetector,
    )

    reference_path = _artifact_dir() / "user_features.parquet"
    if not reference_path.exists():
        return {"features": [], "should_retrain": False, "reason": "no reference snapshot"}

    engine = get_engine(request)
    reference = pd.read_parquet(reference_path)
    current = engine.user_features if not engine.user_features.empty else reference

    detector = DriftDetector(reference, numeric_features=MONITORED_USER_FEATURES)
    report = detector.compare(current)
    detector.export_metrics(report)

    payload = report.to_dict()
    payload["features"] = payload["features"][:limit]
    return payload


@router.get("/distribution", summary="What the engine actually recommends")
def recommendation_distribution(
    request: Request,
    sample_users: Annotated[int, Query(ge=10, le=2000)] = 300,
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, Any]:
    """Most-recommended products and source mix, over a sample of users.

    Sampled live rather than read from the serving log, because in a fresh
    deployment the serving log is empty and the dashboard would show nothing.
    Once real impressions accumulate this should read `recommendations`
    instead - the sampled version answers "what would we recommend", the logged
    version answers "what did we recommend", and the second is the one that
    matters in production.
    """
    engine = get_engine(request)
    if engine.user_features.empty:
        return {"products": [], "sources": {}, "categories": [], "sampled_users": 0}

    user_ids = engine.user_features.index[:sample_users].tolist()
    product_counts: dict[int, int] = {}
    source_counts: dict[str, int] = {}
    category_counts: dict[int, int] = {}

    for user_id in user_ids:
        context = engine.build_context(user_id=int(user_id))
        response = engine.personalised(context, k=k)
        for item in response.items:
            product_counts[item.product.id] = product_counts.get(item.product.id, 0) + 1
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
            category_counts[item.product.category_id] = (
                category_counts.get(item.product.category_id, 0) + 1
            )

    top_products = sorted(product_counts.items(), key=lambda item: -item[1])[:25]
    top_categories = sorted(category_counts.items(), key=lambda item: -item[1])[:12]
    total_impressions = sum(product_counts.values()) or 1

    return {
        "sampled_users": len(user_ids),
        "distinct_products": len(product_counts),
        "catalogue_size": len(engine.catalogue),
        "coverage": round(len(product_counts) / max(len(engine.catalogue), 1), 4),
        "products": [
            {
                "product_id": pid,
                "name": (p.name if (p := engine.catalogue.get(pid)) else str(pid)),
                "count": count,
                "share": round(count / total_impressions, 5),
            }
            for pid, count in top_products
        ],
        "sources": dict(sorted(source_counts.items(), key=lambda item: -item[1])),
        "categories": [
            {
                "category_id": cid,
                "name": engine.catalogue.category_names.get(cid, str(cid)),
                "count": count,
            }
            for cid, count in top_categories
        ],
    }


@router.get("/analytics", summary="Recommendation business metrics (FR-15)")
def analytics(
    request: Request,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> dict[str, Any]:
    """CTR, conversion and attributed revenue, sliced by surface and source.

    Reads the serving log joined to its outcome tables. Returns
    `available: false` when the database is unreachable rather than zeros -
    "no data yet" and "nothing is converting" are very different claims.
    """
    service = getattr(request.app.state, "analytics", None)
    if service is None:
        return {"available": False, "reason": "analytics service not configured"}
    return service.overview(days=days)


@router.get("/analytics/products", summary="Most recommended and converted products")
def analytics_products(
    request: Request,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
) -> dict[str, Any]:
    service = getattr(request.app.state, "analytics", None)
    if service is None:
        return {"available": False, "reason": "analytics service not configured"}
    return service.top_products(days=days, limit=limit)


@router.get(
    "/experiments/{experiment_key}/results",
    summary="A/B test readout with significance and guardrails (FR-16)",
)
def experiment_results(
    request: Request,
    experiment_key: str,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> dict[str, Any]:
    """Control versus treatment for one experiment.

    The sample-ratio check runs before the significance tests and can mark the
    readout not-ready on its own: if traffic did not split as intended, every
    downstream number is invalid, and a broken experiment tends to produce a
    convincing-looking win.
    """
    service = getattr(request.app.state, "analytics", None)
    if service is None:
        return {"available": False, "reason": "analytics service not configured"}

    experiments = getattr(request.app.state, "experiments", None)
    allocation = None
    minimum = 1000
    if experiments is not None:
        definition = next(
            (e for e in experiments.active if e.key == experiment_key), None
        )
        if definition is not None:
            allocation = {v.name: v.allocation for v in definition.variants}

    readout = service.experiment(
        experiment_key,
        days=days,
        minimum_sample_size=minimum,
        expected_allocation=allocation,
    )
    return readout.to_dict()


@router.get("/experiments", summary="Running experiments and their allocations")
def experiments(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "experiments", None)
    if service is None:
        return {"experiments": []}
    return {
        "experiments": [
            {
                "key": experiment.key,
                "status": experiment.status.value,
                "traffic_allocation": experiment.traffic_allocation,
                "variants": [
                    {"name": v.name, "allocation": v.allocation, "config": v.config}
                    for v in experiment.variants
                ],
            }
            for experiment in service.active
        ]
    }


__all__ = ["router"]
