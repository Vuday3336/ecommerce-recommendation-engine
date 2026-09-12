"""Recommendation endpoints (FR-01 to FR-11, FR-14).

Every endpoint here is on the shopper hot path and is bound by NFR-01
(p95 < 80 ms cached). Two consequences show up in the code: nothing does a
database write inline, and nothing raises on a missing model - the degradation
ladder returns a worse answer rather than an error, and reports which rung
served it.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from app.api.deps import (
    get_principal,
    get_recommendation_service,
    get_session_key,
    rate_limit_recommendations,
)
from app.core.config import settings
from app.core.security import Principal
from app.schemas.recommendations import (
    ExplanationResponse,
    HomeRecommendations,
    ProductSummary,
    RecommendationItem,
    RecommendationSection,
)
from app.services.recommendations import RecommendationService, ServedSection

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    dependencies=[Depends(rate_limit_recommendations)],
)

ServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]
PrincipalDep = Annotated[Principal | None, Depends(get_principal)]
SessionDep = Annotated[str, Depends(get_session_key)]


def _to_items(section: ServedSection) -> list[RecommendationItem]:
    return [
        RecommendationItem(
            product=ProductSummary(
                id=item.product_id,
                name=item.name,
                price=item.price,
                category_id=item.category_id,
                brand_id=item.brand_id,
                rating_average=item.rating_average,
            ),
            score=max(item.score, 0.0),
            recommendation_type=item.source,
            explanation=item.explanation,
            explanation_evidence=item.explanation_evidence,
            score_components=item.score_components,
            position=item.position,
            # The id the client echoes back on impression and click. Built from
            # the request id and position so it is unique per served slot
            # without a database round trip on the hot path.
            recommendation_id=f"{section.request_id}:{item.position}",
        )
        for item in section.items
    ]


def _to_section(section: ServedSection) -> RecommendationSection:
    return RecommendationSection(
        section=section.section,
        surface=section.surface,
        items=_to_items(section),
        strategy=section.strategy,
        model_version=section.model_version,
        request_id=section.request_id,
        variant=section.variant,
        experiment_key=section.experiment_key,
        candidate_pool_size=section.candidate_pool_size,
        cache_hit=section.cache_hit,
        latency_ms=round(section.latency_ms, 2),
    )


@router.get(
    "/home/{user_id}",
    response_model=HomeRecommendations,
    summary="Personalised homepage (FR-07)",
)
def home(
    service: ServiceDep,
    session_key: SessionDep,
    user_id: Annotated[int, Path(gt=0)],
    limit: Annotated[int, Query(ge=1, le=50)] = settings.recommendations_per_section,
) -> HomeRecommendations:
    """All homepage rails in one call.

    One call rather than five because the rails share a context and a candidate
    pass; five calls would recompute the most expensive part of the request
    each time.
    """
    sections = service.home(user_id=user_id, session_key=session_key, limit=limit)
    rendered = {name: _to_section(section) for name, section in sections.items()}
    request_id = next(iter(rendered.values())).request_id if rendered else ""

    return HomeRecommendations(
        user_id=user_id,
        for_you=rendered["for_you"].items if "for_you" in rendered else [],
        because_you_viewed=(
            rendered["because_you_viewed"].items if "because_you_viewed" in rendered else []
        ),
        trending=rendered["trending"].items if "trending" in rendered else [],
        frequently_bought_together=(
            rendered["frequently_bought_together"].items
            if "frequently_bought_together" in rendered
            else []
        ),
        continue_shopping=(
            rendered["continue_shopping"].items if "continue_shopping" in rendered else []
        ),
        sections=rendered,
        model_version=(
            next(iter(rendered.values())).model_version if rendered else "unversioned"
        ),
        request_id=request_id,
    )


@router.get(
    "/home",
    response_model=HomeRecommendations,
    summary="Personalised homepage for the current session (anonymous allowed)",
)
def home_for_session(
    service: ServiceDep,
    session_key: SessionDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=50)] = settings.recommendations_per_section,
) -> HomeRecommendations:
    """Homepage without a user id in the path.

    This is the endpoint a real storefront calls: the user is resolved from the
    token when present, and anonymous visitors get session-based cold-start
    recommendations rather than an error (FR-09).
    """
    user_id = principal.user_id if principal else None
    sections = service.home(user_id=user_id, session_key=session_key, limit=limit)
    rendered = {name: _to_section(section) for name, section in sections.items()}
    request_id = next(iter(rendered.values())).request_id if rendered else ""

    return HomeRecommendations(
        user_id=user_id,
        for_you=rendered["for_you"].items if "for_you" in rendered else [],
        because_you_viewed=(
            rendered["because_you_viewed"].items if "because_you_viewed" in rendered else []
        ),
        trending=rendered["trending"].items if "trending" in rendered else [],
        frequently_bought_together=(
            rendered["frequently_bought_together"].items
            if "frequently_bought_together" in rendered
            else []
        ),
        continue_shopping=(
            rendered["continue_shopping"].items if "continue_shopping" in rendered else []
        ),
        sections=rendered,
        model_version=(
            next(iter(rendered.values())).model_version if rendered else "unversioned"
        ),
        request_id=request_id,
    )


@router.get(
    "/similar/{product_id}",
    response_model=RecommendationSection,
    summary="Similar products (FR-02)",
)
def similar(
    service: ServiceDep,
    product_id: Annotated[int, Path(gt=0)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RecommendationSection:
    return _to_section(service.similar(product_id, limit))


@router.get(
    "/frequently-bought/{product_id}",
    response_model=RecommendationSection,
    summary="Frequently bought together (FR-03)",
)
def frequently_bought(
    service: ServiceDep,
    product_id: Annotated[int, Path(gt=0)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> RecommendationSection:
    return _to_section(service.frequently_bought(product_id, limit))


@router.get(
    "/also-viewed/{product_id}",
    response_model=RecommendationSection,
    summary="Customers who viewed this also viewed (FR-04)",
)
def also_viewed(
    service: ServiceDep,
    product_id: Annotated[int, Path(gt=0)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RecommendationSection:
    return _to_section(service.also_viewed(product_id, limit))


@router.get(
    "/trending",
    response_model=RecommendationSection,
    summary="Trending products (FR-05)",
)
def trending(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
    category_id: Annotated[int | None, Query(gt=0)] = None,
) -> RecommendationSection:
    return _to_section(service.trending(limit, category_id=category_id))


@router.get(
    "/recent/{user_id}",
    response_model=RecommendationSection,
    summary="Recently viewed (FR-06)",
)
def recently_viewed(
    service: ServiceDep,
    user_id: Annotated[int, Path(gt=0)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> RecommendationSection:
    return _to_section(service.recently_viewed(user_id, limit))


@router.get(
    "/explain/{user_id}/{product_id}",
    response_model=ExplanationResponse,
    summary="Why this product for this user (FR-11)",
)
def explain(
    service: ServiceDep,
    user_id: Annotated[int, Path(gt=0)],
    product_id: Annotated[int, Path(gt=0)],
) -> ExplanationResponse:
    """Full reasoning for one (user, product) pair.

    Deliberately answers "would we recommend this, and why or why not" rather
    than only explaining items already shown. The negative case - a product the
    user will never see - is usually the more useful debugging answer.
    """
    return ExplanationResponse(**service.explain(user_id, product_id))


__all__ = ["router"]
