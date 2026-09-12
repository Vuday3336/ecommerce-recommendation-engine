"""Recommendation API response schemas.

Every item carries the four things the brief requires - product, score,
recommendation type and explanation - plus the `recommendation_id` the frontend
must echo back on click. Without that id, a click cannot be joined to the
serving decision that produced it, and attribution becomes guesswork.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class ProductSummary(BaseModel):
    """The catalogue fields a rail needs to render."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    price: float
    category_id: int
    brand_id: int
    rating_average: float = 0.0
    image_url: str | None = None


class RecommendationItem(BaseModel):
    """One recommended product."""

    product: ProductSummary
    score: Annotated[float, Field(ge=0.0)]
    recommendation_type: str = Field(
        description="Candidate source that produced this item"
    )
    explanation: str
    explanation_evidence: dict[str, Any] = Field(default_factory=dict)
    score_components: dict[str, float] = Field(default_factory=dict)
    position: int
    #: Echoed back on impression and click so outcomes join to this decision.
    recommendation_id: str | None = None


class RecommendationSection(BaseModel):
    """One rail."""

    section: str
    surface: str
    items: list[RecommendationItem]
    strategy: str = Field(
        description="Degradation-ladder rung that served this section"
    )
    model_version: str
    request_id: str
    variant: str | None = None
    experiment_key: str | None = None
    candidate_pool_size: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0


class HomeRecommendations(BaseModel):
    """The multi-section homepage payload (FR-07)."""

    user_id: int | None
    for_you: list[RecommendationItem] = Field(default_factory=list)
    because_you_viewed: list[RecommendationItem] = Field(default_factory=list)
    trending: list[RecommendationItem] = Field(default_factory=list)
    frequently_bought_together: list[RecommendationItem] = Field(default_factory=list)
    continue_shopping: list[RecommendationItem] = Field(default_factory=list)
    sections: dict[str, RecommendationSection] = Field(
        default_factory=dict,
        description="Per-section metadata: strategy, model version, timings",
    )
    model_version: str = "unversioned"
    request_id: str


class ExplanationResponse(BaseModel):
    """Why a specific product would (or would not) be recommended."""

    product_id: int
    would_recommend: bool
    position: int | None = None
    score: float | None = None
    explanation: dict[str, Any] | None = None
    score_components: dict[str, float] = Field(default_factory=dict)
    strategy: str | None = None
    model_version: str
    reason: str | None = None


class FeedbackIn(BaseModel):
    """Impression, click or conversion reported by the client (FR-12)."""

    model_config = ConfigDict(extra="forbid")

    recommendation_id: Annotated[str, Field(min_length=8, max_length=64)]
    product_id: Annotated[int, Field(gt=0)]
    session_id: Annotated[str, Field(min_length=8, max_length=64)]
    position: Annotated[int, Field(ge=0, le=1000)] = 0
    #: Milliseconds the item was actually visible. Reported by an
    #: IntersectionObserver, so an impression means "a human could see it"
    #: rather than "the API returned it" - which is what keeps CTR meaningful.
    visible_ms: Annotated[int, Field(ge=0, le=3_600_000)] = 0
    dwell_ms: Annotated[int | None, Field(ge=0, le=3_600_000)] = None


class HealthResponse(BaseModel):
    """Recommendation-service health (FR-14)."""

    status: str
    engine_loaded: bool
    model_version: str
    models: list[str] = Field(default_factory=list)
    catalogue_size: int = 0
    has_ranker: bool = False
    database: bool = False
    cache: bool = False


__all__ = [
    "ExplanationResponse",
    "FeedbackIn",
    "HealthResponse",
    "HomeRecommendations",
    "ProductSummary",
    "RecommendationItem",
    "RecommendationSection",
]
