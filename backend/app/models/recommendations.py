"""Recommendation serving log and its outcome tables.

`recommendations` is a **fact table**, deliberately denormalised: one row per
served item, carrying the request context (strategy, model version, experiment
variant, latency) on every row rather than in a separate request dimension.

That is a considered trade, not laziness. These rows are written once and read
by analytics forever, and every dashboard query - CTR by surface, conversion by
source, revenue by model version, position-bias curves - would otherwise pay a
join to a request table it never filters on independently. Item-level grain is
what makes "most recommended products" and "conversion rate by candidate
source" single-table `GROUP BY`s instead of unnesting a JSONB array.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid as uuid_module
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin
from app.db.types import JSONB, Money, Score, pg_enum
from app.models.enums import (
    RecommendationSource,
    RecommendationSurface,
    ServingStrategy,
)

if TYPE_CHECKING:
    from app.models.catalog import Product
    from app.models.orders import Order, OrderItem
    from app.models.users import User


class Recommendation(IdMixin, Base):
    """One product served to one user on one surface."""

    __tablename__ = "recommendations"

    request_id: Mapped[uuid_module.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment="Groups the items returned by a single API call",
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    session_key: Mapped[str] = mapped_column(String(64), nullable=False)
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False
    )

    surface: Mapped[RecommendationSurface] = mapped_column(
        pg_enum(RecommendationSurface, "recommendation_surface"), nullable=False
    )
    source: Mapped[RecommendationSource] = mapped_column(
        pg_enum(RecommendationSource, "recommendation_source"),
        nullable=False,
        comment="Candidate generator that produced this item",
    )
    strategy: Mapped[ServingStrategy] = mapped_column(
        pg_enum(ServingStrategy, "serving_strategy"),
        nullable=False,
        comment="Degradation-ladder rung that served the request",
    )

    position: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, comment="Zero-based slot; needed for position bias"
    )
    score: Mapped[decimal.Decimal] = mapped_column(Score, nullable=False)
    #: Per-source scores before blending, kept so a ranking decision can be
    #: reconstructed months later without re-running the model.
    score_components: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    explanation: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="", comment="Human-readable reason (FR-11)"
    )
    explanation_evidence: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="Machine-readable support: anchor products, SHAP contributions",
    )

    model_version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=True
    )
    experiment_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("experiments.id", ondelete="RESTRICT"), nullable=True
    )
    variant: Mapped[str | None] = mapped_column(String(40), nullable=True)

    candidate_pool_size: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    latency_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cache_hit: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))

    served_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    user: Mapped[User | None] = relationship()
    product: Mapped[Product] = relationship()
    impressions: Mapped[list[RecommendationImpression]] = relationship(
        back_populates="recommendation"
    )
    clicks: Mapped[list[RecommendationClick]] = relationship(
        back_populates="recommendation"
    )
    conversions: Mapped[list[RecommendationConversion]] = relationship(
        back_populates="recommendation"
    )

    __table_args__ = (
        CheckConstraint("position >= 0", name="position_non_negative"),
        CheckConstraint("latency_ms >= 0", name="latency_non_negative"),
        UniqueConstraint(
            "request_id", "position", name="uq_recommendations_request_id_position"
        ),
        Index("ix_recommendations_request_id", "request_id"),
        Index("ix_recommendations_user_id_served_at", "user_id", text("served_at DESC")),
        Index("ix_recommendations_product_id_served_at", "product_id", text("served_at DESC")),
        Index("ix_recommendations_surface_served_at", "surface", text("served_at DESC")),
        Index("ix_recommendations_model_version_id", "model_version_id"),
        Index("ix_recommendations_experiment_variant", "experiment_id", "variant"),
    )


class RecommendationImpression(IdMixin, Base):
    """The item was actually visible to the user.

    Separate from the serving log on purpose: an API response is not an
    impression. A rail below the fold is served but never seen, and counting it
    would depress CTR for exactly the surfaces that are working. The frontend
    reports this from an `IntersectionObserver`, so the denominator of CTR is
    "items a human could see" rather than "items we sent".
    """

    __tablename__ = "recommendation_impressions"

    recommendation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False
    )
    shown_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    visible_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    viewport_position: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    recommendation: Mapped[Recommendation] = relationship(back_populates="impressions")

    __table_args__ = (
        UniqueConstraint(
            "recommendation_id", name="uq_recommendation_impressions_recommendation_id"
        ),
        CheckConstraint("visible_ms >= 0", name="visible_ms_non_negative"),
        Index("ix_recommendation_impressions_shown_at", text("shown_at DESC")),
    )


class RecommendationClick(IdMixin, Base):
    """The user clicked the recommended item."""

    __tablename__ = "recommendation_clicks"

    recommendation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False
    )
    clicked_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    dwell_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time on the destination page; a proxy for click quality",
    )

    recommendation: Mapped[Recommendation] = relationship(back_populates="clicks")

    __table_args__ = (
        Index("ix_recommendation_clicks_recommendation_id", "recommendation_id"),
        Index("ix_recommendation_clicks_clicked_at", text("clicked_at DESC")),
    )


class RecommendationConversion(IdMixin, Base):
    """The recommended item was purchased within the attribution window.

    `attribution_window_hours` and `attribution_model` are stored per row
    because attribution is a *policy*, not a fact. Changing the window from 24h
    to 7d changes every historical number, and without recording the policy in
    force at write time nobody can explain why last quarter's revenue moved.
    """

    __tablename__ = "recommendation_conversions"

    recommendation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False
    )
    order_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("orders.id", ondelete="RESTRICT"), nullable=False
    )
    order_item_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("order_items.id", ondelete="RESTRICT"), nullable=True
    )

    converted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    quantity: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    revenue: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)

    attribution_window_hours: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("168")
    )
    attribution_model: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="last_click"
    )

    recommendation: Mapped[Recommendation] = relationship(back_populates="conversions")
    order: Mapped[Order] = relationship()
    order_item: Mapped[OrderItem | None] = relationship()

    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("revenue >= 0", name="revenue_non_negative"),
        UniqueConstraint(
            "recommendation_id",
            "order_item_id",
            name="uq_recommendation_conversions_recommendation_id_order_item_id",
        ),
        Index("ix_recommendation_conversions_recommendation_id", "recommendation_id"),
        Index("ix_recommendation_conversions_order_id", "order_id"),
        Index("ix_recommendation_conversions_order_item_id", "order_item_id"),
        Index("ix_recommendation_conversions_converted_at", text("converted_at DESC")),
    )
