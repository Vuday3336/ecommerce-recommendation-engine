"""Materialised feature tables and product embeddings.

**Read this before using these tables in training.** They hold the *current*
snapshot, materialised by the batch feature job for online serving. They are
not a time-travel store, and training must never read them for a historical
label: doing so would attach features computed today to an interaction from
three months ago, which is exactly the leakage ADR-007 forbids.

Training features are computed by the same `recsys.features` builders with an
explicit `as_of_ts` cut-off. `as_of_ts` is recorded on every row here so that a
serving-time feature vector can be reconciled against the offline one and the
skew test (NFR-06) has something to compare.
"""

from __future__ import annotations

import datetime as dt
import decimal
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.types import JSONB, Money, Score, pg_enum
from app.models.enums import PriceBand

if TYPE_CHECKING:
    from app.models.catalog import Product
    from app.models.users import User

#: all-MiniLM-L6-v2 output width. Pinned here so a model swap is a deliberate
#: migration rather than a silent dimension mismatch at query time.
EMBEDDING_DIM = 384


class UserFeature(TimestampMixin, Base):
    """Current feature snapshot for one user."""

    __tablename__ = "user_features"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    as_of_ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Cut-off used to build this row; no input event is at or after it",
    )

    # --- Volume and value ----------------------------------------------------
    total_purchases: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_spending: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    avg_order_value: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    customer_lifetime_value: Mapped[decimal.Decimal] = mapped_column(
        Money,
        nullable=False,
        server_default=text("0"),
        comment="Historical margin plus a simple frequency/recency projection",
    )

    # --- Engagement ----------------------------------------------------------
    total_events: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    distinct_products_viewed: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    session_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    avg_session_duration_s: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    active_days: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    # --- Rates ---------------------------------------------------------------
    view_to_cart_rate: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    cart_to_purchase_rate: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    view_to_purchase_rate: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    purchase_frequency_days: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Mean days between orders; NULL for users with fewer than 2 orders",
    )

    # --- Recency -------------------------------------------------------------
    days_since_last_event: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_since_last_purchase: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_since_first_seen: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )

    # --- Price behaviour -----------------------------------------------------
    avg_purchase_price: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    median_purchase_price: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    price_sensitivity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default=text("0"),
        comment="0 = indifferent to price, 1 = strongly prefers the cheap end",
    )
    dominant_price_band: Mapped[PriceBand] = mapped_column(
        pg_enum(PriceBand, "price_band"),
        nullable=False,
        server_default=PriceBand.MID.value,
    )

    # --- Affinities ----------------------------------------------------------
    # Stored as JSONB maps {id: weight} rather than separate rows. A user has
    # a handful of meaningful affinities out of hundreds of categories, so a
    # sparse map is both smaller and read in a single fetch on the hot path -
    # where the alternative is an extra join per recommendation request.
    category_affinity: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    brand_affinity: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    top_categories: Mapped[list[int]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    top_brands: Mapped[list[int]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    #: Mean of the embeddings of products the user interacted with, weighted by
    #: implicit weight. This is the query vector for the content candidate
    #: source, so it is stored rather than recomputed per request.
    taste_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    user: Mapped[User] = relationship(back_populates="features")

    __table_args__ = (
        CheckConstraint(
            "price_sensitivity >= 0 AND price_sensitivity <= 1",
            name="price_sensitivity_range",
        ),
        CheckConstraint(
            "view_to_purchase_rate >= 0 AND view_to_purchase_rate <= 1",
            name="view_to_purchase_rate_range",
        ),
        CheckConstraint("total_purchases >= 0", name="total_purchases_non_negative"),
        Index("ix_user_features_as_of_ts", text("as_of_ts DESC")),
    )


class ProductFeature(TimestampMixin, Base):
    """Current feature snapshot for one product."""

    __tablename__ = "product_features"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), primary_key=True
    )
    as_of_ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # --- Volume --------------------------------------------------------------
    view_count_total: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    view_count_30d: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    view_count_7d: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    cart_count_30d: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    purchase_count_total: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    purchase_count_30d: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    distinct_buyers: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    revenue_30d: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )

    # --- Rates ---------------------------------------------------------------
    view_to_cart_rate: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    conversion_rate: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    repeat_purchase_rate: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )

    # --- Derived scores ------------------------------------------------------
    popularity_score: Mapped[decimal.Decimal] = mapped_column(
        Score,
        nullable=False,
        server_default=text("0"),
        comment="Normalised blend of purchases, views and revenue over the window",
    )
    trending_score: Mapped[decimal.Decimal] = mapped_column(
        Score,
        nullable=False,
        server_default=text("0"),
        comment="Time-decayed velocity; short half-life so it reacts within hours",
    )
    novelty_score: Mapped[decimal.Decimal] = mapped_column(
        Score,
        nullable=False,
        server_default=text("0"),
        comment="Inverse log popularity; used by the diversity re-ranker",
    )
    quality_score: Mapped[decimal.Decimal] = mapped_column(
        Score,
        nullable=False,
        server_default=text("0"),
        comment="Bayesian-smoothed rating, so a single 5-star review cannot top the chart",
    )

    days_since_release: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    is_cold: Mapped[bool] = mapped_column(
        nullable=False,
        server_default=text("true"),
        comment="Too few interactions for collaborative signal; routes to content path",
    )

    product: Mapped[Product] = relationship(back_populates="features")

    __table_args__ = (
        CheckConstraint(
            "conversion_rate >= 0 AND conversion_rate <= 1", name="conversion_rate_range"
        ),
        CheckConstraint("popularity_score >= 0", name="popularity_non_negative"),
        Index("ix_product_features_popularity", text("popularity_score DESC")),
        Index("ix_product_features_trending", text("trending_score DESC")),
        Index("ix_product_features_is_cold", "is_cold"),
    )


class ProductEmbedding(TimestampMixin, Base):
    """Dense vector representation of a product.

    Stored in Postgres rather than a separate index (ADR-004) so that insert
    and index are one transaction: a new product is immediately retrievable by
    the content candidate source, which is what makes new-product cold start
    (FR-10) reliable instead of dependent on a rebuild schedule.

    `source_hash` is a digest of the exact text and attributes that produced
    the vector. The embedding job re-embeds only rows whose hash changed, which
    turns a full re-embed of the catalogue into an incremental update.
    """

    __tablename__ = "product_embeddings"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dim: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text(str(EMBEDDING_DIM))
    )

    product: Mapped[Product] = relationship(back_populates="embedding")

    __table_args__ = (
        CheckConstraint(f"dim = {EMBEDDING_DIM}", name="dim_matches_model"),
        # HNSW rather than IVFFlat: it needs no training step and no periodic
        # rebuild as rows are added, which matters because the catalogue grows
        # continuously and an IVFFlat index degrades until it is retrained.
        # Cosine distance because the sentence-transformer output is directionally
        # meaningful but not length-normalised in a way that makes L2 sensible.
        Index(
            "ix_product_embeddings_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_product_embeddings_model", "model_name", "model_version"),
    )
