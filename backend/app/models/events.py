"""Behavioural events and their rolled-up per-(user, product) aggregate."""

from __future__ import annotations

import datetime as dt
import decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import JSONB, Money, Score, pg_enum
from app.models.enums import DeviceType, EventType

if TYPE_CHECKING:
    from app.models.catalog import Product
    from app.models.users import User


class UserEvent(Base):
    """Append-only behavioural event log.

    Design notes, because this is the table that decides whether the system
    survives contact with real traffic:

    * **Partitioned by month on `occurred_at`.** It is the only table whose
      growth is unbounded (100M events/day at the top of the scale story), and
      partitioning buys three things: retention becomes `DROP PARTITION`
      instead of a multi-hour `DELETE`; the training extract for a date range
      is a partition scan rather than an index scan over everything; and
      autovacuum works per partition instead of fighting one enormous table.

    * **Composite primary key `(id, occurred_at)`.** PostgreSQL requires the
      partition key to be part of every unique constraint on a partitioned
      table. This is not a modelling preference, it is a constraint of the
      feature.

    * **No `updated_at` and no soft delete.** Events are immutable facts. A
      mutable event log cannot be replayed, and replayability is what makes
      feature recomputation and backfills possible.

    * **`product_id` is nullable.** `SEARCH`, `SESSION_START` and `SESSION_END`
      have no product. Forcing a sentinel product would corrupt every count.

    * **Typed columns for what is queried, JSONB for the rest.** Anything the
      feature pipeline filters or groups on is a real column with a real index.
      `event_metadata` holds the per-event-type payload (search terms, rating
      values, cart quantities), validated against a typed Pydantic model at the
      API boundary. That split is what makes FR-21 (new event types with no
      migration) safe rather than a schemaless dumping ground.
    """

    __tablename__ = "user_events"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        comment="Business timestamp and partition key",
    )

    event_type: Mapped[EventType] = mapped_column(
        pg_enum(EventType, "event_type"), nullable=False
    )
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    session_key: Mapped[str] = mapped_column(String(64), nullable=False)
    product_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    source: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default="unknown",
        comment="Surface that produced the event: homepage, pdp, search, rail id",
    )
    device_type: Mapped[DeviceType] = mapped_column(
        pg_enum(DeviceType, "device_type"),
        nullable=False,
        server_default=DeviceType.UNKNOWN.value,
    )

    #: Set when the event followed a recommendation, so post-impression
    #: outcomes can be joined back to the exact serving decision (FR-12).
    recommendation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    event_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        CheckConstraint(
            "event_type NOT IN ("
            "'PRODUCT_VIEW','PRODUCT_CLICK','ADD_TO_CART','REMOVE_FROM_CART',"
            "'WISHLIST','PURCHASE','PRODUCT_SHARE','PRODUCT_RATING','PRODUCT_REVIEW'"
            ") OR product_id IS NOT NULL",
            name="product_scoped_events_have_product",
        ),
        # Foreign keys are deliberately omitted on this table. A partitioned,
        # append-only log ingesting thousands of rows per second cannot afford
        # a referential check per row, and events must still be accepted for a
        # product that is mid-deletion rather than rejected at the API edge.
        # Referential integrity is enforced by the nightly validation step in
        # the pipeline instead, which is the standard trade for event logs.
        Index("ix_user_events_user_id_occurred_at", "user_id", text("occurred_at DESC")),
        Index("ix_user_events_product_id_event_type", "product_id", "event_type"),
        Index("ix_user_events_session_key", "session_key"),
        Index("ix_user_events_occurred_at", text("occurred_at DESC")),
        Index("ix_user_events_recommendation_id", "recommendation_id"),
        {"postgresql_partition_by": "RANGE (occurred_at)"},
    )


class UserProductInteraction(Base):
    """Rolled-up interaction strength for one (user, product) pair.

    This table is the input to collaborative filtering, and it exists so that
    training never scans the raw event log. Rebuilding the interaction matrix
    from `user_events` would mean aggregating tens of millions of rows on every
    training run; reading a pre-aggregated table of at most
    (users x products-they-touched) rows is orders of magnitude cheaper and is
    incrementally maintainable.

    `implicit_weight` is the confidence value fed to ALS (`c_ui = 1 + alpha *
    r_ui`). It is written here rather than computed at training time so that
    the weight actually used is auditable, versionable and reproducible - the
    calibration procedure is ADR-002.
    """

    __tablename__ = "user_product_interactions"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("products.id", ondelete="RESTRICT"),
        primary_key=True,
    )

    view_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    click_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cart_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cart_removal_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    wishlist_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    purchase_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    share_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_quantity: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_spend: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )

    implicit_weight: Mapped[decimal.Decimal] = mapped_column(
        Score,
        nullable=False,
        server_default=text("0"),
        comment="Confidence for ALS; derived from event mix and recency (ADR-002)",
    )
    recency_weight: Mapped[decimal.Decimal] = mapped_column(
        Numeric(10, 6), nullable=False, server_default=text("1")
    )

    first_interaction_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_interaction_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    computed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    user: Mapped[User] = relationship()
    product: Mapped[Product] = relationship()

    __table_args__ = (
        CheckConstraint("view_count >= 0", name="view_count_non_negative"),
        CheckConstraint("purchase_count >= 0", name="purchase_count_non_negative"),
        CheckConstraint("implicit_weight >= 0", name="implicit_weight_non_negative"),
        CheckConstraint(
            "rating IS NULL OR (rating >= 1 AND rating <= 5)", name="rating_range"
        ),
        CheckConstraint(
            "last_interaction_at >= first_interaction_at", name="interaction_ordered"
        ),
        # The (user_id, product_id) primary key already serves user-major
        # lookups. This covers the item-major direction, which item-item
        # neighbour computation and "who else bought this" both need.
        Index("ix_user_product_interactions_product_id", "product_id"),
        Index(
            "ix_user_product_interactions_user_weight",
            "user_id",
            text("implicit_weight DESC"),
        ),
        Index(
            "ix_user_product_interactions_last_interaction_at",
            text("last_interaction_at DESC"),
        ),
    )
