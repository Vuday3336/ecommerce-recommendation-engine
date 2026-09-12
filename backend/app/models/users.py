"""Users and their sessions."""

from __future__ import annotations

import datetime as dt
import decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, SoftDeleteMixin, TimestampMixin, UuidMixin
from app.db.types import JSONB, Money, pg_enum
from app.models.enums import DeviceType, PriceBand, UserRole, UserSegment

if TYPE_CHECKING:
    from app.models.features import UserFeature
    from app.models.orders import Order


class User(IdMixin, UuidMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A registered shopper, analyst or administrator.

    `password_hash` is nullable because the synthetic dataset creates 10k users
    who never authenticate. Making it nullable is honest about that rather than
    seeding fake credential hashes that look real in a database dump.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        pg_enum(UserRole, "user_role"),
        nullable=False,
        server_default=UserRole.CUSTOMER.value,
    )

    country: Mapped[str] = mapped_column(String(2), nullable=False, server_default="US")
    signup_source: Mapped[str] = mapped_column(
        String(40), nullable=False, server_default="organic"
    )
    primary_device: Mapped[DeviceType] = mapped_column(
        pg_enum(DeviceType, "device_type"),
        nullable=False,
        server_default=DeviceType.DESKTOP.value,
    )

    # --- Behavioural profile -------------------------------------------------
    # These are *ground truth* from the simulator (and, for real users, an
    # analyst-facing label). The model never trains on them: they exist so the
    # Phase 2 diagnostic can check that the recommender rediscovers structure
    # it was never told about. Training on them would be circular.
    segment: Mapped[UserSegment] = mapped_column(
        pg_enum(UserSegment, "user_segment"),
        nullable=False,
        server_default=UserSegment.NEW.value,
    )
    preferred_price_band: Mapped[PriceBand] = mapped_column(
        pg_enum(PriceBand, "price_band"),
        nullable=False,
        server_default=PriceBand.MID.value,
    )
    onboarding_categories: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment="Category ids picked at signup; seeds the new-user cold start (FR-09)",
    )

    # --- Denormalised lifetime counters (maintained by the feature job) ------
    lifetime_orders: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    lifetime_spend: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    last_active_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    orders: Mapped[list[Order]] = relationship(back_populates="user")
    sessions: Mapped[list[UserSession]] = relationship(back_populates="user")
    features: Mapped[UserFeature | None] = relationship(
        back_populates="user", uselist=False
    )

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint("lifetime_orders >= 0", name="lifetime_orders_non_negative"),
        CheckConstraint("lifetime_spend >= 0", name="lifetime_spend_non_negative"),
        CheckConstraint("char_length(country) = 2", name="country_is_iso2"),
        Index("ix_users_segment", "segment"),
        Index("ix_users_last_active_at", text("last_active_at DESC")),
    )


class UserSession(IdMixin, TimestampMixin, Base):
    """A browsing session.

    Sessions are a first-class table because session-scoped recommendations are
    the highest-value cold-start lever (architecture.md section 5): a brand new
    visitor has no history, but three views into their first session they have
    a strong short-term intent signal. `user_id` is nullable so anonymous
    traffic is tracked from the first page view and stitched to an account at
    login (FR-22).
    """

    __tablename__ = "user_sessions"

    session_key: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    device_type: Mapped[DeviceType] = mapped_column(
        pg_enum(DeviceType, "device_type"),
        nullable=False,
        server_default=DeviceType.UNKNOWN.value,
    )
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    event_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    converted: Mapped[bool] = mapped_column(
        nullable=False, server_default=text("false")
    )

    user: Mapped[User | None] = relationship(back_populates="sessions")

    __table_args__ = (
        UniqueConstraint("session_key", name="uq_user_sessions_session_key"),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at", name="session_time_ordered"
        ),
        Index("ix_user_sessions_user_id_started_at", "user_id", text("started_at DESC")),
    )
