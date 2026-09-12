"""Orders and order lines."""

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

from app.db.base import Base, IdMixin, TimestampMixin, UuidMixin
from app.db.types import Money, pg_enum
from app.models.enums import OrderStatus

if TYPE_CHECKING:
    from app.models.catalog import Product, ProductVariant
    from app.models.users import User


class Order(IdMixin, UuidMixin, TimestampMixin, Base):
    """A placed order.

    `placed_at` is separate from `created_at`: the synthetic generator writes
    180 days of backdated history in one batch, so `created_at` (row insert
    time) is useless as a business timestamp. Every temporal split, feature
    window and trending calculation reads `placed_at`. Conflating the two is a
    quiet way to destroy the temporal evaluation described in ADR-007.
    """

    __tablename__ = "orders"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    session_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("user_sessions.id", ondelete="SET NULL"), nullable=True
    )
    order_number: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(
        pg_enum(OrderStatus, "order_status"),
        nullable=False,
        server_default=OrderStatus.PAID.value,
    )

    subtotal: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)
    discount_total: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    shipping_total: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    grand_total: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default="USD"
    )

    placed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Business timestamp; all temporal logic reads this, not created_at",
    )

    user: Mapped[User] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("order_number", name="uq_orders_order_number"),
        CheckConstraint("subtotal >= 0", name="subtotal_non_negative"),
        CheckConstraint("grand_total >= 0", name="grand_total_non_negative"),
        CheckConstraint("discount_total >= 0", name="discount_non_negative"),
        # Feature building reads "this user's orders, newest first" constantly.
        Index("ix_orders_user_id_placed_at", "user_id", text("placed_at DESC")),
        Index("ix_orders_placed_at", text("placed_at DESC")),
        Index("ix_orders_status", "status"),
        # Not for a query we run often, but for the foreign key itself: without
        # it, deleting or nulling a session forces a sequential scan of orders.
        Index("ix_orders_session_id", "session_id"),
    )


class OrderItem(IdMixin, TimestampMixin, Base):
    """A line in an order.

    Both `product_id` and `variant_id` are stored. The variant is what was
    actually shipped; the product is the grain every model works on, and
    denormalising it here turns "frequently bought together" from a three-table
    self-join into a single self-join on this table. At the co-purchase matrix
    sizes involved that difference is the difference between a fast batch job
    and a slow one.

    `unit_price` is a snapshot, not a lookup: prices change, and an order from
    three months ago must keep the price the customer actually paid or revenue
    attribution silently rewrites history.
    """

    __tablename__ = "order_items"

    order_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False
    )
    variant_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("product_variants.id", ondelete="RESTRICT"), nullable=True
    )

    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)
    discount: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    line_total: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)

    #: Set when this line is attributable to a recommendation the user clicked.
    #: This is what makes "recommendation revenue" a measured number instead of
    #: an estimate. Nullable because most purchases are not recommendation-driven.
    source_recommendation_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("recommendations.id", ondelete="SET NULL"),
        nullable=True,
    )

    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship()
    variant: Mapped[ProductVariant | None] = relationship()

    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("unit_price >= 0", name="unit_price_non_negative"),
        CheckConstraint("line_total >= 0", name="line_total_non_negative"),
        UniqueConstraint(
            "order_id", "variant_id", name="uq_order_items_order_id_variant_id"
        ),
        Index("ix_order_items_order_id", "order_id"),
        # Drives frequently-bought-together and product purchase counts.
        Index("ix_order_items_product_id", "product_id"),
        Index("ix_order_items_variant_id", "variant_id"),
        # Recommendation revenue attribution joins on this.
        Index(
            "ix_order_items_source_recommendation_id",
            "source_recommendation_id",
            postgresql_where=text("source_recommendation_id IS NOT NULL"),
        ),
    )
