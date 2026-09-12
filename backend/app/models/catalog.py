"""Catalogue: categories, products and product variants."""

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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, SoftDeleteMixin, TimestampMixin, UuidMixin
from app.db.types import JSONB, Money, pg_enum
from app.models.enums import PriceBand

if TYPE_CHECKING:
    from app.models.features import ProductEmbedding, ProductFeature


class Category(IdMixin, TimestampMixin, SoftDeleteMixin, Base):
    """Self-referencing category tree (department -> category -> subcategory).

    Kept as an adjacency list rather than a closure table: the tree is three
    levels deep and read almost entirely as "the ancestors of one node", which
    a recursive CTE answers in microseconds at this depth. A closure table
    would be a maintenance cost with no measurable benefit here.
    """

    __tablename__ = "categories"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(140), nullable=False)
    parent_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("categories.id", ondelete="RESTRICT"),
        nullable=True,
    )
    depth: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    path: Mapped[str] = mapped_column(
        String(400),
        nullable=False,
        comment="Materialised ancestor path, e.g. 'apparel/footwear/running-shoes'",
    )

    parent: Mapped[Category | None] = relationship(
        remote_side="Category.id", back_populates="children"
    )
    children: Mapped[list[Category]] = relationship(back_populates="parent")
    products: Mapped[list[Product]] = relationship(back_populates="category")

    __table_args__ = (
        UniqueConstraint("slug", name="uq_categories_slug"),
        CheckConstraint("depth >= 0 AND depth <= 4", name="depth_range"),
        CheckConstraint("parent_id IS NULL OR parent_id <> id", name="no_self_parent"),
        Index("ix_categories_parent_id", "parent_id"),
        Index("ix_categories_path", "path"),
    )


class Brand(IdMixin, TimestampMixin, SoftDeleteMixin, Base):
    """Brands are a first-class table, not a string column on `products`.

    Brand affinity is one of the candidate sources and one of the ranking
    features, so brands need stable ids to join on. A denormalised string would
    make "products by this brand" a full scan and would let typos create
    phantom brands that fragment the affinity signal.
    """

    __tablename__ = "brands"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(140), nullable=False)
    price_tier: Mapped[PriceBand] = mapped_column(
        pg_enum(PriceBand, "price_band"), nullable=False
    )

    products: Mapped[list[Product]] = relationship(back_populates="brand")

    __table_args__ = (UniqueConstraint("slug", name="uq_brands_slug"),)


class Product(IdMixin, UuidMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A sellable product.

    `rating_average` / `rating_count` / `view_count` / `purchase_count` are
    denormalised counters maintained by the batch feature job. They are
    duplicated in `product_features` deliberately: this copy serves the
    catalogue API (a shopper seeing a star rating), while the feature table
    holds the point-in-time snapshot the model trains on. Conflating those two
    is how popularity leaks into a training set (ADR-007).
    """

    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    category_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("categories.id", ondelete="RESTRICT"), nullable=False
    )
    brand_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("brands.id", ondelete="RESTRICT"), nullable=False
    )

    price: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)
    cost: Mapped[decimal.Decimal] = mapped_column(Money, nullable=False)
    price_band: Mapped[PriceBand] = mapped_column(
        pg_enum(PriceBand, "price_band"), nullable=False
    )

    attributes: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="Category-specific attributes (colour, size, material, ...)",
    )
    tags: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    stock_quantity: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    rating_average: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0")
    )
    rating_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    view_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    purchase_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    released_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Catalogue entry date; drives the new-product cold-start path",
    )

    category: Mapped[Category] = relationship(back_populates="products")
    brand: Mapped[Brand] = relationship(back_populates="products")
    variants: Mapped[list[ProductVariant]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    features: Mapped[ProductFeature | None] = relationship(
        back_populates="product", uselist=False
    )
    embedding: Mapped[ProductEmbedding | None] = relationship(
        back_populates="product", uselist=False
    )

    __table_args__ = (
        UniqueConstraint("sku", name="uq_products_sku"),
        CheckConstraint("price > 0", name="price_positive"),
        CheckConstraint("cost >= 0", name="cost_non_negative"),
        CheckConstraint("stock_quantity >= 0", name="stock_non_negative"),
        CheckConstraint(
            "rating_average >= 0 AND rating_average <= 5", name="rating_range"
        ),
        CheckConstraint("rating_count >= 0", name="rating_count_non_negative"),
        # The catalogue and every candidate source filter on active products,
        # so the index only covers those rows. A partial index here is roughly
        # an order of magnitude smaller than the full one once soft-deleted
        # and out-of-stock rows accumulate.
        Index(
            "ix_products_active_category",
            "category_id",
            "price",
            postgresql_where=text("is_active AND stock_quantity > 0"),
        ),
        Index(
            "ix_products_active_brand",
            "brand_id",
            postgresql_where=text("is_active AND stock_quantity > 0"),
        ),
        # Trending and popularity baselines order by these counters.
        Index("ix_products_purchase_count", text("purchase_count DESC")),
        Index("ix_products_released_at", text("released_at DESC")),
        # Full-text search vector for the keyword stage of personalised search
        # (FR-08). Built as an expression index so no extra column is stored.
        Index(
            "ix_products_search_tsv",
            text("to_tsvector('english', name || ' ' || description)"),
            postgresql_using="gin",
        ),
        Index("ix_products_attributes", "attributes", postgresql_using="gin"),
    )


class ProductVariant(IdMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A purchasable variant (size / colour) of a product.

    Orders reference variants, but recommendations reference products: nobody
    wants "we recommend this shoe in size 9" as a homepage rail. Keeping both
    levels means stock can be tracked accurately while the models still work on
    the product grain, which is where the behavioural signal actually lives.
    """

    __tablename__ = "product_variants"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False
    )
    sku: Mapped[str] = mapped_column(String(72), nullable=False)
    variant_name: Mapped[str] = mapped_column(String(160), nullable=False)
    options: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    price_delta: Mapped[decimal.Decimal] = mapped_column(
        Money, nullable=False, server_default=text("0")
    )
    stock_quantity: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    product: Mapped[Product] = relationship(back_populates="variants")

    __table_args__ = (
        UniqueConstraint("sku", name="uq_product_variants_sku"),
        CheckConstraint("stock_quantity >= 0", name="stock_non_negative"),
        Index("ix_product_variants_product_id", "product_id"),
    )
