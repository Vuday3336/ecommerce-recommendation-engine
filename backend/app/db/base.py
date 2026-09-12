"""Declarative base, naming conventions and shared column mixins.

The naming convention matters more than it looks: without it, Alembic
autogenerate produces migrations full of unnamed constraints that cannot be
dropped by name later, and the "no pending autogenerate diff" CI check
(ADR-013) produces false positives forever.
"""

from __future__ import annotations

import datetime as dt
import uuid as uuid_module
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, MetaData, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Root declarative class for every ORM model."""

    metadata = metadata_obj

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk}>"


class IdMixin:
    """Surrogate BIGINT primary key.

    BIGINT rather than INT because `user_events` and the recommendation log
    tables are expected to pass 2^31 rows at the scale described in
    architecture.md section 7, and widening a primary key later is painful.
    """

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)


class UuidMixin:
    """Public, non-enumerable identifier.

    Internal joins use the BIGINT `id`; anything exposed over the API uses this,
    so external parties cannot enumerate users or infer row counts.
    """

    @declared_attr
    @classmethod
    def public_id(cls) -> Mapped[uuid_module.UUID]:
        return mapped_column(
            UUID(as_uuid=True),
            nullable=False,
            unique=True,
            server_default=text("gen_random_uuid()"),
        )


class TimestampMixin:
    """`created_at` / `updated_at`, maintained by the database."""

    @declared_attr
    @classmethod
    def created_at(cls) -> Mapped[dt.datetime]:
        return mapped_column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            index=True,
        )

    @declared_attr
    @classmethod
    def updated_at(cls) -> Mapped[dt.datetime]:
        return mapped_column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        )


class SoftDeleteMixin:
    """Soft deletion for rows a historical recommendation may reference.

    A recommendation logged six months ago must stay joinable to the product it
    recommended, otherwise attribution analytics silently lose rows. Hard
    deletion is therefore forbidden on catalogue and user tables; `is_active`
    is the flag the hot path filters on (it is cheap to index), and
    `deleted_at` records when the change happened.
    """

    @declared_attr
    @classmethod
    def is_active(cls) -> Mapped[bool]:
        return mapped_column(Boolean, nullable=False, server_default=text("true"))

    @declared_attr
    @classmethod
    def deleted_at(cls) -> Mapped[dt.datetime | None]:
        return mapped_column(DateTime(timezone=True), nullable=True)

    def soft_delete(self, when: dt.datetime | None = None) -> None:
        self.is_active = False
        self.deleted_at = when or dt.datetime.now(dt.UTC)


def utcnow() -> dt.datetime:
    """Timezone-aware current time. Never use naive datetimes in this project."""
    return dt.datetime.now(dt.UTC)


__all__: list[str] = [
    "NAMING_CONVENTION",
    "Any",
    "Base",
    "IdMixin",
    "SoftDeleteMixin",
    "TimestampMixin",
    "UuidMixin",
    "metadata_obj",
    "utcnow",
]
