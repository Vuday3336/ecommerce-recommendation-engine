"""Shared column type helpers."""

from __future__ import annotations

from enum import Enum

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Numeric
from sqlalchemy.dialects.postgresql import JSONB

#: Money is `NUMERIC(12, 2)`, never float. Binary floating point cannot
#: represent 0.10 exactly, so summing float prices across an order drifts by
#: fractions of a cent, and revenue attribution in the analytics dashboard
#: stops reconciling with the orders table.
Money = Numeric(12, 2)

#: Model scores, affinities and similarities. Six decimal places is well past
#: the precision any of these models actually carry, and keeping them NUMERIC
#: makes stored scores comparable across model versions without float noise.
Score = Numeric(10, 6)


_ENUM_REGISTRY: dict[str, SAEnum] = {}


def pg_enum[E: Enum](enum_cls: type[E], name: str) -> SAEnum:
    """Native PostgreSQL ENUM bound to a Python `StrEnum`.

    `values_callable` is essential: without it SQLAlchemy stores the member
    *name* (`HOME_FOR_YOU`) while application code compares against the member
    *value* (`home_for_you`), and the mismatch only surfaces at query time.

    Instances are cached by type name so that a type used on several tables
    (`price_band` appears on both `brands` and `products`) resolves to one
    shared object. Without the cache, Alembic emits a `CREATE TYPE` per column
    and the second one fails with "type already exists".
    """
    cached = _ENUM_REGISTRY.get(name)
    if cached is not None:
        if cached.enum_class is not enum_cls:
            raise ValueError(
                f"PostgreSQL enum {name!r} is already bound to "
                f"{cached.enum_class!r}, cannot rebind to {enum_cls!r}"
            )
        return cached

    sa_enum = SAEnum(
        enum_cls,
        name=name,
        native_enum=True,
        create_type=True,
        validate_strings=True,
        values_callable=lambda enum: [member.value for member in enum],
    )
    _ENUM_REGISTRY[name] = sa_enum
    return sa_enum


__all__ = ["JSONB", "Money", "Score", "pg_enum"]
