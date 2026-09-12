"""Schema invariants.

These are not "does SQLAlchemy work" tests. Each one encodes a design rule from
`docs/architecture.md` or an ADR, so that a future model change that quietly
breaks the rule fails here rather than in a latency graph six weeks later.
"""

from __future__ import annotations

import decimal

import pytest
from app.db.types import _ENUM_REGISTRY
from app.models import Base
from sqlalchemy import Float, Numeric, UniqueConstraint, create_mock_engine
from sqlalchemy.dialects import postgresql

METADATA = Base.metadata
TABLES = METADATA.sorted_tables

#: Tables that are append-only event or log data, where a mutation timestamp
#: and a soft-delete flag would be meaningless.
APPEND_ONLY = {
    "user_events",
    "recommendations",
    "recommendation_impressions",
    "recommendation_clicks",
    "recommendation_conversions",
    "experiment_assignments",
}

#: Derived aggregate tables. These are rebuilt wholesale by a batch job, so
#: `computed_at` (when the derivation ran) is the meaningful timestamp and a
#: row-level created/updated pair would be noise.
DERIVED = {"user_product_interactions"}

#: Tables a historical recommendation may reference, which therefore must never
#: be hard-deleted (see `SoftDeleteMixin`).
SOFT_DELETE_REQUIRED = {
    "users",
    "products",
    "product_variants",
    "categories",
    "brands",
}


def test_expected_tables_are_registered() -> None:
    names = {table.name for table in TABLES}
    required = {
        "users", "products", "categories", "product_variants", "orders",
        "order_items", "user_events", "user_product_interactions",
        "recommendations", "recommendation_impressions", "recommendation_clicks",
        "recommendation_conversions", "user_features", "product_features",
        "model_versions", "experiments", "experiment_assignments",
    }
    missing = required - names
    assert not missing, f"missing required tables: {sorted(missing)}"


def test_every_table_has_a_primary_key() -> None:
    for table in TABLES:
        assert table.primary_key.columns, f"{table.name} has no primary key"


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_every_foreign_key_column_is_indexed(table) -> None:
    """An unindexed foreign key turns every parent delete and every join into a
    sequential scan. This is the single most common schema performance bug, so
    it is asserted rather than trusted to review."""
    indexed: set[str] = set()
    for index in table.indexes:
        columns = list(index.expressions)
        if columns:
            first = columns[0]
            name = getattr(first, "name", None)
            if name:
                indexed.add(name)
    # A column that leads the primary key is already indexed by it.
    pk_columns = list(table.primary_key.columns)
    if pk_columns:
        indexed.add(pk_columns[0].name)
    for constraint in table.constraints:
        if isinstance(constraint, UniqueConstraint):
            cols = list(constraint.columns)
            if cols:
                indexed.add(cols[0].name)

    for fk in table.foreign_key_constraints:
        column = next(iter(fk.columns))
        assert column.name in indexed, (
            f"{table.name}.{column.name} is a foreign key with no index leading "
            f"on it; add one or make it the first column of an existing index"
        )


def test_money_columns_are_numeric_not_float() -> None:
    """Binary floats cannot represent 0.10 exactly. Summing float prices across
    an order drifts, and revenue attribution stops reconciling with orders."""
    money_names = {
        "price", "cost", "unit_price", "line_total", "discount", "subtotal",
        "grand_total", "shipping_total", "discount_total", "revenue",
        "total_spend", "total_spending", "avg_order_value", "lifetime_spend",
        "customer_lifetime_value", "avg_purchase_price", "median_purchase_price",
        "revenue_30d", "price_delta",
    }
    for table in TABLES:
        for column in table.columns:
            if column.name in money_names:
                assert isinstance(column.type, Numeric) and not isinstance(
                    column.type, Float
                ), f"{table.name}.{column.name} holds money but is {column.type!r}"


def test_mutable_tables_carry_timestamps() -> None:
    for table in TABLES:
        if table.name in APPEND_ONLY or table.name in DERIVED:
            continue
        assert "created_at" in table.columns, f"{table.name} lacks created_at"
        assert "updated_at" in table.columns, f"{table.name} lacks updated_at"


def test_derived_tables_record_when_they_were_computed() -> None:
    for name in DERIVED:
        assert "computed_at" in METADATA.tables[name].columns


def test_append_only_tables_have_no_update_timestamp() -> None:
    """An event log with an `updated_at` invites mutation, and a mutable event
    log cannot be replayed - which breaks feature backfills."""
    for name in ("user_events",):
        table = METADATA.tables[name]
        assert "updated_at" not in table.columns


def test_referenced_tables_support_soft_deletion() -> None:
    for name in SOFT_DELETE_REQUIRED:
        table = METADATA.tables[name]
        assert "deleted_at" in table.columns, f"{name} cannot be soft-deleted"
        assert "is_active" in table.columns, f"{name} has no active flag"


def test_catalogue_foreign_keys_restrict_deletion() -> None:
    """A logged recommendation must stay joinable to the product it named."""
    for table_name, column_name in (
        ("products", "category_id"),
        ("products", "brand_id"),
        ("order_items", "product_id"),
        ("recommendations", "product_id"),
    ):
        table = METADATA.tables[table_name]
        fk = next(
            fk
            for fk in table.foreign_key_constraints
            if column_name in {c.name for c in fk.columns}
        )
        assert fk.ondelete == "RESTRICT", (
            f"{table_name}.{column_name} allows the parent row to vanish, which "
            f"would orphan historical recommendation and order records"
        )


def test_user_events_is_partitioned_on_its_timestamp() -> None:
    table = METADATA.tables["user_events"]
    partition_by = table.dialect_kwargs.get("postgresql_partition_by")
    assert partition_by == "RANGE (occurred_at)"

    # PostgreSQL requires the partition key in every unique constraint.
    pk_columns = {c.name for c in table.primary_key.columns}
    assert "occurred_at" in pk_columns, (
        "the partition key must be part of the primary key or PostgreSQL "
        "rejects the table"
    )


def test_enum_types_are_registered_once_each() -> None:
    """A duplicate CREATE TYPE fails the migration on the second table."""
    names = list(_ENUM_REGISTRY)
    assert len(names) == len(set(names))
    assert len(names) >= 12

    used: dict[str, set[str]] = {}
    for table in TABLES:
        for column in table.columns:
            type_name = getattr(column.type, "name", None)
            if type_name in _ENUM_REGISTRY:
                used.setdefault(type_name, set()).add(f"{table.name}.{column.name}")

    shared = {name: cols for name, cols in used.items() if len(cols) > 1}
    assert "price_band" in shared, (
        "price_band is expected on several tables; if that stops being true the "
        "enum-sharing regression this guards against is no longer covered"
    )


def test_vector_columns_declare_a_dimension() -> None:
    from app.models.features import EMBEDDING_DIM

    for table_name, column_name in (
        ("product_embeddings", "embedding"),
        ("user_features", "taste_embedding"),
    ):
        column = METADATA.tables[table_name].columns[column_name]
        assert getattr(column.type, "dim", None) == EMBEDDING_DIM


def test_metadata_compiles_to_postgresql() -> None:
    """The whole schema must render as valid PostgreSQL with no live database."""
    statements: list[str] = []

    def collect(sql, *_args, **_kwargs) -> None:
        statements.append(str(sql.compile(dialect=engine.dialect)))

    engine = create_mock_engine("postgresql+psycopg2://", collect)
    METADATA.create_all(engine, checkfirst=False)

    ddl = "\n".join(statements)
    assert "PARTITION BY RANGE (occurred_at)" in ddl
    assert "USING hnsw" in ddl
    assert "vector_cosine_ops" in ddl
    assert ddl.count("CREATE TYPE") == len(_ENUM_REGISTRY)


def test_no_column_uses_naive_datetime() -> None:
    """Naive timestamps and a UTC pipeline eventually disagree by an hour, and
    the failure shows up as a subtly wrong feature window rather than an error."""
    from sqlalchemy import DateTime

    offenders = [
        f"{table.name}.{column.name}"
        for table in TABLES
        for column in table.columns
        if isinstance(column.type, DateTime) and not column.type.timezone
    ]
    assert not offenders, f"timezone-naive datetime columns: {offenders}"


def test_decimal_precision_is_sufficient_for_money() -> None:
    table = METADATA.tables["orders"]
    column = table.columns["grand_total"]
    assert column.type.precision >= 12
    assert column.type.scale == 2
    assert decimal.Decimal("99999999.99") < decimal.Decimal(10) ** (
        column.type.precision - column.type.scale
    )


def test_postgres_dialect_renders_jsonb_not_json() -> None:
    """JSON has no indexing support; JSONB does, and the event metadata and
    affinity maps are both queried."""
    for table_name, column_name in (
        ("user_events", "event_metadata"),
        ("user_features", "category_affinity"),
        ("products", "attributes"),
    ):
        column = METADATA.tables[table_name].columns[column_name]
        assert isinstance(column.type, postgresql.JSONB)
