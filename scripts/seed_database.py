"""Load the generated dataset into PostgreSQL.

    python scripts/seed_database.py
    python scripts/seed_database.py --data data/synthetic --truncate

Uses `COPY` rather than ORM inserts. At ~440k events the difference is minutes
versus seconds, but the real reason is that this is the same mechanism a
production backfill would use, so the seeding path exercises something we
actually rely on rather than a convenience shortcut.

The script is idempotent with `--truncate`, and refuses to load into a
non-empty database without it, so a half-loaded dataset cannot silently
accumulate duplicate rows.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import psycopg2
from app.core.config import settings
from psycopg2.extensions import connection as PgConnection  # noqa: N812

DEFAULT_DATA = REPO_ROOT / "data" / "synthetic"

#: (parquet file, table, columns to copy). Order is load order: every table
#: appears after the tables it references, so foreign keys hold at every step
#: rather than only at the end.
LOAD_PLAN: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("categories", "categories", ("id", "name", "slug", "parent_id", "depth", "path")),
    ("brands", "brands", ("id", "name", "slug", "price_tier")),
    (
        "products",
        "products",
        (
            "id", "sku", "name", "description", "category_id", "brand_id",
            "price", "cost", "price_band", "attributes", "tags",
            "stock_quantity", "rating_average", "rating_count",
            "view_count", "purchase_count", "released_at",
        ),
    ),
    (
        "product_variants",
        "product_variants",
        ("id", "product_id", "sku", "variant_name", "options", "price_delta", "stock_quantity"),
    ),
    (
        "users",
        "users",
        (
            "id", "email", "full_name", "role", "country", "signup_source",
            "primary_device", "segment", "preferred_price_band",
            "onboarding_categories", "created_at",
        ),
    ),
    (
        "user_sessions",
        "user_sessions",
        ("id", "session_key", "user_id", "device_type", "started_at", "ended_at",
         "event_count", "converted"),
    ),
    (
        "orders",
        "orders",
        ("id", "user_id", "session_id", "order_number", "status", "subtotal",
         "discount_total", "shipping_total", "grand_total", "currency", "placed_at"),
    ),
    (
        "order_items",
        "order_items",
        ("id", "order_id", "product_id", "variant_id", "quantity", "unit_price",
         "discount", "line_total"),
    ),
    (
        "events",
        "user_events",
        ("occurred_at", "event_type", "user_id", "session_key", "product_id",
         "source", "device_type", "event_metadata"),
    ),
)

#: Tables whose `id` is explicitly supplied, so the identity sequence has to be
#: advanced afterwards. Without this the first application insert collides with
#: a seeded row and fails on the primary key - a classic and confusing seeding bug.
SEQUENCE_TABLES: tuple[str, ...] = (
    "categories", "brands", "products", "product_variants",
    "users", "user_sessions", "orders", "order_items",
)

#: Truncation order is the reverse of load order.
TRUNCATE_ORDER: tuple[str, ...] = (
    "recommendation_conversions", "recommendation_clicks",
    "recommendation_impressions", "recommendations",
    "user_events", "user_product_interactions",
    "order_items", "orders", "user_sessions",
    "user_features", "product_features", "product_embeddings",
    "experiment_assignments", "experiments", "model_versions",
    "product_variants", "products", "brands", "categories", "users",
)


def connect() -> PgConnection:
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


def frame_to_csv(frame: pd.DataFrame, columns: tuple[str, ...]) -> io.StringIO:
    """Serialise selected columns to an in-memory CSV for COPY.

    NULL is represented by the empty string with `NULL ''` on the COPY command.
    That is safe here because every text column that could legitimately hold an
    empty string (`description`) is non-null with a default, so an empty field
    unambiguously means missing.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    subset = frame.reindex(columns=list(columns))

    # pandas has no integer dtype that can hold NaN, so any integer column with
    # a single null is silently promoted to float64 - and `categories.parent_id`
    # has exactly one (the root category). COPY then sees "1.0" for a BIGINT
    # column and rejects the entire load. Casting integral float columns back to
    # pandas' nullable Int64 fixes it at the source, rather than string-munging
    # the CSV afterwards and risking mangled decimals.
    for column in subset.columns:
        values = subset[column]
        if values.dtype.kind != "f":
            continue
        present = values.dropna()
        if present.empty or not (present % 1 == 0).all():
            continue
        subset[column] = values.astype("Int64")

    for row in subset.itertuples(index=False, name=None):
        writer.writerow([_render(value) for value in row])
    buffer.seek(0)
    return buffer


def _render(value: object) -> object:
    """Render one value for COPY, with the empty string meaning NULL.

    Every pandas null spelling has to be caught here: `None`, `float('nan')`,
    `pd.NaT` and `pd.NA`. Missing one is not a visible bug in Python - it is a
    literal `<NA>` or `nan` string reaching PostgreSQL, which rejects the whole
    COPY with a type error naming a column that looks perfectly fine.
    """
    if value is None:
        return ""
    # `pd.isna` raises on arrays and returns an array for list-likes, so it is
    # only safe on scalars. Strings and dicts are passed straight through.
    if isinstance(value, (str, bytes, dict, list)):
        return value
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def table_is_empty(conn: PgConnection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)")
        return not cur.fetchone()[0]


def truncate_all(conn: PgConnection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE TABLE "
            + ", ".join(TRUNCATE_ORDER)
            + " RESTART IDENTITY CASCADE"
        )
    conn.commit()


def copy_frame(
    conn: PgConnection, table: str, columns: tuple[str, ...], frame: pd.DataFrame
) -> int:
    if frame.empty:
        return 0
    buffer = frame_to_csv(frame, columns)
    column_list = ", ".join(columns)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({column_list}) FROM STDIN WITH (FORMAT csv, NULL '')",
            buffer,
        )
    return len(frame)


def reset_sequences(conn: PgConnection) -> None:
    with conn.cursor() as cur:
        for table in SEQUENCE_TABLES:
            cur.execute(
                f"""
                SELECT setval(
                    pg_get_serial_sequence(%s, 'id'),
                    COALESCE((SELECT MAX(id) FROM {table}), 1),
                    (SELECT MAX(id) IS NOT NULL FROM {table})
                )
                """,
                (table,),
            )
    conn.commit()


def analyze(conn: PgConnection) -> None:
    """Refresh planner statistics.

    Freshly bulk-loaded tables have no statistics, so the planner assumes tiny
    row counts and picks sequential scans for queries that should use the
    indexes this schema exists to provide. Skipping this step makes the first
    latency measurements meaningless.
    """
    old_isolation = conn.isolation_level
    conn.set_isolation_level(0)
    with conn.cursor() as cur:
        cur.execute("ANALYZE")
    conn.set_isolation_level(old_isolation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear all data first. Required when the database is not empty.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"No dataset at {args.data}. Run: python data-generation/generate.py")
        return 1

    print(f"Connecting to {settings.postgres_host}:{settings.postgres_port}"
          f"/{settings.postgres_db} as {settings.postgres_user}")
    conn = connect()

    try:
        if args.truncate:
            print("Truncating existing data...")
            truncate_all(conn)
        elif not table_is_empty(conn, "products"):
            print(
                "Database already contains data. Re-run with --truncate to "
                "replace it. Refusing to append, which would duplicate rows."
            )
            return 1

        total = 0
        started = time.perf_counter()
        for file_name, table, columns in LOAD_PLAN:
            path = args.data / f"{file_name}.parquet"
            frame = pd.read_parquet(path)
            step = time.perf_counter()
            rows = copy_frame(conn, table, columns, frame)
            conn.commit()
            total += rows
            print(
                f"  {table:<20} {rows:>10,} rows  "
                f"({time.perf_counter() - step:.1f}s)"
            )

        reset_sequences(conn)
        print("  sequences reset")
        analyze(conn)
        print("  statistics refreshed")

        print(f"\nLoaded {total:,} rows in {time.perf_counter() - started:.1f}s")
        print("Next: python scripts/verify_database.py")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
