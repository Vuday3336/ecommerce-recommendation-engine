"""Verify the seeded database - the SQL half of the Phase 2 exit criterion.

`diagnostics.py` proves the *dataset* carries structure. This proves the
*database* faithfully holds it: the schema applied, the partitions exist and
route rows correctly, the indexes are present and used, referential integrity
holds where foreign keys were deliberately omitted, and the distributions in
SQL match the distributions in the Parquet files.

    python scripts/verify_database.py
    python scripts/verify_database.py --explain

Exit code 0 means the database is ready for Phase 3.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import psycopg2
from app.core.config import settings
from psycopg2.extras import DictCursor

EXPECTED_TABLES = 20
EXPECTED_INDEXES = 65
EXPECTED_ENUMS = 12


@dataclass(slots=True)
class Result:
    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}\n        {self.detail}"


def connect():
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        cursor_factory=DictCursor,
    )


def scalar(cur, sql: str, params: tuple | None = None) -> object:
    """Run a query returning a single value.

    `params` defaults to None rather than `()`: psycopg2 only skips its
    `%`-interpolation pass when no parameters are supplied at all. Passing an
    empty tuple still triggers it, and a literal `%` in a LIKE pattern then
    raises `IndexError: tuple index out of range` - a confusing error that
    names nothing useful.
    """
    cur.execute(sql, params) if params else cur.execute(sql)
    row = cur.fetchone()
    return row[0] if row else None


def check_schema(cur) -> list[Result]:
    results: list[Result] = []

    # `user_events_*` are partitions of one logical table, and
    # `alembic_version` is Alembic's own bookkeeping row - neither is part of
    # the application schema, so neither should be counted against it.
    tables = scalar(
        cur,
        r"""
        SELECT count(*) FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
          AND table_name NOT LIKE 'user\_events\_%%'
          AND table_name <> 'alembic_version'
        """,
    )
    results.append(
        Result(
            "Tables created",
            tables == EXPECTED_TABLES,
            f"{tables} base tables (expected {EXPECTED_TABLES}, partitions excluded)",
        )
    )

    indexes = scalar(
        cur,
        r"""
        SELECT count(*) FROM pg_indexes
        WHERE schemaname = 'public'
          AND (indexname LIKE 'ix\_%%' OR indexname LIKE 'uq\_%%')
        """,
    )
    results.append(
        Result(
            "Indexes created",
            indexes >= EXPECTED_INDEXES,
            f"{indexes} named indexes (expected at least {EXPECTED_INDEXES})",
        )
    )

    enums = scalar(
        cur,
        "SELECT count(*) FROM pg_type t "
        "JOIN pg_namespace n ON n.oid = t.typnamespace "
        "WHERE t.typtype = 'e' AND n.nspname = 'public'",
    )
    results.append(
        Result(
            "Enum types created",
            enums == EXPECTED_ENUMS,
            f"{enums} native enum types (expected {EXPECTED_ENUMS})",
        )
    )

    has_vector = scalar(
        cur, "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
    )
    results.append(
        Result("pgvector installed", has_vector == 1, "extension 'vector' present")
    )

    hnsw = scalar(
        cur,
        "SELECT indexdef FROM pg_indexes "
        "WHERE indexname = 'ix_product_embeddings_hnsw_cosine'",
    )
    results.append(
        Result(
            "HNSW vector index",
            hnsw is not None and "hnsw" in str(hnsw),
            str(hnsw) if hnsw else "index missing",
        )
    )

    return results


def check_partitions(cur) -> list[Result]:
    results: list[Result] = []

    partitions = scalar(
        cur,
        """
        SELECT count(*) FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid
        JOIN pg_class p ON p.oid = i.inhparent
        WHERE p.relname = 'user_events'
        """,
    )
    results.append(
        Result(
            "user_events partitioned",
            partitions >= 12,
            f"{partitions} partitions attached",
        )
    )

    in_default = scalar(cur, "SELECT count(*) FROM user_events_default")
    results.append(
        Result(
            "No rows in default partition",
            in_default == 0,
            (
                f"{in_default} rows landed in the catch-all partition. Rows here "
                "mean a month range is missing and cannot be added until they "
                "are moved."
            ),
        )
    )

    populated = scalar(
        cur,
        """
        SELECT count(*) FROM (
            SELECT c.relname, pg_catalog.pg_table_size(c.oid) AS bytes
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'user_events'
        ) s WHERE bytes > 8192
        """,
    )
    results.append(
        Result(
            "Events spread across partitions",
            populated >= 5,
            f"{populated} partitions hold data, so pruning has something to prune",
        )
    )

    return results


def check_row_counts(cur) -> list[Result]:
    expectations = {
        "users": 10_000,
        "products": 5_000,
        "categories": 90,
        "brands": 30,
        "user_events": 100_000,
        "orders": 1_000,
        "order_items": 1_000,
        "user_sessions": 5_000,
        "product_variants": 5_000,
    }
    results: list[Result] = []
    for table, minimum in expectations.items():
        count = scalar(cur, f"SELECT count(*) FROM {table}")
        results.append(
            Result(
                f"Row count: {table}",
                count >= minimum,
                f"{count:,} rows (minimum {minimum:,})",
            )
        )
    return results


def check_referential_integrity(cur) -> list[Result]:
    """Validate the references `user_events` deliberately does not enforce.

    Foreign keys were omitted on the event log for throughput (see the note in
    `app/models/events.py`). That is only a defensible trade if something
    actually validates them, which is this.
    """
    orphan_products = scalar(
        cur,
        """
        SELECT count(*) FROM user_events e
        LEFT JOIN products p ON p.id = e.product_id
        WHERE e.product_id IS NOT NULL AND p.id IS NULL
        """,
    )
    orphan_users = scalar(
        cur,
        """
        SELECT count(*) FROM user_events e
        LEFT JOIN users u ON u.id = e.user_id
        WHERE e.user_id IS NOT NULL AND u.id IS NULL
        """,
    )
    orphan_sessions = scalar(
        cur,
        """
        SELECT count(*) FROM user_events e
        LEFT JOIN user_sessions s ON s.session_key = e.session_key
        WHERE s.session_key IS NULL
        """,
    )
    return [
        Result("No orphan event products", orphan_products == 0, f"{orphan_products} orphans"),
        Result("No orphan event users", orphan_users == 0, f"{orphan_users} orphans"),
        Result("No orphan event sessions", orphan_sessions == 0, f"{orphan_sessions} orphans"),
    ]


def check_distributions(cur) -> list[Result]:
    """Confirm SQL sees the same non-uniform structure the diagnostic measured."""
    results: list[Result] = []

    cur.execute(
        """
        SELECT event_type, count(*) AS n
        FROM user_events GROUP BY event_type ORDER BY n DESC
        """
    )
    mix = cur.fetchall()
    types = len(mix)
    results.append(
        Result(
            "Event type coverage",
            types >= 10,
            f"{types} distinct event types present: "
            + ", ".join(f"{r['event_type']}={r['n']:,}" for r in mix[:4])
            + ", ...",
        )
    )

    top_share = scalar(
        cur,
        """
        WITH views AS (
            SELECT product_id, count(*) AS n
            FROM user_events WHERE event_type = 'PRODUCT_VIEW'
            GROUP BY product_id
        ), ranked AS (
            SELECT n, ntile(100) OVER (ORDER BY n DESC) AS bucket FROM views
        )
        SELECT round(
            100.0 * sum(n) FILTER (WHERE bucket = 1) / NULLIF(sum(n), 0), 2
        ) FROM ranked
        """,
    )
    results.append(
        Result(
            "View popularity is skewed",
            top_share is not None and float(top_share) >= 5.0,
            f"top 1% of viewed products take {top_share}% of views",
        )
    )

    concentration = scalar(
        cur,
        """
        WITH per_user AS (
            SELECT e.user_id, c.path, count(*) AS n
            FROM user_events e
            JOIN products p ON p.id = e.product_id
            JOIN categories c ON c.id = p.category_id
            WHERE e.event_type = 'PRODUCT_VIEW'
            GROUP BY e.user_id, c.path
        ), shares AS (
            SELECT user_id,
                   max(n)::numeric / sum(n) AS top_share,
                   sum(n) AS total
            FROM per_user GROUP BY user_id
        )
        SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY top_share)::numeric, 3)
        FROM shares WHERE total >= 5
        """,
    )
    results.append(
        Result(
            "User taste is concentrated",
            concentration is not None and float(concentration) >= 0.20,
            f"median top-subcategory view share per user = {concentration}",
        )
    )

    span = scalar(
        cur,
        "SELECT max(occurred_at)::date - min(occurred_at)::date FROM user_events",
    )
    results.append(
        Result(
            "Temporal span",
            span is not None and int(span) >= 150,
            f"{span} days of events, enough for a train/validation/test split",
        )
    )

    return results


EXPLAIN_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "Recent events for one user (feature building)",
        """
        SELECT product_id, event_type, occurred_at FROM user_events
        WHERE user_id = 42 ORDER BY occurred_at DESC LIMIT 50
        """,
    ),
    (
        "Active products in a category (candidate generation)",
        """
        SELECT id, price FROM products
        WHERE category_id = 12 AND is_active AND stock_quantity > 0
        ORDER BY price LIMIT 50
        """,
    ),
    (
        "Co-purchase pairs for one product (frequently bought together)",
        """
        SELECT b.product_id, count(*) FROM order_items a
        JOIN order_items b ON b.order_id = a.order_id AND b.product_id <> a.product_id
        WHERE a.product_id = 100 GROUP BY b.product_id ORDER BY count(*) DESC LIMIT 10
        """,
    ),
)


def run_explains(cur) -> None:
    print("\nQuery plans")
    print("=" * 78)
    for label, sql in EXPLAIN_QUERIES:
        cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql)
        plan = "\n".join(row[0] for row in cur.fetchall())
        print(f"\n{label}")
        print("-" * 78)
        print(textwrap.indent(plan, "  "))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain", action="store_true", help="show query plans")
    args = parser.parse_args()

    conn = connect()
    try:
        with conn.cursor() as cur:
            groups = {
                "Schema": check_schema(cur),
                "Partitioning": check_partitions(cur),
                "Row counts": check_row_counts(cur),
                "Referential integrity": check_referential_integrity(cur),
                "Distributions": check_distributions(cur),
            }

            print("=" * 78)
            print("PHASE 2 DATABASE VERIFICATION")
            print("=" * 78)

            all_results: list[Result] = []
            for title, results in groups.items():
                print(f"\n{title}")
                print("-" * 78)
                for result in results:
                    print(result.render())
                all_results.extend(results)

            if args.explain:
                run_explains(cur)

        failed = [r for r in all_results if not r.passed]
        print("\n" + "=" * 78)
        print(f"{len(all_results) - len(failed)} of {len(all_results)} checks passed.")
        if failed:
            print("\nPHASE 2 DATABASE GATE: FAILED")
            for result in failed:
                print(f"  - {result.name}")
            return 1
        print("\nPHASE 2 DATABASE GATE: PASSED")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
