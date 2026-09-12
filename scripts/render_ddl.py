"""Render the full PostgreSQL DDL from the ORM metadata without a database.

This is the offline half of the Phase 2 exit criterion: it proves the models
compile to valid PostgreSQL, that every enum, index, constraint and partition
clause is emitted, and that nothing depends on a live connection.

    python scripts/render_ddl.py            # print DDL
    python scripts/render_ddl.py --out x.sql
    python scripts/render_ddl.py --summary  # table/index/constraint counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.models import Base
from sqlalchemy import create_mock_engine
from sqlalchemy.schema import CreateSchema  # noqa: F401


def render() -> str:
    """Compile `CREATE` statements for the whole metadata as PostgreSQL."""
    statements: list[str] = []

    def dump(sql, *_args, **_kwargs) -> None:
        text = str(sql.compile(dialect=engine.dialect)).strip()
        if text:
            statements.append(text + ";")

    engine = create_mock_engine("postgresql+psycopg2://", dump)
    Base.metadata.create_all(engine, checkfirst=False)
    return "\n\n".join(statements)


def summarise() -> int:
    tables = Base.metadata.sorted_tables
    total_indexes = 0
    total_constraints = 0
    total_columns = 0

    print(f"{'table':<32}{'cols':>6}{'idx':>6}{'ck':>5}{'fk':>5}{'uq':>5}")
    print("-" * 59)
    for table in tables:
        checks = [c for c in table.constraints if type(c).__name__ == "CheckConstraint"]
        fks = list(table.foreign_key_constraints)
        uniques = [
            c for c in table.constraints if type(c).__name__ == "UniqueConstraint"
        ]
        total_indexes += len(table.indexes)
        total_constraints += len(checks) + len(fks) + len(uniques)
        total_columns += len(table.columns)
        print(
            f"{table.name:<32}{len(table.columns):>6}{len(table.indexes):>6}"
            f"{len(checks):>5}{len(fks):>5}{len(uniques):>5}"
        )

    print("-" * 59)
    print(
        f"{len(tables)} tables, {total_columns} columns, "
        f"{total_indexes} indexes, {total_constraints} constraints"
    )

    partitioned = [
        t.name for t in tables if t.dialect_kwargs.get("postgresql_partition_by")
    ]
    print(f"partitioned tables: {', '.join(partitioned) or 'none'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write DDL to this file")
    parser.add_argument(
        "--summary", action="store_true", help="print structure counts instead of DDL"
    )
    args = parser.parse_args()

    if args.summary:
        return summarise()

    ddl = render()
    if args.out:
        args.out.write_text(ddl + "\n", encoding="utf-8")
        print(f"wrote {len(ddl.splitlines())} lines to {args.out}")
    else:
        print(ddl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
