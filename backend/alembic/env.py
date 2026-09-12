"""Alembic environment.

Two things here are load-bearing and easy to get wrong:

1. The database URL comes from the application settings object, never from
   `alembic.ini`. That keeps credentials out of a checked-in file and
   guarantees migrations and the running service always agree on the target.

2. `include_object` hides the `user_events` partitions from autogenerate.
   Without it, every partition PostgreSQL created (`user_events_2026_01`, ...)
   looks like an unmanaged table and autogenerate proposes dropping it - which
   would both destroy data and make the "no pending diff" CI check (ADR-013)
   permanently red.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata

#: Objects autogenerate must ignore.
IGNORED_TABLE_PREFIXES: tuple[str, ...] = ("user_events_",)

#: Expression indexes autogenerate cannot round-trip.
#:
#: PostgreSQL normalises an index expression when it stores it - it adds
#: explicit casts and parentheses, so
#:     to_tsvector('english', name || ' ' || description)
#: comes back as
#:     to_tsvector('english'::regconfig, (name::text || ' '::text) || description)
#: Alembic compares those as strings, finds them different, and proposes
#: dropping and recreating the index on *every* autogenerate run. The index is
#: correct and unchanged; only the textual form differs.
#:
#: Left in, this would make the "no pending diff" CI check permanently red -
#: which trains everyone to ignore it, defeating the purpose of having it. These
#: indexes are managed by hand in migrations instead.
IGNORED_INDEXES: frozenset[str] = frozenset({"ix_products_search_tsv"})


def include_object(obj, name: str, type_: str, reflected: bool, compare_to) -> bool:
    """Filter reflected objects that Alembic should not manage."""
    if type_ == "index" and name in IGNORED_INDEXES:
        return False
    return not (
        type_ == "table"
        and name is not None
        and any(name.startswith(prefix) for prefix in IGNORED_TABLE_PREFIXES)
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running against a connection."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
