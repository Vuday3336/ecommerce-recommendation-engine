"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def build_engine(url: str | None = None, **kwargs: object) -> Engine:
    """Create the SQLAlchemy engine.

    `pool_pre_ping` is on because a recommendation request that fails on a
    stale connection is a user-visible 500 for an entirely avoidable reason -
    connections die when a container restarts or a proxy times them out, and
    the cost of a ping is far below the cost of the alternative.
    """
    engine = create_engine(
        url or settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=settings.db_echo,
        future=True,
        **kwargs,
    )

    @event.listens_for(engine, "connect")
    def _set_session_defaults(dbapi_connection, _record) -> None:
        """Pin every connection to UTC.

        The whole pipeline assumes UTC. A connection inheriting the server's
        local timezone would shift every `date_trunc` and window boundary by
        hours, producing features that are subtly wrong rather than broken.
        """
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")

    return engine


engine: Engine = build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and background jobs."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Cheap liveness probe used by the readiness endpoint."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


__all__ = [
    "SessionLocal",
    "build_engine",
    "check_connection",
    "engine",
    "get_session",
    "session_scope",
]
