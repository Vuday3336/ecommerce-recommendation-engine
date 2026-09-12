"""Shared test fixtures.

The API is tested against the **real** recommendation engine loaded from the
committed artefacts, not against a mock. Mocking the engine would make these
tests assert that FastAPI routes to a function - which is not a risk worth
covering - while missing the failures that actually happen: a response shape
that does not match the schema, an empty rail, a broken explanation.

Redis is faked (`fakeredis`) because its behaviour is fully specified and a
fake is indistinguishable for our purposes. Postgres is *not* faked: the event
sink falls back to an in-memory sink when the database is absent, which is a
real production code path worth exercising.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Must be set before `app.core.config` is imported anywhere.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret-never-used-in-any-deployment")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def artifacts_available() -> bool:
    return (REPO_ROOT / "ml" / "artifacts" / "popularity.joblib").exists()


@pytest.fixture(scope="session")
def app(artifacts_available):
    """The real application, built once for the whole session.

    Session-scoped because start-up loads model artefacts from disk; doing that
    per test would make the suite minutes long instead of seconds.
    """
    from app.main import create_app

    return create_app()


@pytest.fixture(scope="session")
def client(app):
    from fastapi.testclient import TestClient

    # The context manager is what runs the lifespan handler; without it the
    # engine, sink and services are never built and every request 503s.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_token():
    from app.core.security import create_token
    from app.models.enums import UserRole

    return create_token(user_id=1, public_id="test-admin", role=UserRole.ADMIN)


@pytest.fixture
def analyst_token():
    from app.core.security import create_token
    from app.models.enums import UserRole

    return create_token(user_id=2, public_id="test-analyst", role=UserRole.ANALYST)


@pytest.fixture
def customer_token():
    from app.core.security import create_token
    from app.models.enums import UserRole

    return create_token(user_id=42, public_id="test-customer", role=UserRole.CUSTOMER)


@pytest.fixture
def auth(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def fake_redis():
    """An in-process Redis substitute."""
    import fakeredis

    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def counters(fake_redis):
    from app.cache.counters import RealTimeCounters

    return RealTimeCounters(fake_redis)
