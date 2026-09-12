"""Tests that need a live PostgreSQL.

These verify the things only a real database can verify: that the constraints
actually reject bad data, that partitioning routes rows correctly, that the
indexes are used by the planner, and that the auth flow round-trips through
real tables.

Skipped automatically when no database is reachable, so the suite still runs on
a machine without one.

    python scripts/local_postgres.py start
    cd backend && alembic upgrade head
    python scripts/seed_database.py --truncate
    pytest tests/integration/test_database.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "backend", REPO_ROOT / "ml"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("JWT_SECRET_KEY", "db-integration-test-secret")
os.environ.setdefault("LOG_LEVEL", "ERROR")

PREFIX = "/api/v1"


def _database_available() -> bool:
    try:
        from app.db.session import check_connection

        return check_connection()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _database_available(), reason="no PostgreSQL reachable"
)


def _database_seeded() -> bool:
    """Does the database contain the synthetic dataset?

    A migrated-but-empty database is a legitimate state - it is what CI gets
    after `alembic upgrade head`, and what a developer gets before their first
    seed. Tests that assert row counts or sequence positions are asserting a
    *precondition* they never checked, so on an empty database they failed with
    "expected 10,000 users, got 0", which reads as data loss rather than as
    "nothing has been loaded yet".

    Checking `products` rather than `user_events`: the event log is partitioned
    and also written by the API during this very suite, so it is non-empty even
    when nothing has been seeded.
    """
    try:
        from app.db.session import session_scope
        from sqlalchemy import text

        with session_scope() as db:
            return bool(db.execute(text("SELECT 1 FROM products LIMIT 1")).first())
    except Exception:
        return False


#: Applied to tests that read the seeded dataset rather than writing their own.
requires_seed = pytest.mark.skipif(
    not _database_seeded(),
    reason="database is migrated but not seeded - run scripts/seed_database.py",
)


@pytest.fixture(scope="module")
def session():
    from app.db.session import session_scope

    with session_scope() as db:
        yield db


@pytest.fixture(scope="module")
def client():
    from app.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as test_client:
        yield test_client


class TestSchema:
    def test_every_expected_table_exists(self, session):
        from sqlalchemy import text

        rows = session.execute(
            text(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """
            )
        ).scalars().all()
        names = set(rows)
        for expected in (
            "users", "products", "categories", "orders", "order_items",
            "user_events", "user_product_interactions", "recommendations",
            "user_features", "product_features", "product_embeddings",
            "model_versions", "experiments", "experiment_assignments",
        ):
            assert expected in names, f"missing table {expected}"

    def test_pgvector_is_installed_and_usable(self, session):
        from sqlalchemy import text

        # Not just "is the extension listed" - actually compute a distance,
        # because a registered extension with a broken library still lists.
        distance = session.execute(
            text("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector")
        ).scalar()
        assert distance == pytest.approx(1.0)

    def test_the_hnsw_index_exists(self, session):
        from sqlalchemy import text

        definition = session.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'ix_product_embeddings_hnsw_cosine'"
            )
        ).scalar()
        assert definition is not None
        assert "hnsw" in definition
        assert "vector_cosine_ops" in definition


class TestConstraints:
    """The constraints must actually reject bad data, not merely be declared."""

    def test_product_scoped_events_require_a_product(self, session):
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            session.execute(
                text(
                    """
                    INSERT INTO user_events
                        (occurred_at, event_type, session_key, source, device_type)
                    VALUES (now(), 'PURCHASE', 'test-constraint-01', 'test', 'desktop')
                    """
                )
            )
        session.rollback()

    def test_prices_must_be_positive(self, session):
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            session.execute(
                text(
                    """
                    INSERT INTO products
                        (sku, name, description, category_id, brand_id, price, cost,
                         price_band, released_at)
                    VALUES ('TEST-NEG', 'Negative', '', 3, 1, -5.00, 1.00,
                            'budget', now())
                    """
                )
            )
        session.rollback()

    def test_only_one_production_model_version_per_name(self, session):
        """A partial unique index, so two concurrent promotions fail loudly
        rather than leaving two live models."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        session.execute(
            text(
                """
                INSERT INTO model_versions (name, version, kind, stage)
                VALUES ('test-model', 'v1', 'ranker', 'production'),
                       ('test-model', 'v2', 'ranker', 'staging')
                """
            )
        )
        session.commit()
        try:
            with pytest.raises(IntegrityError):
                session.execute(
                    text(
                        "UPDATE model_versions SET stage = 'production' "
                        "WHERE name = 'test-model' AND version = 'v2'"
                    )
                )
            session.rollback()
        finally:
            session.execute(
                text("DELETE FROM model_versions WHERE name = 'test-model'")
            )
            session.commit()

    def test_a_referenced_product_cannot_be_hard_deleted(self, session):
        """ON DELETE RESTRICT keeps historical orders joinable."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        product_id = session.execute(
            text("SELECT product_id FROM order_items LIMIT 1")
        ).scalar()
        if product_id is None:
            pytest.skip("no order items seeded")

        with pytest.raises(IntegrityError):
            session.execute(
                text("DELETE FROM products WHERE id = :pid"), {"pid": product_id}
            )
        session.rollback()


class TestPartitioning:
    @requires_seed
    def test_events_are_routed_into_monthly_partitions(self, session):
        from sqlalchemy import text

        rows = session.execute(
            text(
                """
                SELECT c.relname, pg_catalog.pg_table_size(c.oid) AS bytes
                FROM pg_inherits i
                JOIN pg_class c ON c.oid = i.inhrelid
                JOIN pg_class p ON p.oid = i.inhparent
                WHERE p.relname = 'user_events'
                ORDER BY c.relname
                """
            )
        ).all()
        populated = [name for name, size in rows if size > 8192]
        assert len(rows) >= 12, "monthly partitions were not created"

        # Derive the expectation from the data rather than hardcoding it. The
        # invariant is "every month that has events has a populated partition",
        # which holds at any scale; `>= 5` silently encoded "180 days of data"
        # and failed on a 90-day dataset that was routed perfectly correctly.
        span = session.execute(
            text("SELECT min(occurred_at), max(occurred_at) FROM user_events")
        ).first()
        months = {
            (span[0].year, span[0].month),
            (span[1].year, span[1].month),
        }
        cursor = dt.datetime(span[0].year, span[0].month, 1, tzinfo=span[0].tzinfo)
        while cursor < span[1]:
            months.add((cursor.year, cursor.month))
            cursor = (cursor.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

        assert len(populated) >= len(months), (
            f"events span {len(months)} months but only {len(populated)} "
            f"partitions hold data: {populated}"
        )

    def test_the_default_partition_is_empty(self, session):
        """A row here means a month range is missing, and PostgreSQL will then
        refuse to attach a partition for that month until it is moved."""
        from sqlalchemy import text

        assert session.execute(text("SELECT count(*) FROM user_events_default")).scalar() == 0

    def test_a_user_query_prunes_partitions(self, session):
        from sqlalchemy import text

        plan = "\n".join(
            row[0]
            for row in session.execute(
                text(
                    """
                    EXPLAIN SELECT product_id FROM user_events
                    WHERE user_id = 42 ORDER BY occurred_at DESC LIMIT 50
                    """
                )
            ).all()
        )
        # The composite (user_id, occurred_at DESC) index must be chosen. A
        # sequential scan here would mean the index design is not working.
        assert "Index Scan" in plan, plan
        assert "Seq Scan on user_events" not in plan, plan

    def test_the_maintenance_function_exists(self, session):
        from sqlalchemy import text

        exists = session.execute(
            text(
                "SELECT count(*) FROM pg_proc WHERE proname = "
                "'ensure_user_events_partition'"
            )
        ).scalar()
        assert exists == 1


class TestIndexUsage:
    def test_category_candidate_query_uses_the_partial_index(self, session):
        from sqlalchemy import text

        plan = "\n".join(
            row[0]
            for row in session.execute(
                text(
                    """
                    EXPLAIN SELECT id, price FROM products
                    WHERE category_id = 12 AND is_active AND stock_quantity > 0
                    ORDER BY price LIMIT 50
                    """
                )
            ).all()
        )
        assert "ix_products_active_category" in plan, plan

    def test_co_purchase_query_uses_an_index(self, session):
        from sqlalchemy import text

        plan = "\n".join(
            row[0]
            for row in session.execute(
                text(
                    """
                    EXPLAIN SELECT b.product_id, count(*) FROM order_items a
                    JOIN order_items b
                      ON b.order_id = a.order_id AND b.product_id <> a.product_id
                    WHERE a.product_id = 100
                    GROUP BY b.product_id ORDER BY count(*) DESC LIMIT 10
                    """
                )
            ).all()
        )
        assert "Index Scan" in plan, plan


@requires_seed
class TestSeededData:
    def test_the_expected_volumes_are_present(self, session):
        from sqlalchemy import text

        # Checked against the generator's own manifest instead of full-scale
        # constants. This is a stronger assertion, not a weaker one: "every row
        # the generator produced reached the database" catches a partial load,
        # which `>= 10_000` would happily pass with 90% of the data missing. It
        # also holds at any scale, so CI can seed a small dataset and still run
        # this test rather than skipping it.
        manifest = json.loads(
            (REPO_ROOT / "data" / "synthetic" / "manifest.json").read_text(encoding="utf-8")
        )
        expected = manifest["row_counts"]

        def count(table: str) -> int:
            return session.execute(text(f"SELECT count(*) FROM {table}")).scalar()

        # Tables nothing else writes must match the manifest *exactly*. Equality
        # is the point: `>=` would pass a load that silently dropped 90% of the
        # rows, which is the failure this test exists to catch.
        for table in ("products", "orders", "order_items"):
            actual = count(table)
            assert actual == expected[table], (
                f"{table}: database has {actual:,}, the manifest declares "
                f"{expected[table]:,} - the load was partial"
            )

        # These three grow during the run, so they can only be bounded below.
        # `users` gains the accounts the auth tests register; `user_events`
        # gains the events posted through the API; `user_sessions` gains the
        # sessions the ingestion path now creates for those events. Asserting
        # equality here would make the suite fail depending on which tests ran
        # first, which is worse than a weaker assertion.
        for table, key in (
            ("users", "users"),
            ("user_events", "events"),
            ("user_sessions", "user_sessions"),
        ):
            actual = count(table)
            assert actual >= expected[key], (
                f"{table} has {actual:,}, fewer than the {expected[key]:,} seeded"
            )

    def test_no_orphan_event_references(self, session):
        """`user_events` has no foreign keys by design (throughput), so this is
        the compensating control that makes that trade defensible."""
        from sqlalchemy import text

        orphans = session.execute(
            text(
                """
                SELECT count(*) FROM user_events e
                LEFT JOIN products p ON p.id = e.product_id
                WHERE e.product_id IS NOT NULL AND p.id IS NULL
                """
            )
        ).scalar()
        assert orphans == 0

    def test_identity_sequences_were_advanced_after_seeding(self, session):
        """Without this, the first application insert collides with a seeded
        row - a confusing and very common seeding bug."""
        from sqlalchemy import text

        for table in ("products", "users", "orders"):
            # `pg_get_serial_sequence` returns the sequence *name*, which is not
            # itself selectable - it has to be cast to regclass and read through
            # `pg_sequence_last_value`.
            last_value = session.execute(
                text(
                    "SELECT pg_sequence_last_value("
                    f"    pg_get_serial_sequence('{table}', 'id')::regclass)"
                )
            ).scalar()
            max_id = session.execute(text(f"SELECT max(id) FROM {table}")).scalar()
            assert last_value is not None, f"{table} sequence was never advanced"
            assert last_value >= max_id, (
                f"{table} sequence is at {last_value} but max(id) is {max_id}; "
                "the first application insert would collide with a seeded row"
            )


class TestAuthRoundTrip:
    def test_register_login_and_me(self, client):
        email = f"test-{uuid.uuid4().hex[:12]}@example.com"
        password = "a-sufficiently-long-password"

        registered = client.post(
            f"{PREFIX}/auth/register",
            json={"email": email, "password": password, "full_name": "Test User"},
        )
        assert registered.status_code == 201, registered.text
        user_id = registered.json()["user_id"]

        logged_in = client.post(
            f"{PREFIX}/auth/login", json={"email": email, "password": password}
        )
        assert logged_in.status_code == 200
        token = logged_in.json()["access_token"]

        me = client.get(f"{PREFIX}/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["user_id"] == user_id

    def test_a_wrong_password_is_rejected_with_a_generic_message(self, client):
        email = f"test-{uuid.uuid4().hex[:12]}@example.com"
        client.post(
            f"{PREFIX}/auth/register",
            json={"email": email, "password": "the-real-password", "full_name": "T"},
        )
        response = client.post(
            f"{PREFIX}/auth/login", json={"email": email, "password": "wrong-password"}
        )
        assert response.status_code == 401
        # The same message for a wrong password and an unknown account, so the
        # response does not disclose which accounts exist.
        assert response.json()["detail"] == "invalid email or password"

    def test_an_unknown_account_gives_the_same_message(self, client):
        response = client.post(
            f"{PREFIX}/auth/login",
            json={"email": "nobody-here@example.com", "password": "some-password"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid email or password"

    def test_duplicate_registration_is_rejected(self, client):
        email = f"test-{uuid.uuid4().hex[:12]}@example.com"
        payload = {"email": email, "password": "a-long-enough-password", "full_name": "T"}
        assert client.post(f"{PREFIX}/auth/register", json=payload).status_code == 201
        assert client.post(f"{PREFIX}/auth/register", json=payload).status_code == 409


class TestEventPersistence:
    def test_an_event_posted_to_the_api_reaches_the_database(self, client, session):
        from sqlalchemy import text

        session_key = f"persist-{uuid.uuid4().hex[:16]}"
        response = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_VIEW",
                "session_id": session_key,
                "product_id": 100,
                "source": "pdp",
                "metadata": {"dwell_ms": 1234},
            },
        )
        assert response.status_code == 202

        # The sink is buffered, so the write is not synchronous with the
        # response - that is the point of FR-23.
        client.app.state.event_sink.flush(timeout=10.0)

        session.commit()
        row = session.execute(
            text(
                "SELECT event_type::text, product_id, source, event_metadata "
                "FROM user_events WHERE session_key = :key"
            ),
            {"key": session_key},
        ).first()
        assert row is not None, "the event never reached the database"
        assert row[0] == "PRODUCT_VIEW"
        assert row[1] == 100
        assert row[3]["dwell_ms"] == 1234

    def test_the_live_path_registers_the_session_it_writes_events_for(
        self, client, session
    ):
        """`user_sessions` must be populated by ingestion, not only by the seeder.

        This is a regression test for a bug that only a live database could
        expose. `user_events` carries no foreign keys by design, so nothing
        rejected an event whose `session_key` matched no session - and nothing
        created the session either. In production every real visitor would have
        been orphaned from the session table, and session-scoped cold-start
        retrieval would have had nothing to read for exactly the new users it
        exists to serve.
        """
        from sqlalchemy import text

        session_key = f"session-reg-{uuid.uuid4().hex[:12]}"
        for event_type, product_id in (
            ("PRODUCT_VIEW", 100),
            ("PRODUCT_VIEW", 101),
            ("ADD_TO_CART", 101),
            ("PURCHASE", 101),
        ):
            assert (
                client.post(
                    f"{PREFIX}/events",
                    json={
                        "event_type": event_type,
                        "session_id": session_key,
                        "product_id": product_id,
                        "device_type": "mobile",
                    },
                ).status_code
                == 202
            )
        client.app.state.event_sink.flush(timeout=10.0)
        session.commit()

        row = session.execute(
            text(
                "SELECT event_count, converted, device_type::text, "
                "       started_at, ended_at "
                "FROM user_sessions WHERE session_key = :key"
            ),
            {"key": session_key},
        ).first()
        assert row is not None, "ingestion did not create the session row"
        assert row[0] == 4, f"expected 4 events counted, got {row[0]}"
        assert row[1] is True, "a PURCHASE in the batch must mark the session converted"
        assert row[2] == "mobile", "the device type observed on the events was lost"
        assert row[4] >= row[3], "the session window is inverted"

        orphans = session.execute(
            text(
                "SELECT count(*) FROM user_events e "
                "LEFT JOIN user_sessions s ON s.session_key = e.session_key "
                "WHERE e.session_key = :key AND s.id IS NULL"
            ),
            {"key": session_key},
        ).scalar()
        assert orphans == 0

    def test_a_session_spanning_batches_is_extended_not_duplicated(
        self, client, session
    ):
        """The upsert must accumulate. A session outlives any single batch."""
        from sqlalchemy import text

        session_key = f"session-span-{uuid.uuid4().hex[:12]}"
        for _ in range(2):
            client.post(
                f"{PREFIX}/events",
                json={
                    "event_type": "PRODUCT_VIEW",
                    "session_id": session_key,
                    "product_id": 100,
                },
            )
            # Flushing between posts forces two separate write batches, which
            # is what a real session does over its lifetime.
            client.app.state.event_sink.flush(timeout=10.0)
        session.commit()

        rows = session.execute(
            text("SELECT event_count FROM user_sessions WHERE session_key = :key"),
            {"key": session_key},
        ).all()
        assert len(rows) == 1, "the session was duplicated instead of updated"
        assert rows[0][0] == 2, f"the second batch did not accumulate: {rows[0][0]}"

    def test_the_event_lands_in_the_right_monthly_partition(self, client, session):
        from sqlalchemy import text

        session_key = f"partition-{uuid.uuid4().hex[:16]}"
        client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "PRODUCT_CLICK",
                "session_id": session_key,
                "product_id": 101,
            },
        )
        client.app.state.event_sink.flush(timeout=10.0)
        session.commit()

        partition = session.execute(
            text(
                "SELECT tableoid::regclass::text FROM user_events "
                "WHERE session_key = :key"
            ),
            {"key": session_key},
        ).scalar()
        assert partition is not None
        expected = f"user_events_{dt.datetime.now(dt.UTC):%Y_%m}"
        assert partition == expected, f"landed in {partition}, expected {expected}"


class TestAnalyticsAgainstTheDatabase:
    def test_analytics_reports_itself_available(self, client):
        from app.core.security import create_token
        from app.models.enums import UserRole

        token = create_token(user_id=1, public_id="t", role=UserRole.ANALYST)
        payload = client.get(
            f"{PREFIX}/admin/analytics", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert payload["available"] is True
        assert "total" in payload
        assert "fallback_rate" in payload

    def test_experiment_results_are_computable(self, client):
        from app.core.security import create_token
        from app.models.enums import UserRole

        token = create_token(user_id=1, public_id="t", role=UserRole.ANALYST)
        payload = client.get(
            f"{PREFIX}/admin/experiments/ranker_v1/results",
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        # No served recommendations logged yet, so the readout must say it is
        # not ready rather than reporting a fabricated result.
        assert payload["ready"] is False
        assert payload["notes"]
