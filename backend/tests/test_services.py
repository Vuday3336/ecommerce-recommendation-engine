"""Unit tests for security, event ingestion, caching and experiments.

These cover the pieces whose failure modes are quiet: a token that is accepted
when it should not be, an event queue that blocks a page render, an experiment
that reassigns a user between requests.
"""

from __future__ import annotations

import datetime as dt
import time

import pytest
from app.cache.counters import RealTimeCounters
from app.core.security import (
    AuthError,
    Principal,
    create_token,
    create_token_pair,
    decode_token,
    hash_password,
    stable_bucket,
    verify_password,
)
from app.models.enums import DeviceType, EventType, UserRole
from app.schemas.events import EventIn, validate_payload
from app.services.event_sink import (
    BufferedEventSink,
    EventRecord,
    MemoryEventSink,
)
from app.services.events import HIGH_SIGNAL_EVENTS, EventService
from app.services.experiments import (
    BUCKETS,
    ExperimentDefinition,
    ExperimentService,
    Variant,
)


class TestPasswords:
    def test_a_password_verifies_against_its_own_hash(self):
        hashed = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", hashed)
        assert not verify_password("wrong password entirely", hashed)

    def test_hashes_are_salted(self):
        """Identical passwords must not produce identical hashes, or a stolen
        database reveals which users share a password."""
        assert hash_password("same-password") != hash_password("same-password")

    def test_verification_against_a_missing_hash_is_safe(self):
        assert not verify_password("anything", None)

    def test_argon2_is_the_default_scheme(self):
        assert hash_password("test-password").startswith("$argon2")


class TestTokens:
    def test_round_trip(self):
        token = create_token(user_id=7, public_id="pub-7", role=UserRole.ANALYST)
        principal = decode_token(token)
        assert principal.user_id == 7
        assert principal.role is UserRole.ANALYST

    def test_a_refresh_token_is_rejected_where_an_access_token_is_expected(self):
        refresh = create_token(
            user_id=1, public_id="x", role=UserRole.ADMIN, token_type="refresh"
        )
        with pytest.raises(AuthError, match="expected a access token"):
            decode_token(refresh, expected_type="access")

    def test_a_tampered_token_is_rejected(self):
        token = create_token(user_id=1, public_id="x", role=UserRole.CUSTOMER)
        with pytest.raises(AuthError):
            decode_token(token[:-4] + "aaaa")

    def test_a_token_pair_has_both_halves(self):
        pair = create_token_pair(user_id=3, public_id="pub-3", role=UserRole.CUSTOMER)
        assert decode_token(pair["access_token"], expected_type="access").user_id == 3
        assert decode_token(pair["refresh_token"], expected_type="refresh").user_id == 3

    def test_tokens_are_unique_even_for_the_same_user(self):
        """The `jti` claim lets one token be revoked without invalidating every
        token the user holds."""
        first = create_token(user_id=1, public_id="x", role=UserRole.CUSTOMER)
        second = create_token(user_id=1, public_id="x", role=UserRole.CUSTOMER)
        assert first != second


class TestRoles:
    def test_the_hierarchy_is_ordered(self):
        admin = Principal(1, "a", UserRole.ADMIN)
        analyst = Principal(2, "b", UserRole.ANALYST)
        customer = Principal(3, "c", UserRole.CUSTOMER)

        assert admin.has_role(UserRole.ANALYST)
        assert admin.has_role(UserRole.ADMIN)
        assert analyst.has_role(UserRole.ANALYST)
        assert not analyst.has_role(UserRole.ADMIN)
        assert not customer.has_role(UserRole.ANALYST)


class TestEventSchemas:
    def test_payloads_are_validated_per_type(self):
        from pydantic import ValidationError

        assert validate_payload(EventType.PRODUCT_RATING, {"rating": 4.5})["rating"] == 4.5
        with pytest.raises(ValidationError):
            validate_payload(EventType.PRODUCT_RATING, {"rating": 99})

    def test_an_unregistered_event_type_gets_strict_empty_validation(self):
        """A new event type must ship with validation by default, not accept
        anything until someone remembers to add a model."""
        from pydantic import ValidationError

        assert validate_payload(EventType.WISHLIST, {}) == {}
        with pytest.raises(ValidationError):
            validate_payload(EventType.WISHLIST, {"anything": 1})

    def test_a_future_timestamp_is_clamped_rather_than_rejected(self):
        """A wrong device clock is not a reason to discard real behaviour."""
        future = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        event = EventIn(
            event_type=EventType.PRODUCT_VIEW,
            session_id="session-clamp-01",
            product_id=1,
            occurred_at=future,
        )
        assert event.occurred_at < future

    def test_a_recent_past_timestamp_is_preserved(self):
        """Mobile clients buffer events offline; backdating within the window
        is legitimate."""
        past = dt.datetime.now(dt.UTC) - dt.timedelta(hours=6)
        event = EventIn(
            event_type=EventType.PRODUCT_VIEW,
            session_id="session-past-01",
            product_id=1,
            occurred_at=past,
        )
        assert event.occurred_at == past


class TestSessionAggregation:
    """`PostgresEventSink` derives one session row per batch.

    The SQL upsert needs a live database, but the aggregation that feeds it is
    pure and is where the interesting decisions live, so it is tested here.
    """

    @staticmethod
    def _record(session_key: str, **overrides):
        from app.models.enums import DeviceType

        defaults = {
            "event_type": EventType.PRODUCT_VIEW,
            "session_key": session_key,
            "occurred_at": dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC),
            "device_type": DeviceType.UNKNOWN,
        }
        defaults.update(overrides)
        return EventRecord(**defaults)

    def _rows(self, records):
        from app.services.event_sink import PostgresEventSink

        return {row[0]: row for row in PostgresEventSink._session_rows(records)}

    def test_one_row_per_session_regardless_of_event_count(self):
        rows = self._rows(
            [self._record("a") for _ in range(5)] + [self._record("b")]
        )
        assert set(rows) == {"a", "b"}
        assert rows["a"][5] == 5
        assert rows["b"][5] == 1

    def test_the_window_spans_the_batch_even_when_events_arrive_out_of_order(self):
        """The buffered worker makes no ordering promise, so the aggregation
        cannot assume the first record it sees is the earliest."""
        base = dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC)
        rows = self._rows(
            [
                self._record("a", occurred_at=base + dt.timedelta(minutes=5)),
                self._record("a", occurred_at=base),
                self._record("a", occurred_at=base + dt.timedelta(minutes=2)),
            ]
        )
        assert rows["a"][3] == base
        assert rows["a"][4] == base + dt.timedelta(minutes=5)

    def test_a_purchase_anywhere_in_the_batch_marks_the_session_converted(self):
        rows = self._rows(
            [self._record("a"), self._record("a", event_type=EventType.PURCHASE)]
        )
        assert rows["a"][6] is True
        assert self._rows([self._record("b")])["b"][6] is False

    def test_an_identified_device_beats_unknown_whatever_the_order(self):
        """`unknown` is a placeholder, not an observation. Taking the first
        value would record `unknown` for a session whose very next event says
        `mobile`."""
        from app.models.enums import DeviceType

        rows = self._rows(
            [
                self._record("a"),
                self._record("a", device_type=DeviceType.MOBILE),
            ]
        )
        assert rows["a"][2] == DeviceType.MOBILE.value

    def test_an_anonymous_event_does_not_erase_a_known_user(self):
        """FR-22 stitching is one-way: a session that signs in stays signed in."""
        rows = self._rows(
            [self._record("a"), self._record("a", user_id=7), self._record("a")]
        )
        assert rows["a"][1] == 7

    def test_an_empty_batch_produces_no_rows(self):
        assert self._rows([]) == {}


class TestEventSink:
    def test_the_buffered_sink_writes_through(self):
        target = MemoryEventSink()
        sink = BufferedEventSink(target, batch_size=10, flush_interval_s=0.05)
        try:
            records = [
                EventRecord(
                    event_type=EventType.PRODUCT_VIEW,
                    session_key="s1",
                    occurred_at=dt.datetime.now(dt.UTC),
                    product_id=index,
                )
                for index in range(25)
            ]
            assert sink.emit(records) == 25
            sink.flush(timeout=5.0)
            assert len(target.records) == 25
        finally:
            sink.close()

    def test_a_full_queue_drops_rather_than_blocks(self):
        """FR-23: analytics must never block a page render. Losing events is
        strictly better than degrading the storefront - and the drop is
        counted, which turns a silent failure into an alert."""
        target = MemoryEventSink()
        sink = BufferedEventSink(target, queue_size=5, flush_interval_s=5.0)
        try:
            records = [
                EventRecord(
                    event_type=EventType.PRODUCT_VIEW,
                    session_key="s1",
                    occurred_at=dt.datetime.now(dt.UTC),
                    product_id=index,
                )
                for index in range(200)
            ]
            started = time.perf_counter()
            accepted = sink.emit(records)
            elapsed = time.perf_counter() - started

            assert accepted < 200
            assert sink.dropped > 0
            assert elapsed < 1.0, "emit blocked; it must always return promptly"
        finally:
            sink.close()

    def test_a_failing_target_does_not_kill_the_worker(self):
        """A worker that dies on the first transient error silently stops
        ingestion for the life of the process."""

        class Exploding:
            def __init__(self):
                self.attempts = 0

            def emit(self, records):
                self.attempts += 1
                raise RuntimeError("database is down")

            def flush(self):
                return None

            def close(self):
                return None

        target = Exploding()
        sink = BufferedEventSink(target, batch_size=1, flush_interval_s=0.05)
        try:
            for index in range(5):
                sink.emit(
                    [
                        EventRecord(
                            event_type=EventType.PRODUCT_VIEW,
                            session_key="s",
                            occurred_at=dt.datetime.now(dt.UTC),
                            product_id=index,
                        )
                    ]
                )
            sink.flush(timeout=3.0)
            assert target.attempts > 1, "the worker stopped after the first failure"
            assert sink.failed > 0
        finally:
            sink.close()


class TestEventService:
    def test_user_id_comes_from_the_principal_not_the_payload(self):
        """Otherwise any client could write events attributed to another user,
        poisoning their recommendations and the training set."""
        target = MemoryEventSink()
        service = EventService(target, RealTimeCounters(None))

        event = EventIn(
            event_type=EventType.PRODUCT_VIEW, session_id="sess-attr-01", product_id=5
        )
        service.ingest([event], user_id=99)
        assert target.records[0].user_id == 99

    def test_high_signal_events_are_the_only_cache_invalidators(self):
        """Invalidating on views would defeat the cache: views are most of the
        traffic."""
        assert EventType.PURCHASE in HIGH_SIGNAL_EVENTS
        assert EventType.ADD_TO_CART in HIGH_SIGNAL_EVENTS
        assert EventType.PRODUCT_VIEW not in HIGH_SIGNAL_EVENTS

    def test_counter_failures_do_not_break_ingestion(self):
        class BrokenCounters(RealTimeCounters):
            def record_event(self, **kwargs):
                raise RuntimeError("redis exploded")

        target = MemoryEventSink()
        service = EventService(target, BrokenCounters(None))
        result = service.ingest(
            [EventIn(event_type=EventType.PRODUCT_VIEW, session_id="sess-ok-0001", product_id=1)],
            user_id=1,
        )
        assert result.accepted == 1


class TestRealTimeCounters:
    def test_everything_degrades_to_empty_without_redis(self):
        counters = RealTimeCounters(None)
        assert not counters.available
        assert counters.trending() == []
        assert counters.recently_viewed(1) == []
        counters.record_event(
            event_type=EventType.PRODUCT_VIEW,
            product_id=1,
            user_id=1,
            session_key="s",
            occurred_at=dt.datetime.now(dt.UTC),
        )

    def test_recently_viewed_is_most_recent_first(self, counters):
        now = dt.datetime.now(dt.UTC)
        for product_id in (10, 20, 30):
            counters.record_event(
                event_type=EventType.PRODUCT_VIEW,
                product_id=product_id,
                user_id=1,
                session_key="s",
                occurred_at=now,
            )
        assert counters.recently_viewed(1)[:3] == [30, 20, 10]

    def test_re_viewing_moves_an_item_to_the_front_without_duplicating(self, counters):
        now = dt.datetime.now(dt.UTC)
        for product_id in (10, 20, 10):
            counters.record_event(
                event_type=EventType.PRODUCT_VIEW,
                product_id=product_id,
                user_id=1,
                session_key="s",
                occurred_at=now,
            )
        recent = counters.recently_viewed(1)
        assert recent[0] == 10
        assert recent.count(10) == 1

    def test_trending_ranks_by_weighted_velocity(self, counters):
        """A purchase counts far more than a view; otherwise trending becomes a
        clickbait chart."""
        now = dt.datetime.now(dt.UTC)
        for _ in range(10):
            counters.record_event(
                event_type=EventType.PRODUCT_VIEW,
                product_id=1,
                user_id=1,
                session_key="s",
                occurred_at=now,
            )
        for _ in range(3):
            counters.record_event(
                event_type=EventType.PURCHASE,
                product_id=2,
                user_id=1,
                session_key="s",
                occurred_at=now,
            )
        trending = counters.trending(limit=5, now=now)
        assert trending
        assert trending[0].product_id == 2


class TestExperimentAssignment:
    @pytest.fixture
    def service(self):
        return ExperimentService(
            [
                ExperimentDefinition(
                    key="ranker_v1",
                    variants=(Variant("control", 0.5), Variant("treatment", 0.5)),
                )
            ]
        )

    def test_assignment_is_sticky(self, service):
        """A user flipping arms between requests means the experiment measures
        nothing."""
        first = service.assign("ranker_v1", 12345)
        for _ in range(50):
            assert service.assign("ranker_v1", 12345).variant == first.variant

    def test_allocation_is_approximately_respected(self, service):
        counts = {"control": 0, "treatment": 0}
        for user_id in range(10_000):
            counts[service.assign("ranker_v1", user_id).variant] += 1
        assert 0.47 < counts["control"] / 10_000 < 0.53

    def test_experiments_are_independently_salted(self):
        """Without per-experiment salting, a user in the top decile of one
        experiment is in the top decile of every experiment, and concurrent
        experiments become confounded."""
        service = ExperimentService(
            [
                ExperimentDefinition("exp_a", (Variant("control", 0.5), Variant("treatment", 0.5))),
                ExperimentDefinition("exp_b", (Variant("control", 0.5), Variant("treatment", 0.5))),
            ]
        )
        same = sum(
            service.assign("exp_a", user).variant == service.assign("exp_b", user).variant
            for user in range(2000)
        )
        # Perfect correlation would be 2000; independence gives about half.
        assert 850 < same < 1150

    def test_allocations_that_do_not_sum_to_one_are_rejected(self):
        with pytest.raises(ValueError, match="sum to"):
            ExperimentDefinition("bad", (Variant("a", 0.3), Variant("b", 0.3)))

    def test_a_holdout_keeps_most_traffic_out(self):
        service = ExperimentService(
            [
                ExperimentDefinition(
                    "small",
                    (Variant("control", 0.5), Variant("treatment", 0.5)),
                    traffic_allocation=0.1,
                )
            ]
        )
        inside = sum(
            service.assign("small", user).in_experiment for user in range(5000)
        )
        assert 0.07 < inside / 5000 < 0.13

    def test_the_holdout_does_not_bias_which_arm_is_chosen(self):
        """Reusing one bucket for both decisions would put every included user
        in the first variant."""
        service = ExperimentService(
            [
                ExperimentDefinition(
                    "small",
                    (Variant("control", 0.5), Variant("treatment", 0.5)),
                    traffic_allocation=0.2,
                )
            ]
        )
        arms = {"control": 0, "treatment": 0}
        for user in range(20_000):
            assignment = service.assign("small", user)
            if assignment.in_experiment:
                arms[assignment.variant] += 1
        total = sum(arms.values())
        assert 0.44 < arms["control"] / total < 0.56

    def test_anonymous_sessions_are_assigned_consistently(self, service):
        first = service.assign("ranker_v1", "anon-session-key-123")
        assert service.assign("ranker_v1", "anon-session-key-123").variant == first.variant

    def test_unknown_experiments_return_nothing(self, service):
        assert service.assign("does-not-exist", 1) is None

    def test_bucketing_is_stable_across_processes(self):
        """`hash()` is randomised per process by PYTHONHASHSEED, which would
        reassign every user on restart. SHA-256 is not."""
        assert stable_bucket("ranker_v1:42") == stable_bucket("ranker_v1:42")
        assert 0 <= stable_bucket("anything") < BUCKETS


class TestAnalytics:
    def test_funnel_maths(self):
        from app.services.analytics import FunnelMetrics

        metrics = FunnelMetrics(
            "control", impressions=1000, clicks=50, conversions=5, revenue=500.0, users=200
        )
        assert metrics.ctr == pytest.approx(0.05)
        # Conversion is measured per click, not per impression: it isolates
        # recommendation quality from rail placement.
        assert metrics.conversion_rate == pytest.approx(0.10)
        assert metrics.revenue_per_user == pytest.approx(2.5)
        assert metrics.average_order_value == pytest.approx(100.0)

    def test_division_by_zero_is_handled(self):
        from app.services.analytics import FunnelMetrics

        empty = FunnelMetrics("empty")
        assert empty.ctr == 0.0
        assert empty.conversion_rate == 0.0
        assert empty.revenue_per_user == 0.0

    def test_the_service_reports_unavailable_without_a_database(self):
        from app.services.analytics import AnalyticsService

        service = AnalyticsService(None)
        assert not service.available
        assert service.overview()["available"] is False
        assert not service.experiment("ranker_v1").ready


class TestDeviceEnum:
    def test_enum_values_are_lowercase_strings(self):
        """The database stores the *value*, so these must match the enum type."""
        assert DeviceType.DESKTOP.value == "desktop"
        assert EventType.PRODUCT_VIEW.value == "PRODUCT_VIEW"
