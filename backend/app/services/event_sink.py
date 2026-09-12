"""Event sinks - the seam that lets ingestion move to Kafka later (ADR-009).

The API never writes an event directly. It hands a normalised `EventRecord` to
an `EventSink`, which is free to buffer, batch, drop or forward it. Today the
production sink is `PostgresEventSink` wrapped in `BufferedEventSink`; the
`KafkaEventSink` described in ADR-009 implements the same protocol and would
replace it without the API changing.

The buffering behaviour is the part that matters for FR-23: a page render must
never wait on, or fail because of, analytics. So the request thread does a
non-blocking put onto a bounded queue and returns. If the queue is full, events
are dropped and counted - losing analytics is strictly better than degrading
the storefront, and a visible drop counter is what turns that from a silent
failure into an alert.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import Engine

from app.models.enums import DeviceType, EventType

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 20_000
DEFAULT_BATCH_SIZE = 500
DEFAULT_FLUSH_INTERVAL_S = 1.0


@dataclass(slots=True)
class EventRecord:
    """A validated, enriched event ready to be written.

    This is intentionally a plain dataclass rather than an ORM object: sinks
    must be able to batch thousands of these without the session bookkeeping,
    identity map and flush machinery an ORM instance carries.
    """

    event_type: EventType
    session_key: str
    occurred_at: dt.datetime
    user_id: int | None = None
    product_id: int | None = None
    source: str = "unknown"
    device_type: DeviceType = DeviceType.UNKNOWN
    recommendation_id: int | None = None
    event_metadata: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> tuple[Any, ...]:
        """Column order matches `PostgresEventSink.COLUMNS`."""
        return (
            self.occurred_at,
            self.event_type.value,
            self.user_id,
            self.session_key,
            self.product_id,
            self.source,
            self.device_type.value,
            json.dumps(self.event_metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["device_type"] = self.device_type.value
        return data


@runtime_checkable
class EventSink(Protocol):
    """Where validated events go."""

    def emit(self, records: list[EventRecord]) -> int:
        """Accept records. Returns how many were accepted (not necessarily written)."""
        ...

    def flush(self) -> None:
        """Force any buffered records to their destination."""
        ...

    def close(self) -> None:
        """Release resources. Must flush first."""
        ...


class MemoryEventSink:
    """Collects events in a list. Used by tests and by the dry-run mode."""

    def __init__(self) -> None:
        self.records: list[EventRecord] = []
        self._lock = threading.Lock()

    def emit(self, records: list[EventRecord]) -> int:
        with self._lock:
            self.records.extend(records)
        return len(records)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


class NullEventSink:
    """Discards everything. Used when ingestion is deliberately disabled."""

    def emit(self, records: list[EventRecord]) -> int:
        return len(records)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class PostgresEventSink:
    """Writes events with `COPY`, and registers the sessions they belong to.

    `COPY` rather than `INSERT ... VALUES` because at the target ingest rate
    the per-statement overhead dominates: a 500-row COPY is roughly an order of
    magnitude cheaper than 500 inserts and about three times cheaper than one
    multi-row insert, and it avoids building a huge parameter list.

    **Sessions are upserted from the same batch.** `user_sessions` is not
    populated by any other live code path, so without this the table would
    contain only synthetic rows: every real visitor would generate events whose
    `session_key` matches nothing. Session-scoped retrieval is the highest-value
    cold-start lever (architecture.md section 5), and it would have silently had
    nothing to read for exactly the users it exists to serve. This was found by
    `scripts/verify_database.py`, which is the compensating control that makes
    the absence of foreign keys on `user_events` defensible.

    Doing it here rather than in the request handler is deliberate: the upsert
    is one statement per *batch*, not per event, and it runs on the sink's
    background worker, so it costs the page render nothing.
    """

    COLUMNS = (
        "occurred_at",
        "event_type",
        "user_id",
        "session_key",
        "product_id",
        "source",
        "device_type",
        "event_metadata",
    )

    #: Upsert rather than insert because a session spans many batches: the
    #: first batch creates the row and every later one extends it. `LEAST` and
    #: `GREATEST` keep the window correct even if batches arrive out of order,
    #: which matters because the buffered worker makes no ordering promise.
    #:
    #: `user_id` is coalesced, never overwritten, so the login stitch (FR-22)
    #: is one-way: an anonymous session that signs in keeps the account, and a
    #: later anonymous event cannot detach it.
    UPSERT_SESSION = """
        INSERT INTO user_sessions
            (session_key, user_id, device_type, started_at, ended_at,
             event_count, converted, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
        ON CONFLICT (session_key) DO UPDATE SET
            user_id     = COALESCE(user_sessions.user_id, EXCLUDED.user_id),
            device_type = CASE
                              WHEN user_sessions.device_type = 'unknown'
                              THEN EXCLUDED.device_type
                              ELSE user_sessions.device_type
                          END,
            started_at  = LEAST(user_sessions.started_at, EXCLUDED.started_at),
            ended_at    = GREATEST(user_sessions.ended_at, EXCLUDED.ended_at),
            event_count = user_sessions.event_count + EXCLUDED.event_count,
            converted   = user_sessions.converted OR EXCLUDED.converted,
            updated_at  = now()
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @staticmethod
    def _session_rows(records: list[EventRecord]) -> list[tuple[Any, ...]]:
        """Collapse a batch into one row per session.

        500 events typically belong to a handful of sessions, so aggregating in
        Python turns 500 round trips into a handful - and makes the per-session
        conflict deterministic, rather than depending on statement order.
        """
        sessions: dict[str, dict[str, Any]] = {}
        for record in records:
            state = sessions.get(record.session_key)
            if state is None:
                sessions[record.session_key] = {
                    "user_id": record.user_id,
                    # `unknown` is a placeholder, not an observation: the first
                    # event that actually identifies the device wins.
                    "device_type": (
                        record.device_type.value
                        if record.device_type is not DeviceType.UNKNOWN
                        else None
                    ),
                    "started_at": record.occurred_at,
                    "ended_at": record.occurred_at,
                    "count": 1,
                    "converted": record.event_type is EventType.PURCHASE,
                }
                continue
            if state["user_id"] is None:
                state["user_id"] = record.user_id
            if state["device_type"] is None and record.device_type is not DeviceType.UNKNOWN:
                state["device_type"] = record.device_type.value
            if record.occurred_at < state["started_at"]:
                state["started_at"] = record.occurred_at
            if record.occurred_at > state["ended_at"]:
                state["ended_at"] = record.occurred_at
            state["count"] += 1
            state["converted"] = state["converted"] or record.event_type is EventType.PURCHASE

        return [
            (
                session_key,
                state["user_id"],
                state["device_type"] or DeviceType.UNKNOWN.value,
                state["started_at"],
                state["ended_at"],
                state["count"],
                state["converted"],
            )
            for session_key, state in sessions.items()
        ]

    def emit(self, records: list[EventRecord]) -> int:
        if not records:
            return 0

        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for record in records:
            writer.writerow(
                [
                    value.isoformat() if isinstance(value, dt.datetime)
                    else "" if value is None
                    else value
                    for value in record.as_row()
                ]
            )
        buffer.seek(0)

        statement = (
            f"COPY user_events ({', '.join(self.COLUMNS)}) "
            "FROM STDIN WITH (FORMAT csv, NULL '')"
        )
        raw = self._engine.raw_connection()
        try:
            with raw.cursor() as cursor:
                # Sessions first, in the same transaction: a reader must never
                # see an event whose session row does not exist yet, which is
                # precisely the orphan state this fix removes.
                cursor.executemany(self.UPSERT_SESSION, self._session_rows(records))
                cursor.copy_expert(statement, buffer)
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
        return len(records)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class BufferedEventSink:
    """Non-blocking front for another sink.

    The request thread enqueues and returns; a background worker batches and
    writes. Three properties are deliberate:

    * **Bounded queue.** An unbounded queue turns a database outage into an
      out-of-memory crash of the API - the failure spreads instead of staying
      contained.
    * **Drop rather than block.** When the queue is full, events are dropped
      and `dropped` is incremented. Blocking would push database latency
      straight into the page render, which is the thing FR-23 forbids.
    * **Write failures do not kill the worker.** A failed batch is logged and
      abandoned; the thread keeps draining. A worker that dies on the first
      transient error would silently stop ingestion for the process lifetime.
    """

    def __init__(
        self,
        target: EventSink,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        self._target = target
        self._queue: queue.Queue[EventRecord] = queue.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

        self.accepted = 0
        self.dropped = 0
        self.written = 0
        self.failed = 0

        self._worker = threading.Thread(
            target=self._run, name="event-sink", daemon=True
        )
        self._worker.start()

    # -- producer side ----------------------------------------------------

    def emit(self, records: list[EventRecord]) -> int:
        accepted = 0
        for record in records:
            try:
                self._queue.put_nowait(record)
                accepted += 1
            except queue.Full:
                self.dropped += 1
        self.accepted += accepted
        if accepted:
            self._idle.clear()
        return accepted

    # -- consumer side ----------------------------------------------------

    def _drain(self) -> list[EventRecord]:
        batch: list[EventRecord] = []
        try:
            batch.append(self._queue.get(timeout=self._flush_interval_s))
        except queue.Empty:
            return batch
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._drain()
            if not batch:
                self._idle.set()
                continue
            self._write(batch)
            if self._queue.empty():
                self._idle.set()

        remaining = self._drain_all()
        if remaining:
            self._write(remaining)
        self._idle.set()

    def _drain_all(self) -> list[EventRecord]:
        batch: list[EventRecord] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                return batch

    def _write(self, batch: list[EventRecord]) -> None:
        try:
            self.written += self._target.emit(batch)
        except Exception:
            self.failed += len(batch)
            logger.exception("event sink failed to write %d events", len(batch))

    # -- lifecycle --------------------------------------------------------

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue has drained. For tests and shutdown only."""
        self._idle.wait(timeout=timeout)
        self._target.flush()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=10.0)
        self._target.flush()
        self._target.close()

    def stats(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
            "queued": self._queue.qsize(),
        }


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_FLUSH_INTERVAL_S",
    "DEFAULT_QUEUE_SIZE",
    "BufferedEventSink",
    "EventRecord",
    "EventSink",
    "MemoryEventSink",
    "NullEventSink",
    "PostgresEventSink",
]
