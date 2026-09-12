"""Event ingestion service.

Sits between the API router and the sink. Responsibilities, in order:

1. **Attribute the event to the authenticated principal.** `user_id` comes from
   the token, never from the request body, so a client cannot write events as
   another user.
2. **Stamp a server timestamp** when the client did not supply one.
3. **Update real-time counters** so trending and recently-viewed react
   immediately rather than at the next batch run.
4. **Invalidate the user's cached homepage** on high-signal events only.
5. **Hand the record to the sink**, which buffers and returns immediately.

Nothing here blocks on the database. That is the whole point of FR-23: a page
render must not wait on analytics, and must not fail if analytics does.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from app.cache.counters import RealTimeCounters
from app.models.enums import EventType
from app.schemas.events import EventIn
from app.services.event_sink import EventRecord, EventSink

logger = logging.getLogger(__name__)

#: Events that change a user's recommendations enough to justify dropping their
#: cached homepage. Views are excluded on purpose: they are most of the traffic,
#: so invalidating on them would defeat the cache, and a view already reaches
#: the next refresh through the recently-viewed list.
HIGH_SIGNAL_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.PURCHASE,
        EventType.ADD_TO_CART,
        EventType.WISHLIST,
        EventType.PRODUCT_RATING,
    }
)


@dataclass(slots=True)
class IngestResult:
    accepted: int
    rejected: int
    errors: list[str]


class EventService:
    """Validates, enriches and dispatches behavioural events."""

    def __init__(
        self,
        sink: EventSink,
        counters: RealTimeCounters,
        *,
        category_lookup: dict[int, int] | None = None,
    ) -> None:
        self._sink = sink
        self._counters = counters
        # product_id -> category_id, used to maintain per-category trending.
        # A dict rather than a query: the catalogue is small, static within a
        # process lifetime, and this sits on the ingest hot path.
        self._category_lookup = category_lookup or {}

    def set_category_lookup(self, lookup: dict[int, int]) -> None:
        self._category_lookup = lookup

    def to_record(
        self,
        event: EventIn,
        *,
        user_id: int | None,
        now: dt.datetime | None = None,
    ) -> EventRecord:
        return EventRecord(
            event_type=event.event_type,
            session_key=event.session_id,
            occurred_at=event.occurred_at or (now or dt.datetime.now(dt.UTC)),
            user_id=user_id,
            product_id=event.product_id,
            source=event.source,
            device_type=event.device_type,
            recommendation_id=event.recommendation_id,
            event_metadata=event.metadata,
        )

    def ingest(
        self,
        events: list[EventIn],
        *,
        user_id: int | None,
        now: dt.datetime | None = None,
    ) -> IngestResult:
        """Accept a batch of events."""
        if not events:
            return IngestResult(accepted=0, rejected=0, errors=[])

        stamped = now or dt.datetime.now(dt.UTC)
        records = [self.to_record(event, user_id=user_id, now=stamped) for event in events]

        accepted = self._sink.emit(records)
        rejected = len(records) - accepted

        self._update_realtime(records)

        if rejected:
            logger.warning(
                "event sink dropped %d of %d events; the ingest queue is full",
                rejected,
                len(records),
            )

        return IngestResult(
            accepted=accepted,
            rejected=rejected,
            errors=(
                [f"{rejected} events dropped: ingest buffer full"] if rejected else []
            ),
        )

    def _update_realtime(self, records: list[EventRecord]) -> None:
        """Best-effort counter updates.

        Wrapped so a Redis problem can never turn into a failed event write:
        the durable path is the sink, and this is the disposable one.
        """
        invalidate: set[int] = set()
        for record in records:
            try:
                self._counters.record_event(
                    event_type=record.event_type,
                    product_id=record.product_id,
                    user_id=record.user_id,
                    session_key=record.session_key,
                    occurred_at=record.occurred_at,
                    category_id=(
                        self._category_lookup.get(record.product_id)
                        if record.product_id is not None
                        else None
                    ),
                )
            except Exception:
                logger.debug("counter update failed", exc_info=True)

            if record.user_id is not None and record.event_type in HIGH_SIGNAL_EVENTS:
                invalidate.add(record.user_id)

        for user_id in invalidate:
            try:
                self._counters.invalidate_user_recommendations(user_id)
            except Exception:
                logger.debug("cache invalidation failed", exc_info=True)


__all__ = ["HIGH_SIGNAL_EVENTS", "EventService", "IngestResult"]
