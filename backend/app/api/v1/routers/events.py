"""Event ingestion and recommendation feedback (FR-12, FR-20 to FR-23).

The contract these endpoints keep is FR-23: a page render must never wait on,
or fail because of, analytics. So the handler validates, hands the batch to a
buffered sink, and returns. Nothing here touches the database synchronously,
and a full ingest buffer drops events (counted) rather than blocking.

`user_id` is never read from the request body. It comes from the token, so a
client cannot write events attributed to someone else - which would poison both
that user's recommendations and the training set.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import (
    get_event_service,
    get_principal,
    get_session_key,
    rate_limit_events,
)
from app.core import metrics
from app.core.security import Principal
from app.models.enums import EventType
from app.schemas.events import EventAccepted, EventBatchIn, EventIn
from app.schemas.recommendations import FeedbackIn
from app.services.events import EventService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"], dependencies=[Depends(rate_limit_events)])

ServiceDep = Annotated[EventService, Depends(get_event_service)]
PrincipalDep = Annotated[Principal | None, Depends(get_principal)]
SessionDep = Annotated[str, Depends(get_session_key)]


@router.post(
    "/events",
    response_model=EventAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest one behavioural event",
)
def ingest_event(
    event: EventIn,
    service: ServiceDep,
    principal: PrincipalDep,
    response: Response,
) -> EventAccepted:
    """Accept a single event.

    Returns 202, not 201: the event has been accepted for processing, not
    persisted. Claiming persistence would be a lie the client might act on.
    """
    started = time.perf_counter()
    result = service.ingest([event], user_id=principal.user_id if principal else None)
    metrics.event_ingest_duration_seconds.observe(time.perf_counter() - started)
    metrics.events_ingested_total.labels(event_type=event.event_type.value).inc(
        result.accepted
    )
    if result.rejected:
        metrics.events_dropped_total.inc(result.rejected)

    response.headers["Cache-Control"] = "no-store"
    return EventAccepted(
        accepted=result.accepted, rejected=result.rejected, errors=result.errors
    )


@router.post(
    "/events/batch",
    response_model=EventAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of events",
)
def ingest_batch(
    batch: EventBatchIn,
    service: ServiceDep,
    principal: PrincipalDep,
    response: Response,
) -> EventAccepted:
    """Accept up to 100 events in one call.

    This is what the browser client actually uses: a page that fires eight
    events must not make eight round trips, and the batch is flushed on a timer
    and on `visibilitychange` so events survive the user navigating away.
    """
    started = time.perf_counter()
    result = service.ingest(
        batch.events, user_id=principal.user_id if principal else None
    )
    metrics.event_ingest_duration_seconds.observe(time.perf_counter() - started)

    for event in batch.events:
        metrics.events_ingested_total.labels(event_type=event.event_type.value).inc()
    if result.rejected:
        metrics.events_dropped_total.inc(result.rejected)

    response.headers["Cache-Control"] = "no-store"
    return EventAccepted(
        accepted=result.accepted, rejected=result.rejected, errors=result.errors
    )


# ---------------------------------------------------------------------------
# Recommendation feedback
# ---------------------------------------------------------------------------

feedback_router = APIRouter(prefix="/feedback", tags=["feedback"])


@feedback_router.post(
    "/impression",
    status_code=status.HTTP_202_ACCEPTED,
    summary="A recommended item became visible (FR-12)",
)
def record_impression(
    feedback: FeedbackIn,
    service: ServiceDep,
    principal: PrincipalDep,
    session_key: SessionDep,
) -> EventAccepted:
    """Record that an item was actually on screen.

    Reported from an `IntersectionObserver`, not from the API response. A rail
    below the fold is served but never seen, and counting it would depress CTR
    for exactly the surfaces that are working.
    """
    event = EventIn(
        event_type=EventType.PRODUCT_VIEW,
        session_id=feedback.session_id or session_key,
        product_id=feedback.product_id,
        source="recommendation_impression",
        metadata={"dwell_ms": feedback.visible_ms, "position": feedback.position},
    )
    result = service.ingest([event], user_id=principal.user_id if principal else None)
    metrics.recommendation_impressions_total.labels(surface="unknown").inc()
    return EventAccepted(accepted=result.accepted, rejected=result.rejected)


@feedback_router.post(
    "/click",
    status_code=status.HTTP_202_ACCEPTED,
    summary="A recommended item was clicked (FR-12)",
)
def record_click(
    feedback: FeedbackIn,
    service: ServiceDep,
    principal: PrincipalDep,
    session_key: SessionDep,
) -> EventAccepted:
    event = EventIn(
        event_type=EventType.PRODUCT_CLICK,
        session_id=feedback.session_id or session_key,
        product_id=feedback.product_id,
        source="recommendation_click",
        metadata={"position": feedback.position},
    )
    result = service.ingest([event], user_id=principal.user_id if principal else None)
    metrics.recommendation_clicks_total.labels(surface="unknown").inc()
    return EventAccepted(accepted=result.accepted, rejected=result.rejected)


__all__ = ["feedback_router", "router"]
