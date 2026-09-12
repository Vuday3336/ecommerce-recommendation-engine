"""Event ingestion schemas.

The design goal is FR-21: a new event type must be addable without a schema
migration, *without* turning `event_metadata` into an unvalidated dumping
ground. The resolution is a registry of per-type payload models. The database
column stays JSONB and never changes shape; the API boundary validates the
payload against the model registered for that event type.

Adding an event type is therefore: one enum member, one payload model, one
registry entry. No migration, no new column, no loss of validation.

`user_id` is deliberately absent from every inbound model. It is resolved from
the authenticated principal, never from the request body - otherwise any client
could write events attributed to another user, poisoning both their
recommendations and the training set.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import PRODUCT_SCOPED_EVENTS, DeviceType, EventType

MAX_BATCH_SIZE = 100

#: How far in the past a client-supplied timestamp may be. Mobile clients
#: legitimately buffer events while offline, so backdating is allowed - but
#: unbounded backdating would let a client write into a partition that has
#: already been used for training, silently changing a "frozen" dataset.
MAX_EVENT_AGE = dt.timedelta(days=7)

#: Tolerance for clock skew on client devices. Anything further ahead is
#: clamped to now rather than rejected: dropping the event would lose real
#: behaviour because a phone's clock is wrong.
MAX_CLOCK_SKEW = dt.timedelta(minutes=5)


# ---------------------------------------------------------------------------
# Per-event payloads
# ---------------------------------------------------------------------------


class EventPayload(BaseModel):
    """Base for typed event metadata."""

    model_config = ConfigDict(extra="forbid")


class ViewPayload(EventPayload):
    dwell_ms: Annotated[int, Field(ge=0, le=3_600_000)] = 0
    position: Annotated[int | None, Field(ge=0, le=1000)] = None


class ClickPayload(EventPayload):
    position: Annotated[int | None, Field(ge=0, le=1000)] = None


class SearchPayload(EventPayload):
    query: Annotated[str, Field(min_length=1, max_length=200)]
    results: Annotated[int, Field(ge=0)] = 0
    filters: dict[str, Any] = Field(default_factory=dict)


class CartPayload(EventPayload):
    quantity: Annotated[int, Field(ge=1, le=999)] = 1
    variant_id: int | None = None


class RatingPayload(EventPayload):
    rating: Annotated[float, Field(ge=1.0, le=5.0)]


class ReviewPayload(EventPayload):
    rating: Annotated[float, Field(ge=1.0, le=5.0)]
    length: Annotated[int, Field(ge=0, le=20_000)] = 0


class SharePayload(EventPayload):
    channel: Annotated[str, Field(max_length=40)] = "link"


class SessionPayload(EventPayload):
    entry: Annotated[str, Field(max_length=60)] = "direct"
    referrer: Annotated[str | None, Field(max_length=300)] = None


class EmptyPayload(EventPayload):
    pass


#: The registry FR-21 rests on. An event type with no entry falls back to
#: `EmptyPayload`, which forbids extra keys - so a new type ships with strict
#: validation by default rather than accidentally accepting anything.
PAYLOAD_MODELS: dict[EventType, type[EventPayload]] = {
    EventType.PRODUCT_VIEW: ViewPayload,
    EventType.PRODUCT_CLICK: ClickPayload,
    EventType.SEARCH: SearchPayload,
    EventType.ADD_TO_CART: CartPayload,
    EventType.REMOVE_FROM_CART: CartPayload,
    EventType.WISHLIST: EmptyPayload,
    EventType.PURCHASE: CartPayload,
    EventType.PRODUCT_SHARE: SharePayload,
    EventType.PRODUCT_RATING: RatingPayload,
    EventType.PRODUCT_REVIEW: ReviewPayload,
    EventType.SESSION_START: SessionPayload,
    EventType.SESSION_END: SessionPayload,
}


def validate_payload(event_type: EventType, metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata against the model registered for this event type."""
    model = PAYLOAD_MODELS.get(event_type, EmptyPayload)
    return model.model_validate(metadata).model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------------


class EventIn(BaseModel):
    """One behavioural event as submitted by a client."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    session_id: Annotated[str, Field(min_length=8, max_length=64)]
    product_id: Annotated[int | None, Field(gt=0)] = None
    occurred_at: dt.datetime | None = None
    source: Annotated[str, Field(max_length=64)] = "unknown"
    device_type: DeviceType = DeviceType.UNKNOWN
    recommendation_id: Annotated[int | None, Field(gt=0)] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _bound_timestamp(cls, value: dt.datetime | None) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("occurred_at must include a timezone offset")
        now = dt.datetime.now(dt.UTC)
        if value > now + MAX_CLOCK_SKEW:
            # Clamp rather than reject: a wrong device clock is not a reason to
            # discard real behaviour.
            return now
        if value < now - MAX_EVENT_AGE:
            raise ValueError(
                f"occurred_at is more than {MAX_EVENT_AGE.days} days old; "
                "backdating that far would mutate an already-trained window"
            )
        return value

    @model_validator(mode="after")
    def _check_product_scope(self) -> EventIn:
        """Mirror the database CHECK constraint at the edge.

        Catching it here returns a 422 explaining the problem instead of a 500
        from a constraint violation several layers down.
        """
        if self.event_type in PRODUCT_SCOPED_EVENTS and self.product_id is None:
            raise ValueError(f"{self.event_type.value} requires product_id")
        if self.event_type not in PRODUCT_SCOPED_EVENTS and self.product_id is not None:
            raise ValueError(f"{self.event_type.value} must not carry product_id")
        return self

    @model_validator(mode="after")
    def _validate_metadata(self) -> EventIn:
        object.__setattr__(
            self, "metadata", validate_payload(self.event_type, self.metadata)
        )
        return self


class EventBatchIn(BaseModel):
    """A batch of events.

    Batching exists for NFR-02/FR-23: a page that fires eight events must not
    make eight round trips, and the browser must never wait on any of them.
    """

    model_config = ConfigDict(extra="forbid")

    events: Annotated[list[EventIn], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------


class EventAccepted(BaseModel):
    """Acknowledgement.

    Deliberately says `accepted`, not `stored`. Ingestion is buffered and the
    response returns before the database write completes, so claiming
    persistence here would be a lie the client might act on.
    """

    accepted: int
    rejected: int = 0
    status: Literal["accepted"] = "accepted"
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "MAX_BATCH_SIZE",
    "MAX_CLOCK_SKEW",
    "MAX_EVENT_AGE",
    "PAYLOAD_MODELS",
    "CartPayload",
    "ClickPayload",
    "EmptyPayload",
    "EventAccepted",
    "EventBatchIn",
    "EventIn",
    "EventPayload",
    "RatingPayload",
    "ReviewPayload",
    "SearchPayload",
    "SessionPayload",
    "SharePayload",
    "ViewPayload",
    "validate_payload",
]
