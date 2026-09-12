"""Real-time counters: trending velocity, recently viewed, session context.

These are the parts of the recommendation system that must react within
seconds rather than waiting for the next batch run. They are all derived state
that can be rebuilt from `user_events`, which is why losing Redis costs
freshness and nothing else.

**Trending.** Scores accumulate into hourly sorted sets, and a read is a
weighted union of the recent buckets with exponentially decaying weights. The
alternative - one sorted set with a periodic decay pass over every member - is
O(catalogue) per decay and needs a scheduler. Bucketing makes writes O(1),
reads O(hours), and expiry automatic: an old bucket simply falls out of its TTL.

The decay half-life is a product decision expressed as a number. Six hours
means a burst is visible almost immediately and mostly gone by the next day,
which is what "trending" should mean on a storefront. A longer half-life turns
trending into a slow popularity chart, which the popularity model already is.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError

from app.cache import keys
from app.models.enums import EventType

logger = logging.getLogger(__name__)

#: Weight applied to each event type when accumulating trending velocity.
#: A purchase is a far stronger statement of interest than a view, and using
#: raw view counts makes trending a clickbait chart.
TREND_EVENT_WEIGHTS: dict[EventType, float] = {
    EventType.PRODUCT_VIEW: 1.0,
    EventType.PRODUCT_CLICK: 1.5,
    EventType.WISHLIST: 3.0,
    EventType.ADD_TO_CART: 5.0,
    EventType.PURCHASE: 12.0,
}

TREND_HALF_LIFE_HOURS = 6.0
TREND_WINDOW_HOURS = 24
TREND_UNION_TTL_SECONDS = 300


def _decay_weights(hours: int, half_life: float) -> list[float]:
    """Exponential decay weights, newest first."""
    return [math.pow(0.5, age / half_life) for age in range(hours)]


@dataclass(slots=True)
class ScoredProduct:
    product_id: int
    score: float


class RealTimeCounters:
    """Redis-backed real-time signals.

    Every method degrades to a no-op or an empty result when Redis is absent,
    so callers do not need to branch on availability.
    """

    def __init__(self, client: Redis | None) -> None:
        self._redis = client

    @property
    def available(self) -> bool:
        return self._redis is not None

    # -- writes -----------------------------------------------------------

    def record_event(
        self,
        *,
        event_type: EventType,
        product_id: int | None,
        user_id: int | None,
        session_key: str,
        occurred_at: dt.datetime,
        category_id: int | None = None,
    ) -> None:
        """Update every real-time structure this event affects.

        Written as one pipeline so a single page view costs one round trip
        rather than four.
        """
        if self._redis is None or product_id is None:
            return

        weight = TREND_EVENT_WEIGHTS.get(event_type)
        try:
            pipe = self._redis.pipeline(transaction=False)

            if weight:
                bucket = keys.trend_bucket(occurred_at)
                pipe.zincrby(bucket, weight, product_id)
                pipe.expire(bucket, keys.TREND_BUCKET_TTL_SECONDS)
                if category_id is not None:
                    scoped = keys.trend_bucket(occurred_at, f"category:{category_id}")
                    pipe.zincrby(scoped, weight, product_id)
                    pipe.expire(scoped, keys.TREND_BUCKET_TTL_SECONDS)

            if event_type is EventType.PRODUCT_VIEW:
                # Session list first: it is the only signal an anonymous
                # visitor has, and it is what session-based cold start reads.
                session_key_name = keys.session_products(session_key)
                pipe.lpush(session_key_name, product_id)
                pipe.ltrim(session_key_name, 0, keys.RECENTLY_VIEWED_MAX - 1)
                pipe.expire(session_key_name, 60 * 60 * 6)

                if user_id is not None:
                    recent = keys.recently_viewed(user_id)
                    # Remove before pushing so a re-view moves the product to
                    # the front instead of appearing twice.
                    pipe.lrem(recent, 0, product_id)
                    pipe.lpush(recent, product_id)
                    pipe.ltrim(recent, 0, keys.RECENTLY_VIEWED_MAX - 1)
                    pipe.expire(recent, keys.RECENTLY_VIEWED_TTL_SECONDS)

            pipe.execute()
        except (RedisError, OSError):
            logger.debug("real-time counter update failed", exc_info=True)

    def invalidate_user_recommendations(self, user_id: int) -> None:
        """Drop a user's cached homepage after a high-signal event.

        Only purchases and cart additions trigger this. Invalidating on every
        view would defeat the cache entirely - views are most of the traffic -
        and a view already influences the next refresh through recently-viewed.
        """
        if self._redis is None:
            return
        try:
            pattern = f"{keys.PREFIX_RECOMMENDATION}:home:{user_id}:*"
            for key in self._redis.scan_iter(match=pattern, count=100):
                self._redis.delete(key)
        except (RedisError, OSError):
            logger.debug("cache invalidation failed for user %s", user_id, exc_info=True)

    # -- reads ------------------------------------------------------------

    def trending(
        self,
        limit: int = 50,
        *,
        scope: str = "global",
        now: dt.datetime | None = None,
        hours: int = TREND_WINDOW_HOURS,
        half_life: float = TREND_HALF_LIFE_HOURS,
    ) -> list[ScoredProduct]:
        """Decayed trending list."""
        if self._redis is None:
            return []

        now = now or dt.datetime.now(dt.UTC)
        weights = _decay_weights(hours, half_life)
        bucket_keys = [
            keys.trend_bucket(now - dt.timedelta(hours=age), scope)
            for age in range(hours)
        ]

        destination = keys.trend_union(scope)
        try:
            existing = [
                (key, weight)
                for key, weight in zip(bucket_keys, weights, strict=True)
                if self._redis.exists(key)
            ]
            if not existing:
                return []

            self._redis.zunionstore(destination, dict(existing))
            self._redis.expire(destination, TREND_UNION_TTL_SECONDS)
            rows = self._redis.zrevrange(destination, 0, limit - 1, withscores=True)
        except (RedisError, OSError):
            logger.debug("trending read failed", exc_info=True)
            return []

        return [ScoredProduct(int(member), float(score)) for member, score in rows]

    def recently_viewed(self, user_id: int, limit: int = 20) -> list[int]:
        if self._redis is None:
            return []
        try:
            values = self._redis.lrange(keys.recently_viewed(user_id), 0, limit - 1)
        except (RedisError, OSError):
            return []
        return [int(value) for value in values]

    def session_products(self, session_key: str, limit: int = 20) -> list[int]:
        if self._redis is None:
            return []
        try:
            values = self._redis.lrange(keys.session_products(session_key), 0, limit - 1)
        except (RedisError, OSError):
            return []
        return [int(value) for value in values]

    # -- maintenance ------------------------------------------------------

    def warm_trending(
        self, scores: dict[int, float], *, now: dt.datetime | None = None, scope: str = "global"
    ) -> None:
        """Seed the current bucket from batch-computed scores.

        Needed after a cold start or a Redis restart: without it, trending is
        empty until enough live traffic accumulates, and the trending rail on
        the homepage would be blank for hours.
        """
        if self._redis is None or not scores:
            return
        bucket = keys.trend_bucket(now or dt.datetime.now(dt.UTC), scope)
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.zadd(bucket, {str(pid): score for pid, score in scores.items()})
            pipe.expire(bucket, keys.TREND_BUCKET_TTL_SECONDS)
            pipe.execute()
        except (RedisError, OSError):
            logger.debug("trending warm failed", exc_info=True)


__all__ = [
    "TREND_EVENT_WEIGHTS",
    "TREND_HALF_LIFE_HOURS",
    "TREND_WINDOW_HOURS",
    "RealTimeCounters",
    "ScoredProduct",
]
