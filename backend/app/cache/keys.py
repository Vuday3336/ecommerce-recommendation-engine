"""Redis key namespaces.

Every key the system uses is constructed here. Scattering key strings across
services is how two components end up disagreeing about a format, and how a
cache invalidation quietly stops matching the keys it is supposed to clear.

Namespaces:

* ``rec:``  rendered recommendation payloads (cache)
* ``feat:`` online feature vectors (cache with a database fallback)
* ``rt:``   real-time counters and lists (derived, rebuildable)
* ``rl:``   rate-limit buckets
* ``lock:`` short-lived locks for cache-stampede protection

The model version is embedded in every recommendation key, so promoting a model
rolls the cache over naturally instead of needing a flush and the cold-cache
latency spike that follows one (ADR-006).
"""

from __future__ import annotations

import datetime as dt

PREFIX_RECOMMENDATION = "rec"
PREFIX_FEATURE = "feat"
PREFIX_REALTIME = "rt"
PREFIX_RATELIMIT = "rl"
PREFIX_LOCK = "lock"

#: Trending is accumulated into hourly buckets and read as a decayed union of
#: the recent ones. Hourly is the right granularity: fine enough that a spike
#: shows up within the hour, coarse enough that reading a day of history is a
#: 24-key union rather than a scan.
TREND_BUCKET_FORMAT = "%Y%m%d%H"
TREND_BUCKET_TTL_SECONDS = 60 * 60 * 50  # 50 hours: two days plus slack
RECENTLY_VIEWED_MAX = 50
RECENTLY_VIEWED_TTL_SECONDS = 60 * 60 * 24 * 30


def home_recommendations(user_id: int, model_version: str, variant: str = "control") -> str:
    return f"{PREFIX_RECOMMENDATION}:home:{user_id}:{model_version}:{variant}"


def similar_products(product_id: int, model_version: str) -> str:
    return f"{PREFIX_RECOMMENDATION}:similar:{product_id}:{model_version}"


def frequently_bought(product_id: int, model_version: str) -> str:
    return f"{PREFIX_RECOMMENDATION}:fbt:{product_id}:{model_version}"


def also_viewed(product_id: int, model_version: str) -> str:
    return f"{PREFIX_RECOMMENDATION}:covisit:{product_id}:{model_version}"


def trending(scope: str = "global") -> str:
    return f"{PREFIX_RECOMMENDATION}:trending:{scope}"


def user_features(user_id: int) -> str:
    return f"{PREFIX_FEATURE}:user:{user_id}"


def product_features(product_id: int) -> str:
    return f"{PREFIX_FEATURE}:product:{product_id}"


def trend_bucket(when: dt.datetime, scope: str = "global") -> str:
    return f"{PREFIX_REALTIME}:trend:{scope}:{when.strftime(TREND_BUCKET_FORMAT)}"


def trend_union(scope: str = "global") -> str:
    return f"{PREFIX_REALTIME}:trend:{scope}:union"


def recently_viewed(user_id: int) -> str:
    return f"{PREFIX_REALTIME}:recent:user:{user_id}"


def recently_viewed_session(session_key: str) -> str:
    return f"{PREFIX_REALTIME}:recent:session:{session_key}"


def session_products(session_key: str) -> str:
    """Products seen this session - the cold-start signal for anonymous users."""
    return f"{PREFIX_REALTIME}:session:{session_key}:products"


def rate_limit(scope: str, identity: str, window: str) -> str:
    return f"{PREFIX_RATELIMIT}:{scope}:{identity}:{window}"


def lock(name: str) -> str:
    return f"{PREFIX_LOCK}:{name}"


__all__ = [
    "PREFIX_FEATURE",
    "PREFIX_LOCK",
    "PREFIX_RATELIMIT",
    "PREFIX_REALTIME",
    "PREFIX_RECOMMENDATION",
    "RECENTLY_VIEWED_MAX",
    "RECENTLY_VIEWED_TTL_SECONDS",
    "TREND_BUCKET_FORMAT",
    "TREND_BUCKET_TTL_SECONDS",
    "also_viewed",
    "frequently_bought",
    "home_recommendations",
    "lock",
    "product_features",
    "rate_limit",
    "recently_viewed",
    "recently_viewed_session",
    "session_products",
    "similar_products",
    "trend_bucket",
    "trend_union",
    "trending",
    "user_features",
]
