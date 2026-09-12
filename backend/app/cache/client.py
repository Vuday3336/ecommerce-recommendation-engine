"""Redis client construction and availability handling.

Redis holds only derived state (ADR-006), so the correct behaviour when it is
unavailable is to degrade, not to fail. `get_redis()` returns `None` rather
than raising, and every caller is written to work without it - slower, with
staler trending, but working. That is what makes rungs 5-7 of the degradation
ladder reachable instead of theoretical.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import redis
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Redis | None = None
_unavailable = False


def build_client(url: str | None = None, **kwargs: Any) -> Redis:
    """Create a Redis client.

    `decode_responses=True` because every value stored here is JSON or a plain
    number, and dealing with bytes at each call site is noise.

    Timeouts are short and explicit: this sits on the hot path, and a Redis
    that has become slow must fail fast into the database fallback rather than
    hold the request open and blow the latency budget it exists to protect.
    """
    return redis.from_url(
        url or settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
        retry_on_timeout=False,
        health_check_interval=30,
        **kwargs,
    )


def get_redis() -> Redis | None:
    """Return a live client, or None when Redis is unreachable."""
    global _client, _unavailable

    if _client is not None:
        return _client
    if _unavailable:
        return None

    try:
        client = build_client()
        client.ping()
    except (RedisError, OSError) as exc:
        _unavailable = True
        logger.warning(
            "Redis unavailable at %s:%s (%s). Continuing without cache: "
            "recommendations will be slower and trending staler, but nothing "
            "durable is lost.",
            settings.redis_host,
            settings.redis_port,
            exc,
        )
        return None

    _client = client
    return _client


def set_redis(client: Redis | None) -> None:
    """Inject a client. Used by tests to supply `fakeredis`."""
    global _client, _unavailable
    _client = client
    _unavailable = client is None


def reset_redis() -> None:
    global _client, _unavailable
    if _client is not None:
        with contextlib.suppress(RedisError):
            _client.close()
    _client = None
    _unavailable = False


def check_connection() -> bool:
    client = get_redis()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except (RedisError, OSError):
        return False


__all__ = [
    "build_client",
    "check_connection",
    "get_redis",
    "reset_redis",
    "set_redis",
]
