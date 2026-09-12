"""FastAPI dependencies: authentication, services, rate limiting.

Long-lived objects - the recommendation engine, the event sink, the experiment
definitions - live in `app.state` and are built once during start-up. Building
them per request would load model artefacts from disk on every call.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.exceptions import RedisError

from app.cache import keys
from app.core.config import settings
from app.core.security import AuthError, Principal, decode_token
from app.models.enums import UserRole

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> Principal | None:
    """Resolve the caller, or None for anonymous traffic.

    Anonymous is a first-class case, not an error: the storefront must serve
    recommendations to visitors who have never signed in, and cold-start
    handling depends on that path working.
    """
    if credentials is None or not credentials.credentials:
        return None
    try:
        return decode_token(credentials.credentials, expected_type="access")
    except AuthError:
        # An invalid token is treated as anonymous rather than rejected. A
        # stale token in a browser tab should degrade to public browsing, not
        # break the storefront.
        return None


def require_principal(
    principal: Annotated[Principal | None, Depends(get_principal)],
) -> Principal:
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_role(required: UserRole):
    """Dependency factory enforcing a minimum role."""

    def dependency(
        principal: Annotated[Principal, Depends(require_principal)],
    ) -> Principal:
        if not principal.has_role(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires the {required.value} role or higher",
            )
        return principal

    return dependency


require_analyst = require_role(UserRole.ANALYST)
require_admin = require_role(UserRole.ADMIN)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def get_session_key(
    x_session_id: Annotated[str | None, Header(alias="X-Session-Id")] = None,
) -> str:
    """The browsing session, from a client-supplied header.

    Required rather than generated server-side: the session must be stable
    across requests for co-visitation and session-based cold start to mean
    anything, and only the client can guarantee that.
    """
    if x_session_id and 8 <= len(x_session_id) <= 64:
        return x_session_id
    return "anonymous-session"


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


def get_engine(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the recommendation engine is not loaded",
        )
    return engine


def get_recommendation_service(request: Request):
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the recommendation service is not available",
        )
    return service


def get_event_service(request: Request):
    service = getattr(request.app.state, "event_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="event ingestion is not available",
        )
    return service


def get_redis(request: Request):
    return getattr(request.app.state, "redis", None)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Fixed-window rate limit backed by Redis.

    A fixed window, not a sliding one: it costs a single `INCR` plus an
    `EXPIRE`, where a sliding window needs a sorted set and several commands.
    The known cost is that a caller can send up to 2x the limit across a window
    boundary. For abuse protection on a storefront that is an acceptable trade;
    it would not be for billing.

    **Fails open.** If Redis is unavailable the request is allowed. A cache
    outage must not take down the storefront - denial of service by the rate
    limiter is worse than the abuse it prevents.
    """

    def __init__(self, scope: str, limit_per_minute: int) -> None:
        self.scope = scope
        self.limit = limit_per_minute

    def __call__(
        self,
        request: Request,
        principal: Annotated[Principal | None, Depends(get_principal)] = None,
    ) -> None:
        redis = getattr(request.app.state, "redis", None)
        if redis is None or self.limit <= 0:
            return

        identity = (
            f"user:{principal.user_id}"
            if principal
            else f"ip:{request.client.host if request.client else 'unknown'}"
        )
        window = str(int(time.time() // 60))
        key = keys.rate_limit(self.scope, identity, window)

        try:
            pipe = redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 90)
            count, _ = pipe.execute()
        except (RedisError, OSError):
            return

        if int(count) > self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"rate limit exceeded for {self.scope}",
                headers={"Retry-After": "60"},
            )


rate_limit_auth = RateLimiter("auth", settings.rate_limit_auth_per_minute)
rate_limit_events = RateLimiter("events", settings.rate_limit_events_per_minute)
rate_limit_recommendations = RateLimiter(
    "recommendations", settings.rate_limit_recommendations_per_minute
)


__all__ = [
    "RateLimiter",
    "get_engine",
    "get_event_service",
    "get_principal",
    "get_recommendation_service",
    "get_redis",
    "get_session_key",
    "rate_limit_auth",
    "rate_limit_events",
    "rate_limit_recommendations",
    "require_admin",
    "require_analyst",
    "require_principal",
    "require_role",
]
