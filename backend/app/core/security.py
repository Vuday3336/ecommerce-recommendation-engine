"""Authentication, password hashing and role-based authorisation.

Access tokens are short-lived (30 minutes) and refresh tokens long-lived
(14 days). The split exists so a leaked access token expires quickly while
users are not forced to log in twice a day. Both are signed with the same
secret but carry a `type` claim, and the verifier checks it: without that
check, a refresh token would be accepted as an access token and the short
lifetime would be decorative.

Passwords use Argon2id, which is memory-hard - GPU attacks on bcrypt are
cheap in a way they are not on Argon2. bcrypt is kept as a verification-only
fallback so existing hashes from an older system still authenticate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from typing import Any, Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.models.enums import UserRole

logger = logging.getLogger(__name__)

TokenType = Literal["access", "refresh"]

password_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4,
)

#: Role hierarchy. An admin satisfies an `analyst` requirement; a customer
#: satisfies neither. Encoding this as an ordering rather than a set of
#: explicit grants keeps the dependency in the routers to one comparison.
ROLE_ORDER: dict[str, int] = {
    UserRole.CUSTOMER.value: 0,
    UserRole.ANALYST.value: 1,
    UserRole.ADMIN.value: 2,
}


@dataclass(slots=True, frozen=True)
class Principal:
    """The authenticated caller."""

    user_id: int
    public_id: str
    role: UserRole
    token_type: TokenType = "access"

    def has_role(self, required: UserRole) -> bool:
        return ROLE_ORDER.get(self.role.value, -1) >= ROLE_ORDER.get(required.value, 99)


class AuthError(Exception):
    """Raised when a credential is missing, malformed or expired."""


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, hashed: str | None) -> bool:
    """Verify a password.

    Always runs a hash even when there is no stored hash. Returning early for
    an unknown user makes the response measurably faster for that case, which
    leaks account existence through timing - a small leak, but free to close.
    """
    if not hashed:
        password_context.dummy_verify()
        return False
    try:
        return password_context.verify(password, hashed)
    except ValueError:
        return False


def needs_rehash(hashed: str) -> bool:
    return password_context.needs_update(hashed)


def _expiry(token_type: TokenType) -> dt.datetime:
    now = dt.datetime.now(dt.UTC)
    if token_type == "access":
        return now + dt.timedelta(minutes=settings.access_token_expire_minutes)
    return now + dt.timedelta(days=settings.refresh_token_expire_days)


def create_token(
    *,
    user_id: int,
    public_id: str,
    role: UserRole,
    token_type: TokenType = "access",
) -> str:
    if not settings.jwt_secret_key:
        raise AuthError("JWT_SECRET_KEY is not configured")

    now = dt.datetime.now(dt.UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "pid": public_id,
        "role": role.value,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int(_expiry(token_type).timestamp()),
        # A unique id per token, so a specific token can be revoked without
        # invalidating every token the user holds.
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, expected_type: TokenType = "access") -> Principal:
    if not settings.jwt_secret_key:
        raise AuthError("JWT_SECRET_KEY is not configured")
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise AuthError("invalid or expired token") from exc

    if payload.get("type") != expected_type:
        # Without this, a refresh token would be accepted wherever an access
        # token is, and its 14-day lifetime would silently become the real
        # session length.
        raise AuthError(f"expected a {expected_type} token")

    try:
        return Principal(
            user_id=int(payload["sub"]),
            public_id=str(payload.get("pid", "")),
            role=UserRole(payload.get("role", UserRole.CUSTOMER.value)),
            token_type=expected_type,
        )
    except (KeyError, ValueError) as exc:
        raise AuthError("malformed token payload") from exc


def create_token_pair(*, user_id: int, public_id: str, role: UserRole) -> dict[str, Any]:
    return {
        "access_token": create_token(
            user_id=user_id, public_id=public_id, role=role, token_type="access"
        ),
        "refresh_token": create_token(
            user_id=user_id, public_id=public_id, role=role, token_type="refresh"
        ),
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


def constant_time_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def stable_bucket(key: str, *, buckets: int = 10_000) -> int:
    """Deterministic hash bucket, used for experiment assignment (ADR-011).

    SHA-256 rather than Python's `hash()`: the built-in is randomised per
    process by PYTHONHASHSEED, so a user would land in a different experiment
    arm on every worker and after every restart, which would destroy the
    experiment silently.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


__all__ = [
    "ROLE_ORDER",
    "AuthError",
    "Principal",
    "TokenType",
    "constant_time_compare",
    "create_token",
    "create_token_pair",
    "decode_token",
    "hash_password",
    "needs_rehash",
    "stable_bucket",
    "verify_password",
]
