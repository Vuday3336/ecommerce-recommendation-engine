"""Authentication endpoints.

Password login against the `users` table, plus a development-only token issuer
so the admin dashboard is usable before any user has been seeded.

**The dev endpoint is hard-gated on `APP_ENV == "development"` and returns 404
otherwise.** Not 403 - 404, so its existence is not even discoverable in a
deployed environment. An endpoint that mints admin tokens on request is exactly
the kind of convenience that ends up enabled in production by accident, so the
guard is on the environment rather than on a feature flag someone could flip.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import rate_limit_auth, require_principal
from app.core.config import settings
from app.core.security import (
    AuthError,
    Principal,
    create_token_pair,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.session import get_session
from app.models.enums import UserRole
from app.models.users import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(rate_limit_auth)])

SessionDep = Annotated[Session, Depends(get_session)]


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: Annotated[str, Field(min_length=8, max_length=128)]


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: Annotated[str, Field(min_length=8, max_length=128)]
    full_name: Annotated[str, Field(min_length=1, max_length=200)]


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    role: str


class MeResponse(BaseModel):
    user_id: int
    public_id: str
    role: str


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, session: SessionDep) -> TokenResponse:
    existing = session.scalar(select(User).where(User.email == payload.email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="email already registered"
        )

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=UserRole.CUSTOMER,
    )
    session.add(user)
    session.flush()

    tokens = create_token_pair(
        user_id=user.id, public_id=str(user.public_id), role=user.role
    )
    return TokenResponse(**tokens, user_id=user.id, role=user.role.value)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, session: SessionDep) -> TokenResponse:
    user = session.scalar(select(User).where(User.email == payload.email))

    # One generic message and a hash computed either way, so neither the text
    # nor the timing distinguishes "no such account" from "wrong password".
    if user is None or not verify_password(payload.password, user.password_hash):
        if user is None:
            verify_password(payload.password, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tokens = create_token_pair(
        user_id=user.id, public_id=str(user.public_id), role=user.role
    )
    return TokenResponse(**tokens, user_id=user.id, role=user.role.value)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest) -> TokenResponse:
    try:
        principal = decode_token(payload.refresh_token, expected_type="refresh")
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc

    tokens = create_token_pair(
        user_id=principal.user_id, public_id=principal.public_id, role=principal.role
    )
    return TokenResponse(**tokens, user_id=principal.user_id, role=principal.role.value)


@router.get("/me", response_model=MeResponse)
def me(principal: Annotated[Principal, Depends(require_principal)]) -> MeResponse:
    return MeResponse(
        user_id=principal.user_id,
        public_id=principal.public_id,
        role=principal.role.value,
    )


class DevTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Annotated[int, Field(gt=0)] = 42
    role: UserRole = UserRole.ADMIN


@router.post(
    "/dev-token",
    response_model=TokenResponse,
    summary="Development-only token issuer",
    include_in_schema=False,
)
def dev_token(payload: DevTokenRequest) -> TokenResponse:
    """Mint a token without a password. Development environments only.

    Exists so the admin dashboard and the seeded demo users are usable before
    any account has been created - the synthetic dataset has 10,000 users and
    none of them have password hashes, by design.
    """
    if settings.app_env != "development":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    tokens = create_token_pair(
        user_id=payload.user_id, public_id=f"dev-{payload.user_id}", role=payload.role
    )
    logger.warning(
        "issued a development token for user %s with role %s",
        payload.user_id,
        payload.role.value,
    )
    return TokenResponse(**tokens, user_id=payload.user_id, role=payload.role.value)


__all__ = ["router"]
