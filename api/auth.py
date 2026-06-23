"""Optional API-key authentication for protected endpoints."""

from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def api_key() -> str | None:
    value = os.getenv("TRIS_API_KEY", "").strip()
    return value or None


def api_auth_enabled() -> bool:
    return api_key() is not None


async def require_api_key(
    key: str | None = Security(API_KEY_HEADER),
) -> None:
    """Reject requests when TRIS_API_KEY is set and header is missing or wrong."""
    expected = api_key()
    if expected is None:
        return
    if not key or key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Send X-API-Key header.",
        )