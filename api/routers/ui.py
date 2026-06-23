from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from api.auth import api_auth_enabled
from api.schemas import AuthConfigResponse

router = APIRouter(tags=["ui"])

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "static"


@router.get("/auth/config", response_model=AuthConfigResponse)
def auth_config() -> AuthConfigResponse:
    return AuthConfigResponse(auth_required=api_auth_enabled())


@router.get("/ui")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")