from fastapi import APIRouter, Query

from api.schemas import FixtureItem, StatusResponse
from api.services import get_status, list_upcoming_fixtures

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    data = get_status()
    return StatusResponse(
        version=data["version"],
        leagues=data["leagues"],
        matches_total=data["matches_total"],
        matches_completed=data["matches_completed"],
        caches=data["caches"],
        fixture_readiness=data["fixture_readiness"],
    )


@router.get("/fixtures/upcoming", response_model=list[FixtureItem])
def upcoming_fixtures(
    limit: int = Query(50, ge=0, description="0 = all upcoming Big 5 fixtures"),
) -> list[FixtureItem]:
    return list_upcoming_fixtures(predict_limit=limit)