from fastapi import APIRouter, Query

from api.schemas import BacktestRequest, BacktestResponse
from api.services import run_backtest_summary

router = APIRouter(tags=["backtest"])


def _backtest_from_params(
    *,
    limit: int,
    use_cache: bool,
    refresh_cache: bool,
    edge_margin: float | None,
) -> BacktestResponse:
    data = run_backtest_summary(
        limit=limit,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        edge_margin=edge_margin,
    )
    return BacktestResponse(**data)


@router.get("/backtest", response_model=BacktestResponse)
def backtest_get(
    limit: int = Query(200, ge=0),
    use_cache: bool = Query(True),
    refresh_cache: bool = Query(False),
    edge_margin: float | None = Query(None, ge=0.0, le=1.0),
) -> BacktestResponse:
    return _backtest_from_params(
        limit=limit,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        edge_margin=edge_margin,
    )


@router.post("/backtest", response_model=BacktestResponse)
def backtest_post(body: BacktestRequest) -> BacktestResponse:
    return _backtest_from_params(
        limit=body.limit,
        use_cache=body.use_cache,
        refresh_cache=body.refresh_cache,
        edge_margin=body.edge_margin,
    )