from fastapi import APIRouter, Depends, Query

from api.auth import require_api_key
from api.schemas import PredictRequest, PredictResponse
from api.services import generate_predictions

router = APIRouter(tags=["predictions"], dependencies=[Depends(require_api_key)])


def _predict_from_params(
    *,
    confidence: float,
    train_limit: int,
    predict_limit: int,
    edge_margin: float | None,
    dry_run: bool,
    refresh_cache: bool,
    use_cache: bool,
    model_path: str | None,
) -> PredictResponse:
    data = generate_predictions(
        confidence=confidence,
        train_limit=train_limit,
        predict_limit=predict_limit,
        edge_margin=edge_margin,
        dry_run=dry_run,
        refresh_cache=refresh_cache,
        use_cache=use_cache,
        model_path=model_path,
    )
    return PredictResponse(**data)


@router.get("/predictions", response_model=PredictResponse)
def predict_get(
    confidence: float = Query(75.0, ge=0.0, le=100.0),
    train_limit: int = Query(0, ge=0),
    predict_limit: int = Query(50, ge=0),
    edge_margin: float | None = Query(None, ge=0.0, le=1.0),
    dry_run: bool = Query(False),
    refresh_cache: bool = Query(False),
    use_cache: bool = Query(True),
    model_path: str | None = Query(None),
) -> PredictResponse:
    return _predict_from_params(
        confidence=confidence,
        train_limit=train_limit,
        predict_limit=predict_limit,
        edge_margin=edge_margin,
        dry_run=dry_run,
        refresh_cache=refresh_cache,
        use_cache=use_cache,
        model_path=model_path,
    )


@router.post("/predictions", response_model=PredictResponse)
def predict_post(body: PredictRequest) -> PredictResponse:
    return _predict_from_params(
        confidence=body.confidence,
        train_limit=body.train_limit,
        predict_limit=body.predict_limit,
        edge_margin=body.edge_margin,
        dry_run=body.dry_run,
        refresh_cache=body.refresh_cache,
        use_cache=body.use_cache,
        model_path=body.model_path,
    )