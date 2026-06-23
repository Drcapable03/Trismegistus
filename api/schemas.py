"""Pydantic request/response models for the Trismegistus API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "trismegistus"


class CacheStats(BaseModel):
    understat: int = 0
    statsbomb: int = 0
    fbref: int = 0
    chaos: int = 0


class FixtureReadiness(BaseModel):
    as_of: datetime
    div_codes: list[str]
    completed_big5: int
    uncompleted_big5: int
    upcoming_big5: int
    next_fixture_date: datetime | None = None
    ready_for_live_predict: bool
    guidance: str


class StatusResponse(BaseModel):
    version: str
    leagues: str
    matches_total: int
    matches_completed: int
    caches: CacheStats
    fixture_readiness: FixtureReadiness


class FixtureItem(BaseModel):
    div: str
    date: str
    home_team: str
    away_team: str
    b365_h: float | None = None
    b365_d: float | None = None
    b365_a: float | None = None


class PredictRequest(BaseModel):
    confidence: float = Field(75.0, ge=0.0, le=100.0)
    train_limit: int = Field(0, ge=0, description="0 = all completed matches")
    predict_limit: int = Field(50, ge=0, description="0 = all upcoming fixtures")
    edge_margin: float | None = Field(None, ge=0.0, le=1.0)
    dry_run: bool = Field(False, description="Use ingested odds only; skip live intel scrape")
    refresh_cache: bool = False
    use_cache: bool = True
    model_path: str | None = None


class PredictionProbabilities(BaseModel):
    H: float
    D: float
    A: float


class PredictionItem(BaseModel):
    home: str
    away: str
    date: str
    div: str | None = None
    outcome: str
    outcome_code: int
    confidence: float
    edge: float | None = None
    edge_margin: float | None = None
    expected_goals: float | None = None
    total_goals: int | None = None
    probs: PredictionProbabilities | None = None
    bookie_pick: str | None = None
    b365: list[float] | None = None
    b365_open: list[float] | None = None
    b365_close: list[float] | None = None
    intel: dict[str, float] = Field(default_factory=dict)


class PredictMeta(BaseModel):
    div_codes: list[str]
    upcoming_fixtures: int
    scored_fixtures: int
    dry_run: bool
    confidence_threshold: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    meta: PredictMeta
    message: str | None = None


class BacktestRequest(BaseModel):
    limit: int = Field(200, ge=0, description="0 = all completed matches")
    use_cache: bool = True
    refresh_cache: bool = False
    edge_margin: float | None = None


class BacktestMetrics(BaseModel):
    holdout_accuracy_pct: float
    bookie_accuracy_pct: float
    selective_accuracy_pct: float
    selective_picks: int
    all_picks: int
    selective_roi_pct: float
    train_matches: int | None = None
    test_matches: int | None = None


class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    div_codes: list[str]


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str
    ui: str
    endpoints: dict[str, str]


class AuthConfigResponse(BaseModel):
    auth_required: bool