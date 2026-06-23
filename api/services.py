"""Service layer: wraps pipeline logic for JSON API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import edge_margin_min, league_div_codes, league_summary
from evaluation.edge import selective_accuracy
from evaluation.implied_odds import bookie_accuracy
from evaluation.metrics import flat_stake_roi
from main import DEFAULT_INJURIES, get_future_matches
from predictors.game_forger import GameForger, _is_completed, _parse_dates
from predictors.registry import load_game_forger
from scripts.expand_history import season_coverage
from scripts.prep_caches import cache_snapshot
from scripts.validate_live_predict import _fixture_guidance, fixture_readiness
from utils.db import read_matches


def _serialize_date(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")
    return str(value)


def _serialize_odds(value: tuple[float, float, float] | list[float] | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value]


def serialize_prediction(pred: dict) -> dict:
    """Normalize a GameForger prediction dict for JSON/Pydantic."""
    out = dict(pred)
    out["date"] = _serialize_date(out.get("date"))
    for key in ("b365", "b365_open", "b365_close"):
        if key in out:
            out[key] = _serialize_odds(out.get(key))
    probs = out.get("probs")
    if probs:
        out["probs"] = {k: float(v) for k, v in probs.items()}
    intel = out.get("intel") or {}
    out["intel"] = {k: float(v) for k, v in intel.items()}
    return out


def get_status() -> dict:
    matches = read_matches()
    completed = int(_is_completed(matches).sum()) if not matches.empty else 0
    readiness = fixture_readiness(matches)
    next_date = readiness.get("next_fixture_date")
    if next_date is not None and hasattr(next_date, "isoformat"):
        next_dt = next_date
    else:
        next_dt = None

    return {
        "version": "0.3.0",
        "leagues": league_summary(),
        "matches_total": len(matches),
        "matches_completed": completed,
        "caches": cache_snapshot(),
        "fixture_readiness": {
            "as_of": readiness["as_of"],
            "div_codes": readiness["div_codes"],
            "completed_big5": readiness["completed_big5"],
            "uncompleted_big5": readiness["uncompleted_big5"],
            "upcoming_big5": readiness["upcoming_big5"],
            "next_fixture_date": next_dt,
            "ready_for_live_predict": readiness["ready_for_live_predict"],
            "guidance": _fixture_guidance(readiness),
        },
        "season_coverage": season_coverage(matches).to_dict(orient="records")
        if not matches.empty
        else [],
    }


def list_upcoming_fixtures(*, predict_limit: int = 50) -> list[dict]:
    div_codes = league_div_codes()
    matches = read_matches()
    future = get_future_matches(matches, div_filter=div_codes)
    if future.empty:
        return []
    batch = future.head(predict_limit) if predict_limit > 0 else future
    rows = []
    for _, row in batch.iterrows():
        rows.append({
            "div": str(row.get("Div", "")),
            "date": _serialize_date(row.get("Date")),
            "home_team": str(row["HomeTeam"]),
            "away_team": str(row["AwayTeam"]),
            "b365_h": _optional_float(row.get("B365H")),
            "b365_d": _optional_float(row.get("B365D")),
            "b365_a": _optional_float(row.get("B365A")),
        })
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def generate_predictions(
    *,
    confidence: float = 75.0,
    train_limit: int = 0,
    predict_limit: int = 50,
    edge_margin: float | None = None,
    dry_run: bool = False,
    refresh_cache: bool = False,
    use_cache: bool = True,
    model_path: str | None = None,
) -> dict:
    div_codes = league_div_codes()
    matches = read_matches()
    future = get_future_matches(matches, div_filter=div_codes)

    meta = {
        "div_codes": div_codes,
        "upcoming_fixtures": len(future),
        "scored_fixtures": 0,
        "dry_run": dry_run,
        "confidence_threshold": confidence,
    }

    if future.empty:
        return {
            "predictions": [],
            "meta": meta,
            "message": (
                "No upcoming Big 5 fixtures in DB. Run ingest after football-data "
                "publishes fixtures.csv rows."
            ),
        }

    train_n = None if train_limit == 0 else train_limit
    batch = future.head(predict_limit) if predict_limit > 0 else future
    meta["scored_fixtures"] = len(batch)

    forger = GameForger()
    if model_path:
        load_game_forger(forger, model_path)
        forger.prepare_training_data(
            injuries_df=DEFAULT_INJURIES,
            limit=train_n,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            div_filter=div_codes,
            chaos_cache_only=True,
        )
        forger.fit_dc_baseline()
    else:
        forger.train(
            injuries_df=DEFAULT_INJURIES,
            limit=train_n,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            div_filter=div_codes,
            chaos_cache_only=True,
        )

    forger.prepare_prediction_data(
        batch,
        injuries_df=DEFAULT_INJURIES,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        div_filter=div_codes,
        chaos_cache_only=dry_run,
    )
    raw = forger.predict(
        confidence_threshold=confidence,
        edge_margin=edge_margin,
        use_simulation=False,
    )
    predictions = [serialize_prediction(p) for p in raw]
    message = None
    if not predictions:
        message = "No edge-qualified picks for the requested fixtures and thresholds."

    return {"predictions": predictions, "meta": meta, "message": message}


def run_backtest_summary(
    *,
    limit: int = 200,
    use_cache: bool = True,
    refresh_cache: bool = False,
    edge_margin: float | None = None,
) -> dict:
    div_codes = league_div_codes()
    forger = GameForger()
    forger.train(
        injuries_df=DEFAULT_INJURIES,
        limit=limit,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        div_filter=div_codes,
        chaos_cache_only=not refresh_cache,
    )

    margin = edge_margin if edge_margin is not None else edge_margin_min()
    all_preds = forger.backtest_on_holdout(
        confidence_threshold=0.0,
        require_edge=False,
        edge_margin=0.0,
    )
    selective_preds = forger.backtest_on_holdout(
        confidence_threshold=0.0,
        require_edge=True,
        edge_margin=edge_margin,
    )

    holdout_accuracy = 0.0
    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        holdout_accuracy = float(
            (pred_df["outcome_code"] == pred_df["actual_code"]).mean() * 100
        )

    bookie_acc = 0.0
    matches = read_matches()
    matches = matches[matches["Div"].isin(div_codes)] if not matches.empty else matches
    if all_preds and not matches.empty:
        m = matches.copy()
        m["Date"] = pd.to_datetime(m["Date"], dayfirst=True, errors="coerce").dt.strftime(
            "%d/%m/%Y"
        )
        p = pd.DataFrame(all_preds)
        p["date"] = pd.to_datetime(p["date"], dayfirst=True, errors="coerce").dt.strftime(
            "%d/%m/%Y"
        )
        merged = m.merge(
            p,
            left_on=["HomeTeam", "AwayTeam", "Date"],
            right_on=["home", "away", "date"],
            how="inner",
        )
        if not merged.empty and {"B365H", "B365A", "B365D", "FTR"}.issubset(merged.columns):
            bookie_acc = float(bookie_accuracy(merged))

    sel_acc, sel_n = selective_accuracy(selective_preds)
    roi, roi_bets, _ = flat_stake_roi(selective_preds)
    training = forger.training_metadata or {}

    return {
        "metrics": {
            "holdout_accuracy_pct": holdout_accuracy,
            "bookie_accuracy_pct": bookie_acc,
            "selective_accuracy_pct": float(sel_acc),
            "selective_picks": int(sel_n),
            "all_picks": len(all_preds),
            "selective_roi_pct": float(roi),
            "train_matches": training.get("train_matches"),
            "test_matches": training.get("test_matches"),
        },
        "div_codes": div_codes,
    }