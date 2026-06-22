from datetime import timedelta

import pandas as pd
import pytest

from config.settings import today
from evaluation.odds_lines import resolve_b365_for_live
from predictors.game_forger import _is_completed
from scripts.validate_live_predict import (
    ValidationCheck,
    e2e_smoke_predict,
    fixture_readiness,
    format_validation_report,
    run_validation_checks,
)


def test_fixture_readiness_counts():
    future_date = (today() + timedelta(days=30)).strftime("%d/%m/%Y")
    past_date = (today() - timedelta(days=30)).strftime("%d/%m/%Y")
    df = pd.DataFrame([
        {"Div": "E0", "Date": past_date, "HomeTeam": "A", "AwayTeam": "B", "FTR": "H", "FTHG": 1, "FTAG": 0},
        {"Div": "E0", "Date": future_date, "HomeTeam": "C", "AwayTeam": "D", "B365H": 2.0, "B365D": 3.5, "B365A": 3.2},
        {"Div": "SP2", "Date": future_date, "HomeTeam": "X", "AwayTeam": "Y"},
    ])
    report = fixture_readiness(df)
    assert report["upcoming_big5"] == 1
    assert report["completed_big5"] == 1
    assert report["ready_for_live_predict"] is True


def test_fixture_readiness_empty_big5():
    df = pd.DataFrame([
        {"Div": "SP2", "Date": "30/05/2026", "HomeTeam": "X", "AwayTeam": "Y", "FTR": None},
    ])
    report = fixture_readiness(df)
    assert report["upcoming_big5"] == 0
    assert report["ready_for_live_predict"] is False


def test_format_validation_report():
    checks = [
        ValidationCheck("a", True, "ok"),
        ValidationCheck("b", False, "bad"),
    ]
    text = format_validation_report(checks)
    assert "[PASS] a" in text
    assert "[FAIL] b" in text
    assert "1/2 checks passed" in text


def test_run_validation_checks_without_smoke():
    checks = run_validation_checks(train_limit=50, run_smoke=False)
    names = {c.name for c in checks}
    assert "enabled_leagues" in names
    assert "closing_odds_resolution" in names
    assert "e2e_smoke_predict" not in names
    static_checks = {
        "enabled_leagues",
        "intel_train_policy",
        "per_league_edges",
        "closing_odds_resolution",
    }
    by_name = {c.name: c for c in checks}
    assert all(by_name[name].passed for name in static_checks)


def _db_has_training_data() -> bool:
    return fixture_readiness()["completed_big5"] >= 100


def test_e2e_smoke_predict_scores_synthetic_fixture():
    if not _db_has_training_data():
        pytest.skip("needs ingested Big 5 match DB")
    ok, detail, preds = e2e_smoke_predict(train_limit=60)
    assert ok, detail
    assert len(preds) == 1
    assert preds[0]["home"]
    assert preds[0]["away"]
    assert resolve_b365_for_live(pd.Series({
        "B365CH": 2.0, "B365CD": 3.6, "B365CA": 3.5,
    })) == (2.0, 3.6, 3.5)