from datetime import timedelta
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api.app import create_app
from api.services import serialize_prediction
from config.settings import today


def _client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint():
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "trismegistus"


def test_root_lists_endpoints():
    response = _client().get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Trismegistus"
    assert "/predictions" in body["endpoints"]["predictions"]


def test_status_endpoint():
    response = _client().get("/status")
    assert response.status_code == 200
    body = response.json()
    assert "caches" in body
    assert "fixture_readiness" in body
    assert "matches_total" in body


def test_upcoming_fixtures_endpoint():
    future_date = (today() + timedelta(days=21)).strftime("%d/%m/%Y")
    sample = pd.DataFrame([
        {
            "Div": "E0",
            "Date": future_date,
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "B365H": 2.1,
            "B365D": 3.5,
            "B365A": 3.4,
        },
    ])
    with patch("api.services.read_matches", return_value=sample):
        with patch("api.services.get_future_matches", return_value=sample):
            response = _client().get("/fixtures/upcoming?limit=10")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Arsenal"


def test_predictions_empty_when_no_fixtures():
    empty = pd.DataFrame()
    with patch("api.services.read_matches", return_value=empty):
        with patch("api.services.get_future_matches", return_value=empty):
            response = _client().get("/predictions?train_limit=50&predict_limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == []
    assert body["message"] is not None


def test_backtest_endpoint_small_limit():
    response = _client().get("/backtest?limit=80")
    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body
    assert "holdout_accuracy_pct" in body["metrics"]
    assert body["div_codes"]


def test_serialize_prediction_formats_date_and_odds():
    serialized = serialize_prediction({
        "home": "A",
        "away": "B",
        "date": pd.Timestamp("2026-05-24"),
        "outcome": "Home Win",
        "outcome_code": 1,
        "confidence": 80.0,
        "b365": (2.0, 3.5, 4.0),
        "probs": {"H": 0.5, "D": 0.25, "A": 0.25},
        "intel": {"home_news_attention": 0.2},
    })
    assert serialized["date"] == "24/05/2026"
    assert serialized["b365"] == [2.0, 3.5, 4.0]
    assert serialized["intel"]["home_news_attention"] == 0.2