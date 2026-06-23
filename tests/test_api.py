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
    response = _client().get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Trismegistus"
    assert body["ui"] == "/ui"
    assert "/predictions" in body["endpoints"]["predictions"]


def test_root_redirects_browsers_to_ui():
    response = _client().get("/", headers={"Accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/ui"


def test_ui_dashboard_served():
    response = _client().get("/ui")
    assert response.status_code == 200
    assert "Trismegistus" in response.text


def test_static_assets_served():
    response = _client().get("/static/app.js")
    assert response.status_code == 200
    assert "apiFetch" in response.text


def test_auth_config_endpoint():
    response = _client().get("/auth/config")
    assert response.status_code == 200
    assert response.json()["auth_required"] is False


def test_status_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("TRIS_API_KEY", "test-secret-key")
    fake_status = {
        "version": "0.3.0",
        "leagues": "Big 5",
        "matches_total": 100,
        "matches_completed": 90,
        "caches": {"understat": 90, "statsbomb": 10, "fbref": 0, "chaos": 50},
        "fixture_readiness": {
            "as_of": today(),
            "div_codes": ["E0"],
            "completed_big5": 90,
            "uncompleted_big5": 10,
            "upcoming_big5": 0,
            "next_fixture_date": None,
            "ready_for_live_predict": False,
            "guidance": "No fixtures.",
        },
    }
    with patch("api.routers.status.get_status", return_value=fake_status):
        client = TestClient(create_app())
        assert client.get("/status").status_code == 401
        assert client.get("/status", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/status", headers={"X-API-Key": "test-secret-key"}).status_code == 200


def test_health_public_when_auth_configured(monkeypatch):
    monkeypatch.setenv("TRIS_API_KEY", "test-secret-key")
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/ui").status_code == 200
    assert client.get("/auth/config").status_code == 200


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
    fake = {
        "metrics": {
            "holdout_accuracy_pct": 55.0,
            "bookie_accuracy_pct": 50.0,
            "selective_accuracy_pct": 60.0,
            "selective_picks": 12,
            "all_picks": 80,
            "selective_roi_pct": -5.0,
            "train_matches": 64,
            "test_matches": 16,
        },
        "div_codes": ["E0", "SP1", "D1", "I1", "F1"],
    }
    with patch("api.routers.backtest.run_backtest_summary", return_value=fake):
        response = _client().get("/backtest?limit=80")
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["holdout_accuracy_pct"] == 55.0
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