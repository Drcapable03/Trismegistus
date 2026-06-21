from datetime import UTC, datetime

from utils.chaos_cache import get_cached, save_cached, sentiment_is_pit_safe


def test_sentiment_stripped_when_fetched_after_match(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.chaos_cache.engine", __import__("utils.db", fromlist=["engine"]).engine)
    match_day = datetime(2025, 8, 1)
    save_cached({
        "HomeTeam": "Alpha",
        "AwayTeam": "Beta",
        "Date": "01/08/2025",
        "rain": 1.0,
        "wind": 2.0,
        "home_x_sentiment": 0.8,
        "away_x_sentiment": 0.6,
        "home_injuries": 0,
        "away_injuries": 0,
        "odds_H": 2.0,
        "odds_A": 3.0,
        "odds_D": 3.5,
        "fetched_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    })
    assert not sentiment_is_pit_safe(
        {"fetched_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat()},
        match_day,
    )
    cached = get_cached("Alpha", "Beta", "01/08/2025", match_dt=match_day)
    assert cached is not None
    assert cached["home_x_sentiment"] == 0.0
    assert cached["away_x_sentiment"] == 0.0
    assert cached["rain"] == 1.0