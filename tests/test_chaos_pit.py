from datetime import UTC, datetime

from utils.chaos_cache import get_cached, intel_is_pit_safe, save_cached


def test_intel_stripped_when_fetched_after_match(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.chaos_cache.engine", __import__("utils.db", fromlist=["engine"]).engine)
    match_day = datetime(2025, 8, 1)
    save_cached({
        "HomeTeam": "Alpha",
        "AwayTeam": "Beta",
        "Date": "01/08/2025",
        "rain": 1.0,
        "wind": 2.0,
        "home_news_attention": 0.8,
        "away_news_attention": 0.6,
        "home_news_sentiment": 0.7,
        "away_news_sentiment": 0.6,
        "home_reddit_sentiment": 0.55,
        "away_reddit_sentiment": 0.45,
        "home_youtube_sentiment": 0.5,
        "away_youtube_sentiment": 0.5,
        "home_injuries": 0,
        "away_injuries": 0,
        "odds_H": 2.0,
        "odds_A": 3.0,
        "odds_D": 3.5,
        "fetched_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    })
    assert not intel_is_pit_safe(
        {"fetched_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat()},
        match_day,
    )
    cached = get_cached("Alpha", "Beta", "01/08/2025", match_dt=match_day)
    assert cached is not None
    assert cached["home_news_attention"] == 0.0
    assert cached["away_news_sentiment"] == 0.0
    assert cached["rain"] == 1.0