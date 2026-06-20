from utils.chaos_cache import cache_stats, ensure_chaos_cache, get_cached, save_cached


def test_chaos_cache_roundtrip():
    ensure_chaos_cache()
    record = {
        "HomeTeam": "TestA",
        "AwayTeam": "TestB",
        "Date": "01/01/2026",
        "rain": 1.2,
        "wind": 5.0,
        "home_x_sentiment": 0.2,
        "away_x_sentiment": 0.3,
        "home_injuries": 1,
        "away_injuries": 0,
        "odds_H": 2.1,
        "odds_A": 3.5,
        "odds_D": 3.2,
    }
    save_cached(record)
    cached = get_cached("TestA", "TestB", "01/01/2026")
    assert cached is not None
    assert cached["rain"] == 1.2
    assert cache_stats()["cached_matches"] >= 1