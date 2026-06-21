from utils.chaos_cache import INTEL_COLS, cache_stats, ensure_chaos_cache, get_cached, save_cached


def test_chaos_cache_roundtrip():
    ensure_chaos_cache()
    record = {
        "HomeTeam": "TestA",
        "AwayTeam": "TestB",
        "Date": "01/01/2026",
        "rain": 1.2,
        "wind": 5.0,
        "home_news_attention": 0.2,
        "away_news_attention": 0.3,
        "home_news_sentiment": 0.6,
        "away_news_sentiment": 0.4,
        "home_reddit_sentiment": 0.5,
        "away_reddit_sentiment": 0.5,
        "home_youtube_sentiment": 0.5,
        "away_youtube_sentiment": 0.5,
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
    assert cached["home_news_attention"] == 0.2
    for col in INTEL_COLS:
        assert col in cached
    assert cache_stats()["cached_matches"] >= 1