from datetime import UTC, datetime, timedelta

import pandas as pd

from utils.odds_cache import (
    cache_stats,
    clear_cache,
    ensure_odds_cache,
    is_fresh,
    load_cached,
    save_scrape,
)


def test_odds_cache_roundtrip_and_ttl():
    ensure_odds_cache()
    clear_cache()

    df = pd.DataFrame([{
        "HomeTeam": "Spain",
        "AwayTeam": "Saudi Arabia",
        "Div": "WC26",
        "Date": "20/06/2026",
        "B365H": 1.12,
        "B365D": 8.5,
        "B365A": 19.0,
        "FTHG": None,
        "FTAG": None,
        "FTR": None,
    }])
    save_scrape(df)
    assert is_fresh()
    cached = load_cached()
    assert cached is not None
    assert len(cached) == 1
    assert cached.iloc[0]["B365H"] == 1.12
    stats = cache_stats()
    assert stats["cached_matches"] == 1
    assert stats["fresh"] is True


def test_odds_cache_div_scoped_replace():
    ensure_odds_cache()
    clear_cache()
    wc = pd.DataFrame([{
        "HomeTeam": "Spain", "AwayTeam": "Saudi Arabia", "Div": "WC26", "Date": "20/06/2026",
        "B365H": 1.12, "B365D": 8.5, "B365A": 19.0,
        "FTHG": None, "FTAG": None, "FTR": None,
    }])
    e0 = pd.DataFrame([{
        "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Div": "E0", "Date": "15/08/2026",
        "B365H": 2.0, "B365D": 3.5, "B365A": 3.4,
        "FTHG": None, "FTAG": None, "FTR": None,
    }])
    save_scrape(wc, div_filter=["WC26"])
    save_scrape(e0, div_filter=["E0"])
    cached = load_cached()
    assert cached is not None
    assert len(cached) == 2
    assert set(cached["Div"]) == {"WC26", "E0"}


def test_odds_cache_stale_after_ttl(monkeypatch):
    ensure_odds_cache()
    clear_cache()
    df = pd.DataFrame([{
        "HomeTeam": "A", "AwayTeam": "B", "Div": "WC26", "Date": "01/01/2026",
        "B365H": 2.0, "B365D": 3.0, "B365A": 4.0,
        "FTHG": 1, "FTAG": 0, "FTR": "H",
    }])
    save_scrape(df)

    old = datetime.now(UTC) - timedelta(hours=10)
    monkeypatch.setattr("utils.odds_cache._latest_scraped_at", lambda: old)
    assert is_fresh() is False
    assert load_cached() is None