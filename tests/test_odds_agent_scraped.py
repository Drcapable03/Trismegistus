import pandas as pd

from agents.odds_agent import _match_names, fetch_odds_scraped


def test_match_names_uses_oddsportal_aliases():
    df = pd.DataFrame([{
        "HomeTeam": "Manchester Utd",
        "AwayTeam": "Liverpool",
        "B365H": 2.5,
        "B365D": 3.4,
        "B365A": 2.8,
        "Div": "E0",
    }])
    row = _match_names("Man United", "Liverpool", df)
    assert row is not None
    assert row["B365H"] == 2.5


def test_fetch_odds_scraped_from_cache(monkeypatch):
    cached = pd.DataFrame([{
        "HomeTeam": "Arsenal",
        "AwayTeam": "Chelsea",
        "B365H": 2.1,
        "B365D": 3.5,
        "B365A": 3.2,
        "Div": "E0",
    }])

    monkeypatch.setattr("agents.odds_agent._scraped_odds_df", lambda force_refresh=False: cached)
    odds = fetch_odds_scraped("Arsenal", "Chelsea", "2026-08-15", div_code="E0")
    assert odds is not None
    assert odds["source"] == "oddsportal"
    assert odds["H"] == 2.1