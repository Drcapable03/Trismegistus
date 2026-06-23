import pandas as pd
import pytest

from utils.fbref_cache import parse_fbref_fixtures, save_match_rows, load_fbref_xg, FBREF_TABLE
from utils.db import engine
from sqlalchemy import text


def _clear_fbref_cache() -> None:
    from utils.fbref_cache import ensure_fbref_cache
    ensure_fbref_cache()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {FBREF_TABLE}"))
        conn.commit()


def test_parse_fbref_fixtures():
    df = pd.DataFrame({
        "team_home": ["Arsenal", "Chelsea"],
        "team_away": ["Chelsea", "Liverpool"],
        "xg_home": [1.2, 0.8],
        "xg_away": [0.9, 1.5],
        "datetime": pd.to_datetime(["2024-08-17", "2024-08-24"]),
    })
    rows = parse_fbref_fixtures(df, "E0", "2425")
    assert len(rows) == 2
    assert rows[0]["xg_home"] == pytest.approx(1.2)
    assert rows[0]["Div"] == "E0"


def test_fbref_cache_roundtrip():
    _clear_fbref_cache()
    row = {
        "HomeTeam": "Arsenal",
        "AwayTeam": "Chelsea",
        "MatchDate": "17/08/2024",
        "Div": "E0",
        "xg_home": 1.2,
        "xg_away": 0.9,
        "season": "2425",
    }
    assert save_match_rows([row]) == 1
    loaded = load_fbref_xg("E0")
    assert len(loaded) == 1
    assert loaded.iloc[0]["xg_home"] == pytest.approx(1.2)
    _clear_fbref_cache()