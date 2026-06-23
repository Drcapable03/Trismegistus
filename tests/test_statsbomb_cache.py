import pandas as pd
import pytest

from utils.statsbomb_cache import (
    STATSBOMB_TABLE,
    match_xg_from_events,
    parse_statsbomb_match,
    save_match_rows,
    load_statsbomb_xg,
)
from utils.db import engine
from sqlalchemy import text


def _clear_statsbomb_cache() -> None:
    from utils.statsbomb_cache import ensure_statsbomb_cache
    ensure_statsbomb_cache()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {STATSBOMB_TABLE}"))
        conn.commit()


def test_match_xg_from_events():
    events = pd.DataFrame({
        "type": ["Shot", "Shot", "Pass"],
        "team": ["Barcelona", "Celta Vigo", "Barcelona"],
        "shot_statsbomb_xg": [0.4, 0.2, None],
    })
    home, away = match_xg_from_events(events, "Barcelona", "Celta Vigo")
    assert home == pytest.approx(0.4)
    assert away == pytest.approx(0.2)


def test_statsbomb_cache_roundtrip():
    _clear_statsbomb_cache()  # isolate from other tests / prior runs
    match = pd.Series({
        "match_id": 99,
        "match_date": "2020-10-31",
        "home_team": "Barcelona",
        "away_team": "Celta Vigo",
        "competition_id": 11,
        "season_id": 90,
    })
    row = parse_statsbomb_match(match, 1.1, 0.7, "SP1")
    assert save_match_rows([row]) == 1
    loaded = load_statsbomb_xg("SP1")
    assert len(loaded) == 1
    assert loaded.iloc[0]["xg_home"] == pytest.approx(1.1)
    _clear_statsbomb_cache()