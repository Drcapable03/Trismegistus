import pandas as pd
import pytest
from sqlalchemy import text

from utils.db import engine
from utils.elo_cache import ELO_TABLE, ensure_elo_cache, save_elo_history
from utils.pit_features import compute_pit_form_and_h2h


def _seed_elo(team: str, elo: float, date_from: str, date_to: str) -> None:
    ensure_elo_cache()
    history = pd.DataFrame({
        "club_elo_name": [team],
        "Elo": [elo],
        "date_from": [date_from],
        "date_to": [date_to],
    })
    save_elo_history(team, history)


def test_pit_elo_columns_from_cache():
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {ELO_TABLE}"))
        conn.commit()
    _seed_elo("Alpha", 1600.0, "2025-01-01", "2025-12-31")
    _seed_elo("Beta", 1400.0, "2025-01-01", "2025-12-31")

    history = pd.DataFrame({
        "HomeTeam": ["Alpha", "Beta"],
        "AwayTeam": ["Beta", "Alpha"],
        "FTHG": [2, 1],
        "FTAG": [1, 2],
        "FTR": ["H", "A"],
        "Date": ["01/01/2025", "01/02/2025"],
    })
    target = pd.DataFrame({
        "HomeTeam": ["Alpha"],
        "AwayTeam": ["Beta"],
        "Date": ["01/03/2025"],
    })
    out = compute_pit_form_and_h2h(target, history)
    assert out.iloc[0]["elo_home"] == pytest.approx(1600.0)
    assert out.iloc[0]["elo_away"] == pytest.approx(1400.0)
    assert out.iloc[0]["elo_diff"] == pytest.approx(200.0)