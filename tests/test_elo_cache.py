import pandas as pd
import pytest
from sqlalchemy import text

from utils.db import engine
from utils.elo_cache import (
    ELO_DEFAULT,
    ELO_TABLE,
    elo_on_date,
    ensure_elo_cache,
    save_elo_history,
)


def _clear_elo_cache() -> None:
    ensure_elo_cache()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {ELO_TABLE}"))
        conn.commit()


def test_elo_on_date_from_history():
    _clear_elo_cache()
    history = pd.DataFrame({
        "club_elo_name": ["Arsenal"],
        "Elo": [1820.5],
        "date_from": ["2024-08-01"],
        "date_to": ["2024-12-31"],
    })
    save_elo_history("Arsenal", history)

    match_dt = pd.Timestamp("2024-10-15")
    assert elo_on_date("Arsenal", match_dt) == pytest.approx(1820.5)


def test_elo_on_date_defaults_when_missing():
    _clear_elo_cache()
    assert elo_on_date("Unknown FC", pd.Timestamp("2024-10-15")) == ELO_DEFAULT


def test_elo_on_date_defaults_outside_range():
    _clear_elo_cache()
    history = pd.DataFrame({
        "club_elo_name": ["Chelsea"],
        "Elo": [1750.0],
        "date_from": ["2024-01-01"],
        "date_to": ["2024-06-30"],
    })
    save_elo_history("Chelsea", history)
    assert elo_on_date("Chelsea", pd.Timestamp("2024-12-01")) == ELO_DEFAULT