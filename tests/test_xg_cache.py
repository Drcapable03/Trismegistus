import pytest
from sqlalchemy import text

from utils.db import engine
from utils.xg_cache import (
    XG_TABLE,
    ensure_xg_cache,
    parse_understat_match,
    save_match_rows,
    xg_cache_stats,
)


def _clear_xg_cache() -> None:
    ensure_xg_cache()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {XG_TABLE}"))
        conn.commit()


def test_parse_understat_match_completed():
    raw = {
        "isResult": True,
        "datetime": "2024-08-16 20:00:00",
        "h": {"title": "Arsenal"},
        "a": {"title": "Wolves"},
        "xG": {"h": "1.82", "a": "0.91"},
    }
    row = parse_understat_match(raw, div="E0", season="2024")
    assert row is not None
    assert row["HomeTeam"] == "Arsenal"
    assert row["AwayTeam"] == "Wolves"
    assert row["MatchDate"] == "16/08/2024"
    assert row["xg_home"] == pytest.approx(1.82)
    assert row["xg_away"] == pytest.approx(0.91)


def test_parse_understat_match_skips_unplayed():
    assert parse_understat_match({"isResult": False}, "E0", "2024") is None


def test_xg_cache_roundtrip():
    _clear_xg_cache()
    rows = [{
        "HomeTeam": "Arsenal",
        "AwayTeam": "Chelsea",
        "MatchDate": "01/09/2024",
        "Div": "E0",
        "xg_home": 1.5,
        "xg_away": 1.1,
        "season": "2024",
    }]
    n = save_match_rows(rows)
    assert n == 1
    stats = xg_cache_stats()
    assert stats["understat_matches"] == 1