"""SQLite cache for Understat match-level xG."""

from datetime import datetime

import pandas as pd
from sqlalchemy import inspect, text

from utils.db import engine
from utils.team_aliases import from_understat

XG_TABLE = "understat_xg"


def ensure_xg_cache() -> None:
    if inspect(engine).has_table(XG_TABLE):
        return
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE {XG_TABLE} (
                HomeTeam TEXT,
                AwayTeam TEXT,
                MatchDate TEXT,
                Div TEXT,
                xg_home REAL,
                xg_away REAL,
                season TEXT,
                fetched_at TEXT,
                PRIMARY KEY (HomeTeam, AwayTeam, MatchDate)
            )
        """))
        conn.commit()


def save_match_rows(rows: list[dict]) -> int:
    ensure_xg_cache()
    if not rows:
        return 0
    ts = datetime.utcnow().isoformat()
    with engine.connect() as conn:
        for row in rows:
            conn.execute(
                text(f"""
                    INSERT OR REPLACE INTO {XG_TABLE}
                    (HomeTeam, AwayTeam, MatchDate, Div, xg_home, xg_away, season, fetched_at)
                    VALUES (:HomeTeam, :AwayTeam, :MatchDate, :Div, :xg_home, :xg_away, :season, :fetched_at)
                """),
                {**row, "fetched_at": ts},
            )
        conn.commit()
    return len(rows)


def load_xg_matches(div: str | None = None) -> pd.DataFrame:
    ensure_xg_cache()
    if div:
        return pd.read_sql(
            f'SELECT * FROM {XG_TABLE} WHERE "Div" = :div',
            engine,
            params={"div": div},
        )
    return pd.read_sql(f"SELECT * FROM {XG_TABLE}", engine)


def xg_cache_stats() -> dict:
    ensure_xg_cache()
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {XG_TABLE}", engine)["n"][0]
    return {"understat_matches": int(n)}


def parse_understat_match(match: dict, div: str, season: str) -> dict | None:
    if not match.get("isResult"):
        return None
    dt_raw = match.get("datetime")
    if not dt_raw:
        return None
    dt = datetime.strptime(dt_raw, "%Y-%m-%d %H:%M:%S")
    home = from_understat(match["h"]["title"])
    away = from_understat(match["a"]["title"])
    return {
        "HomeTeam": home,
        "AwayTeam": away,
        "MatchDate": dt.strftime("%d/%m/%Y"),
        "Div": div,
        "xg_home": float(match["xG"]["h"]),
        "xg_away": float(match["xG"]["a"]),
        "season": season,
    }