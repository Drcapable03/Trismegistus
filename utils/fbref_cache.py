"""SQLite cache for FBref match-level xG (via penaltyblog scraper)."""

from datetime import datetime

import pandas as pd
from sqlalchemy import inspect, text

from utils.db import engine
from utils.team_aliases import from_fbref

FBREF_TABLE = "fbref_xg"


def ensure_fbref_cache() -> None:
    if inspect(engine).has_table(FBREF_TABLE):
        return
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE {FBREF_TABLE} (
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
    ensure_fbref_cache()
    if not rows:
        return 0
    ts = datetime.utcnow().isoformat()
    with engine.connect() as conn:
        for row in rows:
            conn.execute(
                text(f"""
                    INSERT OR REPLACE INTO {FBREF_TABLE}
                    (HomeTeam, AwayTeam, MatchDate, Div, xg_home, xg_away, season, fetched_at)
                    VALUES (:HomeTeam, :AwayTeam, :MatchDate, :Div, :xg_home, :xg_away, :season, :fetched_at)
                """),
                {**row, "fetched_at": ts},
            )
        conn.commit()
    return len(rows)


def load_fbref_xg(div: str | None = None) -> pd.DataFrame:
    ensure_fbref_cache()
    if div:
        return pd.read_sql(
            f'SELECT * FROM {FBREF_TABLE} WHERE "Div" = :div',
            engine,
            params={"div": div},
        )
    return pd.read_sql(f"SELECT * FROM {FBREF_TABLE}", engine)


def fbref_cache_stats() -> dict:
    ensure_fbref_cache()
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {FBREF_TABLE}", engine)["n"][0]
    return {"fbref_matches": int(n)}


def parse_fbref_fixtures(df: pd.DataFrame, div: str, season: str) -> list[dict]:
    """Convert penaltyblog FBref fixtures frame to cache rows."""
    rows: list[dict] = []
    if df is None or df.empty:
        return rows

    frame = df.reset_index(drop=True)
    for _, row in frame.iterrows():
        xg_home = row.get("xg_home")
        xg_away = row.get("xg_away")
        if pd.isna(xg_home) or pd.isna(xg_away):
            continue
        dt = row.get("datetime") or row.get("date")
        if dt is None or pd.isna(dt):
            continue
        dt_parsed = pd.to_datetime(dt)
        home = from_fbref(str(row.get("team_home", "")))
        away = from_fbref(str(row.get("team_away", "")))
        if not home or not away:
            continue
        rows.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "MatchDate": dt_parsed.strftime("%d/%m/%Y"),
            "Div": div,
            "xg_home": float(xg_home),
            "xg_away": float(xg_away),
            "season": season,
        })
    return rows