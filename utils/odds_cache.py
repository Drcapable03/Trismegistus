"""Persistent TTL cache for OddsPortal World Cup scrape results."""

from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import inspect, text

from config.settings import get_env
from utils.db import engine

CACHE_TABLE = "odds_portal_cache"
DEFAULT_TTL_HOURS = 6


def ttl_hours() -> float:
    return float(get_env("ODDS_CACHE_TTL_HOURS", str(DEFAULT_TTL_HOURS)))


def ensure_odds_cache() -> None:
    if inspect(engine).has_table(CACHE_TABLE):
        return
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE {CACHE_TABLE} (
                HomeTeam TEXT,
                AwayTeam TEXT,
                Div TEXT,
                Date TEXT,
                B365H REAL,
                B365D REAL,
                B365A REAL,
                FTHG REAL,
                FTAG REAL,
                FTR TEXT,
                scraped_at TEXT,
                PRIMARY KEY (HomeTeam, AwayTeam)
            )
        """))
        conn.commit()


def _latest_scraped_at() -> datetime | None:
    ensure_odds_cache()
    row = pd.read_sql(f"SELECT MAX(scraped_at) AS ts FROM {CACHE_TABLE}", engine)
    ts = row["ts"].iloc[0]
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    dt = datetime.fromisoformat(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def is_fresh() -> bool:
    scraped_at = _latest_scraped_at()
    if scraped_at is None:
        return False
    return datetime.now(UTC) - scraped_at < timedelta(hours=ttl_hours())


def load_cached() -> pd.DataFrame | None:
    if not is_fresh():
        return None
    ensure_odds_cache()
    df = pd.read_sql(f"SELECT * FROM {CACHE_TABLE}", engine)
    return df if not df.empty else None


def save_scrape(df: pd.DataFrame) -> None:
    ensure_odds_cache()
    scraped_at = datetime.now(UTC).isoformat()
    cols = [
        "HomeTeam", "AwayTeam", "Div", "Date",
        "B365H", "B365D", "B365A", "FTHG", "FTAG", "FTR",
    ]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = None
    out["scraped_at"] = scraped_at
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {CACHE_TABLE}"))
        conn.commit()
    out[cols + ["scraped_at"]].to_sql(CACHE_TABLE, engine, if_exists="append", index=False)
    print(f"Odds cache saved: {len(out)} matches (TTL {ttl_hours():g}h)")


def clear_cache() -> None:
    ensure_odds_cache()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {CACHE_TABLE}"))
        conn.commit()


def cache_stats() -> dict:
    ensure_odds_cache()
    n = int(pd.read_sql(f"SELECT COUNT(*) AS n FROM {CACHE_TABLE}", engine)["n"].iloc[0])
    scraped_at = _latest_scraped_at()
    return {
        "cached_matches": n,
        "scraped_at": scraped_at.isoformat() if scraped_at else None,
        "fresh": is_fresh(),
        "ttl_hours": ttl_hours(),
    }