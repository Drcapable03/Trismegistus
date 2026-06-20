from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import inspect, text

from utils.db import engine

CACHE_TABLE = "chaos_cache"


def ensure_chaos_cache() -> None:
    if inspect(engine).has_table(CACHE_TABLE):
        return
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE {CACHE_TABLE} (
                HomeTeam TEXT,
                AwayTeam TEXT,
                Date TEXT,
                rain REAL,
                wind REAL,
                home_x_sentiment REAL,
                away_x_sentiment REAL,
                home_injuries INTEGER,
                away_injuries INTEGER,
                odds_H REAL,
                odds_A REAL,
                odds_D REAL,
                fetched_at TEXT,
                PRIMARY KEY (HomeTeam, AwayTeam, Date)
            )
        """))
        conn.commit()


def _match_key(home: str, away: str, date: str) -> tuple[str, str, str]:
    return home, away, str(date)


def get_cached(home: str, away: str, date: str) -> dict | None:
    ensure_chaos_cache()
    row = pd.read_sql(
        f"SELECT * FROM {CACHE_TABLE} WHERE HomeTeam = :h AND AwayTeam = :a AND Date = :d",
        engine,
        params={"h": home, "a": away, "d": str(date)},
    )
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def save_cached(record: dict) -> None:
    ensure_chaos_cache()
    record = {**record, "fetched_at": datetime.now(UTC).isoformat()}
    cols = [
        "HomeTeam", "AwayTeam", "Date", "rain", "wind",
        "home_x_sentiment", "away_x_sentiment", "home_injuries", "away_injuries",
        "odds_H", "odds_A", "odds_D", "fetched_at",
    ]
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT OR REPLACE INTO {CACHE_TABLE}
                ({", ".join(cols)})
                VALUES ({", ".join(f":{c}" for c in cols)})
            """),
            {c: record.get(c) for c in cols},
        )
        conn.commit()


def cache_stats() -> dict:
    ensure_chaos_cache()
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {CACHE_TABLE}", engine)["n"][0]
    return {"cached_matches": int(n)}