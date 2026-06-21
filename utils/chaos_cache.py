from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import inspect, text

from config.settings import pit_cache_sentiment
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


def _parse_fetched_at(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _strip_sentiment(record: dict) -> dict:
    out = dict(record)
    out["home_x_sentiment"] = 0.0
    out["away_x_sentiment"] = 0.0
    return out


def sentiment_is_pit_safe(record: dict, match_dt: datetime) -> bool:
    """True if cached sentiment could have existed before kickoff."""
    if not pit_cache_sentiment():
        return False
    fetched = _parse_fetched_at(record.get("fetched_at"))
    if fetched is None:
        return False
    kickoff = match_dt.replace(hour=23, minute=59, second=59)
    if fetched.tzinfo is not None:
        kickoff = kickoff.replace(tzinfo=UTC)
    return fetched <= kickoff


def get_cached(
    home: str,
    away: str,
    date: str,
    match_dt: datetime | None = None,
    allow_sentiment: bool = True,
) -> dict | None:
    ensure_chaos_cache()
    row = pd.read_sql(
        f"SELECT * FROM {CACHE_TABLE} WHERE HomeTeam = :h AND AwayTeam = :a AND Date = :d",
        engine,
        params={"h": home, "a": away, "d": str(date)},
    )
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    if not allow_sentiment:
        return _strip_sentiment(record)
    if match_dt is not None and not sentiment_is_pit_safe(record, match_dt):
        return _strip_sentiment(record)
    return record


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