from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import inspect, text

from config.settings import pit_cache_intel
from utils.db import engine

CACHE_TABLE = "chaos_cache"

INTEL_COLS = [
    "home_news_attention",
    "away_news_attention",
    "home_news_sentiment",
    "away_news_sentiment",
    "home_reddit_sentiment",
    "away_reddit_sentiment",
    "home_youtube_sentiment",
    "away_youtube_sentiment",
]

LEGACY_INTEL_MAP = {
    "home_x_sentiment": "home_news_attention",
    "away_x_sentiment": "away_news_attention",
}

BASE_COLS = ["HomeTeam", "AwayTeam", "Date", "rain", "wind"]
TAIL_COLS = ["home_injuries", "away_injuries", "odds_H", "odds_A", "odds_D", "fetched_at"]
ALL_CACHE_COLS = [*BASE_COLS, *INTEL_COLS, *TAIL_COLS]


def _table_columns() -> set[str]:
    if not inspect(engine).has_table(CACHE_TABLE):
        return set()
    return {c["name"] for c in inspect(engine).get_columns(CACHE_TABLE)}


def _migrate_chaos_cache() -> None:
    cols = _table_columns()
    if not cols:
        return
    with engine.connect() as conn:
        for new_col in INTEL_COLS:
            if new_col not in cols:
                conn.execute(text(f"ALTER TABLE {CACHE_TABLE} ADD COLUMN {new_col} REAL DEFAULT 0"))
        for legacy, new_col in LEGACY_INTEL_MAP.items():
            if legacy in cols and new_col in cols:
                conn.execute(
                    text(f"""
                        UPDATE {CACHE_TABLE}
                        SET {new_col} = COALESCE({new_col}, {legacy})
                        WHERE {legacy} IS NOT NULL AND ({new_col} IS NULL OR {new_col} = 0)
                    """)
                )
        conn.commit()


def ensure_chaos_cache() -> None:
    if not inspect(engine).has_table(CACHE_TABLE):
        intel_defs = ", ".join(f"{c} REAL" for c in INTEL_COLS)
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE {CACHE_TABLE} (
                    HomeTeam TEXT,
                    AwayTeam TEXT,
                    Date TEXT,
                    rain REAL,
                    wind REAL,
                    {intel_defs},
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
        return
    _migrate_chaos_cache()


def _match_key(home: str, away: str, date: str) -> tuple[str, str, str]:
    return home, away, str(date)


def _parse_fetched_at(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_record(record: dict) -> dict:
    out = dict(record)
    for legacy, new_col in LEGACY_INTEL_MAP.items():
        if legacy in out and (new_col not in out or out.get(new_col) in (None, 0)):
            out[new_col] = out.get(legacy, 0.0)
    for col in INTEL_COLS:
        out.setdefault(col, 0.0)
    return out


def _strip_intel(record: dict) -> dict:
    out = _normalize_record(record)
    for col in INTEL_COLS:
        out[col] = 0.0
    return out


def intel_is_pit_safe(record: dict, match_dt: datetime) -> bool:
    """True if cached intel could have existed before kickoff."""
    if not pit_cache_intel():
        return False
    fetched = _parse_fetched_at(record.get("fetched_at"))
    if fetched is None:
        return False
    kickoff = match_dt.replace(hour=23, minute=59, second=59)
    if fetched.tzinfo is not None:
        kickoff = kickoff.replace(tzinfo=UTC)
    return fetched <= kickoff


def sentiment_is_pit_safe(record: dict, match_dt: datetime) -> bool:
    """Legacy alias."""
    return intel_is_pit_safe(record, match_dt)


def get_cached(
    home: str,
    away: str,
    date: str,
    match_dt: datetime | None = None,
    allow_intel: bool = True,
    allow_sentiment: bool | None = None,
) -> dict | None:
    if allow_sentiment is not None:
        allow_intel = allow_sentiment
    ensure_chaos_cache()
    row = pd.read_sql(
        f"SELECT * FROM {CACHE_TABLE} WHERE HomeTeam = :h AND AwayTeam = :a AND Date = :d",
        engine,
        params={"h": home, "a": away, "d": str(date)},
    )
    if row.empty:
        return None
    record = _normalize_record(row.iloc[0].to_dict())
    if not allow_intel:
        return _strip_intel(record)
    if match_dt is not None and not intel_is_pit_safe(record, match_dt):
        return _strip_intel(record)
    return record


def save_cached(record: dict) -> None:
    ensure_chaos_cache()
    record = _normalize_record({**record, "fetched_at": datetime.now(UTC).isoformat()})
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT OR REPLACE INTO {CACHE_TABLE}
                ({", ".join(ALL_CACHE_COLS)})
                VALUES ({", ".join(f":{c}" for c in ALL_CACHE_COLS)})
            """),
            {c: record.get(c) for c in ALL_CACHE_COLS},
        )
        conn.commit()


def cache_stats() -> dict:
    ensure_chaos_cache()
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {CACHE_TABLE}", engine)["n"][0]
    return {"cached_matches": int(n)}