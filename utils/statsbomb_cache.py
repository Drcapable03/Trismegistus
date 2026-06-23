"""SQLite cache for StatsBomb open-data match xG (aggregated from shot events)."""

from datetime import datetime

import pandas as pd
from sqlalchemy import inspect, text

from utils.db import engine
from utils.team_aliases import from_statsbomb

STATSBOMB_TABLE = "statsbomb_xg"


def ensure_statsbomb_cache() -> None:
    if inspect(engine).has_table(STATSBOMB_TABLE):
        return
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE {STATSBOMB_TABLE} (
                match_id INTEGER,
                HomeTeam TEXT,
                AwayTeam TEXT,
                MatchDate TEXT,
                Div TEXT,
                xg_home REAL,
                xg_away REAL,
                competition_id INTEGER,
                season_id INTEGER,
                fetched_at TEXT,
                PRIMARY KEY (match_id)
            )
        """))
        conn.commit()


def save_match_rows(rows: list[dict]) -> int:
    ensure_statsbomb_cache()
    if not rows:
        return 0
    ts = datetime.utcnow().isoformat()
    with engine.connect() as conn:
        for row in rows:
            conn.execute(
                text(f"""
                    INSERT OR REPLACE INTO {STATSBOMB_TABLE}
                    (match_id, HomeTeam, AwayTeam, MatchDate, Div, xg_home, xg_away,
                     competition_id, season_id, fetched_at)
                    VALUES (:match_id, :HomeTeam, :AwayTeam, :MatchDate, :Div,
                            :xg_home, :xg_away, :competition_id, :season_id, :fetched_at)
                """),
                {**row, "fetched_at": ts},
            )
        conn.commit()
    return len(rows)


def load_statsbomb_xg(div: str | None = None) -> pd.DataFrame:
    ensure_statsbomb_cache()
    if div:
        return pd.read_sql(
            f'SELECT * FROM {STATSBOMB_TABLE} WHERE "Div" = :div',
            engine,
            params={"div": div},
        )
    if not inspect(engine).has_table(STATSBOMB_TABLE):
        return pd.DataFrame()
    return pd.read_sql(f"SELECT * FROM {STATSBOMB_TABLE}", engine)


def statsbomb_cache_stats() -> dict:
    ensure_statsbomb_cache()
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {STATSBOMB_TABLE}", engine)["n"][0]
    return {"statsbomb_matches": int(n)}


def match_xg_from_events(events: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float]:
    shots = events[events["type"] == "Shot"]
    home_xg = float(shots.loc[shots["team"] == home_team, "shot_statsbomb_xg"].sum())
    away_xg = float(shots.loc[shots["team"] == away_team, "shot_statsbomb_xg"].sum())
    return home_xg, away_xg


def parse_statsbomb_match(
    match_row: pd.Series,
    xg_home: float,
    xg_away: float,
    div: str,
) -> dict:
    dt = pd.to_datetime(match_row["match_date"])
    return {
        "match_id": int(match_row["match_id"]),
        "HomeTeam": from_statsbomb(str(match_row["home_team"])),
        "AwayTeam": from_statsbomb(str(match_row["away_team"])),
        "MatchDate": dt.strftime("%d/%m/%Y"),
        "Div": div,
        "xg_home": xg_home,
        "xg_away": xg_away,
        "competition_id": int(match_row["competition_id"]),
        "season_id": int(match_row["season_id"]),
    }