"""Club Elo rating cache (http://api.clubelo.com)."""

from datetime import datetime
from io import StringIO

import pandas as pd
import requests
from sqlalchemy import inspect, text

from utils.db import engine
from utils.team_aliases import to_club_elo

ELO_TABLE = "club_elo"
ELO_DEFAULT = 1500.0
ELO_API = "http://api.clubelo.com/{team}"


def ensure_elo_cache() -> None:
    if inspect(engine).has_table(ELO_TABLE):
        return
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE {ELO_TABLE} (
                team TEXT,
                club_elo_name TEXT,
                elo REAL,
                date_from TEXT,
                date_to TEXT,
                fetched_at TEXT
            )
        """))
        conn.commit()


def fetch_team_elo_history(team: str, timeout: int = 15) -> pd.DataFrame:
    club = to_club_elo(team)
    url = ELO_API.format(team=requests.utils.quote(club))
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception:
        return pd.DataFrame()
    if df.empty or "Elo" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={"From": "date_from", "To": "date_to"})
    df["team"] = team
    df["club_elo_name"] = club
    return df


def save_elo_history(team: str, df: pd.DataFrame) -> int:
    ensure_elo_cache()
    if df.empty:
        return 0
    ts = datetime.utcnow().isoformat()
    with engine.connect() as conn:
        conn.execute(text(f'DELETE FROM {ELO_TABLE} WHERE team = :team'), {"team": team})
        for _, row in df.iterrows():
            conn.execute(
                text(f"""
                    INSERT INTO {ELO_TABLE}
                    (team, club_elo_name, elo, date_from, date_to, fetched_at)
                    VALUES (:team, :club_elo_name, :elo, :date_from, :date_to, :fetched_at)
                """),
                {
                    "team": team,
                    "club_elo_name": row["club_elo_name"],
                    "elo": float(row["Elo"]),
                    "date_from": str(row["date_from"]),
                    "date_to": str(row["date_to"]),
                    "fetched_at": ts,
                },
            )
        conn.commit()
    return len(df)


def load_elo_history() -> pd.DataFrame:
    ensure_elo_cache()
    if not inspect(engine).has_table(ELO_TABLE):
        return pd.DataFrame()
    return pd.read_sql(f"SELECT * FROM {ELO_TABLE}", engine)


def elo_on_date(team: str, match_dt: pd.Timestamp, history: pd.DataFrame | None = None) -> float:
    if history is None:
        history = load_elo_history()
    if history.empty or team not in history["team"].values:
        return ELO_DEFAULT
    team_hist = history[history["team"] == team].copy()
    day = match_dt.strftime("%Y-%m-%d")
    for _, row in team_hist.iterrows():
        if str(row["date_from"]) <= day <= str(row["date_to"]):
            return float(row["elo"])
    return ELO_DEFAULT


def elo_cache_stats() -> dict:
    ensure_elo_cache()
    if not inspect(engine).has_table(ELO_TABLE):
        return {"elo_rows": 0, "teams": 0}
    stats = pd.read_sql(
        f"SELECT COUNT(*) AS rows, COUNT(DISTINCT team) AS teams FROM {ELO_TABLE}",
        engine,
    ).iloc[0]
    return {"elo_rows": int(stats["rows"]), "teams": int(stats["teams"])}