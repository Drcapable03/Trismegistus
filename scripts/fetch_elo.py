"""Fetch Club Elo histories for teams in the matches table."""

import time

import pandas as pd

from config.settings import league_div_codes
from utils.db import engine
from utils.elo_cache import elo_cache_stats, fetch_team_elo_history, save_elo_history


def fetch_elo(div_filter: list[str] | None = None, sleep_s: float = 0.3) -> int:
    codes = div_filter or league_div_codes()
    placeholders = ",".join(f"'{c}'" for c in codes)
    teams = pd.read_sql(
        f"""
        SELECT DISTINCT HomeTeam AS team FROM matches WHERE Div IN ({placeholders})
        UNION
        SELECT DISTINCT AwayTeam FROM matches WHERE Div IN ({placeholders})
        """,
        engine,
    )["team"].tolist()

    saved_rows = 0
    ok_teams = 0
    for team in sorted(teams):
        df = fetch_team_elo_history(team)
        if df.empty:
            print(f"  {team}: no Club Elo data (API down or unknown name)")
            if sleep_s:
                time.sleep(sleep_s)
            continue
        n = save_elo_history(team, df)
        saved_rows += n
        ok_teams += 1
        if sleep_s:
            time.sleep(sleep_s)

    stats = elo_cache_stats()
    print(f"Club Elo cache: {ok_teams} teams, {stats['elo_rows']} rating rows")
    return saved_rows


if __name__ == "__main__":
    fetch_elo()