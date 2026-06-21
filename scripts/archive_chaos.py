"""Pre-cache weather chaos for historical matches (PIT-safe; no live news)."""

import time

import pandas as pd

from config.settings import league_div_codes
from scripts.fetch_chaos import _city_for_team, _fetch_weather, _load_team_cities, _normalize_date
from agents.intel_agent import stripped_intel_fields
from utils.chaos_cache import get_cached, save_cached
from utils.db import engine


def archive_chaos(
    div_filter: list[str] | None = None,
    limit: int | None = 500,
    sleep_s: float = 0.5,
) -> int:
    """Cache rain/wind for completed matches missing from chaos_cache."""
    codes = div_filter or league_div_codes()
    placeholders = ",".join(f"'{c}'" for c in codes)
    matches = pd.read_sql(
        f"SELECT * FROM matches WHERE Div IN ({placeholders}) AND FTR IN ('H','D','A')",
        engine,
    )
    if matches.empty:
        print("No completed matches to archive.")
        return 0

    matches = matches.sort_values("Date", ascending=False)
    if limit:
        matches = matches.head(limit)

    team_cities = _load_team_cities()
    archived = 0
    for _, row in matches.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        _, date_str, date_display = _normalize_date(date)
        if get_cached(home, away, date_display):
            continue

        city = _city_for_team(home, team_cities)
        rain, wind = _fetch_weather(city, date_str)
        record = {
            "HomeTeam": home,
            "AwayTeam": away,
            "Date": date_display,
            "rain": rain,
            "wind": wind,
            **stripped_intel_fields(),
            "home_injuries": 0,
            "away_injuries": 0,
            "odds_H": float(row.get("B365H") or 0.0),
            "odds_A": float(row.get("B365A") or 0.0),
            "odds_D": float(row.get("B365D") or 0.0),
        }
        save_cached(record)
        archived += 1
        if sleep_s:
            time.sleep(sleep_s)

    print(f"Archived weather for {archived} matches ({len(matches)} scanned).")
    return archived


if __name__ == "__main__":
    archive_chaos()