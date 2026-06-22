import time
from datetime import datetime as dt
from pathlib import Path

import pandas as pd
import requests
import yaml

from agents.injuries_agent import fetch_injuries
from agents.intel_agent import fetch_team_intel, intel_to_chaos_fields, stripped_intel_fields
from agents.odds_agent import fetch_odds
from config.settings import today, use_intel_in_train
from utils.chaos_cache import INTEL_COLS, cache_stats, get_cached, save_cached

TEAM_CITIES_PATH = Path(__file__).resolve().parent.parent / "config" / "team_cities.yaml"

CHAOS_DEFAULTS = {
    "rain": 0.0,
    "wind": 0.0,
    **stripped_intel_fields(),
    "home_injuries": 0,
    "away_injuries": 0,
}


def _load_team_cities() -> dict[str, str]:
    if not TEAM_CITIES_PATH.exists():
        return {}
    with open(TEAM_CITIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _city_for_team(team: str, lookup: dict[str, str]) -> str:
    if team in lookup:
        return lookup[team]
    return team.split()[-1]


def _normalize_date(date) -> tuple[dt, str, str]:
    current_date = today()
    try:
        if isinstance(date, dt):
            date_obj = date
        else:
            date_obj = dt.strptime(str(date), "%d/%m/%Y")
        return date_obj, date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%d/%m/%Y")
    except ValueError:
        return current_date, current_date.strftime("%Y-%m-%d"), current_date.strftime("%d/%m/%Y")


def _fetch_weather(city: str, date_str: str) -> tuple[float, float]:
    try:
        geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        geo_response = requests.get(geo_url, headers={"User-Agent": "Trismegistus"}, timeout=5)
        geo_data = geo_response.json()
        lat = float(geo_data[0]["lat"]) if geo_data else 51.5074
        lon = float(geo_data[0]["lon"]) if geo_data else -0.1278
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&daily=precipitation_sum,wind_speed_10m_max&timezone=auto"
        )
        weather_response = requests.get(weather_url, timeout=5)
        weather_response.raise_for_status()
        weather = weather_response.json()
        if "daily" not in weather:
            return 0, 0
        rain = weather["daily"]["precipitation_sum"][0] or 0
        wind = weather["daily"]["wind_speed_10m_max"][0] or 0
        return rain, wind
    except Exception as e:
        print(f"Weather fetch failed: {e} — using defaults")
        return 0, 0


def _odds_from_row(row: pd.Series) -> dict[str, float]:
    if {"B365H", "B365D", "B365A"}.issubset(row.index):
        h, d, a = row["B365H"], row["B365D"], row["B365A"]
        if pd.notna(h) and pd.notna(d) and pd.notna(a) and min(h, d, a) > 0:
            return {"H": float(h), "D": float(d), "A": float(a)}
    return {"H": 0.0, "D": 0.0, "A": 0.0}


def _default_record(home_team: str, away_team: str, date_display: str, row: pd.Series) -> dict:
    odds = _odds_from_row(row)
    return {
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "Date": date_display,
        **CHAOS_DEFAULTS,
        "odds_H": odds["H"],
        "odds_A": odds["A"],
        "odds_D": odds["D"],
    }


def _apply_intel_policy(record: dict, match_dt: dt, training: bool) -> dict:
    out = dict(record)
    if training and not use_intel_in_train():
        out.update(stripped_intel_fields())
    return out


def _chaos_output_keys() -> list[str]:
    return [
        "HomeTeam", "AwayTeam", "Date", "rain", "wind",
        *INTEL_COLS,
        "home_injuries", "away_injuries",
        "odds_H", "odds_A", "odds_D",
    ]


def get_chaos_data(
    matches,
    injuries_df=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_only: bool = False,
):
    """Enrich matches with chaos features. One output row per input row.

    cache_only=True (training/backtest): use SQLite cache or row defaults — no live scrape.
    """
    chaos_data = []
    team_cities = _load_team_cities()
    cache_hits = 0
    cache_misses = 0
    live_fetches = 0
    total = len(matches)
    output_keys = _chaos_output_keys()

    for n, (_, row) in enumerate(matches.iterrows(), start=1):
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        date_obj, date_str, date_display = _normalize_date(date)
        allow_intel = not (cache_only and not use_intel_in_train())

        if use_cache and not refresh:
            cached = get_cached(
                home_team, away_team, date_display,
                match_dt=date_obj, allow_intel=allow_intel,
            )
            if cached:
                cache_hits += 1
                record = {k: cached[k] for k in output_keys}
                chaos_data.append(_apply_intel_policy(record, date_obj, training=cache_only))
                continue

        if cache_only:
            cache_misses += 1
            chaos_data.append(
                _apply_intel_policy(
                    _default_record(home_team, away_team, date_display, row),
                    date_obj,
                    training=True,
                )
            )
            continue

        print(f"Fetching chaos for {home_team} vs. {away_team} on {date_display} ({n}/{total})")
        live_fetches += 1

        city = _city_for_team(home_team, team_cities)
        rain, wind = _fetch_weather(city, date_str)

        div_code = str(row.get("Div")) if row.get("Div") else None
        home_intel = fetch_team_intel(
            home_team, date_str, opponent=away_team, div_code=div_code,
        )
        away_intel = fetch_team_intel(
            away_team, date_str, opponent=home_team, div_code=div_code,
        )

        odds = fetch_odds(
            home_team, away_team, date_str,
            force_refresh=refresh,
            div_code=str(div_code) if div_code else None,
        ) or _odds_from_row(row)

        if injuries_df is not None and not injuries_df.empty:
            home_injury_count = len(
                injuries_df[injuries_df["team"] == home_team]["status"].tolist()
            )
            away_injury_count = len(
                injuries_df[injuries_df["team"] == away_team]["status"].tolist()
            )
        else:
            home_injury_count = fetch_injuries(home_team)
            away_injury_count = fetch_injuries(away_team)

        record = {
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Date": date_display,
            "rain": rain,
            "wind": wind,
            **intel_to_chaos_fields("home", home_intel),
            **intel_to_chaos_fields("away", away_intel),
            "home_injuries": home_injury_count,
            "away_injuries": away_injury_count,
            "odds_H": odds["H"],
            "odds_A": odds["A"],
            "odds_D": odds["D"],
        }
        if use_cache:
            save_cached(record)
        chaos_data.append(record)
        time.sleep(1)

    stats = cache_stats()
    print(
        f"Chaos: {cache_hits} cache hits, {cache_misses} defaults"
        f"{f', {live_fetches} live fetches' if live_fetches else ''}"
        f" ({stats['cached_matches']} rows in cache)"
    )
    return pd.DataFrame(chaos_data)