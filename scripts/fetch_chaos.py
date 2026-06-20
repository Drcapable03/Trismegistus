import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

from agents.odds_agent import fetch_odds
from agents.news_agent import fetch_news
from config.settings import today
from utils.chaos_cache import cache_stats, get_cached, save_cached

TEAM_CITIES_PATH = Path(__file__).resolve().parent.parent / "config" / "team_cities.yaml"


def _load_team_cities() -> dict[str, str]:
    if not TEAM_CITIES_PATH.exists():
        return {}
    with open(TEAM_CITIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _city_for_team(team: str, lookup: dict[str, str]) -> str:
    if team in lookup:
        return lookup[team]
    return team.split()[-1]


def _normalize_date(date) -> tuple[datetime, str, str]:
    current_date = today()
    try:
        if isinstance(date, datetime):
            date_obj = date
        else:
            date_obj = datetime.strptime(str(date), "%d/%m/%Y")
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


def get_chaos_data(matches, injuries_df=None, use_cache: bool = True, refresh: bool = False):
    chaos_data = []
    total_matches = min(len(matches), 50)
    current_date = today()
    team_cities = _load_team_cities()
    cache_hits = 0

    subset = matches.head(total_matches)
    for n, (_, row) in enumerate(subset.iterrows(), start=1):
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        date_obj, date_str, date_display = _normalize_date(date)

        if use_cache and not refresh:
            cached = get_cached(home_team, away_team, date_display)
            if cached:
                cache_hits += 1
                chaos_data.append({k: cached[k] for k in [
                    "HomeTeam", "AwayTeam", "Date", "rain", "wind",
                    "home_x_sentiment", "away_x_sentiment", "home_injuries", "away_injuries",
                    "odds_H", "odds_A", "odds_D",
                ]})
                continue

        print(f"Fetching chaos for {home_team} vs. {away_team} on {date_display} ({n}/{total_matches})")

        city = _city_for_team(home_team, team_cities)
        rain, wind = _fetch_weather(city, date_str)

        if date_obj.date() >= current_date.date():
            home_sentiment = fetch_news(home_team, date_str)
            away_sentiment = fetch_news(away_team, date_str)
        else:
            home_sentiment, away_sentiment = 0.1, 0.1

        odds = fetch_odds(home_team, away_team, date_str) or {"H": 0, "A": 0, "D": 0}

        home_injuries = (
            injuries_df[injuries_df["team"] == home_team]["status"].tolist()
            if injuries_df is not None else []
        )
        away_injuries = (
            injuries_df[injuries_df["team"] == away_team]["status"].tolist()
            if injuries_df is not None else []
        )

        record = {
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Date": date_display,
            "rain": rain,
            "wind": wind,
            "home_x_sentiment": home_sentiment,
            "away_x_sentiment": away_sentiment,
            "home_injuries": len(home_injuries),
            "away_injuries": len(away_injuries),
            "odds_H": odds["H"],
            "odds_A": odds["A"],
            "odds_D": odds["D"],
        }
        if use_cache:
            save_cached(record)
        chaos_data.append(record)
        time.sleep(1)

    stats = cache_stats()
    print(f"Chaos cache: {cache_hits} hits, {stats['cached_matches']} total cached rows")
    return pd.DataFrame(chaos_data)