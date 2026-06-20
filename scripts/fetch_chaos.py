import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

from agents.odds_agent import fetch_odds
from agents.news_agent import fetch_news
from config.settings import today

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


def get_chaos_data(matches, injuries_df=None):
    chaos_data = []
    total_matches = min(len(matches), 50)
    current_date = today()
    team_cities = _load_team_cities()

    subset = matches.head(total_matches)
    for n, (_, row) in enumerate(subset.iterrows(), start=1):
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        print(f"Fetching chaos for {home_team} vs. {away_team} on {date} ({n}/{total_matches})")

        try:
            if isinstance(date, datetime):
                date_obj = date
                date_str = date_obj.strftime("%Y-%m-%d")
                date_display = date_obj.strftime("%d/%m/%Y")
            else:
                date_obj = datetime.strptime(str(date), "%d/%m/%Y")
                date_str = date_obj.strftime("%Y-%m-%d")
                date_display = str(date)
        except ValueError:
            print(f"Invalid date: {date} — using defaults")
            date_obj = current_date
            date_str = current_date.strftime("%Y-%m-%d")
            date_display = current_date.strftime("%d/%m/%Y")

        city = _city_for_team(home_team, team_cities)
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
                rain, wind = 0, 0
            else:
                rain = weather["daily"]["precipitation_sum"][0] or 0
                wind = weather["daily"]["wind_speed_10m_max"][0] or 0
        except Exception as e:
            print(f"Weather fetch failed: {e} — using defaults")
            rain, wind = 0, 0

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

        chaos_data.append({
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
        })
        time.sleep(1)

    return pd.DataFrame(chaos_data)