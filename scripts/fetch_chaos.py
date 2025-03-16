import pandas as pd
import requests
import time
from agents.odds_agent import fetch_odds
from agents.news_agent import fetch_news
from datetime import datetime

def get_chaos_data(matches, weather_api_key=None, injuries_df=None):
    chaos_data = []
    total_matches = min(len(matches), 50)
    current_date = datetime(2025, 3, 15)  # Today
    
    for idx, row in matches.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        print(f"Fetching chaos for {home_team} vs. {away_team} on {date} ({idx+1}/{total_matches})")
        
        # Convert date
        try:
            date_obj = datetime.strptime(date, "%d/%m/%Y")
            date_str = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Invalid date: {date}—using default")
            date_str = "2025-03-17"

        # Weather (open-meteo)
        city = home_team.split()[-1]
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            geo_response = requests.get(geo_url, headers={"User-Agent": "Trismegistus"})
            geo_data = geo_response.json()
            lat = float(geo_data[0]["lat"]) if geo_data else 51.5074
            lon = float(geo_data[0]["lon"]) if geo_data else -0.1278
            weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&daily=precipitation_sum,wind_speed_10m_max&timezone=auto"
            weather_response = requests.get(weather_url, timeout=5)
            weather_response.raise_for_status()
            weather = weather_response.json()
            if "daily" not in weather:
                print(f"Weather missing 'daily': {weather}")
                rain, wind = 0, 0
            else:
                rain = weather["daily"]["precipitation_sum"][0] or 0
                wind = weather["daily"]["wind_speed_10m_max"][0] or 0
        except Exception as e:
            print(f"Weather fetch failed: {e}—using defaults")
            rain, wind = 0, 0

        # News (only if future)
        if date_obj >= current_date:
            home_sentiment = fetch_news(home_team, date_str)
            away_sentiment = fetch_news(away_team, date_str)
        else:
            print(f"Skipping NewsAPI for past date: {date}")
            home_sentiment, away_sentiment = 0.1, 0.1

        # Odds
        odds = fetch_odds(home_team, away_team, date_str) or {"H": 0, "A": 0, "D": 0}
        odds_H, odds_A, odds_D = odds["H"], odds["A"], odds["D"]

        # Injuries
        home_injuries = injuries_df[injuries_df["team"] == home_team]["status"].tolist() if injuries_df is not None else []
        away_injuries = injuries_df[injuries_df["team"] == away_team]["status"].tolist() if injuries_df is not None else []

        chaos_data.append({
            "HomeTeam": home_team, "AwayTeam": away_team, "Date": date,
            "rain": rain, "wind": wind, "home_x_sentiment": home_sentiment, "away_x_sentiment": away_sentiment,
            "home_injuries": len(home_injuries), "away_injuries": len(away_injuries),
            "odds_H": odds_H, "odds_A": odds_A, "odds_D": odds_D
        })
        time.sleep(1)

    return pd.DataFrame(chaos_data)