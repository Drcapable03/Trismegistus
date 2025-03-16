import pandas as pd
import requests
import time
from agents.odds_agent import fetch_odds
from agents.news_agent import fetch_news

def get_chaos_data(matches, weather_api_key=None, injuries_df=None):
    chaos_data = []
    total_matches = min(len(matches), 50)  # Cap at 50 for testing
    
    for idx, row in matches.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        print(f"Fetching chaos for {home_team} vs. {away_team} on {date} ({idx+1}/{total_matches})")
        
        # Weather (open-meteo)
        city = home_team.split()[-1]
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            geo_response = requests.get(geo_url, headers={"User-Agent": "Trismegistus"})
            geo_data = geo_response.json()
            lat = float(geo_data[0]["lat"]) if geo_data else 0
            lon = float(geo_data[0]["lon"]) if geo_data else 0
            weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&daily=precipitation_sum,wind_speed_10m_max&timezone=auto"
            weather_response = requests.get(weather_url, timeout=5)
            weather = weather_response.json()["daily"]
            rain = weather["precipitation_sum"][0] or 0
            wind = weather["wind_speed_10m_max"][0] or 0
        except Exception as e:
            print(f"Weather fetch failed: {e}")
            rain, wind = 0, 0

        # News sentiment
        home_sentiment = fetch_news(home_team, date)
        away_sentiment = fetch_news(away_team, date)

        # Injuries placeholder
        home_injuries = injuries_df[injuries_df["team"] == home_team]["status"].tolist() if injuries_df is not None else []
        away_injuries = injuries_df[injuries_df["team"] == away_team]["status"].tolist() if injuries_df is not None else []

        chaos_data.append({
            "HomeTeam": home_team, "AwayTeam": away_team, "Date": date,
            "rain": rain, "wind": wind, "home_x_sentiment": home_sentiment, "away_x_sentiment": away_sentiment,
            "home_injuries": len(home_injuries), "away_injuries": len(away_injuries)
        })
        time.sleep(1)  # Rate limit

    return pd.DataFrame(chaos_data)