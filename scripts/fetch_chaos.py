import requests
import feedparser
from textblob import TextBlob
import pandas as pd

def fetch_weather(city, date, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5).json()
        rain = response.get("rain", {}).get("1h", 0)
        wind = response.get("wind", {}).get("speed", 0)
        print(f"Weather for {city}: rain={rain}, wind={wind}")
        return {"rain": rain, "wind": wind}
    except Exception as e:
        print(f"Weather fetch failed for {city}: {e}")
        return {"rain": 0, "wind": 0}

def fetch_sentiment(team, date):
    rss_url = "https://feeds.bbci.co.uk/sport/football/rss.xml"
    try:
        feed = feedparser.parse(rss_url)
        sentiment = 0
        count = 0
        for entry in feed.entries[:10]:
            if team.lower() in entry.title.lower() or team.lower() in entry.summary.lower():
                blob = TextBlob(entry.summary)
                sentiment += blob.sentiment.polarity
                count += 1
        score = sentiment / max(count, 1) if count else 0
        print(f"Sentiment for {team}: {score}")
        return score
    except Exception as e:
        print(f"Sentiment fetch failed for {team}: {e}")
        return 0

def get_chaos_data(matches, weather_api_key):
    chaos_data = []
    for i, row in matches.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        home_city = home.split()[0]
        print(f"Fetching chaos for {home} vs. {away} on {date} ({i+1}/{len(matches)})")
        weather = fetch_weather(home_city, date, weather_api_key)
        home_sentiment = fetch_sentiment(home, date)
        away_sentiment = fetch_sentiment(away, date)
        chaos_data.append({
            "HomeTeam": home, "AwayTeam": away, "Date": date,
            "rain": weather["rain"], "wind": weather["wind"],
            "home_sentiment": home_sentiment, "away_sentiment": away_sentiment
        })
    return pd.DataFrame(chaos_data)