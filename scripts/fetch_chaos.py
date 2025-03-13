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

def fetch_rss_sentiment(team, date):
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
        print(f"RSS Sentiment for {team}: {score}")
        return score
    except Exception as e:
        print(f"RSS Sentiment fetch failed for {team}: {e}")
        return 0

def fetch_x_sentiment(team, date):
    # Placeholder—swap with real X API later
    x_posts = [
        f"{team} gonna smash it today!",
        f"{team} looking shaky, injuries piling up",
        f"Come on {team} let’s go!"
    ]
    sentiment = 0
    count = 0
    for post in x_posts:
        blob = TextBlob(post)
        sentiment += blob.sentiment.polarity
        count += 1
    score = sentiment / max(count, 1) if count else 0
    print(f"X Sentiment for {team}: {score}")
    return score

def fetch_injuries(team, date, injuries_df):
    try:
        team_injuries = injuries_df[injuries_df["team"] == team]
        key_players_out = len(team_injuries[team_injuries["status"] == "out"])
        print(f"Injuries for {team}: {key_players_out} key players out")
        return key_players_out
    except Exception as e:
        print(f"Injury fetch failed for {team}: {e}")
        return 0

def get_chaos_data(matches, weather_api_key, injuries_df):
    chaos_data = []
    limited_matches = matches.head(5)  # LIMIT TO 5 FOR TESTING—ADJUST FOR PRODUCTION
    for i, row in limited_matches.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        home_city = home.split()[0]
        print(f"Fetching chaos for {home} vs. {away} on {date} ({i+1}/{len(limited_matches)})")
        weather = fetch_weather(home_city, date, weather_api_key)
        home_rss = fetch_rss_sentiment(home, date)
        away_rss = fetch_rss_sentiment(away, date)
        home_x = fetch_x_sentiment(home, date)
        away_x = fetch_x_sentiment(away, date)
        home_injuries = fetch_injuries(home, date, injuries_df)
        away_injuries = fetch_injuries(away, date, injuries_df)
        chaos_data.append({
            "HomeTeam": home, "AwayTeam": away, "Date": date,
            "rain": weather["rain"], "wind": weather["wind"],
            "home_rss_sentiment": home_rss, "away_rss_sentiment": away_rss,
            "home_x_sentiment": home_x, "away_x_sentiment": away_x,
            "home_injuries": home_injuries, "away_injuries": away_injuries
        })
    for i, row in matches.iloc[5:].iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        chaos_data.append({
            "HomeTeam": home, "AwayTeam": away, "Date": date,
            "rain": 0, "wind": 0,
            "home_rss_sentiment": 0, "away_rss_sentiment": 0,
            "home_x_sentiment": 0, "away_x_sentiment": 0,
            "home_injuries": 0, "away_injuries": 0
        })
    return pd.DataFrame(chaos_data)