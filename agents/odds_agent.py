import os
from dotenv import load_dotenv
import requests

load_dotenv()

def fetch_odds(home_team, away_team, date):
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("No ODDS_API_KEY found!")
        return None
    
    # Try multiple leagues
    sports = ["soccer_epl", "soccer_netherlands_eredivisie"]  # Add more as needed
    for sport in sports:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={api_key}&regions=eu&markets=h2h&date={date}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for event in data:
                if event["home_team"] == home_team and event["away_team"] == away_team:
                    odds = event["bookmakers"][0]["markets"][0]["outcomes"]
                    return {"H": odds[0]["price"], "A": odds[1]["price"], "D": odds[2]["price"]}
    print(f"No odds found for {home_team} vs {away_team} on {date}")
    return None