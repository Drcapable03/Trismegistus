import requests
import pandas as pd
from utils.db import engine

def scrape_matches(league_url, league_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(league_url, headers=headers)
    if "text/csv" not in response.headers.get("Content-Type", ""):
        print(f"Error: Got {response.headers.get('Content-Type')} instead of CSV")
        print("Response Snippet:", response.text[:200])
        raise ValueError(f"Invalid response from {league_url}")
    csv_path = f"data/raw_matches_{league_name}.csv"
    with open(csv_path, "wb") as f:
        f.write(response.content)
    print(f"Scraped matches from {league_url} to {csv_path}")
    return csv_path  # Return path for loading