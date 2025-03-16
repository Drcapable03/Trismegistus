import os
from dotenv import load_dotenv 
from scripts.scrape_football_data import scrape_matches
from utils.db import load_csv_to_db, engine, reset_table
import pandas as pd
from utils.features import calculate_team_form
from predictors.game_forger import GameForger
from scripts.backtest import backtest_predictions
from datetime import datetime

PAST_URLS = {
    "Premier League": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
}
FUTURE_URL = "https://www.football-data.co.uk/fixtures.csv"

def explore_data():
    df = pd.read_sql("SELECT * FROM matches", engine)
    print(df.head())
    print(df.columns)

load_dotenv()

def run_agent_task():
    print("Agent disabled—news fetching skipped for now.")

def predict_matches():
    weather_api_key = os.getenv("WEATHER_API_KEY")
    injuries_df = pd.DataFrame({
        "team": ["Sheffield Weds", "Southampton"],
        "player": ["Player1", "Player2"],
        "status": ["out", "out"]
    })
    matches = pd.read_sql("SELECT * FROM matches", engine)
    
    # Filter future matches
    matches["Date"] = pd.to_datetime(matches["Date"], format="%d/%m/%Y")
    future_date = datetime(2025, 3, 2)  # Start March 2
    future_matches = matches[matches["Date"] >= future_date]
    if len(future_matches) == 0:
        print("No future matches found after March 2, 2025!")
        return
    
    print(f"Processing {len(future_matches)} future matches")
    forger = GameForger(weather_api_key)
    forger.train(injuries_df, limit=min(50, len(future_matches)))
    predictions = forger.predict(confidence_threshold=75.0)
    print("Predictions:")
    for pred in predictions:
        print(pred)
    print("Backtest skipped—future matches have no results.")

if __name__ == "__main__":
    print("Global Football Predictor is alive!")
    # Reset DB for clean run
    reset_table("matches")
    # Scrape past data
    for league, url in PAST_URLS.items():
        csv_path = scrape_matches(url, league)
        load_csv_to_db(csv_path, "matches", if_exists="append")
    # Scrape future fixtures
    csv_path = scrape_matches(FUTURE_URL, "Fixtures")
    load_csv_to_db(csv_path, "matches", if_exists="append")
    calculate_team_form()
    explore_data()
    predict_matches()
    run_agent_task()
    print("EVERYTHING DID WORK WELL!!")