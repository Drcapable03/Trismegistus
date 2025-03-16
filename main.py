import os
from dotenv import load_dotenv 
from scripts.scrape_football_data import scrape_matches
from utils.db import load_csv_to_db, engine
import pandas as pd
from utils.features import calculate_team_form
from predictors.game_forger import GameForger
from scripts.backtest import backtest_predictions
from datetime import datetime

LEAGUE_URLS = {
    "Eredivisie": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
}

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
    
    # Filter for future matches (post-March 17, 2025)
    matches["Date"] = pd.to_datetime(matches["Date"], format="%d/%m/%Y")
    future_date = datetime(2025, 3, 17)
    future_matches = matches[matches["Date"] >= future_date]
    if len(future_matches) == 0:
        print("No future matches found after March 17, 2025!")
        return
    
    print(f"Processing {len(future_matches)} future matches")
    forger = GameForger(weather_api_key)
    forger.train(injuries_df, limit=min(50, len(future_matches)))  # Cap at 50
    predictions = forger.predict(confidence_threshold=75.0)
    print("Predictions:")
    for pred in predictions:
        print(pred)
    backtest_predictions(predictions, future_matches)

if __name__ == "__main__":
    print("Global Football Predictor is alive!")
    for league, url in LEAGUE_URLS.items():
        csv_path = scrape_matches(url, league)
        load_csv_to_db(csv_path, "matches")
    calculate_team_form()
    explore_data()
    predict_matches()
    run_agent_task()
    print("EVERYTHING DID WORK WELL!!")