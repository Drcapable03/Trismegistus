import os
from dotenv import load_dotenv 
from scripts.scrape_football_data import scrape_matches
from utils.db import load_csv_to_db
import pandas as pd
from utils.db import engine
from utils.features import calculate_team_form
from predictors.game_forger import GameForger
from scripts.backtest import backtest_predictions

def explore_data():
    df = pd.read_sql("SELECT * FROM matches", engine)
    print(df.head())
    print(df.columns)

load_dotenv()

def run_agent_task():
    print("Agent disabled—news fetching skipped for now.")

def predict_matches():
    weather_api_key = os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        raise ValueError("WEATHER_API_KEY missing in .env!")
    
    injuries_df = pd.DataFrame({
        "team": ["Sheffield Weds", "Southampton", "Blackburn"],
        "player": ["Player1", "Player2", "Player3"],
        "status": ["out", "out", "doubtful"]
    })
    
    forger = GameForger(weather_api_key, sim_runs=100)
    forger.train(injuries_df)
    predictions = forger.predict(injuries_df=injuries_df)
    print("Predictions:")
    for pred in predictions:
        print(pred)
    matches = pd.read_sql("SELECT * FROM matches", engine)
    backtest_predictions(predictions, matches)

if __name__ == "__main__":
    print("Global Football Predictor is alive!")
    scrape_matches()
    load_csv_to_db("C:/Users/ASUS/Trismegistus/data/raw_matches.csv", "matches")
    calculate_team_form()
    explore_data()
    predict_matches()
    run_agent_task()
    
    print("EVERYTHING DID WORK WELL!!")