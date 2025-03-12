import os
from dotenv import load_dotenv 
from scripts.scrape_football_data import scrape_matches
from utils.db import load_csv_to_db
import pandas as pd
from sqlalchemy import create_engine
from utils.db import engine
from utils.features import calculate_team_form
from predictors.blunder_sniffer import BlunderSniffer

def explore_data():
    df = pd.read_sql("SELECT * FROM matches", engine)
    print(df.head())
    print(df.columns)

load_dotenv()

def run_agent_task():
    print("Agent disabled—news fetching skipped for now.")

def predict_matches():
    sniffer = BlunderSniffer()
    sniffer.train()
    predictions = sniffer.predict()
    print("Predictions:")
    for pred in predictions:
        print(pred)

if __name__ == "__main__":
    print("Global Football Predictor is alive!")
    scrape_matches()
    load_csv_to_db("C:/Users/ASUS/Trismegistus/data/raw_matches.csv", "matches")
    calculate_team_form()
    explore_data()
    predict_matches()
    run_agent_task()
    
    print("EVERYTHING DID WORK WELL!!")