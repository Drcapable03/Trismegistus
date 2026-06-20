import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from config.settings import fixtures_url, league_urls, today
from predictors.blunder_sniffer import BlunderSniffer
from predictors.game_forger import _is_completed, _parse_dates
from predictors.game_forger import GameForger
from scripts.backtest import backtest_predictions, format_prediction
from scripts.scrape_football_data import scrape_matches
from utils.db import ensure_schema, load_csv_to_db, read_matches, reset_table
from utils.features import calculate_team_form

load_dotenv()

DEFAULT_INJURIES = pd.DataFrame({
    "team": ["Sheffield Weds", "Southampton"],
    "player": ["Player1", "Player2"],
    "status": ["out", "out"],
})


def ingest_data(reset: bool = False) -> None:
    if reset:
        reset_table("matches")
    else:
        ensure_schema("matches")

    for league, url in league_urls().items():
        csv_path = scrape_matches(url, league)
        load_csv_to_db(csv_path, "matches")

    csv_path = scrape_matches(fixtures_url(), "Fixtures")
    load_csv_to_db(csv_path, "matches")
    calculate_team_form()
    print("Ingest complete.")


def get_future_matches(matches: pd.DataFrame) -> pd.DataFrame:
    matches = _parse_dates(matches)
    now = today()
    upcoming = matches[~_is_completed(matches)]
    return upcoming[upcoming["Date"] >= now].sort_values("Date")


def run_backtest(limit: int = 200) -> None:
    forger = GameForger()
    forger.train(injuries_df=DEFAULT_INJURIES, limit=limit)
    forger.evaluate_holdout()
    predictions = forger.backtest_on_holdout(confidence_threshold=0.0)
    matches = read_matches()
    backtest_predictions(predictions, matches)

    sniffer = BlunderSniffer()
    sniffer.train()
    blunders = sniffer.find_blunders(limit=10)
    if blunders:
        print("\nBookie Blunders (BlunderSniffer):")
        for b in blunders:
            print(f"  {b}")


def run_predict(confidence: float = 75.0, limit: int = 50) -> None:
    matches = read_matches()
    future = get_future_matches(matches)
    if future.empty:
        print("No upcoming fixtures found.")
        return

    print(f"Processing {len(future)} future matches")
    forger = GameForger()
    forger.train(injuries_df=DEFAULT_INJURIES, limit=limit)
    forger.prepare_prediction_data(future.head(50), injuries_df=DEFAULT_INJURIES)
    predictions = forger.predict(confidence_threshold=confidence)
    print("\nPredictions:")
    for pred in predictions:
        print(format_prediction(pred))


def explore_data() -> None:
    df = read_matches()
    print(df.head())
    print(df.columns.tolist())
    completed = _is_completed(df).sum()
    print(f"Completed: {completed}, Total: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Trismegistus football prediction pipeline")
    parser.add_argument("--ingest", action="store_true", help="Scrape and load match data")
    parser.add_argument("--predict", action="store_true", help="Predict upcoming fixtures")
    parser.add_argument("--backtest", action="store_true", help="Evaluate on held-out historical matches")
    parser.add_argument("--explore", action="store_true", help="Print data summary")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate matches table before ingest")
    parser.add_argument("--confidence", type=float, default=75.0, help="Min confidence %% for predictions")
    parser.add_argument("--limit", type=int, default=200, help="Max completed matches for training")
    args = parser.parse_args()

    run_all = not any([args.ingest, args.predict, args.backtest, args.explore])

    print("Trismegistus is alive!")
    if args.ingest or run_all:
        ingest_data(reset=args.reset)
    if args.explore or run_all:
        explore_data()
    if args.backtest or run_all:
        run_backtest(limit=args.limit)
    if args.predict:
        run_predict(confidence=args.confidence, limit=args.limit)
    print("Done.")


if __name__ == "__main__":
    main()