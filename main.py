import argparse

import pandas as pd
from dotenv import load_dotenv

from config.settings import enabled_leagues, fixtures_url, league_summary, league_urls, today
from predictors.blunder_sniffer import BlunderSniffer
from predictors.game_forger import GameForger, _is_completed, _parse_dates
from predictors.registry import latest_model_path, load_game_forger, save_game_forger
from scripts.backtest import backtest_predictions, format_prediction
from scripts.scrape_football_data import scrape_matches
from utils.chaos_cache import cache_stats
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

    print(league_summary())
    for league, url in league_urls().items():
        csv_path = scrape_matches(url, league)
        load_csv_to_db(csv_path, "matches")

    csv_path = scrape_matches(fixtures_url(), "Fixtures")
    load_csv_to_db(csv_path, "matches")
    league_codes = [info["code"] for info in enabled_leagues().values()]
    calculate_team_form(div_filter=league_codes)
    print("Ingest complete.")


def get_future_matches(matches: pd.DataFrame) -> pd.DataFrame:
    matches = _parse_dates(matches)
    now = today()
    upcoming = matches[~_is_completed(matches)]
    return upcoming[upcoming["Date"] >= now].sort_values("Date")


def run_backtest(limit: int = 200, use_cache: bool = True, refresh_cache: bool = False) -> GameForger:
    forger = GameForger()
    forger.train(
        injuries_df=DEFAULT_INJURIES,
        limit=limit,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
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
    return forger


def run_predict(
    confidence: float = 75.0,
    limit: int = 50,
    use_cache: bool = True,
    refresh_cache: bool = False,
    model_path: str | None = None,
) -> None:
    matches = read_matches()
    future = get_future_matches(matches)
    if future.empty:
        print("No upcoming fixtures found.")
        return

    print(f"Processing {len(future)} future matches")
    forger = GameForger()

    if model_path:
        load_game_forger(forger, model_path)
    else:
        forger.train(
            injuries_df=DEFAULT_INJURIES,
            limit=limit,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )

    forger.prepare_prediction_data(
        future.head(50),
        injuries_df=DEFAULT_INJURIES,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
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
    print(league_summary())
    print(f"Chaos cache: {cache_stats()}")


def main():
    parser = argparse.ArgumentParser(description="Trismegistus football prediction pipeline")
    parser.add_argument("--ingest", action="store_true", help="Scrape and load match data")
    parser.add_argument("--predict", action="store_true", help="Predict upcoming fixtures")
    parser.add_argument("--backtest", action="store_true", help="Evaluate on held-out historical matches")
    parser.add_argument("--explore", action="store_true", help="Print data summary")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate matches table before ingest")
    parser.add_argument("--confidence", type=float, default=75.0, help="Min confidence %% for predictions")
    parser.add_argument("--limit", type=int, default=200, help="Max completed matches for training")
    parser.add_argument("--save-model", action="store_true", help="Save trained model after backtest")
    parser.add_argument("--load-model", type=str, default=None, help="Path to saved model for --predict")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-fetch chaos data ignoring cache")
    parser.add_argument("--no-cache", action="store_true", help="Disable chaos cache reads/writes")
    parser.add_argument("--worldcup-scrape", action="store_true", help="Test Scrapling World Cup scrape")
    parser.add_argument("--worldcup-ingest", action="store_true", help="Scrape and ingest World Cup matches")
    parser.add_argument("--worldcup-predict", action="store_true", help="Predict upcoming World Cup fixtures")
    args = parser.parse_args()

    use_cache = not args.no_cache
    run_all = not any([
        args.ingest, args.predict, args.backtest, args.explore,
        args.worldcup_scrape, args.worldcup_ingest, args.worldcup_predict,
    ])

    print("Trismegistus is alive!")
    if args.ingest or run_all:
        ingest_data(reset=args.reset)
    if args.explore or run_all:
        explore_data()
    if args.backtest or run_all:
        forger = run_backtest(
            limit=args.limit,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
        )
        if args.save_model:
            save_game_forger(forger)
        elif run_all:
            path = latest_model_path()
            if path:
                print(f"Tip: use --save-model to persist, or --load-model {path}")
    if args.predict:
        run_predict(
            confidence=args.confidence,
            limit=args.limit,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
            model_path=args.load_model,
        )
    if args.worldcup_scrape:
        from scripts.worldcup_pipeline import test_scrape_sample
        test_scrape_sample(limit=8)
    if args.worldcup_ingest:
        from scripts.worldcup_pipeline import ingest_worldcup
        ingest_worldcup()
    if args.worldcup_predict:
        from scripts.worldcup_pipeline import predict_worldcup
        predict_worldcup(confidence=args.confidence)
    print("Done.")


if __name__ == "__main__":
    main()