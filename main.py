import argparse

import pandas as pd
from dotenv import load_dotenv

from config.settings import (
    edge_margin_min,
    fixtures_url,
    league_div_codes,
    league_summary,
    league_urls,
    today,
)
from predictors.blunder_sniffer import BlunderSniffer
from predictors.game_forger import GameForger, _apply_div_filter, _is_completed, _parse_dates
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
    calculate_team_form(div_filter=league_div_codes())
    print("Ingest complete.")


def get_future_matches(matches: pd.DataFrame, div_filter: list[str] | None = None) -> pd.DataFrame:
    matches = _parse_dates(matches)
    if div_filter:
        matches = _apply_div_filter(matches, div_filter)
    now = today()
    upcoming = matches[~_is_completed(matches)]
    return upcoming[upcoming["Date"] >= now].sort_values("Date")


def run_backtest(limit: int = 200, use_cache: bool = True, refresh_cache: bool = False) -> GameForger:
    div_codes = league_div_codes()
    print(f"Backtest scope: Big 5 leagues {div_codes}")
    forger = GameForger()
    forger.train(
        injuries_df=DEFAULT_INJURIES,
        limit=limit,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        div_filter=div_codes,
        chaos_cache_only=not refresh_cache,
    )
    forger.evaluate_holdout()
    margin = edge_margin_min()
    all_preds = forger.backtest_on_holdout(
        confidence_threshold=0.0, require_edge=False, edge_margin=0.0,
    )
    selective_preds = forger.backtest_on_holdout(
        confidence_threshold=0.0, require_edge=True, edge_margin=margin,
    )
    matches = read_matches()
    matches = _apply_div_filter(matches, div_codes)
    backtest_predictions(
        all_preds, matches,
        selective_predictions=selective_preds,
        edge_margin=margin,
    )
    meta = forger.training_metadata
    if meta:
        print(
            f"Split: walk-forward {meta.get('train_matches')} train / "
            f"{meta.get('test_matches')} test"
        )

    sniffer = BlunderSniffer()
    blunders = sniffer.find_blunders(forger, limit=10, edge_margin=margin)
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
    edge_margin: float | None = None,
) -> None:
    div_codes = league_div_codes()
    matches = read_matches()
    future = get_future_matches(matches, div_filter=div_codes)
    if future.empty:
        print("No upcoming league fixtures found.")
        return

    print(f"Predict scope: Big 5 leagues {div_codes}")
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
            div_filter=div_codes,
            chaos_cache_only=not refresh_cache,
        )

    forger.prepare_prediction_data(
        future.head(50),
        injuries_df=DEFAULT_INJURIES,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        div_filter=div_codes,
        chaos_cache_only=False,
    )
    margin = edge_margin if edge_margin is not None else edge_margin_min()
    print(f"Edge filter: only picks with ≥{margin:.0%} edge vs bookie implied")
    predictions = forger.predict(
        confidence_threshold=confidence,
        edge_margin=margin,
        use_simulation=False,
    )
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
    parser.add_argument(
        "--edge-margin", type=float, default=None,
        help="Min edge vs bookie implied prob to emit a pick (default from leagues.yaml)",
    )
    parser.add_argument("--limit", type=int, default=200, help="Max completed matches for training")
    parser.add_argument("--save-model", action="store_true", help="Save trained model after backtest")
    parser.add_argument("--load-model", type=str, default=None, help="Path to saved model for --predict")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-fetch chaos data ignoring cache")
    parser.add_argument("--no-cache", action="store_true", help="Disable chaos cache reads/writes")
    parser.add_argument("--worldcup-scrape", action="store_true", help="Test Scrapling World Cup scrape")
    parser.add_argument("--worldcup-ingest", action="store_true", help="Scrape and ingest World Cup matches")
    parser.add_argument("--worldcup-predict", action="store_true", help="Predict upcoming World Cup fixtures")
    parser.add_argument("--tune-blend", action="store_true", help="Grid-search bookie blend weight on league holdout")
    parser.add_argument(
        "--tune-leagues", action="store_true",
        help="Tune per-league bookie blend weights on holdout subsets",
    )
    parser.add_argument(
        "--archive-chaos", action="store_true",
        help="Pre-cache historical weather into chaos_cache (PIT-safe, no news)",
    )
    parser.add_argument(
        "--build-cities", action="store_true",
        help="Regenerate config/team_cities.yaml for Big 5 teams",
    )
    parser.add_argument("--fetch-xg", action="store_true", help="Cache Understat match xG into SQLite")
    parser.add_argument("--fetch-elo", action="store_true", help="Cache Club Elo ratings into SQLite")
    parser.add_argument(
        "--tune-edge", action="store_true",
        help="Grid-search edge margin on selective holdout ROI",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    run_all = not any([
        args.ingest, args.predict, args.backtest, args.explore,
        args.worldcup_scrape, args.worldcup_ingest, args.worldcup_predict,
        args.tune_blend, args.tune_leagues, args.archive_chaos, args.build_cities,
        args.fetch_xg, args.fetch_elo, args.tune_edge,
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
            edge_margin=args.edge_margin,
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
    if args.tune_blend:
        from scripts.tune_blend import tune_blend
        tune_blend(limit=args.limit, use_cache=use_cache)
    if args.tune_leagues:
        from scripts.tune_leagues import tune_leagues
        tune_leagues(limit=args.limit, use_cache=use_cache)
    if args.archive_chaos:
        from scripts.archive_chaos import archive_chaos
        archive_chaos(div_filter=league_div_codes(), limit=args.limit)
    if args.build_cities:
        from scripts.build_team_cities import write_team_cities
        write_team_cities()
    if args.fetch_xg:
        from scripts.fetch_xg import fetch_xg
        fetch_xg()
    if args.fetch_elo:
        from scripts.fetch_elo import fetch_elo
        fetch_elo(div_filter=league_div_codes())
    if args.tune_edge:
        from scripts.tune_edge import tune_edge
        tune_edge(limit=args.limit, use_cache=use_cache)
    print("Done.")


if __name__ == "__main__":
    main()