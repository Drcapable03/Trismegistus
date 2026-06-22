import argparse

import pandas as pd
from dotenv import load_dotenv

from config.settings import (
    current_season,
    edge_margin_min,
    fixtures_url,
    league_div_codes,
    league_summary,
    league_urls,
    per_league_edge_margins,
    today,
)
from scripts.expand_history import season_coverage, season_from_league_label
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
        season = season_from_league_label(league)
        load_csv_to_db(csv_path, "matches", season=season)

    csv_path = scrape_matches(fixtures_url(), "Fixtures")
    load_csv_to_db(csv_path, "matches", season=current_season())
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
        confidence_threshold=0.0, require_edge=True, edge_margin=None,
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
    train_limit: int = 0,
    predict_limit: int = 50,
    use_cache: bool = True,
    refresh_cache: bool = False,
    model_path: str | None = None,
    edge_margin: float | None = None,
    dry_run: bool = False,
) -> None:
    div_codes = league_div_codes()
    matches = read_matches()
    future = get_future_matches(matches, div_filter=div_codes)
    if future.empty:
        print("No upcoming league fixtures found. Run --ingest to refresh fixtures.csv.")
        return

    train_n = None if train_limit == 0 else train_limit
    batch = future.head(predict_limit) if predict_limit > 0 else future

    print(f"Predict scope: Big 5 leagues {div_codes}")
    print(f"Upcoming fixtures: {len(future)} (scoring {len(batch)})")
    margins = per_league_edge_margins()
    if edge_margin is not None:
        print(f"Edge filter: global ≥{edge_margin:.0%} vs closing implied")
    elif margins:
        print(f"Edge filter: per-league margins {margins}")
    else:
        print(f"Edge filter: global ≥{edge_margin_min():.0%} vs closing implied")
    if dry_run:
        print("Dry run: ON — ingested odds only, no live intel/odds scrape")
    else:
        print(
            "Live intel: ON (news/Reddit/YouTube)"
            + (" — refreshing cache" if refresh_cache else " — cache + live fetch on miss")
        )

    forger = GameForger()
    if model_path:
        load_game_forger(forger, model_path)
        forger.prepare_training_data(
            injuries_df=DEFAULT_INJURIES,
            limit=train_n,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            div_filter=div_codes,
            chaos_cache_only=True,
        )
        forger.fit_dc_baseline()
    else:
        forger.train(
            injuries_df=DEFAULT_INJURIES,
            limit=train_n,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            div_filter=div_codes,
            chaos_cache_only=True,
        )

    forger.prepare_prediction_data(
        batch,
        injuries_df=DEFAULT_INJURIES,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        div_filter=div_codes,
        chaos_cache_only=dry_run,
    )
    predictions = forger.predict(
        confidence_threshold=confidence,
        edge_margin=edge_margin,
        use_simulation=False,
    )
    if not predictions:
        print("\nNo edge-qualified picks for upcoming fixtures.")
        return
    print(f"\nLive picks ({len(predictions)}):")
    for pred in predictions:
        print(format_prediction(pred))


def explore_data() -> None:
    from scripts.validate_live_predict import fixture_readiness, _fixture_guidance

    df = read_matches()
    print(df.head())
    print(df.columns.tolist())
    completed = _is_completed(df).sum()
    print(f"Completed: {completed}, Total: {len(df)}")
    coverage = season_coverage(df)
    if not coverage.empty:
        print("Season coverage (completed / total):")
        for season in sorted(coverage["Season"].unique()):
            rows = coverage[coverage["Season"] == season]
            done = int(rows["completed"].sum())
            total = int(rows["total"].sum())
            print(f"  {season}: {done} / {total}")
    print(league_summary())
    print(f"Chaos cache: {cache_stats()}")
    from utils.odds_cache import cache_stats as odds_cache_stats

    print(f"Odds cache: {odds_cache_stats()}")
    readiness = fixture_readiness(df)
    print(f"Live predict: {_fixture_guidance(readiness)}")


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
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Max completed matches for training/backtest (0 = all ingested)",
    )
    parser.add_argument(
        "--predict-limit", type=int, default=50,
        help="Max upcoming fixtures to score on --predict (0 = all)",
    )
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
    parser.add_argument(
        "--tune-edge-leagues", action="store_true",
        help="Tune per-league edge margins on selective holdout ROI",
    )
    parser.add_argument(
        "--tune-dc-blend", action="store_true",
        help="Grid-search Dixon-Coles blend weight on selective holdout ROI",
    )
    parser.add_argument(
        "--expand-history", action="store_true",
        help="Verify historical season CSV URLs and report DB coverage",
    )
    parser.add_argument(
        "--kelly-sim", action="store_true",
        help="Run fractional Kelly bankroll sim on selective holdout picks",
    )
    parser.add_argument(
        "--validate-live", action="store_true",
        help="Run live-predict pre-flight checks and E2E smoke test",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --predict: score using ingested odds only (no live intel scrape)",
    )
    parser.add_argument(
        "--fetch-odds", action="store_true",
        help="Scrape and cache live Big 5 odds from OddsPortal (Scrapling)",
    )
    parser.add_argument(
        "--fetch-odds-league", type=str, default=None,
        help="Scrape one league only (E0, SP1, D1, I1, F1) with --fetch-odds",
    )
    parser.add_argument(
        "--calibrate-intel", action="store_true",
        help="Run intel source pre-flight checks (news, Reddit, YouTube)",
    )
    parser.add_argument(
        "--calibrate-intel-live", action="store_true",
        help="With --calibrate-intel: fetch live probe intel (network)",
    )
    parser.add_argument(
        "--intel-roi", action="store_true",
        help="Report chaos-cache intel coverage and holdout ablation ROI",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    run_all = not any([
        args.ingest, args.predict, args.backtest, args.explore,
        args.worldcup_scrape, args.worldcup_ingest, args.worldcup_predict,
        args.tune_blend, args.tune_leagues, args.archive_chaos, args.build_cities,
        args.fetch_xg, args.fetch_elo, args.tune_edge, args.tune_edge_leagues,
        args.tune_dc_blend, args.expand_history,
        args.kelly_sim, args.validate_live, args.fetch_odds,
        args.calibrate_intel, args.intel_roi,
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
    if args.validate_live:
        from scripts.validate_live_predict import run_validate_live

        exit_code = run_validate_live(
            train_limit=args.limit if args.limit > 0 else 80,
            injuries_df=DEFAULT_INJURIES,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)
    if args.predict:
        run_predict(
            confidence=args.confidence,
            train_limit=args.limit if args.limit > 0 else 0,
            predict_limit=args.predict_limit,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
            model_path=args.load_model,
            edge_margin=args.edge_margin,
            dry_run=args.dry_run,
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
    if args.calibrate_intel:
        from scripts.calibrate_intel import run_calibrate_intel

        exit_code = run_calibrate_intel(live_fetch=args.calibrate_intel_live)
        if exit_code != 0:
            raise SystemExit(exit_code)
    if args.intel_roi:
        from scripts.intel_roi import run_intel_roi

        run_intel_roi(limit=args.limit if args.limit > 0 else 200, use_cache=use_cache)
    if args.fetch_odds:
        from scripts.fetch_odds import fetch_big5_odds, fetch_league_odds
        from utils.odds_cache import cache_stats

        if args.fetch_odds_league:
            code = args.fetch_odds_league.upper()
            df = fetch_league_odds(code, include_results=True)
        else:
            df = fetch_big5_odds(
                div_filter=league_div_codes(),
                include_results=True,
                force_refresh=True,
            )
        print(f"Fetched {len(df)} odds rows. Cache: {cache_stats()}")
    if args.tune_edge:
        from scripts.tune_edge import tune_edge
        tune_edge(limit=args.limit, use_cache=use_cache)
    if args.tune_edge_leagues:
        from scripts.tune_edge_leagues import tune_edge_leagues
        tune_edge_leagues(limit=args.limit, use_cache=use_cache)
    if args.tune_dc_blend:
        from scripts.tune_dc_blend import tune_dc_blend
        tune_dc_blend(limit=args.limit, use_cache=use_cache)
    if args.expand_history:
        from scripts.expand_history import expand_history
        expand_history()
    if args.kelly_sim:
        from evaluation.kelly import kelly_simulation
        from config.settings import kelly_fraction

        div_codes = league_div_codes()
        forger = GameForger()
        forger.train(limit=args.limit, use_cache=use_cache, div_filter=div_codes, chaos_cache_only=True)
        preds = forger.backtest_on_holdout(require_edge=True)
        if not preds:
            print("Kelly sim: no selective picks on holdout.")
        else:
            for line in ("close", "open"):
                sim = kelly_simulation(preds, kelly_fraction=kelly_fraction(), odds_line=line)
                print(
                    f"Kelly ({kelly_fraction():.0%}, {line}): "
                    f"ROI {sim['roi_pct']:+.1f}%, bankroll {sim['final_bankroll']:.1f}, "
                    f"max DD {sim['max_drawdown_pct']:.1f}% ({sim['bets']} bets)"
                )
    print("Done.")


if __name__ == "__main__":
    main()