"""World Cup 2026 scrape → ingest → predict test pipeline.

World Cup is a temporary live-data harness while Big 5 leagues may be off-season.
Core product remains the leagues in config/leagues.yaml.
"""

import pandas as pd

from agents.odds_agent import clear_odds_cache, fetch_odds
from agents.news_agent import fetch_news
from agents.injuries_agent import fetch_injuries
from predictors.game_forger import GameForger, _is_completed
from scrapers.oddsportal import scrape_worldcup_odds
from scripts.backtest import format_prediction
from utils.db import load_csv_to_db, read_matches
from utils.features import calculate_team_form

WC_DIV = "WC26"
# Sparse WC sample — lean on bookie more than league-tuned default (0.0).
WC_BOOKIE_BLEND_WEIGHT = 0.55


def ingest_worldcup(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else scrape_worldcup_odds(include_results=True)
    if df.empty:
        print("No World Cup data scraped.")
        return df

    out = df[[
        "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "B365H", "B365D", "B365A",
    ]].copy()
    tmp = "data/worldcup_scraped.csv"
    out.to_csv(tmp, index=False)
    load_csv_to_db(tmp, "matches")
    calculate_team_form(div_filter=WC_DIV)
    print(f"Ingested {len(out)} World Cup matches (form scoped to {WC_DIV})")
    return out


def test_scrape_sample(limit: int = 5) -> None:
    print("=== World Cup Scrape Test (temporary harness; core leagues in leagues.yaml) ===")
    clear_odds_cache()
    df = scrape_worldcup_odds(include_results=True)
    print(df.head(limit).to_string())
    print(f"\nTotal matches: {len(df)}")
    upcoming = df[df["FTR"].isna()]
    finished = df[df["FTR"].notna()]
    print(f"Finished: {len(finished)}, Upcoming: {len(upcoming)}")

    if not upcoming.empty:
        row = upcoming.iloc[0]
        home, away = row["HomeTeam"], row["AwayTeam"]
        print(f"\n--- Live enrichment test: {home} vs {away} ---")
        odds = fetch_odds(home, away, "2026-06-20")
        print("Odds:", odds)
        print("Home news:", fetch_news(home, "2026-06-20"))
        print("Away news:", fetch_news(away, "2026-06-20"))
        print("Home injuries:", fetch_injuries(home))
        print("Away injuries:", fetch_injuries(away))


def predict_worldcup(confidence: float = 60.0, max_upcoming: int = 5) -> None:
    matches = read_matches()
    wc = matches[matches["Div"] == WC_DIV] if "Div" in matches.columns else matches
    if wc.empty:
        print("No WC26 matches in DB — run ingest first.")
        return

    completed = wc[_is_completed(wc)]
    upcoming = wc[~_is_completed(wc)]
    if completed.empty:
        print("No completed WC matches for training.")
        return
    if upcoming.empty:
        print("No upcoming WC matches to predict.")
        return

    print(
        f"Training on {len(completed)} completed WC26 matches "
        f"(not mixing with league data)"
    )
    forger = GameForger(bookie_blend_weight=WC_BOOKIE_BLEND_WEIGHT)
    forger.train(
        limit=len(completed),
        use_cache=True,
        div_filter=WC_DIV,
    )
    forger.prepare_prediction_data(
        upcoming.head(max_upcoming),
        use_cache=True,
        div_filter=WC_DIV,
    )
    preds = forger.predict(confidence_threshold=confidence)
    print("\nWorld Cup Predictions (test harness — core product is Big 5 leagues):")
    for p in preds:
        print(format_prediction(p))