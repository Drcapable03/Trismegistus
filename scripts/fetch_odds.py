"""Phase 6B: scrape and cache live Big 5 odds from OddsPortal."""

from __future__ import annotations

import pandas as pd

from config.settings import league_div_codes, oddsportal_league_urls
from scrapers.oddsportal import scrape_big5_odds, scrape_league_odds, scrape_worldcup_odds
from utils.odds_cache import cache_stats, save_scrape


def fetch_big5_odds(
    *,
    div_filter: list[str] | None = None,
    include_results: bool = True,
    force_refresh: bool = True,
) -> pd.DataFrame:
    """Scrape OddsPortal for enabled Big 5 leagues and persist to SQLite cache."""
    codes = div_filter or league_div_codes()
    configured = oddsportal_league_urls(codes)
    missing = [c for c in codes if c not in configured]
    if missing:
        print(f"OddsPortal URLs missing for: {missing}")

    df = scrape_big5_odds(list(configured.keys()), include_results=include_results)
    if df.empty:
        print("No Big 5 odds scraped.")
        return df

    if force_refresh:
        save_scrape(df, div_filter=list(configured.keys()))
    print(f"Odds cache stats: {cache_stats()}")
    return df


def fetch_league_odds(
    div_code: str,
    *,
    include_results: bool = True,
) -> pd.DataFrame:
    df = scrape_league_odds(div_code, include_results=include_results)
    if not df.empty:
        save_scrape(df, div_filter=[div_code])
    return df


def fetch_worldcup_odds(*, include_results: bool = True) -> pd.DataFrame:
    df = scrape_worldcup_odds(include_results=include_results)
    if not df.empty:
        save_scrape(df, div_filter=["WC26"])
    return df