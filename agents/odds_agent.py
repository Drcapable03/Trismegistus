import os
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def _worldcup_odds_df() -> pd.DataFrame:
    from scrapers.oddsportal import scrape_worldcup_odds
    return scrape_worldcup_odds(include_results=True)


def _match_names(home: str, away: str, df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    exact = df[(df["HomeTeam"] == home) & (df["AwayTeam"] == away)]
    if not exact.empty:
        return exact.iloc[0]
    # Fuzzy: case-insensitive
    h, a = home.lower(), away.lower()
    fuzzy = df[
        df["HomeTeam"].str.lower().eq(h) & df["AwayTeam"].str.lower().eq(a)
    ]
    return fuzzy.iloc[0] if not fuzzy.empty else None


def fetch_odds_scraped(home_team: str, away_team: str, date: str | None = None) -> dict | None:
    try:
        df = _worldcup_odds_df()
        row = _match_names(home_team, away_team, df)
        if row is None:
            print(f"Scraped odds: no match for {home_team} vs {away_team}")
            return None
        return {"H": row["B365H"], "D": row["B365D"], "A": row["B365A"], "source": "oddsportal"}
    except Exception as e:
        print(f"Scraped odds failed: {e}")
        return None


def fetch_odds_api(home_team: str, away_team: str, date: str) -> dict | None:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return None

    sports = [
        "soccer_epl",
        "soccer_netherlands_eredivisie",
        "soccer_fifa_world_cup",
    ]
    for sport in sports:
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            f"?apiKey={api_key}&regions=eu&markets=h2h&date={date}"
        )
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            continue
        for event in response.json():
            if event["home_team"] == home_team and event["away_team"] == away_team:
                odds = event["bookmakers"][0]["markets"][0]["outcomes"]
                return {"H": odds[0]["price"], "A": odds[1]["price"], "D": odds[2]["price"], "source": "odds_api"}
    return None


def fetch_odds(home_team: str, away_team: str, date: str) -> dict | None:
    """Scrapling (OddsPortal) first, then The Odds API if key present."""
    if _scrapling_enabled():
        scraped = fetch_odds_scraped(home_team, away_team, date)
        if scraped:
            print(f"Odds via OddsPortal: {home_team} vs {away_team} -> H={scraped['H']}")
            return scraped

    api = fetch_odds_api(home_team, away_team, date)
    if api:
        print(f"Odds via API: {home_team} vs {away_team}")
        return api

    print(f"No odds found for {home_team} vs {away_team} on {date}")
    return None


def clear_odds_cache() -> None:
    _worldcup_odds_df.cache_clear()