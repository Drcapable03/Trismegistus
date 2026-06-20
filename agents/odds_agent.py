import os
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv

from utils.db import engine

load_dotenv()


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


def _match_names(home: str, away: str, df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    exact = df[(df["HomeTeam"] == home) & (df["AwayTeam"] == away)]
    if not exact.empty:
        return exact.iloc[0]
    h, a = home.lower(), away.lower()
    fuzzy = df[
        df["HomeTeam"].str.lower().eq(h) & df["AwayTeam"].str.lower().eq(a)
    ]
    return fuzzy.iloc[0] if not fuzzy.empty else None


def fetch_odds_from_db(home_team: str, away_team: str, date: str | None = None) -> dict | None:
    """Use B365 odds already stored in matches (ingested fixtures/results)."""
    try:
        df = pd.read_sql(
            """
            SELECT HomeTeam, AwayTeam, Date, B365H, B365D, B365A
            FROM matches
            WHERE HomeTeam = :h AND AwayTeam = :a
            """,
            engine,
            params={"h": home_team, "a": away_team},
        )
        if df.empty:
            h, a = home_team.lower(), away_team.lower()
            all_df = pd.read_sql(
                "SELECT HomeTeam, AwayTeam, Date, B365H, B365D, B365A FROM matches",
                engine,
            )
            df = all_df[
                all_df["HomeTeam"].str.lower().eq(h) & all_df["AwayTeam"].str.lower().eq(a)
            ]
        if df.empty:
            return None

        if date and "Date" in df.columns:
            dated = df[df["Date"].astype(str).str.contains(str(date)[:10], na=False)]
            if not dated.empty:
                df = dated

        row = df.dropna(subset=["B365H", "B365D", "B365A"], how="any").iloc[-1]
        if row[["B365H", "B365D", "B365A"]].le(0).any():
            return None
        return {
            "H": float(row["B365H"]),
            "D": float(row["B365D"]),
            "A": float(row["B365A"]),
            "source": "db",
        }
    except Exception as e:
        print(f"DB odds lookup failed: {e}")
        return None


@lru_cache(maxsize=1)
def _worldcup_odds_df(force_refresh: bool = False) -> pd.DataFrame:
    from scrapers.oddsportal import scrape_worldcup_odds
    from utils.odds_cache import is_fresh, load_cached, save_scrape

    if not force_refresh:
        cached = load_cached()
        if cached is not None:
            print(f"Odds cache hit: {len(cached)} matches")
            return cached

    df = scrape_worldcup_odds(include_results=True)
    if not df.empty:
        save_scrape(df)
    return df


def fetch_odds_scraped(
    home_team: str,
    away_team: str,
    date: str | None = None,
    *,
    force_refresh: bool = False,
) -> dict | None:
    try:
        df = _worldcup_odds_df(force_refresh=force_refresh)
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


def fetch_odds(
    home_team: str,
    away_team: str,
    date: str,
    *,
    force_refresh: bool = False,
) -> dict | None:
    """DB (ingested B365) → cached/live OddsPortal scrape → The Odds API."""
    db_odds = fetch_odds_from_db(home_team, away_team, date)
    if db_odds:
        print(f"Odds via DB: {home_team} vs {away_team} -> H={db_odds['H']}")
        return db_odds

    if _scrapling_enabled():
        scraped = fetch_odds_scraped(home_team, away_team, date, force_refresh=force_refresh)
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
    from utils.odds_cache import clear_cache
    clear_cache()