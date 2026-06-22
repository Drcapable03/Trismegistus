import os
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv

from config.settings import league_div_codes
from utils.db import engine
from utils.team_aliases import teams_match

load_dotenv()


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


def _auto_scrape_odds() -> bool:
    return os.getenv("TRIS_AUTO_SCRAPE_ODDS", "false").lower() in {"1", "true", "yes"}


def _match_names(home: str, away: str, df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    for _, row in df.iterrows():
        if teams_match(home, row["HomeTeam"]) and teams_match(away, row["AwayTeam"]):
            return row
    return None


def _load_odds_rows(home_team: str, away_team: str) -> pd.DataFrame:
    base_cols = ["HomeTeam", "AwayTeam", "Date", "B365H", "B365D", "B365A"]
    close_cols = ["B365CH", "B365CD", "B365CA"]
    try:
        df = pd.read_sql(
            f"""
            SELECT {", ".join(base_cols + close_cols)}
            FROM matches
            WHERE HomeTeam = :h AND AwayTeam = :a
            """,
            engine,
            params={"h": home_team, "a": away_team},
        )
    except Exception:
        df = pd.read_sql(
            f"""
            SELECT {", ".join(base_cols)}
            FROM matches
            WHERE HomeTeam = :h AND AwayTeam = :a
            """,
            engine,
            params={"h": home_team, "a": away_team},
        )
    if not df.empty:
        return df

    try:
        all_df = pd.read_sql(f"SELECT {', '.join(base_cols + close_cols)} FROM matches", engine)
    except Exception:
        all_df = pd.read_sql(f"SELECT {', '.join(base_cols)} FROM matches", engine)
    if all_df.empty:
        return all_df
    mask = all_df.apply(
        lambda r: teams_match(home_team, r["HomeTeam"]) and teams_match(away_team, r["AwayTeam"]),
        axis=1,
    )
    return all_df[mask]


def fetch_odds_from_db(home_team: str, away_team: str, date: str | None = None) -> dict | None:
    """Use B365 opening/closing odds stored in matches (ingested fixtures/results)."""
    try:
        df = _load_odds_rows(home_team, away_team)
        if df.empty:
            return None

        if date and "Date" in df.columns:
            dated = df[df["Date"].astype(str).str.contains(str(date)[:10], na=False)]
            if not dated.empty:
                df = dated

        row = df.iloc[-1]
        opening = None
        if pd.notna(row.get("B365H")) and pd.notna(row.get("B365D")) and pd.notna(row.get("B365A")):
            if min(row["B365H"], row["B365D"], row["B365A"]) > 0:
                opening = (float(row["B365H"]), float(row["B365D"]), float(row["B365A"]))

        closing = None
        if pd.notna(row.get("B365CH")) and pd.notna(row.get("B365CD")) and pd.notna(row.get("B365CA")):
            if min(row["B365CH"], row["B365CD"], row["B365CA"]) > 0:
                closing = (float(row["B365CH"]), float(row["B365CD"]), float(row["B365CA"]))

        current = closing or opening
        if current is None:
            return None
        h, d, a = current
        out = {
            "H": h,
            "D": d,
            "A": a,
            "source": "db_closing" if closing else "db_opening",
        }
        if opening:
            out["open_H"], out["open_D"], out["open_A"] = opening
        if closing:
            out["close_H"], out["close_D"], out["close_A"] = closing
        return out
    except Exception as e:
        print(f"DB odds lookup failed: {e}")
        return None


def _row_to_odds_dict(row: pd.Series, *, source: str) -> dict:
    return {
        "H": float(row["B365H"]),
        "D": float(row["B365D"]),
        "A": float(row["B365A"]),
        "source": source,
        "div": row.get("Div"),
        "scraped_home": row.get("HomeTeam"),
        "scraped_away": row.get("AwayTeam"),
    }


@lru_cache(maxsize=1)
def _scraped_odds_df_cached() -> pd.DataFrame:
    from utils.odds_cache import load_cached

    cached = load_cached()
    return cached if cached is not None else pd.DataFrame()


def _scraped_odds_df(force_refresh: bool = False) -> pd.DataFrame:
    """Load cached OddsPortal scrape; live scrape only on force or TRIS_AUTO_SCRAPE_ODDS."""
    from utils.odds_cache import load_cached

    if not force_refresh:
        cached = load_cached()
        if cached is not None and not cached.empty:
            print(f"Odds cache hit: {len(cached)} matches")
            return cached

    if not force_refresh and not _auto_scrape_odds():
        return pd.DataFrame()

    from scripts.fetch_odds import fetch_big5_odds

    df = fetch_big5_odds(force_refresh=True, include_results=True)
    _scraped_odds_df_cached.cache_clear()
    if not df.empty:
        return df

    if force_refresh or _auto_scrape_odds():
        from scrapers.oddsportal import scrape_worldcup_odds
        from utils.odds_cache import save_scrape

        wc = scrape_worldcup_odds(include_results=True)
        if not wc.empty:
            save_scrape(wc, div_filter=["WC26"])
        _scraped_odds_df_cached.cache_clear()
        return wc

    return pd.DataFrame()


def fetch_odds_scraped(
    home_team: str,
    away_team: str,
    date: str | None = None,
    *,
    force_refresh: bool = False,
    div_code: str | None = None,
) -> dict | None:
    try:
        df = _scraped_odds_df(force_refresh=force_refresh)
        if df.empty:
            print(f"Scraped odds: cache empty for {home_team} vs {away_team}")
            return None

        scoped = df
        if div_code and "Div" in df.columns:
            league_rows = df[df["Div"] == div_code]
            if not league_rows.empty:
                scoped = league_rows

        row = _match_names(home_team, away_team, scoped)
        if row is None and scoped is not df:
            row = _match_names(home_team, away_team, df)
        if row is None:
            print(f"Scraped odds: no match for {home_team} vs {away_team}")
            return None
        return _row_to_odds_dict(row, source="oddsportal")
    except Exception as e:
        print(f"Scraped odds failed: {e}")
        return None


def fetch_odds_api(home_team: str, away_team: str, date: str) -> dict | None:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return None

    sports = {
        "E0": "soccer_epl",
        "SP1": "soccer_spain_la_liga",
        "D1": "soccer_germany_bundesliga",
        "I1": "soccer_italy_serie_a",
        "F1": "soccer_france_ligue_one",
        "WC26": "soccer_fifa_world_cup",
    }
    for sport in sports.values():
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            f"?apiKey={api_key}&regions=eu&markets=h2h&date={date}"
        )
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            continue
        for event in response.json():
            if teams_match(home_team, event["home_team"]) and teams_match(away_team, event["away_team"]):
                odds = event["bookmakers"][0]["markets"][0]["outcomes"]
                return {"H": odds[0]["price"], "A": odds[1]["price"], "D": odds[2]["price"], "source": "odds_api"}
    return None


def fetch_odds(
    home_team: str,
    away_team: str,
    date: str,
    *,
    force_refresh: bool = False,
    div_code: str | None = None,
) -> dict | None:
    """DB (ingested B365) → cached/live OddsPortal scrape → The Odds API."""
    db_odds = fetch_odds_from_db(home_team, away_team, date)
    if db_odds:
        print(
            f"Odds via DB ({db_odds['source']}): {home_team} vs {away_team} "
            f"-> H={db_odds['H']}"
        )
        return db_odds

    if _scrapling_enabled():
        scraped = fetch_odds_scraped(
            home_team, away_team, date,
            force_refresh=force_refresh,
            div_code=div_code,
        )
        if scraped:
            print(
                f"Odds via OddsPortal ({scraped['source']}): {home_team} vs {away_team} "
                f"-> H={scraped['H']}"
            )
            return scraped

    api = fetch_odds_api(home_team, away_team, date)
    if api:
        print(f"Odds via API: {home_team} vs {away_team}")
        return api

    print(f"No odds found for {home_team} vs {away_team} on {date}")
    return None


def clear_odds_cache() -> None:
    _scraped_odds_df_cached.cache_clear()
    from utils.odds_cache import clear_cache
    clear_cache()