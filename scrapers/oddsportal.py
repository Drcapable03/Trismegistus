"""Scrape World Cup odds and fixtures from OddsPortal via Scrapling."""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from scrapers.browser import fetch_page

ODDS_TRIPLE = re.compile(r"(\d+\.\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s*$")
SCORE_PATTERN = re.compile(r"(\d+)\s*–\s*(\d+)")


def _load_wc_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "worldcup.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_event_row(row, div_code: str) -> dict | None:
    names = row.css(".participant-name")
    if len(names) < 2:
        return None

    home = (names[0].text or "").strip()
    away = (names[1].text or "").strip()
    if not home or not away:
        return None

    blob = row.get_all_text(strip=True)
    odds_m = ODDS_TRIPLE.search(blob.replace(",", "."))
    if not odds_m:
        return None

    h_odds, d_odds, a_odds = map(float, odds_m.groups())
    score_m = SCORE_PATTERN.search(blob)
    fthg = ftag = None
    ftr = None
    if score_m and "Finished" in blob:
        fthg, ftag = int(score_m.group(1)), int(score_m.group(2))
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"

    date_label = _extract_date_label(blob)
    return {
        "Div": div_code,
        "Date": date_label,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": fthg,
        "FTAG": ftag,
        "FTR": ftr,
        "B365H": float(h_odds),
        "B365D": float(d_odds),
        "B365A": float(a_odds),
        "source": "oddsportal",
        "scraped_at": datetime.now().isoformat(),
    }


def _extract_date_label(blob: str) -> str:
    for token in ("Today", "Yesterday", "Tomorrow"):
        if token in blob:
            idx = blob.find(token)
            snippet = blob[idx:idx + 30]
            return snippet.split("Finished")[0].split("1X2")[0].strip()[:24]
    return datetime.now().strftime("%d/%m/%Y")


def scrape_oddsportal_page(url: str, div_code: str = "WC26") -> list[dict]:
    page = fetch_page(url, force_stealth=True)
    rows = page.css('[class*="eventRow"]')
    matches = []
    seen = set()
    for row in rows:
        parsed = _parse_event_row(row, div_code)
        if not parsed:
            continue
        key = (parsed["HomeTeam"], parsed["AwayTeam"])
        if key in seen:
            continue
        seen.add(key)
        matches.append(parsed)
    return matches


def scrape_worldcup_odds(include_results: bool = True) -> pd.DataFrame:
    cfg = _load_wc_config()
    div = cfg.get("div_code", "WC26")
    urls = [cfg["oddsportal"]["fixtures_url"]]
    if include_results:
        urls.append(cfg["oddsportal"]["results_url"])

    all_matches = []
    for url in urls:
        print(f"Scraping OddsPortal: {url}")
        all_matches.extend(scrape_oddsportal_page(url, div_code=div))

    if not all_matches:
        return pd.DataFrame()

    df = pd.DataFrame(all_matches).drop_duplicates(subset=["HomeTeam", "AwayTeam"], keep="last")
    print(f"Scraped {len(df)} World Cup matches from OddsPortal")
    return df