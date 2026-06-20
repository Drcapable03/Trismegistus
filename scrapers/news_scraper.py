"""Scrape team news sentiment via Google News search (Scrapling stealth)."""

from pathlib import Path
from urllib.parse import quote_plus

import yaml

from scrapers.browser import fetch_page


def _wc_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "worldcup.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scrape_team_news_sentiment(team: str, query_template: str | None = None) -> float:
    """
    Returns 0.1–1.0 sentiment proxy based on headline count from Google News search.
    Free alternative to NewsAPI.
    """
    cfg = _wc_config()
    template = query_template or cfg["news"]["query_template"]
    query = template.format(team=team)
    url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=nws&hl=en"

    try:
        page = fetch_page(url, force_stealth=True)
        headlines = page.css("a[href*='/url?']") or page.css("div[role='heading']")
        count = len(headlines)
        if count == 0:
            # Fallback: count article-like anchors
            headlines = page.css("a")
            count = sum(1 for h in headlines if h.text and len(h.text) > 20)
        sentiment = min(1.0, 0.1 + 0.04 * count)
        print(f"News scrape {team}: {count} items -> sentiment {sentiment:.2f}")
        return sentiment
    except Exception as e:
        print(f"News scrape failed for {team}: {e}")
        return 0.1