"""Scrape team news headlines via Google News (Scrapling) and score with VADER."""

from pathlib import Path
from urllib.parse import quote_plus

import yaml

from scrapers.browser import fetch_page
from utils.intel_score import attention_from_count, score_texts

_INTEL_PATH = Path(__file__).resolve().parent.parent / "config" / "intel.yaml"
_WC_PATH = Path(__file__).resolve().parent.parent / "config" / "worldcup.yaml"


def _load_query_template() -> str:
    if _INTEL_PATH.exists():
        with open(_INTEL_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("news") or {}).get("query_template", "{team} football news")
    with open(_WC_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["news"]["query_template"]


def scrape_team_news_headlines(team: str, query_template: str | None = None) -> list[str]:
    template = query_template or _load_query_template()
    query = template.format(team=team)
    url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=nws&hl=en"

    try:
        page = fetch_page(url, force_stealth=True)
        headlines: list[str] = []
        for node in page.css("a[href*='/url?']"):
            text = (node.text or "").strip()
            if len(text) > 20:
                headlines.append(text)
        if not headlines:
            for node in page.css("div[role='heading']"):
                text = (node.text or "").strip()
                if len(text) > 10:
                    headlines.append(text)
        return headlines
    except Exception as e:
        print(f"News scrape failed for {team}: {e}")
        return []


def scrape_team_news_intel(team: str, query_template: str | None = None) -> dict[str, float]:
    """Return news_attention (buzz) and news_sentiment (VADER) for a team."""
    headlines = scrape_team_news_headlines(team, query_template=query_template)
    attention = attention_from_count(len(headlines))
    sentiment = score_texts(headlines) if headlines else 0.5
    print(
        f"News intel {team}: {len(headlines)} headlines -> "
        f"attention {attention:.2f}, sentiment {sentiment:.2f}"
    )
    return {"attention": attention, "sentiment": sentiment}


def scrape_team_news_sentiment(team: str, query_template: str | None = None) -> float:
    """Legacy alias — returns news_attention only."""
    return scrape_team_news_intel(team, query_template=query_template)["attention"]