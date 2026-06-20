"""Scrape injury/squad signals via Google search (Scrapling stealth)."""

import re
from pathlib import Path
from urllib.parse import quote_plus

import yaml

from scrapers.browser import fetch_page

INJURY_WORDS = re.compile(
    r"\b(injur|doubt|out|sidelined|miss|ruled out|absent|unavailable|squad)\b",
    re.I,
)


def _wc_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "worldcup.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scrape_team_injuries(team: str, query_template: str | None = None) -> int:
    """
    Returns injury concern count (0+) from search snippets mentioning injuries.
    """
    cfg = _wc_config()
    template = query_template or cfg["injuries"]["query_template"]
    query = template.format(team=team)
    url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en"

    try:
        page = fetch_page(url, force_stealth=True)
        snippets = page.css("div[data-sncf], span, div")
        hits = 0
        for el in snippets:
            text = (el.text or "").strip()
            if len(text) < 15 or len(text) > 300:
                continue
            if INJURY_WORDS.search(text) and team.lower() in text.lower():
                hits += 1
        # Cap to avoid noise inflation
        count = min(hits, 10)
        print(f"Injury scrape {team}: {count} concern signals")
        return count
    except Exception as e:
        print(f"Injury scrape failed for {team}: {e}")
        return 0