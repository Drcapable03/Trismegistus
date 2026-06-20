import os

from dotenv import load_dotenv

load_dotenv()


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


def fetch_injuries(team: str) -> int:
    """Return injury concern count via Scrapling search."""
    if not _scrapling_enabled():
        return 0
    from scrapers.injuries_scraper import scrape_team_injuries
    return scrape_team_injuries(team)