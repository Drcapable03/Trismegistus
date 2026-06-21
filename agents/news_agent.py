"""Legacy news entry point — delegates to intel_agent."""

from agents.intel_agent import fetch_team_intel


def fetch_news_scraped(team: str, date: str | None = None) -> float:
    intel = fetch_team_intel(team, date or "")
    return intel["news_attention"]


def fetch_news_api(team: str, date: str) -> float | None:
    from agents.intel_agent import _fetch_news_api_attention

    return _fetch_news_api_attention(team, date)


def fetch_news(team: str, date: str) -> float:
    """Return news_attention (legacy float API)."""
    return fetch_news_scraped(team, date)