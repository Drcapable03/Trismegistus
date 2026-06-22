"""Aggregate live intel from news, Reddit, and YouTube (live prediction only)."""

import os

from config.settings import intel_config
from scrapers.news_scraper import scrape_team_news_intel
from scrapers.reddit_scraper import scrape_reddit_sentiment
from scrapers.youtube_scraper import scrape_youtube_sentiment

NEUTRAL_SENTIMENT = 0.5


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


def fetch_team_intel(
    team: str,
    date: str,
    opponent: str | None = None,
    div_code: str | None = None,
) -> dict[str, float]:
    """Buzz + morale features for one team. date reserved for future time filtering."""
    _ = date
    cfg = intel_config()
    out = {
        "news_attention": 0.0,
        "news_sentiment": NEUTRAL_SENTIMENT,
        "reddit_sentiment": NEUTRAL_SENTIMENT,
        "youtube_sentiment": NEUTRAL_SENTIMENT,
    }

    news_cfg = cfg.get("news") or {}
    if news_cfg.get("enabled", True) and _scrapling_enabled():
        news = scrape_team_news_intel(team, query_template=news_cfg.get("query_template"))
        out["news_attention"] = news["attention"]
        out["news_sentiment"] = news["sentiment"]
    elif news_cfg.get("enabled", True):
        api_sentiment = _fetch_news_api_attention(team, date)
        if api_sentiment is not None:
            out["news_attention"] = api_sentiment
            out["news_sentiment"] = NEUTRAL_SENTIMENT

    reddit_cfg = cfg.get("reddit") or {}
    if reddit_cfg.get("enabled", True):
        out["reddit_sentiment"] = scrape_reddit_sentiment(
            team,
            opponent=opponent,
            subreddits=reddit_cfg.get("subreddits"),
            search_limit=int(reddit_cfg.get("search_limit", 15)),
            comment_limit=int(reddit_cfg.get("comment_limit", 25)),
        )

    youtube_cfg = cfg.get("youtube") or {}
    if youtube_cfg.get("enabled", True) and _scrapling_enabled():
        out["youtube_sentiment"] = scrape_youtube_sentiment(
            team,
            opponent=opponent,
            video_urls=youtube_cfg.get("video_urls"),
            search_query_template=youtube_cfg.get("search_query_template"),
            comment_limit=int(youtube_cfg.get("comment_limit", 40)),
            max_videos=int(youtube_cfg.get("max_videos", 3)),
            div_code=div_code,
        )

    return out


def _fetch_news_api_attention(team: str, date: str) -> float | None:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return None
    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key=api_key)
    try:
        articles = newsapi.get_everything(
            q=f"{team} football",
            from_param=date,
            to=date,
            language="en",
            sort_by="relevancy",
        )
        count = len(articles.get("articles") or [])
        if count == 0:
            return 0.0
        return min(1.0, 0.1 + 0.05 * count)
    except Exception as e:
        print(f"NewsAPI failed: {e}")
        return None


def intel_to_chaos_fields(prefix: str, intel: dict[str, float]) -> dict[str, float]:
    """Map team intel dict to chaos row keys (home_* / away_*)."""
    return {
        f"{prefix}_news_attention": intel["news_attention"],
        f"{prefix}_news_sentiment": intel["news_sentiment"],
        f"{prefix}_reddit_sentiment": intel["reddit_sentiment"],
        f"{prefix}_youtube_sentiment": intel["youtube_sentiment"],
    }


def stripped_intel_fields() -> dict[str, float]:
    """Values used when intel is disabled (training / PIT-unsafe cache)."""
    return {
        "home_news_attention": 0.0,
        "away_news_attention": 0.0,
        "home_news_sentiment": 0.0,
        "away_news_sentiment": 0.0,
        "home_reddit_sentiment": 0.0,
        "away_reddit_sentiment": 0.0,
        "home_youtube_sentiment": 0.0,
        "away_youtube_sentiment": 0.0,
    }