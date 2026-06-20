import os

from dotenv import load_dotenv

load_dotenv()


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


def fetch_news_scraped(team: str, date: str | None = None) -> float:
    from scrapers.news_scraper import scrape_team_news_sentiment
    return scrape_team_news_sentiment(team)


def fetch_news_api(team: str, date: str) -> float | None:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return None

    from newsapi import NewsApiClient
    newsapi = NewsApiClient(api_key=api_key)
    query = f"{team} football"
    try:
        articles = newsapi.get_everything(
            q=query, from_param=date, to=date, language="en", sort_by="relevancy",
        )
        if articles["articles"]:
            return min(1.0, 0.1 + 0.05 * len(articles["articles"]))
        return 0.1
    except Exception as e:
        print(f"NewsAPI failed: {e}")
        return None


def fetch_news(team: str, date: str) -> float:
    """Scrapling Google News first, NewsAPI fallback."""
    if _scrapling_enabled():
        return fetch_news_scraped(team, date)

    api_result = fetch_news_api(team, date)
    if api_result is not None:
        return api_result

    print("No news source available — using neutral sentiment")
    return 0.1