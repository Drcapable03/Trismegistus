from scrapers.oddsportal import scrape_worldcup_odds
from scrapers.news_scraper import scrape_team_news_sentiment
from scrapers.injuries_scraper import scrape_team_injuries

__all__ = [
    "scrape_worldcup_odds",
    "scrape_team_news_sentiment",
    "scrape_team_injuries",
]