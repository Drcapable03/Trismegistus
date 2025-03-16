import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()

def fetch_news(team, date):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("No NEWS_API_KEY found!")
        return 0.1  # Default sentiment
    
    newsapi = NewsApiClient(api_key=api_key)
    query = f"{team} football"
    try:
        articles = newsapi.get_everything(q=query, from_param=date, to=date, language="en", sort_by="relevancy")
        if articles["articles"]:
            # Simple sentiment: more articles = positive buzz
            return min(1.0, 0.1 + 0.05 * len(articles["articles"]))
        return 0.1  # Neutral if no news
    except Exception as e:
        print(f"NewsAPI failed: {e}")
        return 0.1