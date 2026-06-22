"""Reddit match-thread morale via PRAW (optional — needs API credentials)."""

import os

from utils.intel_score import score_texts


def reddit_credentials_configured() -> bool:
    return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))


def _reddit_client():
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "Trismegistus/0.3")
    if not client_id or not client_secret:
        return None
    import praw

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def verify_reddit_connection() -> tuple[bool, str]:
    """Check Reddit API credentials without a full team search."""
    if not reddit_credentials_configured():
        return False, "REDDIT_CLIENT_ID/SECRET not set in .env"
    reddit = _reddit_client()
    if reddit is None:
        return False, "PRAW client failed to initialize"
    try:
        name = reddit.subreddit("soccer").display_name
        return True, f"Reddit API OK (r/{name})"
    except Exception as exc:
        return False, f"Reddit API error: {exc}"


def scrape_reddit_sentiment(
    team: str,
    opponent: str | None = None,
    subreddits: list[str] | None = None,
    search_limit: int = 15,
    comment_limit: int = 25,
) -> float:
    """Return 0-1 morale from recent Reddit posts/comments mentioning the team."""
    reddit = _reddit_client()
    if reddit is None:
        return 0.5

    subs = subreddits or ["soccer"]
    query_parts = [team, "football", "soccer"]
    if opponent:
        query_parts.extend([opponent, "vs"])
    query = " ".join(query_parts)

    texts: list[str] = []
    try:
        for sub_name in subs:
            sub = reddit.subreddit(sub_name)
            for post in sub.search(query, limit=search_limit, sort="new"):
                if post.title:
                    texts.append(post.title)
                if post.selftext:
                    texts.append(post.selftext)
                post.comments.replace_more(limit=0)
                for comment in post.comments[:comment_limit]:
                    body = getattr(comment, "body", "")
                    if body:
                        texts.append(body)
            if len(texts) >= search_limit:
                break
    except Exception as e:
        print(f"Reddit scrape failed for {team}: {e}")
        return 0.5

    return score_texts(texts)