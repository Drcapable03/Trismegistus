"""YouTube comment morale via youtube-comment-downloader."""

import re
from urllib.parse import parse_qs, urlparse

from scrapers.browser import fetch_page
from utils.intel_score import score_texts

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def _extract_video_ids(page, limit: int) -> list[str]:
    ids: list[str] = []
    for node in page.css("a[href*='youtube.com'], a[href*='youtu.be']"):
        href = node.attrib.get("href", "")
        if not href:
            continue
        match = _VIDEO_ID_RE.search(href)
        if match and match.group(1) not in ids:
            ids.append(match.group(1))
        if len(ids) >= limit:
            break
    return ids


def _google_youtube_video_ids(query: str, max_videos: int) -> list[str]:
    from urllib.parse import quote_plus

    url = f"https://www.google.com/search?q={quote_plus(query + ' site:youtube.com')}&hl=en"
    try:
        page = fetch_page(url, force_stealth=True)
        return _extract_video_ids(page, max_videos)
    except Exception as e:
        print(f"YouTube discovery failed: {e}")
        return []


def _comments_for_video(video_id: str, limit: int) -> list[str]:
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader
    except ImportError:
        return []

    texts: list[str] = []
    try:
        downloader = YoutubeCommentDownloader()
        for comment in downloader.get_comments_from_url(
            f"https://www.youtube.com/watch?v={video_id}",
            sort_by=0,
        ):
            text = comment.get("text", "")
            if text:
                texts.append(text)
            if len(texts) >= limit:
                break
    except Exception as e:
        print(f"YouTube comments failed for {video_id}: {e}")
    return texts


def _video_ids_from_urls(urls: list[str]) -> list[str]:
    ids: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        if "youtu.be" in parsed.netloc:
            vid = parsed.path.strip("/").split("/")[0]
        else:
            vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid and len(vid) == 11 and vid not in ids:
            ids.append(vid)
    return ids


def scrape_youtube_sentiment(
    team: str,
    opponent: str | None = None,
    video_urls: list[str] | None = None,
    search_query_template: str | None = None,
    comment_limit: int = 40,
    max_videos: int = 3,
) -> float:
    """Return 0-1 morale from YouTube comments on match-related videos."""
    video_ids = _video_ids_from_urls(video_urls or [])
    if not video_ids:
        template = search_query_template or "{team} {opponent} match preview"
        query = template.format(team=team, opponent=opponent or "").strip()
        video_ids = _google_youtube_video_ids(query, max_videos)

    if not video_ids:
        return 0.5

    texts: list[str] = []
    per_video = max(5, comment_limit // max(1, len(video_ids)))
    for vid in video_ids[:max_videos]:
        texts.extend(_comments_for_video(vid, per_video))

    return score_texts(texts)