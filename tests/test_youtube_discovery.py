from scrapers.youtube_scraper import discover_youtube_video_ids


def test_discover_uses_pinned_urls_first():
    ids = discover_youtube_video_ids(
        "Arsenal",
        "Chelsea",
        video_urls=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        max_videos=3,
    )
    assert ids == ["dQw4w9WgXcQ"]


def test_discover_channel_queries(monkeypatch):
    calls: list[str] = []

    def fake_google(query, max_videos):
        calls.append(query)
        return ["abc12345678"] if "Sky Sports" in query else []

    monkeypatch.setattr(
        "scrapers.youtube_scraper._google_youtube_video_ids",
        fake_google,
    )
    ids = discover_youtube_video_ids(
        "Arsenal",
        "Chelsea",
        div_code="E0",
        max_videos=2,
    )
    assert ids
    assert any("Sky Sports Football" in q for q in calls)