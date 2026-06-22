from agents.intel_agent import fetch_team_intel, intel_to_chaos_fields, stripped_intel_fields


def test_intel_to_chaos_fields():
    fields = intel_to_chaos_fields("home", {
        "news_attention": 0.4,
        "news_sentiment": 0.6,
        "reddit_sentiment": 0.55,
        "youtube_sentiment": 0.45,
    })
    assert fields["home_news_attention"] == 0.4
    assert fields["home_youtube_sentiment"] == 0.45


def test_fetch_team_intel_mocked(monkeypatch):
    monkeypatch.setattr(
        "agents.intel_agent.scrape_team_news_intel",
        lambda team, query_template=None: {"attention": 0.3, "sentiment": 0.7},
    )
    monkeypatch.setattr("agents.intel_agent.scrape_reddit_sentiment", lambda *a, **k: 0.6)
    captured = {}

    def fake_youtube(*a, **k):
        captured.update(k)
        return 0.4

    monkeypatch.setattr("agents.intel_agent.scrape_youtube_sentiment", fake_youtube)
    intel = fetch_team_intel("Arsenal", "2026-06-20", opponent="Chelsea", div_code="E0")
    assert captured.get("div_code") == "E0"
    assert intel["news_attention"] == 0.3
    assert intel["reddit_sentiment"] == 0.6


def test_stripped_intel_fields_are_zeroed():
    stripped = stripped_intel_fields()
    assert stripped["home_news_attention"] == 0.0
    assert stripped["away_reddit_sentiment"] == 0.0