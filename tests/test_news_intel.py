from scrapers.news_scraper import scrape_team_news_intel


class _FakeNode:
    def __init__(self, text: str):
        self.text = text


class _FakePage:
    def __init__(self, nodes):
        self._nodes = nodes

    def css(self, selector):
        if "heading" in selector:
            return []
        return self._nodes


def test_scrape_team_news_intel_from_headlines(monkeypatch):
    headlines = [
        _FakeNode("Arsenal secure brilliant victory in north London derby"),
        _FakeNode("Gunners fans celebrate after dominant performance"),
    ]
    monkeypatch.setattr(
        "scrapers.news_scraper.fetch_page",
        lambda *a, **k: _FakePage(headlines),
    )
    intel = scrape_team_news_intel("Arsenal")
    assert intel["attention"] > 0
    assert intel["sentiment"] > 0.5