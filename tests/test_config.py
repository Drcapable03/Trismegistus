from config.settings import fixtures_url, league_urls, load_leagues_config


def test_league_urls_use_configured_season():
    cfg = load_leagues_config()
    urls = league_urls()
    assert "Premier League" in urls
    assert cfg["season"] in urls["Premier League"]
    assert urls["Premier League"].endswith("/E0.csv")


def test_fixtures_url():
    url = fixtures_url()
    assert "fixtures.csv" in url