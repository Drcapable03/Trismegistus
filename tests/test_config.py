from config.settings import enabled_leagues, fixtures_url, league_urls, load_leagues_config


def test_enabled_leagues_respects_flag():
    leagues = enabled_leagues()
    assert "Premier League" in leagues
    assert "Championship" not in leagues  # disabled in config


def test_league_urls_include_historical_seasons():
    cfg = load_leagues_config()
    urls = league_urls()
    assert any("Premier League" in name for name in urls)
    assert any(cfg["season"] in url for url in urls.values())
    if cfg.get("historical_seasons"):
        assert any(s in url for s in cfg["historical_seasons"] for url in urls.values())


def test_fixtures_url():
    url = fixtures_url()
    assert "fixtures.csv" in url