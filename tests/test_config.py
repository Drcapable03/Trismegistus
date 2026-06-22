from config.settings import (
    all_seasons,
    enabled_leagues,
    fixtures_url,
    historical_seasons,
    league_div_codes,
    league_urls,
    load_leagues_config,
    oddsportal_league_urls,
)


def test_enabled_leagues_respects_flag():
    leagues = enabled_leagues()
    assert "Premier League" in leagues
    assert "Championship" not in leagues  # disabled in config


def test_league_urls_include_historical_seasons():
    cfg = load_leagues_config()
    urls = league_urls()
    assert any("Premier League" in name for name in urls)
    assert any(cfg["season"] in url for url in urls.values())
    for season in historical_seasons():
        assert season in all_seasons()
        assert any(season in url for url in urls.values())
    assert len(urls) == len(league_div_codes()) * len(all_seasons())


def test_fixtures_url():
    url = fixtures_url()
    assert "fixtures.csv" in url


def test_oddsportal_urls_for_big5():
    urls = oddsportal_league_urls(league_div_codes())
    assert set(urls) == {"E0", "SP1", "D1", "I1", "F1"}
    assert "premier-league" in urls["E0"]["fixtures_url"]