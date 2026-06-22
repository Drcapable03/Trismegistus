import pandas as pd

from config.settings import (
    all_seasons,
    current_season,
    football_data_league_url,
    historical_seasons,
    league_urls,
)
from scripts.expand_history import season_coverage, season_from_league_label, verify_season_urls


def test_historical_seasons_expanded():
    seasons = historical_seasons()
    assert seasons == ["2425", "2324", "2223", "2122"]
    assert all_seasons() == [current_season(), *seasons]


def test_league_urls_cover_all_seasons():
    urls = league_urls()
    seasons = all_seasons()
    assert len(urls) == 5 * len(seasons)
    assert football_data_league_url("2324", "E0") in urls.values()
    assert any("2324" in label for label in urls)


def test_season_from_league_label():
    assert season_from_league_label("Premier League (2324)") == "2324"
    assert season_from_league_label("Premier League") is None


def test_season_coverage_groups_completed():
    matches = pd.DataFrame({
        "Div": ["E0", "E0", "SP1"],
        "Season": ["2324", "2324", "2425"],
        "Date": ["01/08/2023", "02/08/2023", "03/08/2024"],
        "HomeTeam": ["A", "B", "C"],
        "AwayTeam": ["B", "C", "D"],
        "FTR": ["H", None, "D"],
        "FTHG": [2, None, 1],
        "FTAG": [1, None, 1],
    })
    coverage = season_coverage(matches)
    e0 = coverage[(coverage["Season"] == "2324") & (coverage["Div"] == "E0")].iloc[0]
    assert int(e0.completed) == 1
    assert int(e0.total) == 2


def test_verify_season_urls(monkeypatch):
    class _Response:
        status_code = 200
        content = b"Div,Date,HomeTeam,AwayTeam\nE0,01/08/23,A,B\n"

        def headers_get(self, key, default=""):
            return "text/csv" if key == "Content-Type" else default

        @property
        def headers(self):
            return {"Content-Type": "text/csv"}

    def _fake_get(url, headers=None, timeout=None):
        return _Response()

    monkeypatch.setattr("scripts.expand_history.requests.get", _fake_get)
    health = verify_season_urls()
    assert health
    assert all(health.values())