import pandas as pd

from utils.pit_features import compute_pit_form_and_h2h
from utils.xg_sources import load_xg_caches_by_priority


def test_load_xg_caches_by_priority(monkeypatch):
    monkeypatch.setattr(
        "utils.xg_cache.load_xg_matches",
        lambda: pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"], "MatchDate": ["01/01/2024"],
                              "Div": ["E0"], "xg_home": [1.0], "xg_away": [0.5]}),
    )
    monkeypatch.setattr("utils.statsbomb_cache.load_statsbomb_xg", lambda: pd.DataFrame())
    monkeypatch.setattr("utils.fbref_cache.load_fbref_xg", lambda: pd.DataFrame())
    monkeypatch.setattr("utils.xg_sources.xg_source_priority", lambda: ["understat", "statsbomb"])

    caches = load_xg_caches_by_priority()
    assert len(caches) == 1
    assert caches[0][0] == "understat"


def test_enrichment_xg_uses_cached_prior(monkeypatch):
    monkeypatch.setattr(
        "utils.pit_features.load_xg_caches_by_priority",
        lambda: [("statsbomb", pd.DataFrame({
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "C"],
            "MatchDate": ["01/01/2024", "01/02/2024"],
            "Div": ["E0", "E0"],
            "xg_home": [2.0, 1.0],
            "xg_away": [0.5, 2.0],
        }))],
    )

    history = pd.DataFrame({
        "HomeTeam": ["A", "A", "B"],
        "AwayTeam": ["B", "C", "A"],
        "FTHG": [2, 1, 0],
        "FTAG": [1, 0, 2],
        "FTR": ["H", "H", "A"],
        "HS": [10, 8, 5],
        "AS": [6, 4, 12],
        "HST": [5, 4, 2],
        "AST": [3, 1, 7],
        "Date": ["01/01/2024", "01/02/2024", "01/03/2024"],
    })
    target = pd.DataFrame({
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "Date": ["01/03/2024"],
    })
    enriched = compute_pit_form_and_h2h(target, history, enrichment_xg=True)
    shots_only = compute_pit_form_and_h2h(target, history, enrichment_xg=False)
    assert enriched.iloc[0]["avg_xg_for_home"] != shots_only.iloc[0]["avg_xg_for_home"]