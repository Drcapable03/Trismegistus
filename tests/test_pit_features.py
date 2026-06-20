import pandas as pd

from utils.pit_features import compute_pit_form_and_h2h


def test_pit_h2h_excludes_current_and_future():
    history = pd.DataFrame({
        "HomeTeam": ["A", "A", "A"],
        "AwayTeam": ["B", "B", "B"],
        "FTHG": [2, 0, 3],
        "FTAG": [1, 0, 1],
        "FTR": ["H", "D", "H"],
        "Date": ["01/01/2025", "01/06/2025", "01/01/2026"],
    })
    target = pd.DataFrame({
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "Date": ["01/06/2025"],
    })
    out = compute_pit_form_and_h2h(target, history)
    assert out.iloc[0]["h2h_home_win_pct"] == 1.0
    assert out.iloc[0]["h2h_avg_home_goals"] == 2.0


def test_pit_form_uses_only_prior_matches():
    history = pd.DataFrame({
        "HomeTeam": ["X", "X"],
        "AwayTeam": ["Y", "Z"],
        "FTHG": [3, 0],
        "FTAG": [0, 2],
        "FTR": ["H", "A"],
        "Date": ["01/01/2025", "01/02/2025"],
    })
    target = pd.DataFrame({
        "HomeTeam": ["X"],
        "AwayTeam": ["W"],
        "Date": ["01/03/2025"],
    })
    out = compute_pit_form_and_h2h(target, history)
    assert out.iloc[0]["avg_goals_scored_home"] == 1.5