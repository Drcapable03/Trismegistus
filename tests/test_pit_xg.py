import pandas as pd

from utils.pit_features import compute_pit_form_and_h2h


def test_pit_xg_proxy_from_shots():
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
        "Date": ["01/01/2025", "01/02/2025", "01/03/2025"],
    })
    target = pd.DataFrame({
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "Date": ["01/03/2025"],
    })
    out = compute_pit_form_and_h2h(target, history)
    assert "avg_xg_for_home" in out.columns
    assert out.iloc[0]["avg_xg_for_home"] > 0
    assert out.iloc[0]["avg_xg_against_home"] > 0