import pandas as pd

from predictors.game_forger import _walk_forward_split


def test_walk_forward_aligns_outcome_and_goals():
    data = pd.DataFrame({
        "Date": pd.to_datetime([f"2025-{i:02d}-01" for i in range(1, 11)]),
        "HomeTeam": [f"H{i}" for i in range(10)],
        "FTR": ["H"] * 10,
    })
    X_o = pd.DataFrame({"f1": range(10)}, index=data.index)
    y_o = pd.Series([1] * 10, index=data.index)
    X_g = pd.DataFrame({"g1": range(10, 20)}, index=data.index)
    y_g = pd.Series(range(10), index=data.index)

    X_train_o, X_test_o, _, _, X_train_g, X_test_g, _, _, train_idx, test_idx = (
        _walk_forward_split(data, X_o, y_o, X_g, y_g, test_fraction=0.2)
    )
    assert len(test_idx) == 2
    assert len(train_idx) == 8
    assert list(X_train_o.index) == list(X_train_g.index)
    assert list(X_test_o.index) == list(X_test_g.index)
    assert X_test_o.iloc[0]["f1"] == X_test_g.iloc[0]["g1"] - 10