import numpy as np
import pandas as pd

from predictors.game_forger import GameForger


def test_feature_row_df_aligns_series_columns():
    forger = GameForger()
    forger.outcome_features = ["b", "a", "c"]
    series = pd.Series({"a": 1.0, "b": 2.0, "extra": 9.0})
    df = forger._feature_row_df(series, forger.outcome_features)
    assert list(df.columns) == ["b", "a", "c"]
    assert df.iloc[0]["a"] == 1.0
    assert df.iloc[0]["b"] == 2.0
    assert df.iloc[0]["c"] == 0.0
    assert not np.isnan(df.values).any()