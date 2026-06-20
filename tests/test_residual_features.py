import pandas as pd

from predictors.game_forger import GameForger, _add_div_features


def test_model_features_exclude_bookie():
    forger = GameForger()
    data = pd.DataFrame({
        "home_x_sentiment": [0.1],
        "away_x_sentiment": [0.2],
        "home_injuries": [1],
        "away_injuries": [0],
        "rain": [0.0],
        "wind": [5.0],
        "h2h_home_win_pct": [0.5],
        "h2h_avg_home_goals": [1.3],
        "h2h_avg_away_goals": [1.1],
        "avg_goals_scored_home": [1.4],
        "avg_goals_scored_away": [1.2],
        "avg_goals_conceded_home": [1.0],
        "avg_goals_conceded_away": [1.3],
        "Div": ["E0"],
        "B365H": [2.0],
        "B365D": [3.5],
        "B365A": [4.0],
        "implied_prob_H": [0.5],
        "odds_H": [2.0],
    })
    data = _add_div_features(data)
    outcome_cols, _ = forger._model_feature_columns(data)
    assert "B365H" not in outcome_cols
    assert "implied_prob_H" not in outcome_cols
    assert "odds_H" not in outcome_cols
    assert "home_injuries" in outcome_cols
    assert any(c.startswith("div_") for c in outcome_cols)