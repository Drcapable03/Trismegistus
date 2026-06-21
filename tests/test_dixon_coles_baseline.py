import numpy as np

from predictors.dixon_coles_baseline import DixonColesBaseline


def test_dixon_coles_fit_creates_league_model():
    frame = __import__("pandas").DataFrame({
        "HomeTeam": ["A", "B", "A", "C", "B", "A"] * 8,
        "AwayTeam": ["B", "C", "C", "A", "A", "B"] * 8,
        "FTHG": [2, 1, 0, 1, 2, 3] * 8,
        "FTAG": [1, 0, 1, 2, 1, 0] * 8,
        "FTR": ["H", "H", "A", "A", "H", "H"] * 8,
        "Div": ["E0"] * 48,
    })
    model = DixonColesBaseline(min_matches=30)
    model.fit(frame, div_codes=["E0"])
    assert "E0" in model.models


def test_predict_probs_returns_normalized_array(monkeypatch):
    model = DixonColesBaseline()

    class _Grid:
        draw = 0.25
        home_win = 0.45
        away_win = 0.30

    class _FakeDC:
        fitted = True

        def predict(self, home, away, max_goals=15):
            return _Grid()

    model.models["E0"] = _FakeDC()
    probs = model.predict_probs("Arsenal", "Chelsea", div="E0")
    assert probs is not None
    assert abs(probs.sum() - 1.0) < 1e-6
    assert np.isclose(probs[1], 0.45)