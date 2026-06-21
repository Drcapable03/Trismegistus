import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from predictors.calibration import OutcomeCalibrator
from predictors.game_forger import GameForger
from predictors.registry import load_game_forger, save_game_forger


def test_registry_roundtrip_blend_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("predictors.registry.MODELS_DIR", tmp_path)
    forger = GameForger(bookie_blend_weight=0.25, edge_margin=0.08)
    forger.per_league_blend = {"E0": 0.15, "SP1": 0.05}
    forger.training_metadata = {"split_method": "walk_forward", "test_matches": 10}
    forger.outcome_features = ["a"]
    forger.goals_features = ["b"]

    base = GradientBoostingClassifier(n_estimators=5, random_state=42)
    X = pd.DataFrame({"a": [0.0, 1.0, 0.5, 1.5, 2.0, 0.2, 1.1, 0.9, 1.3, 0.7]})
    y = pd.Series([0, 1, 2, 1, 0, 1, 2, 0, 1, 2])
    base.fit(X.iloc[:7], y.iloc[:7])
    forger.outcome_model = base
    calibrator = OutcomeCalibrator()
    calibrator.fit(base, X.iloc[7:], y.iloc[7:])
    forger.calibrator = calibrator

    path = save_game_forger(forger)

    loaded = GameForger()
    load_game_forger(loaded, path)
    assert loaded.bookie_blend_weight == 0.25
    assert loaded.edge_margin == 0.08
    assert loaded.per_league_blend == {"E0": 0.15, "SP1": 0.05}
    assert loaded.training_metadata["test_matches"] == 10
    assert loaded.calibrator.is_fitted == forger.calibrator.is_fitted
    probs = loaded.calibrator.predict_proba(loaded.outcome_model, X.iloc[:1])
    assert np.isclose(probs.sum(), 1.0)