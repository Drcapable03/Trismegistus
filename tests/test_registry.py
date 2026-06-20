from predictors.game_forger import GameForger
from predictors.registry import load_game_forger, save_game_forger
from config.settings import MODELS_DIR


def test_registry_roundtrip_blend_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("predictors.registry.MODELS_DIR", tmp_path)
    forger = GameForger(bookie_blend_weight=0.25)
    forger.training_metadata = {"split_method": "walk_forward", "test_matches": 10}
    forger.outcome_features = ["a"]
    forger.goals_features = ["b"]
    path = save_game_forger(forger)

    loaded = GameForger()
    load_game_forger(loaded, path)
    assert loaded.bookie_blend_weight == 0.25
    assert loaded.training_metadata["test_matches"] == 10