import numpy as np

from predictors.game_forger import GameForger


def test_compose_final_probs_bookie_weight():
    forger = GameForger(bookie_blend_weight=0.5)
    forger.dc_blend_weight = 0.0
    model_probs = np.array([0.2, 0.5, 0.3])
    model_only, _, _ = forger._compose_final_probs(
        model_probs, "Arsenal", "Chelsea", (2.0, 3.5, 4.0), blend_weight=0.0,
    )
    assert np.allclose(model_only, model_probs)
    bookie_only, _, implied = forger._compose_final_probs(
        model_probs, "Arsenal", "Chelsea", (2.0, 3.5, 4.0), blend_weight=1.0,
    )
    assert implied is not None
    assert not np.allclose(bookie_only, model_probs)
    mid, _, _ = forger._compose_final_probs(
        model_probs, "Arsenal", "Chelsea", (2.0, 3.5, 4.0), blend_weight=0.5,
    )
    assert np.allclose(mid, 0.5 * model_probs + 0.5 * bookie_only)