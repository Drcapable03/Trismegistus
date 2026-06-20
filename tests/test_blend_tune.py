import numpy as np

from predictors.game_forger import GameForger


def test_blend_with_bookie_weight():
    forger = GameForger(bookie_blend_weight=0.5)
    model_probs = np.array([0.2, 0.5, 0.3])
    model_only = forger._blend_with_bookie(model_probs, (2.0, 3.5, 4.0), blend_weight=0.0)
    assert np.allclose(model_only, model_probs)
    bookie_only = forger._blend_with_bookie(model_probs, (2.0, 3.5, 4.0), blend_weight=1.0)
    mid = forger._blend_with_bookie(model_probs, (2.0, 3.5, 4.0), blend_weight=0.5)
    assert not np.allclose(bookie_only, model_probs)
    assert np.allclose(mid, 0.5 * model_probs + 0.5 * bookie_only)