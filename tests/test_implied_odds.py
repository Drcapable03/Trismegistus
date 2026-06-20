import pytest

from evaluation.implied_odds import bookie_favorite, implied_probs_from_odds, strip_overround
import numpy as np


def test_strip_overround_sums_to_one():
    probs = strip_overround(np.array([0.4, 0.35, 0.35]))
    assert probs.sum() == pytest.approx(1.0)


def test_bookie_favorite_picks_shortest_odds():
    assert bookie_favorite(1.5, 4.0, 6.0) == 1


def test_implied_probs_valid():
    p_d, p_h, p_a = implied_probs_from_odds(2.0, 3.5, 4.0)
    assert abs(p_d + p_h + p_a - 1.0) < 0.01