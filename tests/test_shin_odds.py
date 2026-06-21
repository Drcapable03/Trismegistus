import pytest

from evaluation.implied_odds import _proportional_probs, _shin_probs, implied_probs_from_odds


def test_shin_probs_sum_to_one():
    p_d, p_h, p_a = _shin_probs(1.5, 4.0, 6.0)
    assert p_d + p_h + p_a == pytest.approx(1.0, abs=1e-6)


def test_shin_differs_from_proportional_on_skewed_market():
    shin = _shin_probs(1.5, 4.0, 6.0)
    prop = _proportional_probs(1.5, 4.0, 6.0)
    assert shin != prop


def test_implied_probs_uses_config_method(monkeypatch):
    monkeypatch.setattr("evaluation.implied_odds.devig_method", lambda: "proportional")
    prop = implied_probs_from_odds(2.0, 3.5, 4.0)
    assert sum(prop) == pytest.approx(1.0, abs=1e-6)

    monkeypatch.setattr("evaluation.implied_odds.devig_method", lambda: "shin")
    shin = implied_probs_from_odds(2.0, 3.5, 4.0)
    assert sum(shin) == pytest.approx(1.0, abs=1e-6)


def test_shin_favorite_still_home_on_heavy_favourite():
    _, p_h, _ = _shin_probs(1.2, 6.0, 12.0)
    assert p_h > 0.5