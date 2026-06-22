import numpy as np

from config.settings import dixon_coles_blend_weight, set_dixon_coles_blend_weight
from predictors.game_forger import GameForger
from scripts.tune_dc_blend import tune_dc_blend


def test_set_dixon_coles_blend_weight(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.ROOT", tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    leagues = config_dir / "leagues.yaml"
    leagues.write_text(
        "model:\n  dixon_coles_blend_weight: 0.05\n  bookie_blend_weight: 0.1\n",
        encoding="utf-8",
    )

    set_dixon_coles_blend_weight(0.15)
    text = leagues.read_text(encoding="utf-8")
    assert "dixon_coles_blend_weight: 0.15" in text
    assert dixon_coles_blend_weight() == 0.15


def test_compose_final_probs_dc_weight():
    forger = GameForger(bookie_blend_weight=0.0)
    forger.dc_blend_weight = 0.2
    model_probs = np.array([0.2, 0.5, 0.3])

    class _FakeDC:
        def predict_probs(self, home, away, div=None):
            return np.array([0.1, 0.7, 0.2])

    forger.dc_baseline = _FakeDC()
    final, _, _ = forger._compose_final_probs(
        model_probs, "Arsenal", "Chelsea", None, blend_weight=0.0,
    )
    dc_probs = np.array([0.1, 0.7, 0.2])
    expected = 0.8 * model_probs + 0.2 * dc_probs
    expected = expected / expected.sum()
    assert np.allclose(final, expected)


def test_tune_dc_blend_selects_roi_weight(monkeypatch):
    roi_by_weight = {
        0.0: -5.0,
        0.05: -3.0,
        0.10: 2.0,
        0.15: 8.0,
        0.20: 1.0,
    }
    saved: list[float] = []

    class _FakeForger:
        dc_blend_weight = 0.05

        def train(self, **kwargs):
            return None

        def holdout_blend_accuracy(self, **kwargs):
            return 50.0

        def backtest_on_holdout(self, **kwargs):
            return [{"outcome_code": 1, "actual_code": 1, "b365": (2.0, 3.0, 4.0)}]

    def _fake_summarize(preds, label=""):
        w = float(label.rsplit("=", 1)[-1])
        bets = 10
        return {
            "n": bets,
            "roi_bets": bets,
            "roi_pct": roi_by_weight[w],
            "accuracy": 50.0,
        }

    monkeypatch.setattr("scripts.tune_dc_blend.GameForger", _FakeForger)
    monkeypatch.setattr("scripts.tune_dc_blend.summarize_predictions", _fake_summarize)
    monkeypatch.setattr("scripts.tune_dc_blend.dixon_coles_blend_weight", lambda: 0.05)
    monkeypatch.setattr(
        "scripts.tune_dc_blend.set_dixon_coles_blend_weight",
        lambda w: saved.append(w),
    )

    best, _ = tune_dc_blend(limit=10, persist=True, step=0.05, max_weight=0.20)
    assert best == 0.15
    assert saved == [0.15]