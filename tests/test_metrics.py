import numpy as np
import pytest

from evaluation.metrics import (
    flat_stake_roi,
    multiclass_brier,
    multiclass_log_loss,
    probs_to_matrix,
    summarize_predictions,
)


def test_probs_to_matrix_order():
    preds = [{"probs": {"H": 0.5, "D": 0.3, "A": 0.2}}]
    matrix = probs_to_matrix(preds)
    assert matrix.shape == (1, 3)
    assert matrix[0].tolist() == [0.3, 0.5, 0.2]


def test_log_loss_and_brier_perfect_prediction():
    y_true = [1, 0, 2]
    probs = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert multiclass_log_loss(y_true, probs) == pytest.approx(0.0, abs=1e-6)
    assert multiclass_brier(y_true, probs) == pytest.approx(0.0, abs=1e-6)


def test_flat_stake_roi_win_and_loss():
    preds = [
        {
            "outcome_code": 1,
            "actual_code": 1,
            "b365": (2.0, 3.5, 4.0),
        },
        {
            "outcome_code": 2,
            "actual_code": 1,
            "b365": (2.0, 3.5, 4.0),
        },
    ]
    roi, bets, profit = flat_stake_roi(preds)
    assert bets == 2
    assert profit == pytest.approx(0.0)
    assert roi == pytest.approx(0.0)


def test_summarize_predictions():
    preds = [
        {
            "outcome_code": 1,
            "actual_code": 1,
            "b365": (2.0, 3.5, 4.0),
            "bookie_code": 1,
            "probs": {"H": 0.6, "D": 0.2, "A": 0.2},
        },
    ]
    summary = summarize_predictions(preds, "test")
    assert summary["n"] == 1
    assert summary["accuracy"] == 100.0
    assert summary["roi_bets"] == 1
    assert summary["roi_profit_units"] == pytest.approx(1.0)