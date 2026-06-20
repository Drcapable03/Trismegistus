import pytest

from scripts.backtest import backtest_predictions


def test_backtest_predictions_accuracy():
    predictions = [
        {"outcome_code": 1, "actual_code": 1},
        {"outcome_code": 0, "actual_code": 1},
        {"outcome_code": 2, "actual_code": 2},
    ]
    accuracy, _ = backtest_predictions(predictions)
    assert accuracy == pytest.approx(66.7, rel=0.1)


def test_backtest_empty():
    accuracy, bookie = backtest_predictions([])
    assert accuracy == 0.0
    assert bookie == 0.0