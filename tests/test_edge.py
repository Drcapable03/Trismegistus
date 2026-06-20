import numpy as np
import pytest

from evaluation.edge import best_outcome_and_edge, passes_edge_filter, selective_accuracy


def test_best_outcome_and_edge():
    final = np.array([0.2, 0.6, 0.2])
    implied = np.array([0.3, 0.4, 0.3])
    pred, edge = best_outcome_and_edge(final, implied)
    assert pred == 1
    assert edge == pytest.approx(0.2)


def test_passes_edge_filter():
    assert passes_edge_filter(0.06, 0.05)
    assert not passes_edge_filter(0.04, 0.05)


def test_selective_accuracy():
    preds = [
        {"outcome_code": 1, "actual_code": 1},
        {"outcome_code": 0, "actual_code": 1},
    ]
    acc, n = selective_accuracy(preds)
    assert n == 2
    assert acc == 50.0