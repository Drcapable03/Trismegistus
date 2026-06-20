"""Edge detection: model probability vs bookie implied (residual value layer)."""

import numpy as np

from evaluation.implied_odds import implied_probs_from_odds

OUTCOME_CODES = (0, 1, 2)  # Draw, Home, Away


def implied_probs_array(b365: tuple[float, float, float] | None) -> np.ndarray | None:
    if b365 is None:
        return None
    h, d, a = b365
    if min(h, d, a) <= 0:
        return None
    p_d, p_h, p_a = implied_probs_from_odds(h, d, a)
    return np.array([p_d, p_h, p_a], dtype=float)


def best_outcome_and_edge(
    final_probs: np.ndarray,
    implied_probs: np.ndarray,
) -> tuple[int, float]:
    """Return (outcome_code, edge) for the highest-probability pick."""
    pred = int(np.argmax(final_probs))
    edge = float(final_probs[pred] - implied_probs[pred])
    return pred, edge


def passes_edge_filter(edge: float, margin: float) -> bool:
    return edge >= margin


def selective_accuracy(predictions: list[dict]) -> tuple[float, int]:
    """Accuracy on predictions that include actual_code (holdout rows)."""
    if not predictions:
        return 0.0, 0
    correct = sum(1 for p in predictions if p["outcome_code"] == p["actual_code"])
    return correct / len(predictions) * 100, len(predictions)