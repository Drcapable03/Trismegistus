"""Bookie implied probabilities with simple overround removal."""

import numpy as np
import pandas as pd

OUTCOME_FROM_COL = {"B365H": 1, "B365A": 2, "B365D": 0}


def strip_overround(probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    if total <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    return probs / total


def implied_probs_from_odds(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Return (p_draw, p_home, p_away) matching GameForger outcome codes 0,1,2."""
    odds = np.array([d, h, a], dtype=float)
    odds = np.where(odds <= 0, np.nan, odds)
    if np.any(np.isnan(odds)):
        return (1 / 3, 1 / 3, 1 / 3)
    raw = 1 / odds
    norm = strip_overround(raw)
    return float(norm[0]), float(norm[1]), float(norm[2])  # D, H, A


def bookie_favorite(h: float, d: float, a: float) -> int:
    p_d, p_h, p_a = implied_probs_from_odds(h, d, a)
    probs = {0: p_d, 1: p_h, 2: p_a}
    return max(probs, key=probs.get)


def bookie_predictions(df: pd.DataFrame) -> pd.Series:
    """Predict outcome code from B365 odds for each row."""
    return df.apply(
        lambda r: bookie_favorite(r.get("B365H", 0), r.get("B365D", 0), r.get("B365A", 0)),
        axis=1,
    )


def bookie_accuracy(merged: pd.DataFrame, actual_col: str = "FTR") -> float:
    outcome_map = {"H": 1, "A": 2, "D": 0}
    actual = merged[actual_col].map(outcome_map)
    predicted = bookie_predictions(merged)
    return float((actual == predicted).mean() * 100)