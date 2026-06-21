"""Bookie implied probabilities with Shin de-vig (penaltyblog) or proportional fallback."""

import numpy as np
import pandas as pd

from config.settings import devig_method

OUTCOME_FROM_COL = {"B365H": 1, "B365A": 2, "B365D": 0}


def strip_overround(probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    if total <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    return probs / total


def _shin_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    try:
        from penaltyblog.implied import ImpliedMethod, OddsFormat, calculate_implied

        result = calculate_implied(
            [d, h, a],
            method=ImpliedMethod.SHIN,
            odds_format=OddsFormat.DECIMAL,
        )
        p_d, p_h, p_a = result.probabilities
        return float(p_d), float(p_h), float(p_a)
    except Exception:
        return _proportional_probs(h, d, a)


def _proportional_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    odds = np.array([d, h, a], dtype=float)
    odds = np.where(odds <= 0, np.nan, odds)
    if np.any(np.isnan(odds)):
        return (1 / 3, 1 / 3, 1 / 3)
    raw = 1 / odds
    norm = strip_overround(raw)
    return float(norm[0]), float(norm[1]), float(norm[2])


def implied_probs_from_odds(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Return (p_draw, p_home, p_away) matching GameForger outcome codes 0,1,2."""
    if devig_method() == "shin":
        return _shin_probs(h, d, a)
    return _proportional_probs(h, d, a)


def bookie_favorite(h: float, d: float, a: float) -> int:
    p_d, p_h, p_a = implied_probs_from_odds(h, d, a)
    probs = {0: p_d, 1: p_h, 2: p_a}
    return max(probs, key=probs.get)


def bookie_predictions(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: bookie_favorite(r.get("B365H", 0), r.get("B365D", 0), r.get("B365A", 0)),
        axis=1,
    )


def bookie_accuracy(merged: pd.DataFrame, actual_col: str = "FTR") -> float:
    outcome_map = {"H": 1, "A": 2, "D": 0}
    actual = merged[actual_col].map(outcome_map)
    predicted = bookie_predictions(merged)
    return float((actual == predicted).mean() * 100)