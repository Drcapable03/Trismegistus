"""Probabilistic and value metrics for holdout evaluation."""

import numpy as np

OUTCOME_CODES = (0, 1, 2)


def probs_to_matrix(predictions: list[dict]) -> np.ndarray:
    rows = []
    for p in predictions:
        pr = p.get("probs", {})
        rows.append([pr.get("D", 0.0), pr.get("H", 0.0), pr.get("A", 0.0)])
    return np.array(rows, dtype=float)


def multiclass_log_loss(y_true: list[int], prob_matrix: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    eps = 1e-15
    loss = 0.0
    for i, y in enumerate(y_true):
        p = max(eps, min(1 - eps, prob_matrix[i, y]))
        loss -= np.log(p)
    return float(loss / len(y_true))


def multiclass_brier(y_true: list[int], prob_matrix: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    score = 0.0
    for i, y in enumerate(y_true):
        one_hot = np.zeros(3)
        one_hot[y] = 1.0
        score += float(np.sum((prob_matrix[i] - one_hot) ** 2))
    return score / len(y_true)


def _decimal_odds(b365: tuple[float, float, float], outcome_code: int) -> float:
    h, d, a = b365
    return {0: d, 1: h, 2: a}[outcome_code]


def flat_stake_roi(predictions: list[dict]) -> tuple[float, int, float]:
    """1-unit flat stakes on model pick; needs b365 tuple and actual_code per row."""
    profit = 0.0
    bets = 0
    for p in predictions:
        b365 = p.get("b365")
        if b365 is None:
            continue
        odds = _decimal_odds(b365, p["outcome_code"])
        if odds <= 0:
            continue
        bets += 1
        if p["outcome_code"] == p["actual_code"]:
            profit += odds - 1.0
        else:
            profit -= 1.0
    roi = (profit / bets * 100) if bets else 0.0
    return roi, bets, profit


def bookie_flat_stake_roi(predictions: list[dict]) -> tuple[float, int, float]:
    """ROI if betting bookie favorite each row."""
    profit = 0.0
    bets = 0
    for p in predictions:
        b365 = p.get("b365")
        bookie_code = p.get("bookie_code")
        if b365 is None or bookie_code is None:
            continue
        odds = _decimal_odds(b365, bookie_code)
        if odds <= 0:
            continue
        bets += 1
        if bookie_code == p["actual_code"]:
            profit += odds - 1.0
        else:
            profit -= 1.0
    roi = (profit / bets * 100) if bets else 0.0
    return roi, bets, profit


def summarize_predictions(predictions: list[dict], label: str = "holdout") -> dict:
    if not predictions:
        return {"label": label, "n": 0}
    y_true = [p["actual_code"] for p in predictions]
    probs = probs_to_matrix(predictions)
    roi, bets, profit = flat_stake_roi(predictions)
    broi, bbets, bprofit = bookie_flat_stake_roi(predictions)
    acc = sum(1 for p in predictions if p["outcome_code"] == p["actual_code"]) / len(predictions) * 100
    return {
        "label": label,
        "n": len(predictions),
        "accuracy": acc,
        "log_loss": multiclass_log_loss(y_true, probs),
        "brier": multiclass_brier(y_true, probs),
        "roi_pct": roi,
        "roi_bets": bets,
        "roi_profit_units": profit,
        "bookie_roi_pct": broi,
        "bookie_roi_bets": bbets,
    }


def print_metrics_summary(summary: dict) -> None:
    if summary.get("n", 0) == 0:
        print(f"{summary.get('label', 'Metrics')}: no predictions")
        return
    print(
        f"{summary['label']} ({summary['n']} picks): "
        f"acc {summary['accuracy']:.1f}%, "
        f"log-loss {summary['log_loss']:.3f}, "
        f"Brier {summary['brier']:.3f}"
    )
    print(
        f"  Flat-stake ROI: {summary['roi_pct']:+.1f}% "
        f"({summary['roi_profit_units']:+.1f} units / {summary['roi_bets']} bets)"
    )
    print(
        f"  Bookie-fav ROI: {summary['bookie_roi_pct']:+.1f}% "
        f"({summary['bookie_roi_bets']} bets)"
    )