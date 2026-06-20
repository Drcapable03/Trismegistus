import pandas as pd

from evaluation.implied_odds import bookie_accuracy

OUTCOME_MAP = {"H": 1, "A": 2, "D": 0}


def backtest_predictions(predictions: list[dict], matches: pd.DataFrame | None = None) -> tuple[float, float]:
    """Compare model predictions against actual results and bookmaker baseline."""
    if not predictions:
        print("No predictions to backtest.")
        return 0.0, 0.0

    pred_df = pd.DataFrame(predictions)
    if "actual_code" not in pred_df.columns:
        print("Predictions lack actual_code — run backtest_on_holdout(), not predict().")
        return 0.0, 0.0

    accuracy = (pred_df["outcome_code"] == pred_df["actual_code"]).mean() * 100

    bookie_acc = 0.0
    if matches is not None and not matches.empty:
        m = matches.copy()
        m["Date"] = pd.to_datetime(m["Date"], dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y")
        p = pred_df.copy()
        p["date"] = pd.to_datetime(p["date"], dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y")
        merged = m.merge(
            p,
            left_on=["HomeTeam", "AwayTeam", "Date"],
            right_on=["home", "away", "date"],
            how="inner",
        )
        if not merged.empty and {"B365H", "B365A", "B365D", "FTR"}.issubset(merged.columns):
            bookie_acc = bookie_accuracy(merged)

    print(f"Trismegistus Accuracy (deterministic holdout): {accuracy:.1f}%")
    print(f"Bookie Accuracy on same holdout rows (B365): {bookie_acc:.1f}%")
    return accuracy, bookie_acc


def format_prediction(pred: dict) -> str:
    blunder = " (Bookie Blunder!)" if pred.get("odds_error") else ""
    goals = pred.get("expected_goals", pred.get("total_goals"))
    goals_str = f"{goals:.1f}" if isinstance(goals, float) else str(goals)
    bookie = ""
    if pred.get("bookie_pick"):
        bookie = f", Bookie: {pred['bookie_pick']}"
    probs = pred.get("probs")
    prob_str = ""
    if probs:
        prob_str = (
            f" [H {probs['H']:.0%} / D {probs['D']:.0%} / A {probs['A']:.0%}]"
        )
    return (
        f"Match: {pred['home']} vs. {pred['away']}, {pred['date']}, "
        f"Model: {pred['outcome']}, {pred['confidence']:.1f}%, "
        f"~{goals_str} xG{bookie}{prob_str}{blunder}"
    )