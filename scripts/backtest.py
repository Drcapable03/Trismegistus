import pandas as pd

OUTCOME_MAP = {"H": 1, "A": 2, "D": 0}
ODDS_COL_MAP = {"B365H": 1, "B365A": 2, "B365D": 0}


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

    if matches is not None and not matches.empty:
        merged = matches.merge(
            pred_df, left_on=["HomeTeam", "AwayTeam", "Date"],
            right_on=["home", "away", "date"], how="inner",
        )
        if not merged.empty and {"B365H", "B365A", "B365D"}.issubset(merged.columns):
            bookie_probs = merged[["B365H", "B365A", "B365D"]].apply(lambda x: 1 / x).fillna(0)
            bookie_pred = bookie_probs.idxmin(axis=1).map(ODDS_COL_MAP)
            actual = merged["FTR"].map(OUTCOME_MAP)
            bookie_accuracy = (actual == bookie_pred).mean() * 100
        else:
            bookie_accuracy = 0.0
    else:
        bookie_accuracy = 0.0

    print(f"Trismegistus Accuracy: {accuracy:.1f}%")
    print(f"Bookie Accuracy: {bookie_accuracy:.1f}%")
    return accuracy, bookie_accuracy


def format_prediction(pred: dict) -> str:
    blunder = " (Bookie Blunder!)" if pred.get("odds_error") else ""
    return (
        f"Match: {pred['home']} vs. {pred['away']}, {pred['date']}, "
        f"{pred['outcome']}, {pred['confidence']:.1f}%, "
        f"~{pred['total_goals']} goals{blunder}"
    )