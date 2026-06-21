import pandas as pd

from evaluation.edge import selective_accuracy
from evaluation.implied_odds import bookie_accuracy

OUTCOME_MAP = {"H": 1, "A": 2, "D": 0}


def backtest_predictions(
    predictions: list[dict],
    matches: pd.DataFrame | None = None,
    selective_predictions: list[dict] | None = None,
    edge_margin: float = 0.05,
) -> tuple[float, float]:
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

    print(f"Holdout accuracy (all test picks): {accuracy:.1f}%")
    print(f"Bookie accuracy on same rows (B365): {bookie_acc:.1f}%")

    if selective_predictions is not None:
        sel_acc, n = selective_accuracy(selective_predictions)
        print(
            f"Selective holdout (edge ≥{edge_margin:.0%}): "
            f"{sel_acc:.1f}% on {n} picks"
        )

    return accuracy, bookie_acc


def format_prediction(pred: dict) -> str:
    blunder = " (Bookie Blunder!)" if pred.get("odds_error") else ""
    goals = pred.get("expected_goals", pred.get("total_goals"))
    goals_str = f"{goals:.1f}" if isinstance(goals, float) else str(goals)
    bookie = ""
    if pred.get("bookie_pick"):
        bookie = f", Bookie: {pred['bookie_pick']}"
    edge = pred.get("edge")
    edge_str = f", Edge: {edge:.1%}" if edge is not None else ""
    margin = pred.get("edge_margin")
    margin_str = f" (margin ≥{margin:.0%})" if margin is not None else ""
    div = pred.get("div")
    div_str = f" [{div}]" if div else ""
    probs = pred.get("probs")
    prob_str = ""
    if probs:
        prob_str = (
            f" [H {probs['H']:.0%} / D {probs['D']:.0%} / A {probs['A']:.0%}]"
        )
    odds_str = ""
    if pred.get("b365_close"):
        h, d, a = pred["b365_close"]
        odds_str = f", Closing: {h:.2f}/{d:.2f}/{a:.2f}"
    elif pred.get("b365"):
        h, d, a = pred["b365"]
        odds_str = f", Odds: {h:.2f}/{d:.2f}/{a:.2f}"
    intel = pred.get("intel") or {}
    intel_str = ""
    if intel.get("home_news_attention", 0) > 0 or intel.get("away_news_attention", 0) > 0:
        intel_str = (
            f", Intel attn H/A: {intel.get('home_news_attention', 0):.2f}/"
            f"{intel.get('away_news_attention', 0):.2f}"
        )
    return (
        f"Match: {pred['home']} vs. {pred['away']}, {pred['date']}{div_str}, "
        f"Model: {pred['outcome']}, {pred['confidence']:.1f}%, "
        f"~{goals_str} xG{edge_str}{margin_str}{bookie}{odds_str}{intel_str}"
        f"{prob_str}{blunder}"
    )