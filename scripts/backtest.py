import pandas as pd
from utils.db import engine

def backtest_predictions(predictions, matches):
    pred_df = pd.DataFrame([
        {"HomeTeam": p.split(" vs. ")[0].split(": ")[1], 
         "AwayTeam": p.split(" vs. ")[1].split(", ")[0], 
         "Date": p.split(", ")[1], 
         "Predicted": p.split(", ")[2].split(", ")[0]}
        for p in predictions
    ])
    matches["FTR"] = matches["FTR"].map({"H": 1, "A": 2, "D": 0})
    pred_df["Predicted"] = pred_df["Predicted"].map({"Home Win": 1, "Away Win": 2, "Draw": 0})

    merged = matches.merge(pred_df, on=["HomeTeam", "AwayTeam", "Date"])
    
    # Our accuracy
    accuracy = (merged["FTR"] == merged["Predicted"]).mean() * 100
    
    # Bookie accuracy (implied odds winner)
    bookie_probs = merged[["B365H", "B365A", "B365D"]].apply(lambda x: 1/x).fillna(0)
    bookie_pred = bookie_probs.idxmin(axis=1).map({"B365H": 1, "B365A": 2, "B365D": 0})
    bookie_accuracy = (merged["FTR"] == bookie_pred).mean() * 100

    print(f"Trismegistus Accuracy: {accuracy:.1f}%")
    print(f"Bookie Accuracy: {bookie_accuracy:.1f}%")
    return accuracy, bookie_accuracy