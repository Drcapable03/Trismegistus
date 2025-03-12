import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from utils.db import engine

class BlunderSniffer:
    def __init__(self):
        self.outcome_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.goals_model = LinearRegression()

    def prepare_data(self):
        matches = pd.read_sql("SELECT * FROM matches", engine)
        form = pd.read_sql("SELECT * FROM team_form", engine)

        # Merge home/away form
        home_form = form.rename(columns={"team": "HomeTeam"})
        away_form = form.rename(columns={"team": "AwayTeam"})
        data = matches.merge(home_form, on="HomeTeam").merge(away_form, on="AwayTeam", suffixes=("_home", "_away"))

        # Features for outcome
        X_outcome = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            "B365H", "B365A", "B365D"
        ]].fillna(0)
        y_outcome = data["FTR"].map({"H": 1, "A": 2, "D": 0}).fillna(0)

        # Features for total goals
        X_goals = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away"
        ]].fillna(0)
        y_goals = (data["FTHG"] + data["FTAG"]).fillna(0)

        # Blunder flag
        data["implied_prob_H"] = 1 / data["B365H"]
        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        return X_outcome, y_outcome, X_goals, y_goals, data[["HomeTeam", "AwayTeam", "Date", "odds_error"]]

    def train(self):
        X_outcome, y_outcome, X_goals, y_goals, _ = self.prepare_data()
        self.outcome_model.fit(X_outcome, y_outcome)
        self.goals_model.fit(X_goals, y_goals)

    def predict(self, limit=5):
        X_outcome, y_outcome, X_goals, y_goals, context = self.prepare_data()
        probs = self.outcome_model.predict_proba(X_outcome)
        outcome_preds = self.outcome_model.predict(X_outcome)
        goals_preds = self.goals_model.predict(X_goals)

        results = []
        for i, (pred, prob, goals) in enumerate(zip(outcome_preds, probs, goals_preds)):
            if i >= limit:
                break
            home, away, date = context.iloc[i][["HomeTeam", "AwayTeam", "Date"]]
            outcome = {1: "Home Win", 2: "Away Win", 0: "Draw"}[pred]
            confidence = max(prob) * 100
            total_goals = max(0, round(goals))  # No negative goals
            blunder = " (Bookie Blunder!)" if context.iloc[i]["odds_error"] else ""
            results.append(f"Match: {home} vs. {away}, {date}, {outcome}, {confidence:.1f}%, ~{total_goals} goals{blunder}")
        
        return results