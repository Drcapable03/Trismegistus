import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from utils.db import engine
from scripts.fetch_chaos import get_chaos_data

class GameForger:
    def __init__(self, weather_api_key, sim_runs=1000):
        self.outcome_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.goals_model = LinearRegression()
        self.weather_api_key = weather_api_key
        self.sim_runs = sim_runs
        self.outcome_features = None
        self.goals_features = None

    def prepare_data(self):
        matches = pd.read_sql("SELECT * FROM matches", engine)
        form = pd.read_sql("SELECT * FROM team_form", engine)
        chaos = get_chaos_data(matches, self.weather_api_key)

        data = matches.merge(form.rename(columns={"team": "HomeTeam"}), on="HomeTeam") \
                      .merge(form.rename(columns={"team": "AwayTeam"}), on="AwayTeam", suffixes=("_home", "_away")) \
                      .merge(chaos, on=["HomeTeam", "AwayTeam", "Date"])

        X_outcome = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            "B365H", "B365A", "B365D",
            "rain", "wind", "home_rss_sentiment", "away_rss_sentiment",
            "home_x_sentiment", "away_x_sentiment"
        ]].fillna(0)
        y_outcome = data["FTR"].map({"H": 1, "A": 2, "D": 0}).fillna(0)

        X_goals = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            "rain", "wind"
        ]].fillna(0)
        y_goals = (data["FTHG"] + data["FTAG"]).fillna(0)

        data["implied_prob_H"] = 1 / data["B365H"]
        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        self.outcome_features = X_outcome.columns
        self.goals_features = X_goals.columns

        return X_outcome, y_outcome, X_goals, y_goals, data[["HomeTeam", "AwayTeam", "Date", "odds_error"]]

    def train(self):
        X_outcome, y_outcome, X_goals, y_goals, _ = self.prepare_data()
        self.outcome_model.fit(X_outcome, y_outcome)
        self.goals_model.fit(X_goals, y_goals)

    def simulate_match(self, outcome_features, goals_features):
        outcome_df = pd.DataFrame([outcome_features], columns=self.outcome_features)
        goals_df = pd.DataFrame([goals_features], columns=self.goals_features)
        
        probs = self.outcome_model.predict_proba(outcome_df)[0]
        goals_pred = self.goals_model.predict(goals_df)[0]
        outcomes = np.random.choice([1, 2, 0], size=self.sim_runs, p=probs)
        goals = np.random.poisson(max(0, goals_pred), size=self.sim_runs)
        return outcomes, goals

    def predict(self, limit=5):
        X_outcome, y_outcome, X_goals, y_goals, context = self.prepare_data()
        results = []
        for i in range(min(limit, len(X_outcome))):
            home, away, date = context.iloc[i][["HomeTeam", "AwayTeam", "Date"]]
            outcome_features = X_outcome.iloc[i]
            goals_features = X_goals.iloc[i]
            outcomes, goals = self.simulate_match(outcome_features, goals_features)
            
            pred = np.bincount(outcomes).argmax()
            confidence = np.mean(outcomes == pred) * 100
            total_goals = max(0, round(np.mean(goals)))
            outcome = {1: "Home Win", 2: "Away Win", 0: "Draw"}[pred]
            blunder = " (Bookie Blunder!)" if context.iloc[i]["odds_error"] else ""
            results.append(f"Match: {home} vs. {away}, {date}, {outcome}, {confidence:.1f}%, ~{total_goals} goals{blunder}")
        
        return results