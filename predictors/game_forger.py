import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils.db import engine
from scripts.fetch_chaos import get_chaos_data
from scripts.fetch_referee import fetch_referee_bias

class GameForger:
    def __init__(self, weather_api_key, sim_runs=100):
        self.outcome_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        self.goals_model = LinearRegression()
        self.weather_api_key = weather_api_key
        self.sim_runs = sim_runs
        self.outcome_features = None
        self.goals_features = None

    def prepare_data(self, injuries_df=None):
        matches = pd.read_sql("SELECT * FROM matches", engine)
        form = pd.read_sql("SELECT * FROM team_form", engine)
        chaos = get_chaos_data(matches, self.weather_api_key, injuries_df)
        refs = fetch_referee_bias(matches)

        data = matches.merge(form.rename(columns={"team": "HomeTeam"}), on="HomeTeam") \
                      .merge(form.rename(columns={"team": "AwayTeam"}), on="AwayTeam", suffixes=("_home", "_away")) \
                      .merge(chaos, on=["HomeTeam", "AwayTeam", "Date"]) \
                      .merge(refs, on=["HomeTeam", "AwayTeam", "Date"])

        X_outcome = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            "B365H", "B365A", "B365D",
            "rain", "wind", "home_rss_sentiment", "away_rss_sentiment",
            "home_x_sentiment", "away_x_sentiment",
            "home_injuries", "away_injuries",
            "home_win_pct", "yellows_per_game"
        ]].fillna(0)
        y_outcome = data["FTR"].map({"H": 1, "A": 2, "D": 0}).fillna(0)

        X_goals = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            "rain", "wind", "home_injuries", "away_injuries",
            "yellows_per_game"
        ]].fillna(0)
        y_goals = (data["FTHG"] + data["FTAG"]).fillna(0)

        data["implied_prob_H"] = 1 / data["B365H"]
        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        self.outcome_features = X_outcome.columns
        self.goals_features = X_goals.columns

        # Split: 80% train, 20% test
        X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_outcome, y_outcome, test_size=0.2, random_state=42)
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_goals, y_goals, test_size=0.2, random_state=42)
        test_context = data.iloc[X_test_o.index][["HomeTeam", "AwayTeam", "Date", "odds_error"]]

        return (X_train_o, y_train_o, X_train_g, y_train_g), (X_test_o, y_test_o, X_test_g, y_test_g), test_context

    def train(self, injuries_df=None):
        (X_train_o, y_train_o, X_train_g, y_train_g), (_, _, _, _), _ = self.prepare_data(injuries_df)
        self.outcome_model.fit(X_train_o, y_train_o)
        self.goals_model.fit(X_train_g, y_train_g)
        raw_preds = self.outcome_model.predict(X_train_o)
        raw_accuracy = (raw_preds == y_train_o).mean() * 100
        print(f"Raw Model Accuracy (train): {raw_accuracy:.1f}%")

    def simulate_match(self, outcome_features, goals_features):
        outcome_df = pd.DataFrame([outcome_features], columns=self.outcome_features)
        goals_df = pd.DataFrame([goals_features], columns=self.goals_features)
        
        probs = self.outcome_model.predict_proba(outcome_df)[0]
        print(f"Prediction Probabilities: H={probs[1]:.2f}, A={probs[2]:.2f}, D={probs[0]:.2f}")
        # Fix sims: Match probs order
        outcomes = np.random.choice([0, 1, 2], size=self.sim_runs, p=[probs[0], probs[1], probs[2]])  # D, H, A
        goals_pred = self.goals_model.predict(goals_df)[0]
        goals = np.random.poisson(max(0, goals_pred), size=self.sim_runs)
        return outcomes, goals

    def predict(self, limit=5, injuries_df=None):
        (_, _, _, _), (X_test_o, y_test_o, X_test_g, y_test_g), context = self.prepare_data(injuries_df)
        results = []
        test_preds = self.outcome_model.predict(X_test_o)
        test_accuracy = (test_preds == y_test_o).mean() * 100
        print(f"Test Model Accuracy (raw): {test_accuracy:.1f}%")
        
        for i in range(min(limit, len(X_test_o))):
            home, away, date = context.iloc[i][["HomeTeam", "AwayTeam", "Date"]]
            outcome_features = X_test_o.iloc[i]
            goals_features = X_test_g.iloc[i]
            outcomes, goals = self.simulate_match(outcome_features, goals_features)
            
            pred = np.bincount(outcomes).argmax()
            confidence = np.mean(outcomes == pred) * 100
            total_goals = max(0, round(np.mean(goals)))
            outcome = {1: "Home Win", 2: "Away Win", 0: "Draw"}[pred]
            blunder = " (Bookie Blunder!)" if context.iloc[i]["odds_error"] else ""
            results.append(f"Match: {home} vs. {away}, {date}, {outcome}, {confidence:.1f}%, ~{total_goals} goals{blunder}")
        
        return results