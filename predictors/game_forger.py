import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from utils.db import engine
from scripts.fetch_chaos import get_chaos_data
from scripts.fetch_referee import fetch_referee_bias

class GameForger:
    def __init__(self, weather_api_key, sim_runs=1000):
        self.outcome_model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.01, max_depth=3,  # Slower learning
            min_samples_split=20, min_samples_leaf=10, random_state=42  # More regularization
        )
        self.goals_model = PoissonRegressor(alpha=0.5)  # Increase alpha
        self.weather_api_key = weather_api_key
        self.sim_runs = sim_runs
        self.outcome_features = None
        self.goals_features = None
        self.train_data = None
        self.test_data = None
        self.context = None

    def prepare_data(self, injuries_df=None, limit=None):
        matches = pd.read_sql("SELECT * FROM matches", engine)
        limit = min(limit or len(matches), 50)  # Default to 50
        matches = matches.sort_values("Date", ascending=False).head(limit)
        form = pd.read_sql("SELECT * FROM team_form", engine)
        chaos = get_chaos_data(matches, self.weather_api_key, injuries_df)
        refs = fetch_referee_bias(matches)

        data = matches.merge(form.rename(columns={"team": "HomeTeam"}), on="HomeTeam", how="left", suffixes=("", "_home")) \
                      .merge(form.rename(columns={"team": "AwayTeam"}), on="AwayTeam", how="left", suffixes=("_home", "_away")) \
                      .merge(chaos, on=["HomeTeam", "AwayTeam", "Date"], how="left") \
                      .merge(refs, on=["HomeTeam", "AwayTeam", "Date"], how="left")

        print("Merged Data Columns:", data.columns.tolist())

        X_outcome = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "B365H", "B365A", "B365D",
            "home_x_sentiment", "away_x_sentiment",
            "rain", "wind", "home_win_pct", "yellows_per_game",
            "odds_H"  # New: OddsAPI integration
        ]].fillna(0)
        y_outcome = data["FTR"].map({"H": 1, "A": 2, "D": 0}).fillna(0)

        X_goals = data[["avg_goals_scored_home", "avg_goals_scored_away", "rain", "wind"]].fillna(0)
        y_goals = (data["FTHG"] + data["FTAG"]).fillna(0)

        data["implied_prob_H"] = 1 / data["B365H"]
        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        self.outcome_features = X_outcome.columns
        self.goals_features = X_goals.columns

        X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_outcome, y_outcome, test_size=0.2, random_state=42)
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_goals, y_goals, test_size=0.2, random_state=42)
        test_context = data.iloc[X_test_o.index][["HomeTeam", "AwayTeam", "Date", "odds_error"]]

        self.train_data = (X_train_o, y_train_o, X_train_g, y_train_g)
        self.test_data = (X_test_o, y_test_o, X_test_g, y_test_g)
        self.context = test_context

    def train(self, injuries_df=None, limit=None):
        if self.train_data is None:
            self.prepare_data(injuries_df, limit)
        X_train_o, y_train_o, X_train_g, y_train_g = self.train_data
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
        outcomes = np.random.choice([0, 1, 2], size=self.sim_runs, p=[probs[0], probs[1], probs[2]])
        goals_pred = self.goals_model.predict(goals_df)[0]
        goals = np.random.poisson(max(0, goals_pred), size=self.sim_runs)
        return outcomes, goals

    def predict(self, confidence_threshold=75.0):
        if self.test_data is None:
            raise ValueError("Run train() first!")
        X_test_o, y_test_o, X_test_g, y_test_g = self.test_data
        test_preds = self.outcome_model.predict(X_test_o)
        test_accuracy = (test_preds == y_test_o).mean() * 100
        print(f"Test Model Accuracy (raw): {test_accuracy:.1f}%")

        results = []
        for i in range(len(X_test_o)):
            home, away, date = self.context.iloc[i][["HomeTeam", "AwayTeam", "Date"]]
            outcome_features = X_test_o.iloc[i]
            goals_features = X_test_g.iloc[i]
            outcomes, goals = self.simulate_match(outcome_features, goals_features)
            pred = np.bincount(outcomes).argmax()
            confidence = np.mean(outcomes == pred) * 100
            if confidence >= confidence_threshold:
                total_goals = max(0, round(np.mean(goals)))
                outcome = {1: "Home Win", 2: "Away Win", 0: "Draw"}[pred]
                blunder = " (Bookie Blunder!)" if self.context.iloc[i]["odds_error"] else ""
                results.append(f"Match: {home} vs. {away}, {date}, {outcome}, {confidence:.1f}%, ~{total_goals} goals{blunder}")
        
        print(f"Filtered {len(results)} predictions with confidence >= {confidence_threshold}%")
        return results