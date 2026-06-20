import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sqlalchemy import inspect

from utils.db import engine, read_matches


class BlunderSniffer:
    def __init__(self):
        self.outcome_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.goals_model = LinearRegression()

    def prepare_data(self):
        matches = read_matches()
        matches = matches[matches["FTR"].isin(["H", "D", "A"])]
        form = (
            pd.read_sql("SELECT * FROM team_form", engine)
            if inspect(engine).has_table("team_form")
            else pd.DataFrame()
        )

        home_form = form.rename(columns={"team": "HomeTeam"})
        away_form = form.rename(columns={"team": "AwayTeam"})
        data = matches.merge(home_form, on="HomeTeam", how="left").merge(
            away_form, on="AwayTeam", how="left", suffixes=("_home", "_away"),
        )

        X_outcome = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            "B365H", "B365A", "B365D",
        ]].fillna(0)
        y_outcome = data["FTR"].map({"H": 1, "A": 2, "D": 0})
        X_goals = data[[
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
        ]].fillna(0)
        y_goals = data["FTHG"] + data["FTAG"]

        data["implied_prob_H"] = 1 / data["B365H"].replace(0, float("nan"))
        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        return X_outcome, y_outcome, X_goals, y_goals, data[
            ["HomeTeam", "AwayTeam", "Date", "FTR", "odds_error"]
        ]

    def train(self):
        X_outcome, y_outcome, X_goals, y_goals, _ = self.prepare_data()
        self.outcome_model.fit(X_outcome, y_outcome)
        self.goals_model.fit(X_goals, y_goals)

    def find_blunders(self, limit: int = 10) -> list[str]:
        _, _, _, _, context = self.prepare_data()
        blunders = context[context["odds_error"]].head(limit)
        results = []
        for _, row in blunders.iterrows():
            results.append(
                f"{row['HomeTeam']} vs {row['AwayTeam']} ({row['Date']}): "
                f"bookie favored home (>70%), actual={row['FTR']}"
            )
        return results