import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import inspect

from utils.db import engine
from scripts.fetch_chaos import get_chaos_data
from scripts.fetch_referee import fetch_referee_bias

OUTCOME_MAP = {"H": 1, "A": 2, "D": 0}
OUTCOME_LABELS = {1: "Home Win", 2: "Away Win", 0: "Draw"}


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    return df.dropna(subset=["Date"])


def _is_completed(df: pd.DataFrame) -> pd.Series:
    return df["FTR"].isin(["H", "D", "A"])


def _merge_features(
    matches: pd.DataFrame,
    injuries_df=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    matches = matches.copy()
    if pd.api.types.is_datetime64_any_dtype(matches["Date"]):
        matches["Date"] = matches["Date"].dt.strftime("%d/%m/%Y")

    form = (
        pd.read_sql("SELECT * FROM team_form", engine)
        if inspect(engine).has_table("team_form")
        else pd.DataFrame()
    )
    chaos = get_chaos_data(
        matches, injuries_df=injuries_df, use_cache=use_cache, refresh=refresh_cache,
    )
    refs = fetch_referee_bias(matches)
    all_matches = pd.read_sql("SELECT * FROM matches", engine)
    all_matches = _parse_dates(all_matches)
    completed = all_matches[_is_completed(all_matches)]
    h2h = completed.groupby(["HomeTeam", "AwayTeam"]).agg(
        FTR=("FTR", lambda x: (x == "H").mean()),
        FTHG=("FTHG", "mean"),
        FTAG=("FTAG", "mean"),
    ).reset_index().rename(columns={
        "FTR": "h2h_home_win_pct",
        "FTHG": "h2h_avg_home_goals",
        "FTAG": "h2h_avg_away_goals",
    })

    data = matches.merge(
        form.rename(columns={"team": "HomeTeam"}), on="HomeTeam", how="left", suffixes=("", "_home")
    ).merge(
        form.rename(columns={"team": "AwayTeam"}), on="AwayTeam", how="left", suffixes=("_home", "_away")
    ).merge(chaos, on=["HomeTeam", "AwayTeam", "Date"], how="left")
    data = data.merge(refs, on=["HomeTeam", "AwayTeam", "Date"], how="left")
    data = data.merge(h2h, on=["HomeTeam", "AwayTeam"], how="left")
    return data


class GameForger:
    def __init__(self, sim_runs: int = 1000):
        self.outcome_model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.01, max_depth=3,
            min_samples_split=20, min_samples_leaf=10, random_state=42,
        )
        self.goals_model = PoissonRegressor(alpha=0.5)
        self.sim_runs = sim_runs
        self.outcome_features = None
        self.goals_features = None
        self.train_data = None
        self.test_data = None
        self.context = None
        self.prediction_data = None

    def _feature_columns(self, data: pd.DataFrame) -> tuple[list[str], list[str]]:
        outcome_cols = [
            "B365H", "B365A", "B365D", "home_x_sentiment", "away_x_sentiment",
            "rain", "wind", "home_win_pct", "yellows_per_game", "odds_H", "odds_A", "odds_D",
            "h2h_home_win_pct", "h2h_avg_home_goals", "h2h_avg_away_goals",
        ]
        if "avg_goals_scored_home" in data.columns:
            outcome_cols.extend(["avg_goals_scored_home", "avg_goals_scored_away"])
        goals_cols = ["rain", "wind", "h2h_avg_home_goals", "h2h_avg_away_goals"]
        if "avg_goals_scored_home" in data.columns:
            goals_cols.extend(["avg_goals_scored_home", "avg_goals_scored_away"])
        return outcome_cols, goals_cols

    def prepare_training_data(
        self,
        injuries_df=None,
        limit: int | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ):
        matches = _parse_dates(pd.read_sql("SELECT * FROM matches", engine))
        completed = matches[_is_completed(matches)].sort_values("Date", ascending=False)
        if limit:
            completed = completed.head(limit)
        if completed.empty:
            raise ValueError("No completed matches with results found for training.")

        data = _merge_features(
            completed, injuries_df, use_cache=use_cache, refresh_cache=refresh_cache,
        )
        outcome_cols, goals_cols = self._feature_columns(data)

        X_outcome = data[outcome_cols].fillna(0)
        y_outcome = data["FTR"].map(OUTCOME_MAP)
        X_goals = data[goals_cols].fillna(0)
        y_goals = data["FTHG"] + data["FTAG"]

        data["implied_prob_H"] = 1 / data["B365H"].replace(0, np.nan)
        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        self.outcome_features = list(X_outcome.columns)
        self.goals_features = list(X_goals.columns)

        stratify = y_outcome if y_outcome.value_counts().min() >= 2 else None
        X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
            X_outcome, y_outcome, test_size=0.2, random_state=42, stratify=stratify,
        )
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
            X_goals, y_goals, test_size=0.2, random_state=42,
        )
        test_context = data.loc[X_test_o.index, ["HomeTeam", "AwayTeam", "Date", "odds_error", "FTR"]]

        self.train_data = (X_train_o, y_train_o, X_train_g, y_train_g)
        self.test_data = (X_test_o, y_test_o, X_test_g, y_test_g)
        self.context = test_context
        print(f"Prepared training data: {len(completed)} completed matches")

    def prepare_prediction_data(
        self,
        future_matches: pd.DataFrame,
        injuries_df=None,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ):
        future_matches = _parse_dates(future_matches)
        if future_matches.empty:
            self.prediction_data = None
            return
        data = _merge_features(
            future_matches, injuries_df, use_cache=use_cache, refresh_cache=refresh_cache,
        )
        outcome_cols, goals_cols = self._feature_columns(data)
        self.prediction_data = {
            "X_outcome": data[outcome_cols].fillna(0),
            "X_goals": data[goals_cols].fillna(0),
            "context": data[["HomeTeam", "AwayTeam", "Date"]].copy(),
        }
        print(f"Prepared {len(future_matches)} future matches for prediction")

    def train(
        self,
        injuries_df=None,
        limit: int | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ):
        if self.train_data is None:
            self.prepare_training_data(
                injuries_df, limit, use_cache=use_cache, refresh_cache=refresh_cache,
            )
        X_train_o, y_train_o, X_train_g, y_train_g = self.train_data
        self.outcome_model.fit(X_train_o, y_train_o)
        self.goals_model.fit(X_train_g, y_train_g)
        raw_preds = self.outcome_model.predict(X_train_o)
        raw_accuracy = (raw_preds == y_train_o).mean() * 100
        print(f"Raw Model Accuracy (train): {raw_accuracy:.1f}%")

    def evaluate_holdout(self) -> float:
        if self.test_data is None:
            raise ValueError("Run prepare_training_data() first")
        X_test_o, y_test_o, _, _ = self.test_data
        test_preds = self.outcome_model.predict(X_test_o)
        accuracy = (test_preds == y_test_o).mean() * 100
        print(f"Holdout Model Accuracy: {accuracy:.1f}%")
        return accuracy

    def simulate_match(self, outcome_features, goals_features):
        outcome_df = pd.DataFrame([outcome_features], columns=self.outcome_features)
        goals_df = pd.DataFrame([goals_features], columns=self.goals_features)
        probs = self.outcome_model.predict_proba(outcome_df)[0]
        outcomes = np.random.choice([0, 1, 2], size=self.sim_runs, p=probs)
        goals_pred = self.goals_model.predict(goals_df)[0]
        goals = np.random.poisson(max(0, goals_pred), size=self.sim_runs)
        return outcomes, goals, probs

    def predict(self, confidence_threshold: float = 75.0) -> list[dict]:
        if self.prediction_data is None:
            raise ValueError("Run prepare_prediction_data() with future matches first")
        results = []
        X_outcome = self.prediction_data["X_outcome"]
        X_goals = self.prediction_data["X_goals"]
        context = self.prediction_data["context"]

        for i in range(len(X_outcome)):
            home = context.iloc[i]["HomeTeam"]
            away = context.iloc[i]["AwayTeam"]
            date = context.iloc[i]["Date"]
            outcomes, goals, probs = self.simulate_match(X_outcome.iloc[i], X_goals.iloc[i])
            pred = int(np.bincount(outcomes).argmax())
            confidence = float(np.mean(outcomes == pred) * 100)
            if confidence >= confidence_threshold:
                results.append({
                    "home": home,
                    "away": away,
                    "date": date,
                    "outcome": OUTCOME_LABELS[pred],
                    "outcome_code": pred,
                    "confidence": confidence,
                    "total_goals": max(0, round(float(np.mean(goals)))),
                    "probs": {"H": float(probs[1]), "D": float(probs[0]), "A": float(probs[2])},
                })
        print(f"Filtered {len(results)} predictions with confidence >= {confidence_threshold}%")
        return results

    def backtest_on_holdout(self, confidence_threshold: float = 0.0) -> list[dict]:
        """Predict on held-out test split (matches with known results)."""
        if self.test_data is None or self.context is None:
            raise ValueError("Run train() first")
        X_test_o, y_test_o, X_test_g, _ = self.test_data
        results = []
        for i in range(len(X_test_o)):
            outcomes, goals, probs = self.simulate_match(X_test_o.iloc[i], X_test_g.iloc[i])
            pred = int(np.bincount(outcomes).argmax())
            confidence = float(np.mean(outcomes == pred) * 100)
            if confidence < confidence_threshold:
                continue
            row = self.context.iloc[i]
            actual = OUTCOME_MAP.get(row["FTR"], -1)
            results.append({
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "date": row["Date"],
                "outcome": OUTCOME_LABELS[pred],
                "outcome_code": pred,
                "actual_code": actual,
                "confidence": confidence,
                "total_goals": max(0, round(float(np.mean(goals)))),
                "odds_error": bool(row.get("odds_error", False)),
            })
        return results