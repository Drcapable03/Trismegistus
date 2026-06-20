import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sqlalchemy import inspect

from config.settings import bookie_blend_weight as default_bookie_blend_weight
from evaluation.implied_odds import bookie_favorite, implied_probs_from_odds
from utils.db import engine
from utils.pit_features import compute_pit_form_and_h2h
from scripts.fetch_chaos import get_chaos_data

OUTCOME_MAP = {"H": 1, "A": 2, "D": 0}
OUTCOME_LABELS = {1: "Home Win", 2: "Away Win", 0: "Draw"}
DEFAULT_TEST_FRACTION = 0.2


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    return df.dropna(subset=["Date"])


def _is_completed(df: pd.DataFrame) -> pd.Series:
    return df["FTR"].isin(["H", "D", "A"])


def _date_str_col(df: pd.DataFrame) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        return df["Date"].dt.strftime("%d/%m/%Y")
    return df["Date"].astype(str)


def _add_implied_odds_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    implied = data.apply(
        lambda r: implied_probs_from_odds(
            r.get("B365H", 0), r.get("B365D", 0), r.get("B365A", 0),
        ),
        axis=1,
    )
    data["implied_prob_D"] = [p[0] for p in implied]
    data["implied_prob_H"] = [p[1] for p in implied]
    data["implied_prob_A"] = [p[2] for p in implied]
    return data


def _apply_div_filter(matches: pd.DataFrame, div_filter: str | list[str] | None) -> pd.DataFrame:
    if not div_filter or "Div" not in matches.columns:
        return matches
    codes = [div_filter] if isinstance(div_filter, str) else list(div_filter)
    return matches[matches["Div"].isin(codes)]


def _merge_features(
    matches: pd.DataFrame,
    injuries_df=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    div_filter: str | list[str] | None = None,
    chaos_cache_only: bool = False,
) -> pd.DataFrame:
    matches = _parse_dates(matches.copy())
    matches["Date"] = _date_str_col(matches)

    all_matches = _parse_dates(pd.read_sql("SELECT * FROM matches", engine))
    all_matches = _apply_div_filter(all_matches, div_filter)
    completed_history = all_matches[_is_completed(all_matches)].copy()
    completed_history["Date"] = _date_str_col(completed_history)

    data = compute_pit_form_and_h2h(matches, completed_history)
    chaos = get_chaos_data(
        matches,
        injuries_df=injuries_df,
        use_cache=use_cache,
        refresh=refresh_cache,
        cache_only=chaos_cache_only,
    )
    data = data.merge(chaos, on=["HomeTeam", "AwayTeam", "Date"], how="left")
    return _add_implied_odds_features(data)


def _b365_from_row(row: pd.Series) -> tuple[float, float, float] | None:
    if not {"B365H", "B365D", "B365A"}.issubset(row.index):
        return None
    h, d, a = row["B365H"], row["B365D"], row["B365A"]
    if pd.isna(h) or pd.isna(d) or pd.isna(a) or min(h, d, a) <= 0:
        return None
    return float(h), float(d), float(a)


def _sort_chronologically(data: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(data["Date"], format="%d/%m/%Y", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(data["Date"], errors="coerce")
    return data.assign(_sort_dt=dt).sort_values("_sort_dt").drop(columns=["_sort_dt"])


def _walk_forward_split(
    data: pd.DataFrame,
    X_outcome: pd.DataFrame,
    y_outcome: pd.Series,
    X_goals: pd.DataFrame,
    y_goals: pd.Series,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple:
    """Chronological train/test split — shared indices for outcome and goals."""
    ordered = _sort_chronologically(data)
    n_test = max(1, int(len(ordered) * test_fraction))
    train_idx = ordered.index[:-n_test]
    test_idx = ordered.index[-n_test:]

    return (
        X_outcome.loc[train_idx], X_outcome.loc[test_idx],
        y_outcome.loc[train_idx], y_outcome.loc[test_idx],
        X_goals.loc[train_idx], X_goals.loc[test_idx],
        y_goals.loc[train_idx], y_goals.loc[test_idx],
        train_idx, test_idx,
    )


class GameForger:
    def __init__(self, sim_runs: int = 1000, bookie_blend_weight: float | None = None):
        self.bookie_blend_weight = (
            default_bookie_blend_weight()
            if bookie_blend_weight is None
            else float(bookie_blend_weight)
        )
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
        self.training_metadata: dict = {}

    def _feature_columns(self, data: pd.DataFrame) -> tuple[list[str], list[str]]:
        outcome_cols = [
            "B365H", "B365A", "B365D",
            "implied_prob_H", "implied_prob_D", "implied_prob_A",
            "home_x_sentiment", "away_x_sentiment",
            "rain", "wind", "odds_H", "odds_A", "odds_D",
            "h2h_home_win_pct", "h2h_avg_home_goals", "h2h_avg_away_goals",
            "avg_goals_scored_home", "avg_goals_scored_away",
        ]
        goals_cols = [
            "rain", "wind", "h2h_avg_home_goals", "h2h_avg_away_goals",
            "avg_goals_scored_home", "avg_goals_scored_away",
        ]
        return outcome_cols, goals_cols

    def prepare_training_data(
        self,
        injuries_df=None,
        limit: int | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
        div_filter: str | list[str] | None = None,
        chaos_cache_only: bool = True,
        test_fraction: float = DEFAULT_TEST_FRACTION,
    ):
        matches = _parse_dates(pd.read_sql("SELECT * FROM matches", engine))
        matches = _apply_div_filter(matches, div_filter)
        completed = matches[_is_completed(matches)].sort_values("Date", ascending=False)
        if limit:
            completed = completed.head(limit)
        completed = completed.sort_values("Date", ascending=True).reset_index(drop=True)
        if completed.empty:
            scope = f" for Div={div_filter}" if div_filter else ""
            raise ValueError(f"No completed matches with results found for training{scope}.")

        data = _merge_features(
            completed,
            injuries_df,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            div_filter=div_filter,
            chaos_cache_only=chaos_cache_only,
        ).reset_index(drop=True)
        completed_dates = completed["Date"].copy()
        outcome_cols, goals_cols = self._feature_columns(data)

        X_outcome = data[outcome_cols].fillna(0)
        y_outcome = data["FTR"].map(OUTCOME_MAP)
        X_goals = data[goals_cols].fillna(0)
        y_goals = data["FTHG"] + data["FTAG"]

        data["odds_error"] = (data["FTR"] != "H") & (data["implied_prob_H"] > 0.7)

        self.outcome_features = list(X_outcome.columns)
        self.goals_features = list(X_goals.columns)

        (
            X_train_o, X_test_o, y_train_o, y_test_o,
            X_train_g, X_test_g, y_train_g, y_test_g,
            train_idx, test_idx,
        ) = _walk_forward_split(data, X_outcome, y_outcome, X_goals, y_goals, test_fraction)

        context_cols = ["HomeTeam", "AwayTeam", "Date", "odds_error", "FTR"]
        for col in ("B365H", "B365D", "B365A"):
            if col in data.columns:
                context_cols.append(col)
        test_context = data.loc[test_idx, context_cols].copy()
        test_context["Date"] = _date_str_col(test_context)

        self.train_data = (X_train_o, y_train_o, X_train_g, y_train_g)
        self.test_data = (X_test_o, y_test_o, X_test_g, y_test_g)
        self.context = test_context.reset_index(drop=True)

        train_dates = completed_dates.loc[train_idx]
        test_dates = completed_dates.loc[test_idx]
        self.training_metadata = {
            "div_filter": div_filter,
            "split_method": "walk_forward",
            "test_fraction": test_fraction,
            "train_matches": len(train_idx),
            "test_matches": len(test_idx),
            "train_date_from": str(train_dates.min().date()) if len(train_dates) else None,
            "train_date_to": str(train_dates.max().date()) if len(train_dates) else None,
            "test_date_from": str(test_dates.min().date()) if len(test_dates) else None,
            "test_date_to": str(test_dates.max().date()) if len(test_dates) else None,
            "chaos_cache_only": chaos_cache_only,
            "bookie_blend_weight": self.bookie_blend_weight,
        }

        scope = f" ({div_filter})" if div_filter else ""
        print(
            f"Prepared training data: {len(completed)} matches{scope}, "
            f"walk-forward split {len(train_idx)} train / {len(test_idx)} test"
        )
        print(
            f"  Train: {self.training_metadata['train_date_from']} → "
            f"{self.training_metadata['train_date_to']}"
        )
        print(
            f"  Test:  {self.training_metadata['test_date_from']} → "
            f"{self.training_metadata['test_date_to']}"
        )

    def prepare_prediction_data(
        self,
        future_matches: pd.DataFrame,
        injuries_df=None,
        use_cache: bool = True,
        refresh_cache: bool = False,
        div_filter: str | list[str] | None = None,
        chaos_cache_only: bool = False,
    ):
        future_matches = _parse_dates(future_matches)
        if future_matches.empty:
            self.prediction_data = None
            return
        data = _merge_features(
            future_matches,
            injuries_df,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            div_filter=div_filter,
            chaos_cache_only=chaos_cache_only,
        )
        outcome_cols, goals_cols = self._feature_columns(data)
        context_cols = ["HomeTeam", "AwayTeam", "Date"]
        for col in ("B365H", "B365D", "B365A"):
            if col in data.columns:
                context_cols.append(col)
        self.prediction_data = {
            "X_outcome": data[outcome_cols].fillna(0),
            "X_goals": data[goals_cols].fillna(0),
            "context": data[context_cols].copy(),
        }
        print(f"Prepared {len(future_matches)} future matches for prediction")

    def train(
        self,
        injuries_df=None,
        limit: int | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
        div_filter: str | list[str] | None = None,
        chaos_cache_only: bool = True,
    ):
        if self.train_data is None:
            self.prepare_training_data(
                injuries_df,
                limit,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                div_filter=div_filter,
                chaos_cache_only=chaos_cache_only,
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
        model_acc = self.holdout_blend_accuracy(blend_weight=0.0)
        blend_acc = self.holdout_blend_accuracy()
        print(f"Holdout accuracy (model only, deterministic): {model_acc:.1f}%")
        print(
            f"Holdout accuracy (blend w={self.bookie_blend_weight:.2f}, deterministic): "
            f"{blend_acc:.1f}%"
        )
        return blend_acc

    def _blend_with_bookie(
        self,
        model_probs: np.ndarray,
        b365: tuple[float, float, float] | None,
        blend_weight: float | None = None,
    ) -> np.ndarray:
        if b365 is None:
            return model_probs
        h, d, a = b365
        if min(h, d, a) <= 0:
            return model_probs
        p_d, p_h, p_a = implied_probs_from_odds(h, d, a)
        bookie = np.array([p_d, p_h, p_a], dtype=float)
        w = self.bookie_blend_weight if blend_weight is None else float(blend_weight)
        blended = (1 - w) * model_probs + w * bookie
        total = blended.sum()
        return blended / total if total > 0 else model_probs

    def predict_outcome_probs(
        self,
        outcome_features,
        b365: tuple[float, float, float] | None = None,
        blend_weight: float | None = None,
    ) -> np.ndarray:
        outcome_df = pd.DataFrame([outcome_features], columns=self.outcome_features)
        model_probs = self.outcome_model.predict_proba(outcome_df)[0]
        return self._blend_with_bookie(model_probs, b365, blend_weight=blend_weight)

    def holdout_blend_accuracy(self, blend_weight: float | None = None) -> float:
        """Deterministic holdout accuracy at a given bookie blend weight."""
        if self.test_data is None or self.context is None:
            raise ValueError("Run train() first")
        X_test_o, y_test_o, _, _ = self.test_data
        correct = 0
        for i in range(len(X_test_o)):
            row = self.context.iloc[i]
            b365 = _b365_from_row(row)
            probs = self.predict_outcome_probs(
                X_test_o.iloc[i], b365=b365, blend_weight=blend_weight,
            )
            pred = int(np.argmax(probs))
            if pred == int(y_test_o.iloc[i]):
                correct += 1
        return correct / len(X_test_o) * 100 if len(X_test_o) else 0.0

    def simulate_match(
        self,
        outcome_features,
        goals_features,
        b365: tuple[float, float, float] | None = None,
        blend_weight: float | None = None,
    ):
        goals_df = pd.DataFrame([goals_features], columns=self.goals_features)
        probs = self.predict_outcome_probs(outcome_features, b365=b365, blend_weight=blend_weight)
        outcomes = np.random.choice([0, 1, 2], size=self.sim_runs, p=probs)
        goals_pred = self.goals_model.predict(goals_df)[0]
        goals = np.random.poisson(max(0, goals_pred), size=self.sim_runs)
        return outcomes, goals, probs

    def predict(self, confidence_threshold: float = 75.0, use_simulation: bool = True) -> list[dict]:
        if self.prediction_data is None:
            raise ValueError("Run prepare_prediction_data() with future matches first")
        results = []
        X_outcome = self.prediction_data["X_outcome"]
        X_goals = self.prediction_data["X_goals"]
        context = self.prediction_data["context"]

        for i in range(len(X_outcome)):
            row = context.iloc[i]
            b365 = _b365_from_row(row)
            goals_df = pd.DataFrame([X_goals.iloc[i]], columns=self.goals_features)
            expected_goals = max(0.0, float(self.goals_model.predict(goals_df)[0]))
            probs = self.predict_outcome_probs(X_outcome.iloc[i], b365=b365)

            if use_simulation:
                outcomes, goals, probs = self.simulate_match(
                    X_outcome.iloc[i], X_goals.iloc[i], b365=b365,
                )
                pred = int(np.bincount(outcomes).argmax())
                confidence = float(np.mean(outcomes == pred) * 100)
                total_goals = max(0, round(float(np.mean(goals))))
            else:
                pred = int(np.argmax(probs))
                confidence = float(np.max(probs) * 100)
                total_goals = max(0, round(expected_goals))

            if confidence >= confidence_threshold:
                bookie_pick = None
                if b365 is not None:
                    bookie_code = bookie_favorite(*b365)
                    bookie_pick = OUTCOME_LABELS[bookie_code]
                results.append({
                    "home": row["HomeTeam"],
                    "away": row["AwayTeam"],
                    "date": row["Date"],
                    "outcome": OUTCOME_LABELS[pred],
                    "outcome_code": pred,
                    "confidence": confidence,
                    "expected_goals": expected_goals,
                    "total_goals": total_goals,
                    "probs": {"H": float(probs[1]), "D": float(probs[0]), "A": float(probs[2])},
                    "bookie_pick": bookie_pick,
                })
        print(f"Filtered {len(results)} predictions with confidence >= {confidence_threshold}%")
        return results

    def backtest_on_holdout(self, confidence_threshold: float = 0.0) -> list[dict]:
        """Deterministic predictions on chronological holdout test set."""
        if self.test_data is None or self.context is None:
            raise ValueError("Run train() first")
        X_test_o, y_test_o, X_test_g, _ = self.test_data
        results = []
        for i in range(len(X_test_o)):
            row = self.context.iloc[i]
            b365 = _b365_from_row(row)
            probs = self.predict_outcome_probs(X_test_o.iloc[i], b365=b365)
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs) * 100)
            if confidence < confidence_threshold:
                continue
            goals_df = pd.DataFrame([X_test_g.iloc[i]], columns=self.goals_features)
            expected_goals = max(0.0, float(self.goals_model.predict(goals_df)[0]))
            actual = OUTCOME_MAP.get(row["FTR"], -1)
            results.append({
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "date": row["Date"],
                "outcome": OUTCOME_LABELS[pred],
                "outcome_code": pred,
                "actual_code": actual,
                "confidence": confidence,
                "expected_goals": expected_goals,
                "total_goals": max(0, round(expected_goals)),
                "odds_error": bool(row.get("odds_error", False)),
                "probs": {"H": float(probs[1]), "D": float(probs[0]), "A": float(probs[2])},
            })
        return results