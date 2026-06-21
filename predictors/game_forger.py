import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sqlalchemy import inspect

from config.settings import bookie_blend_weight as default_bookie_blend_weight
from config.settings import edge_margin_min as default_edge_margin
from config.settings import per_league_blend_weights
from evaluation.metrics import print_metrics_summary, summarize_predictions
from predictors.calibration import OutcomeCalibrator
from evaluation.edge import (
    best_outcome_and_edge,
    implied_probs_array,
    passes_edge_filter,
    selective_accuracy,
)
from evaluation.implied_odds import bookie_favorite, implied_probs_from_odds
from utils.db import engine
from utils.pit_features import compute_pit_form_and_h2h
from scripts.fetch_chaos import get_chaos_data

OUTCOME_MAP = {"H": 1, "A": 2, "D": 0}
OUTCOME_LABELS = {1: "Home Win", 2: "Away Win", 0: "Draw"}
DEFAULT_TEST_FRACTION = 0.2
CALIBRATION_FRACTION = 0.15
SHOT_FEATURE_COLS = [
    "avg_shots_on_target_home", "avg_shots_on_target_away",
    "avg_shots_home", "avg_shots_away",
]
XG_FEATURE_COLS = [
    "avg_xg_for_home", "avg_xg_against_home",
    "avg_xg_for_away", "avg_xg_against_away",
]
ELO_FEATURE_COLS = ["elo_home", "elo_away", "elo_diff"]


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
    """Bookie implied probs — used at inference only, not model features."""
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


def _add_div_features(data: pd.DataFrame) -> pd.DataFrame:
    if "Div" not in data.columns:
        return data
    dummies = pd.get_dummies(data["Div"], prefix="div", dtype=float)
    return pd.concat([data, dummies], axis=1)


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
    data = _add_div_features(data)
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
    """Residual edge model: learns from form/chaos/league; bookie used only at inference."""

    def __init__(
        self,
        sim_runs: int = 1000,
        bookie_blend_weight: float | None = None,
        edge_margin: float | None = None,
    ):
        self.bookie_blend_weight = (
            default_bookie_blend_weight()
            if bookie_blend_weight is None
            else float(bookie_blend_weight)
        )
        self.per_league_blend = per_league_blend_weights()
        self.edge_margin = (
            default_edge_margin() if edge_margin is None else float(edge_margin)
        )
        self.calibrator = OutcomeCalibrator()
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

    def _div_columns(self, data: pd.DataFrame) -> list[str]:
        return sorted(c for c in data.columns if c.startswith("div_"))

    def _model_feature_columns(self, data: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Residual features — no raw bookie odds in the learner."""
        outcome_cols = [
            "home_x_sentiment", "away_x_sentiment",
            "home_injuries", "away_injuries",
            "rain", "wind",
            "h2h_home_win_pct", "h2h_avg_home_goals", "h2h_avg_away_goals",
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            *self._div_columns(data),
        ]
        goals_cols = [
            "rain", "wind",
            "h2h_avg_home_goals", "h2h_avg_away_goals",
            "avg_goals_scored_home", "avg_goals_scored_away",
            "avg_goals_conceded_home", "avg_goals_conceded_away",
            *self._div_columns(data),
        ]
        for col in (*SHOT_FEATURE_COLS, *XG_FEATURE_COLS, *ELO_FEATURE_COLS):
            if col in data.columns:
                outcome_cols.append(col)
                if col not in goals_cols:
                    goals_cols.append(col)
        return outcome_cols, goals_cols

    def _blend_weight_for_div(self, div: str | None, blend_weight: float | None) -> float:
        if blend_weight is not None:
            return float(blend_weight)
        if div and div in self.per_league_blend:
            return float(self.per_league_blend[div])
        return self.bookie_blend_weight

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
        outcome_cols, goals_cols = self._model_feature_columns(data)

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
        for col in ("B365H", "B365D", "B365A", "Div"):
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
            "model_type": "residual_edge",
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
            "per_league_blend": self.per_league_blend,
            "edge_margin_min": self.edge_margin,
            "calibrated": self.calibrator.is_fitted,
            "outcome_features": self.outcome_features,
        }

        scope = f" ({div_filter})" if div_filter else ""
        print(
            f"Prepared residual training data: {len(completed)} matches{scope}, "
            f"walk-forward {len(train_idx)} train / {len(test_idx)} test"
        )
        print(f"  Model features ({len(outcome_cols)}): no bookie odds in learner")
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
        outcome_cols, goals_cols = self._model_feature_columns(data)
        context_cols = ["HomeTeam", "AwayTeam", "Date"]
        for col in ("B365H", "B365D", "B365A", "Div"):
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
        n_cal = max(10, int(len(X_train_o) * CALIBRATION_FRACTION))
        X_fit, y_fit = X_train_o.iloc[:-n_cal], y_train_o.iloc[:-n_cal]
        X_cal, y_cal = X_train_o.iloc[-n_cal:], y_train_o.iloc[-n_cal:]

        self.outcome_model.fit(X_fit, y_fit)
        self.calibrator.fit(self.outcome_model, X_cal, y_cal)
        self.goals_model.fit(X_train_g, y_train_g)

        raw_preds = self.outcome_model.predict(X_fit)
        raw_accuracy = (raw_preds == y_fit).mean() * 100
        cal_status = "on" if self.calibrator.is_fitted else "skipped"
        print(f"Residual model train accuracy: {raw_accuracy:.1f}% (calibration {cal_status})")

    def _raw_model_probs(self, outcome_features) -> np.ndarray:
        outcome_df = pd.DataFrame([outcome_features], columns=self.outcome_features)
        return self.calibrator.predict_proba(self.outcome_model, outcome_df)

    def _blend_with_bookie(
        self,
        model_probs: np.ndarray,
        b365: tuple[float, float, float] | None,
        blend_weight: float | None = None,
        div: str | None = None,
    ) -> np.ndarray:
        if b365 is None:
            return model_probs
        implied = implied_probs_array(b365)
        if implied is None:
            return model_probs
        w = self._blend_weight_for_div(div, blend_weight)
        blended = (1 - w) * model_probs + w * implied
        total = blended.sum()
        return blended / total if total > 0 else model_probs

    def predict_outcome_probs(
        self,
        outcome_features,
        b365: tuple[float, float, float] | None = None,
        blend_weight: float | None = None,
        div: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (final_probs, model_probs, implied_probs)."""
        model_probs = self._raw_model_probs(outcome_features)
        implied = implied_probs_array(b365)
        final = self._blend_with_bookie(
            model_probs, b365, blend_weight=blend_weight, div=div,
        )
        return final, model_probs, implied

    def _evaluate_row(
        self,
        outcome_features,
        b365: tuple[float, float, float] | None,
        blend_weight: float | None,
        edge_margin: float,
        require_edge: bool,
        div: str | None = None,
    ) -> tuple[int, float, float, np.ndarray] | None:
        final, _, implied = self.predict_outcome_probs(
            outcome_features, b365=b365, blend_weight=blend_weight, div=div,
        )
        if implied is None:
            pred = int(np.argmax(final))
            edge = 0.0
        else:
            pred, edge = best_outcome_and_edge(final, implied)
        if require_edge and implied is not None and not passes_edge_filter(edge, edge_margin):
            return None
        confidence = float(np.max(final) * 100)
        return pred, confidence, edge, final

    def holdout_blend_accuracy(
        self,
        blend_weight: float | None = None,
        edge_margin: float | None = None,
        require_edge: bool = False,
    ) -> float:
        if self.test_data is None or self.context is None:
            raise ValueError("Run train() first")
        margin = self.edge_margin if edge_margin is None else edge_margin
        X_test_o, y_test_o, _, _ = self.test_data
        correct = 0
        total = 0
        for i in range(len(X_test_o)):
            row = self.context.iloc[i]
            evaluated = self._evaluate_row(
                X_test_o.iloc[i],
                _b365_from_row(row),
                blend_weight,
                margin,
                require_edge,
                div=row.get("Div"),
            )
            if evaluated is None:
                continue
            pred, _, _, _ = evaluated
            total += 1
            if pred == int(y_test_o.iloc[i]):
                correct += 1
        return correct / total * 100 if total else 0.0

    def evaluate_holdout(self) -> float:
        if self.test_data is None:
            raise ValueError("Run prepare_training_data() first")
        model_acc = self.holdout_blend_accuracy(blend_weight=0.0)
        blend_acc = self.holdout_blend_accuracy()
        selective_acc = self.holdout_blend_accuracy(require_edge=True)
        n_selective = len(self.backtest_on_holdout(edge_margin=self.edge_margin))
        n_all = len(self.test_data[1])

        print(f"Holdout — residual model only (w=0): {model_acc:.1f}%")
        print(
            f"Holdout — blend w={self.bookie_blend_weight:.2f} (all picks): {blend_acc:.1f}%"
        )
        print(
            f"Holdout — selective edge ≥{self.edge_margin:.0%}: "
            f"{selective_acc:.1f}% on {n_selective}/{n_all} matches"
        )
        all_preds = self.backtest_on_holdout(require_edge=False, edge_margin=0.0)
        sel_preds = self.backtest_on_holdout(require_edge=True)
        print_metrics_summary(summarize_predictions(all_preds, "Value metrics (all picks)"))
        print_metrics_summary(summarize_predictions(sel_preds, "Value metrics (selective)"))
        return blend_acc

    def simulate_match(
        self,
        outcome_features,
        goals_features,
        b365: tuple[float, float, float] | None = None,
        blend_weight: float | None = None,
    ):
        goals_df = pd.DataFrame([goals_features], columns=self.goals_features)
        final, _, _ = self.predict_outcome_probs(
            outcome_features, b365=b365, blend_weight=blend_weight,
        )
        outcomes = np.random.choice([0, 1, 2], size=self.sim_runs, p=final)
        goals_pred = self.goals_model.predict(goals_df)[0]
        goals = np.random.poisson(max(0, goals_pred), size=self.sim_runs)
        return outcomes, goals, final

    def predict(
        self,
        confidence_threshold: float = 75.0,
        edge_margin: float | None = None,
        use_simulation: bool = False,
    ) -> list[dict]:
        if self.prediction_data is None:
            raise ValueError("Run prepare_prediction_data() with future matches first")
        margin = self.edge_margin if edge_margin is None else edge_margin
        results = []
        X_outcome = self.prediction_data["X_outcome"]
        X_goals = self.prediction_data["X_goals"]
        context = self.prediction_data["context"]

        for i in range(len(X_outcome)):
            row = context.iloc[i]
            b365 = _b365_from_row(row)
            goals_df = pd.DataFrame([X_goals.iloc[i]], columns=self.goals_features)
            expected_goals = max(0.0, float(self.goals_model.predict(goals_df)[0]))

            evaluated = self._evaluate_row(
                X_outcome.iloc[i], b365, None, margin, require_edge=True,
                div=row.get("Div"),
            )
            if evaluated is None:
                continue
            pred, confidence, edge, final = evaluated

            if confidence < confidence_threshold:
                continue

            if use_simulation:
                outcomes, goals, final = self.simulate_match(
                    X_outcome.iloc[i], X_goals.iloc[i], b365=b365,
                )
                pred = int(np.bincount(outcomes).argmax())
                confidence = float(np.mean(outcomes == pred) * 100)
                total_goals = max(0, round(float(np.mean(goals))))
            else:
                total_goals = max(0, round(expected_goals))

            bookie_pick = None
            if b365 is not None:
                bookie_pick = OUTCOME_LABELS[bookie_favorite(*b365)]

            results.append({
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "date": row["Date"],
                "outcome": OUTCOME_LABELS[pred],
                "outcome_code": pred,
                "confidence": confidence,
                "edge": edge,
                "expected_goals": expected_goals,
                "total_goals": total_goals,
                "probs": {"H": float(final[1]), "D": float(final[0]), "A": float(final[2])},
                "bookie_pick": bookie_pick,
            })
        print(
            f"Edge-filtered picks: {len(results)} "
            f"(margin ≥{margin:.0%}, confidence ≥{confidence_threshold}%)"
        )
        return results

    def backtest_on_holdout(
        self,
        confidence_threshold: float = 0.0,
        edge_margin: float | None = None,
        require_edge: bool = True,
    ) -> list[dict]:
        if self.test_data is None or self.context is None:
            raise ValueError("Run train() first")
        margin = self.edge_margin if edge_margin is None else edge_margin
        X_test_o, y_test_o, X_test_g, _ = self.test_data
        results = []
        for i in range(len(X_test_o)):
            row = self.context.iloc[i]
            b365 = _b365_from_row(row)
            evaluated = self._evaluate_row(
                X_test_o.iloc[i], b365, None, margin, require_edge,
                div=row.get("Div"),
            )
            if evaluated is None:
                continue
            pred, confidence, edge, final = evaluated
            if confidence < confidence_threshold:
                continue
            goals_df = pd.DataFrame([X_test_g.iloc[i]], columns=self.goals_features)
            expected_goals = max(0.0, float(self.goals_model.predict(goals_df)[0]))
            actual = OUTCOME_MAP.get(row["FTR"], -1)
            bookie_code = bookie_favorite(*b365) if b365 else None
            results.append({
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "date": row["Date"],
                "div": row.get("Div"),
                "outcome": OUTCOME_LABELS[pred],
                "outcome_code": pred,
                "actual_code": actual,
                "confidence": confidence,
                "edge": edge,
                "expected_goals": expected_goals,
                "total_goals": max(0, round(expected_goals)),
                "odds_error": bool(row.get("odds_error", False)),
                "probs": {"H": float(final[1]), "D": float(final[0]), "A": float(final[2])},
                "b365": b365,
                "bookie_code": bookie_code,
                "bookie_pick": OUTCOME_LABELS[bookie_code] if bookie_code is not None else None,
            })
        return results