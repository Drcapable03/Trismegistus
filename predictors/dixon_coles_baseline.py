"""Dixon-Coles goal model baseline via penaltyblog."""

import numpy as np
import pandas as pd

MIN_MATCHES = 40


class DixonColesBaseline:
    """Per-league Dixon-Coles models with global fallback."""

    def __init__(self, min_matches: int = MIN_MATCHES):
        self.min_matches = min_matches
        self.models: dict[str, object] = {}
        self.global_model = None

    def _fit_one(self, frame: pd.DataFrame):
        from penaltyblog.models import DixonColesGoalModel

        goals_home = pd.to_numeric(frame["FTHG"], errors="coerce").fillna(0).astype(int).tolist()
        goals_away = pd.to_numeric(frame["FTAG"], errors="coerce").fillna(0).astype(int).tolist()
        model = DixonColesGoalModel(
            goals_home,
            goals_away,
            frame["HomeTeam"].astype(str).tolist(),
            frame["AwayTeam"].astype(str).tolist(),
        )
        model.fit()
        return model

    def fit(self, matches: pd.DataFrame, div_codes: list[str] | None = None) -> None:
        frame = matches.copy()
        frame = frame[frame["FTR"].isin(["H", "D", "A"])]
        if frame.empty:
            self.models = {}
            self.global_model = None
            return

        self.models = {}
        codes = div_codes or sorted(frame["Div"].dropna().unique().tolist())
        for div in codes:
            subset = frame[frame["Div"] == div]
            if len(subset) >= self.min_matches:
                try:
                    self.models[str(div)] = self._fit_one(subset)
                except Exception as e:
                    print(f"Dixon-Coles fit failed for {div}: {e}")

        if len(frame) >= self.min_matches:
            try:
                self.global_model = self._fit_one(frame)
            except Exception as e:
                print(f"Dixon-Coles global fit failed: {e}")
                self.global_model = None

    def predict_probs(
        self,
        home_team: str,
        away_team: str,
        div: str | None = None,
    ) -> np.ndarray | None:
        """Return [p_draw, p_home, p_away] aligned with GameForger outcome codes."""
        model = None
        if div and div in self.models:
            model = self.models[div]
        elif self.global_model is not None:
            model = self.global_model
        if model is None:
            return None
        try:
            grid = model.predict(home_team, away_team)
            return np.array([grid.draw, grid.home_win, grid.away_win], dtype=float)
        except Exception:
            return None

    def holdout_accuracy(
        self,
        context: pd.DataFrame,
        y_true: pd.Series,
    ) -> tuple[float, int]:
        correct = 0
        total = 0
        for pos, (_, row) in enumerate(context.iterrows()):
            probs = self.predict_probs(
                row["HomeTeam"],
                row["AwayTeam"],
                div=row.get("Div"),
            )
            if probs is None:
                continue
            pred = int(np.argmax(probs))
            actual = int(y_true.iloc[pos])
            total += 1
            if pred == actual:
                correct += 1
        acc = correct / total * 100 if total else 0.0
        return acc, total