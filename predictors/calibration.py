"""Isotonic probability calibration for multiclass outcome model."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def _calibration_cv_splits(y_cal: pd.Series) -> int | None:
    """Pick CV folds that fit the smallest outcome class (sklearn stratified limit)."""
    counts = y_cal.value_counts()
    if len(counts) < 2:
        return None
    min_class = int(counts.min())
    if min_class < 2:
        return None
    return min(5, min_class)


class OutcomeCalibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator: CalibratedClassifierCV | None = None
        self.classes_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.calibrator is not None

    def fit(self, base_estimator, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        cv = _calibration_cv_splits(y_cal)
        if len(X_cal) < 10 or cv is None:
            self.calibrator = None
            self.classes_ = getattr(base_estimator, "classes_", None)
            return
        self.calibrator = CalibratedClassifierCV(
            FrozenEstimator(base_estimator),
            method=self.method,
            cv=cv,
        )
        self.calibrator.fit(X_cal, y_cal)
        self.classes_ = self.calibrator.classes_

    def predict_proba(self, base_estimator, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            return base_estimator.predict_proba(X)[0]
        return self.calibrator.predict_proba(X)[0]