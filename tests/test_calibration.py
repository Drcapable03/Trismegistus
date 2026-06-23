import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from predictors.calibration import OutcomeCalibrator, _calibration_cv_splits


def _synthetic_calibration_data(n: int = 120):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 3, size=n))
    return X, y


def test_calibrator_fits_and_predicts():
    X, y = _synthetic_calibration_data()
    base = GradientBoostingClassifier(n_estimators=20, random_state=42)
    base.fit(X.iloc[:80], y.iloc[:80])

    calibrator = OutcomeCalibrator()
    calibrator.fit(base, X.iloc[80:], y.iloc[80:])
    assert calibrator.is_fitted

    probs = calibrator.predict_proba(base, X.iloc[:1])
    assert probs.shape == (3,)
    assert pytest.approx(probs.sum(), rel=1e-3) == 1.0
    assert (probs >= 0).all()


def test_calibrator_skips_small_calibration_set():
    X, y = _synthetic_calibration_data(n=20)
    base = GradientBoostingClassifier(n_estimators=10, random_state=42)
    base.fit(X.iloc[:15], y.iloc[:15])

    calibrator = OutcomeCalibrator()
    calibrator.fit(base, X.iloc[15:], y.iloc[15:])
    assert not calibrator.is_fitted

    probs = calibrator.predict_proba(base, X.iloc[:1])
    assert probs.shape == (3,)


def test_calibration_cv_splits_uses_smallest_class():
    y = pd.Series([0] * 7 + [1] * 3 + [2] * 2)
    assert _calibration_cv_splits(y) == 2


def test_calibration_cv_splits_none_when_singleton_class():
    y = pd.Series([0] * 8 + [1, 2])
    assert _calibration_cv_splits(y) is None


def test_calibrator_adapts_cv_for_sparse_classes():
    """WC-sized cal sets can have <5 samples in a class; must not use cv=5."""
    X, y = _synthetic_calibration_data(n=80)
    y_cal = pd.Series([0] * 7 + [1] * 3 + [2] * 2)
    X_cal = X.iloc[:12].copy()
    base = GradientBoostingClassifier(n_estimators=10, random_state=42)
    base.fit(X.iloc[12:], y.iloc[12:])

    calibrator = OutcomeCalibrator()
    calibrator.fit(base, X_cal, y_cal)
    assert calibrator.is_fitted
    assert _calibration_cv_splits(y_cal) == 2