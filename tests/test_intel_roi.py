import pandas as pd

from scripts.intel_roi import _flat_roi, chaos_intel_coverage, intel_sentiment_summary


def test_flat_roi_calculation():
    preds = [{
        "outcome_code": 1,
        "actual_code": 1,
        "b365": (2.0, 3.5, 4.0),
    }, {
        "outcome_code": 0,
        "actual_code": 1,
        "b365": (3.0, 3.5, 2.5),
    }]
    roi = _flat_roi(preds)
    assert roi == 0.0


def test_chaos_intel_coverage_empty():
    holdout = pd.DataFrame(columns=["HomeTeam", "AwayTeam", "Date", "Div", "FTR"])
    report = chaos_intel_coverage(holdout)
    assert report["holdout_matches"] == 0
    assert report["coverage_pct"] == 0.0


def test_intel_sentiment_summary_no_cache(monkeypatch):
    monkeypatch.setattr(
        "scripts.intel_roi.pd.read_sql",
        lambda *a, **k: pd.DataFrame(),
    )
    holdout = pd.DataFrame([{
        "HomeTeam": "A", "AwayTeam": "B",
        "Date": pd.Timestamp("2026-01-01"), "Div": "E0", "FTR": "H",
    }])
    summary = intel_sentiment_summary(holdout)
    assert summary["samples"] == 0