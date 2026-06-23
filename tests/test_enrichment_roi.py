import pandas as pd

from scripts.enrichment_roi import enrichment_coverage, format_enrichment_report


def test_enrichment_coverage_counts_holdout_hit(monkeypatch):
    holdout = pd.DataFrame({
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "Date": pd.to_datetime(["2024-08-17"]),
        "Div": ["E0"],
        "FTR": ["H"],
    })

    monkeypatch.setattr(
        "scripts.enrichment_roi.load_xg_matches",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "scripts.enrichment_roi.load_statsbomb_xg",
        lambda: pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "MatchDate": ["17/08/2024"],
            "Div": ["E0"],
            "xg_home": [1.0],
            "xg_away": [0.5],
        }),
    )
    monkeypatch.setattr(
        "scripts.enrichment_roi.load_fbref_xg",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "scripts.enrichment_roi.xg_source_priority",
        lambda: ["understat", "statsbomb", "fbref"],
    )

    coverage = enrichment_coverage(holdout)
    assert coverage["statsbomb"]["holdout_matched"] == 1
    assert coverage["statsbomb"]["coverage_pct"] == 100.0


def test_format_enrichment_report():
    text = format_enrichment_report(
        {"statsbomb": {"cache_rows": 10, "holdout_matched": 2, "coverage_pct": 20.0}},
        {"understat": 0, "statsbomb": 10, "fbref": 0},
    )
    assert "statsbomb" in text
    assert "20.0%" in text