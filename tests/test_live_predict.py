import pandas as pd

from evaluation.odds_lines import resolve_b365_for_live
from scripts.backtest import format_prediction


def test_resolve_b365_prefers_closing_then_chaos():
    row = pd.Series({
        "B365H": 2.5, "B365D": 3.4, "B365A": 3.1,
        "B365CH": 2.2, "B365CD": 3.5, "B365CA": 3.4,
        "odds_H": 2.0, "odds_D": 3.6, "odds_A": 3.8,
    })
    assert resolve_b365_for_live(row) == (2.2, 3.5, 3.4)


def test_resolve_b365_falls_back_to_chaos_odds():
    row = pd.Series({
        "odds_H": 1.9, "odds_D": 3.6, "odds_A": 4.1,
    })
    assert resolve_b365_for_live(row) == (1.9, 3.6, 4.1)


def test_format_prediction_live_fields():
    text = format_prediction({
        "home": "Arsenal",
        "away": "Chelsea",
        "date": "01/08/2026",
        "div": "E0",
        "outcome": "Home Win",
        "confidence": 78.5,
        "edge": 0.12,
        "edge_margin": 0.13,
        "expected_goals": 2.7,
        "bookie_pick": "Away Win",
        "b365_close": (2.1, 3.5, 3.4),
        "probs": {"H": 0.5, "D": 0.25, "A": 0.25},
        "intel": {"home_news_attention": 0.3, "away_news_attention": 0.2},
    })
    assert "[E0]" in text
    assert "Closing: 2.10/3.50/3.40" in text
    assert "Intel attn" in text
    assert "margin ≥13%" in text