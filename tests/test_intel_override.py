import pandas as pd

from predictors.game_forger import GameForger


def test_apply_intel_override_pins_sentiment():
    forger = GameForger()
    forger.outcome_features = [
        "home_news_attention", "home_news_sentiment", "home_reddit_sentiment",
    ]
    row = pd.Series({
        "home_news_attention": 0.8,
        "home_news_sentiment": 0.7,
        "home_reddit_sentiment": 0.6,
    })
    out = forger._apply_intel_override(row, 0.5)
    assert out["home_news_attention"] == 0.0
    assert out["home_news_sentiment"] == 0.5
    assert out["home_reddit_sentiment"] == 0.5