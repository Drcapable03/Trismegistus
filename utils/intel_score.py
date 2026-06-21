"""VADER-based text sentiment helpers for live intel."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def compound_to_unit(compound: float) -> float:
    """Map VADER compound [-1, 1] to [0, 1] with 0.5 neutral."""
    return max(0.0, min(1.0, (compound + 1.0) / 2.0))


def score_texts(texts: list[str]) -> float:
    """Average VADER sentiment for non-empty strings; 0.5 if none."""
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return 0.5
    analyzer = _analyzer()
    compounds = [analyzer.polarity_scores(t)["compound"] for t in cleaned]
    return compound_to_unit(sum(compounds) / len(compounds))


def attention_from_count(count: int, base: float = 0.1, step: float = 0.04) -> float:
    """Buzz proxy from item count (same scale as legacy headline-count sentiment)."""
    if count <= 0:
        return 0.0
    return min(1.0, base + step * count)