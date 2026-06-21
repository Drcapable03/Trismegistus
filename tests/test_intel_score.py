import pytest

from utils.intel_score import attention_from_count, compound_to_unit, score_texts


def test_compound_to_unit_neutral():
    assert compound_to_unit(0.0) == 0.5


def test_score_texts_positive_headlines():
    score = score_texts(["Great win for the team", "Fans are delighted"])
    assert score > 0.5


def test_score_texts_empty_is_neutral():
    assert score_texts([]) == 0.5


def test_attention_from_count():
    assert attention_from_count(0) == 0.0
    assert attention_from_count(5) == pytest.approx(0.3)