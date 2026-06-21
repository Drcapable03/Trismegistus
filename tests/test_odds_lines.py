import pandas as pd

from evaluation.odds_lines import (
    b365_from_row,
    closing_b365_from_row,
    line_movement,
    opening_b365_from_row,
)


def test_b365_prefers_closing():
    row = pd.Series({
        "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
        "B365CH": 1.9, "B365CD": 3.6, "B365CA": 4.2,
    })
    assert closing_b365_from_row(row) == (1.9, 3.6, 4.2)
    assert opening_b365_from_row(row) == (2.0, 3.5, 4.0)
    assert b365_from_row(row) == (1.9, 3.6, 4.2)


def test_line_movement_present_when_both_lines_exist():
    row = pd.Series({
        "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
        "B365CH": 1.8, "B365CD": 3.8, "B365CA": 4.5,
    })
    move = line_movement(row)
    assert move is not None
    assert "H" in move