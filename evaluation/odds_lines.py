"""Opening vs closing Bet365 odds from football-data.co.uk columns."""

import pandas as pd

from evaluation.implied_odds import implied_probs_from_odds

OPEN_COLS = ("B365H", "B365D", "B365A")
CLOSE_COLS = ("B365CH", "B365CD", "B365CA")


def _tuple_from_cols(row: pd.Series, cols: tuple[str, str, str]) -> tuple[float, float, float] | None:
    if not set(cols).issubset(row.index):
        return None
    h, d, a = row[cols[0]], row[cols[1]], row[cols[2]]
    if pd.isna(h) or pd.isna(d) or pd.isna(a):
        return None
    try:
        h, d, a = float(h), float(d), float(a)
    except (TypeError, ValueError):
        return None
    if min(h, d, a) <= 0:
        return None
    return h, d, a


def opening_b365_from_row(row: pd.Series) -> tuple[float, float, float] | None:
    return _tuple_from_cols(row, OPEN_COLS)


def closing_b365_from_row(row: pd.Series) -> tuple[float, float, float] | None:
    return _tuple_from_cols(row, CLOSE_COLS)


def b365_from_row(row: pd.Series, prefer_closing: bool = True) -> tuple[float, float, float] | None:
    """Return closing odds when available, else opening."""
    if prefer_closing:
        closing = closing_b365_from_row(row)
        if closing is not None:
            return closing
    return opening_b365_from_row(row)


def _tuple_from_chaos_odds(row: pd.Series) -> tuple[float, float, float] | None:
    cols = ("odds_H", "odds_D", "odds_A")
    if not set(cols).issubset(row.index):
        return None
    return _tuple_from_cols(row, cols)


def resolve_b365_for_live(row: pd.Series) -> tuple[float, float, float] | None:
    """Closing/current line: B365C* → live chaos odds → opening B365*."""
    closing = closing_b365_from_row(row)
    if closing is not None:
        return closing
    chaos = _tuple_from_chaos_odds(row)
    if chaos is not None:
        return chaos
    return opening_b365_from_row(row)


def opening_and_closing_from_row(
    row: pd.Series,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Return (opening, closing/current) Bet365 tuples when available."""
    opening = opening_b365_from_row(row)
    closing = closing_b365_from_row(row) or _tuple_from_chaos_odds(row)
    return opening, closing


def line_movement(row: pd.Series) -> dict[str, float] | None:
    """Implied-prob shift (closing minus opening) per outcome code 0=D, 1=H, 2=A."""
    opening = opening_b365_from_row(row)
    closing = closing_b365_from_row(row)
    if opening is None or closing is None:
        return None
    p_open = implied_probs_from_odds(*opening)
    p_close = implied_probs_from_odds(*closing)
    return {
        "D": p_close[0] - p_open[0],
        "H": p_close[1] - p_open[1],
        "A": p_close[2] - p_open[2],
    }