"""Unit tests for OddsPortal row parsing (no network)."""

from unittest.mock import MagicMock

from scrapers.oddsportal import _parse_event_row


def _mock_row(text: str, home: str, away: str):
    row = MagicMock()
    home_el = MagicMock()
    home_el.text = home
    away_el = MagicMock()
    away_el.text = away
    row.css.return_value = [home_el, away_el]
    row.get_all_text.return_value = text
    return row


def test_parse_finished_match():
    blob = (
        "Yesterday, 19 Jun Finished FIN USA 2 2 – 0 Australia 0 "
        "1.70 4.49 6.50"
    )
    row = _mock_row(blob, "USA", "Australia")
    parsed = _parse_event_row(row, "WC26")
    assert parsed is not None
    assert parsed["HomeTeam"] == "USA"
    assert parsed["FTR"] == "H"
    assert parsed["B365H"] == 1.70
    assert parsed["B365A"] == 6.50


def test_parse_upcoming_match():
    blob = "Today, 20 Jun 13:00 Netherlands – Sweden 1.80 4.24 4.96"
    row = _mock_row(blob, "Netherlands", "Sweden")
    parsed = _parse_event_row(row, "WC26")
    assert parsed is not None
    assert parsed["FTR"] is None
    assert parsed["B365D"] == 4.24