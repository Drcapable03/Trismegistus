import pandas as pd

from utils.db import ensure_schema, engine, load_csv_to_db, table_exists


def test_ensure_schema_creates_matches_table():
    ensure_schema("matches")
    assert table_exists("matches")


def test_load_csv_deduplicates(tmp_path):
    from utils.db import reset_table
    reset_table("matches")
    ensure_schema("matches")
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "Div": ["E0", "E0"],
        "Date": ["01/01/2025", "02/01/2025"],
        "HomeTeam": ["TeamA", "TeamB"],
        "AwayTeam": ["TeamC", "TeamD"],
        "FTHG": [1, 2],
        "FTAG": [0, 1],
        "FTR": ["H", "H"],
        "B365H": [2.0, 1.8],
        "B365D": [3.5, 3.4],
        "B365A": [4.0, 4.5],
    })
    df.to_csv(csv_path, index=False)
    # Use matches table for integration test
    first = load_csv_to_db(str(csv_path), "matches")
    second = load_csv_to_db(str(csv_path), "matches")
    assert first == 2
    assert second == 0