import pandas as pd
from sqlalchemy import text

from utils.db import engine, ensure_schema
from utils.features import calculate_team_form


def test_calculate_team_form():
    ensure_schema("matches")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM matches"))
        conn.commit()

    df = pd.DataFrame({
        "HomeTeam": ["Alpha", "Alpha", "Beta"],
        "AwayTeam": ["Beta", "Gamma", "Alpha"],
        "FTHG": [2, 1, 0],
        "FTAG": [1, 2, 2],
        "FTR": ["H", "A", "A"],
        "Date": ["01/01/2025", "08/01/2025", "15/01/2025"],
        "Div": ["E0", "E0", "E0"],
    })
    df.to_sql("matches", engine, if_exists="append", index=False)
    calculate_team_form(div_filter="E0")

    form = pd.read_sql('SELECT * FROM team_form WHERE "Div" = \'E0\'', engine)
    assert "team" in form.columns
    assert "Div" in form.columns
    assert "avg_goals_scored" in form.columns
    assert len(form) >= 2