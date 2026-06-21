import pandas as pd
from sqlalchemy import text

from agents.odds_agent import fetch_odds_from_db
from utils.db import engine, ensure_schema


def test_fetch_odds_from_db():
    ensure_schema("matches")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM matches"))
        conn.commit()

    pd.DataFrame([{
        "HomeTeam": "Spain",
        "AwayTeam": "Saudi Arabia",
        "Date": "20/06/2026",
        "Div": "WC26",
        "B365H": 1.12,
        "B365D": 8.5,
        "B365A": 19.0,
        "FTR": None,
    }]).to_sql("matches", engine, if_exists="append", index=False)

    odds = fetch_odds_from_db("Spain", "Saudi Arabia")
    assert odds is not None
    assert odds["source"] == "db_opening"
    assert odds["H"] == 1.12