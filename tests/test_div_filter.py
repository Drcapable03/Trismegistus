import pandas as pd
from sqlalchemy import text

from predictors.game_forger import GameForger, _apply_div_filter
from utils.db import engine, ensure_schema
from utils.features import calculate_team_form


def test_apply_div_filter():
    df = pd.DataFrame({"Div": ["E0", "WC26", "E0"], "x": [1, 2, 3]})
    filtered = _apply_div_filter(df, "WC26")
    assert len(filtered) == 1
    assert filtered.iloc[0]["Div"] == "WC26"


def test_calculate_team_form_div_filter():
    ensure_schema("matches")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM matches"))
        conn.commit()

    df = pd.DataFrame({
        "HomeTeam": ["Spain", "Spain", "Arsenal"],
        "AwayTeam": ["Brazil", "France", "Chelsea"],
        "FTHG": [2, 1, 3],
        "FTAG": [1, 1, 0],
        "FTR": ["H", "D", "H"],
        "Date": ["01/06/2026", "08/06/2026", "01/08/2025"],
        "Div": ["WC26", "WC26", "E0"],
    })
    df.to_sql("matches", engine, if_exists="append", index=False)
    calculate_team_form(div_filter="WC26")
    form = pd.read_sql("SELECT * FROM team_form", engine)
    assert "Arsenal" not in form["team"].values
    assert "Spain" in form["team"].values


def _fake_chaos(matches, **kwargs):
    m = matches.copy()
    if pd.api.types.is_datetime64_any_dtype(m["Date"]):
        m["Date"] = m["Date"].dt.strftime("%d/%m/%Y")
    return pd.DataFrame({
        "HomeTeam": m["HomeTeam"],
        "AwayTeam": m["AwayTeam"],
        "Date": m["Date"],
        "rain": 0.0,
        "wind": 0.0,
        "home_x_sentiment": 0.0,
        "away_x_sentiment": 0.0,
        "home_injuries": 1,
        "away_injuries": 2,
        "odds_H": m["B365H"] if "B365H" in m.columns else 2.0,
        "odds_A": m["B365A"] if "B365A" in m.columns else 3.0,
        "odds_D": m["B365D"] if "B365D" in m.columns else 3.5,
    })


def test_game_forger_training_div_filter(monkeypatch):
    monkeypatch.setattr("predictors.game_forger.get_chaos_data", _fake_chaos)
    ensure_schema("matches")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM matches"))
        conn.commit()

    rows = []
    for i in range(12):
        rows.append({
            "HomeTeam": f"WC_H{i}",
            "AwayTeam": f"WC_A{i}",
            "FTHG": 2,
            "FTAG": 1,
            "FTR": "H",
            "Date": f"{(i + 1):02d}/06/2026",
            "Div": "WC26",
            "B365H": 1.5,
            "B365D": 4.0,
            "B365A": 6.0,
        })
    for i in range(20):
        rows.append({
            "HomeTeam": f"PL_H{i}",
            "AwayTeam": f"PL_A{i}",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
            "Date": f"{(i + 1):02d}/08/2025",
            "Div": "E0",
            "B365H": 2.0,
            "B365D": 3.2,
            "B365A": 3.8,
        })
    pd.DataFrame(rows).to_sql("matches", engine, if_exists="append", index=False)

    forger = GameForger()
    forger.prepare_training_data(limit=50, use_cache=False, div_filter="WC26", chaos_cache_only=True)
    assert forger.train_data is not None
    X_train_o, y_train_o, _, y_train_g = forger.train_data
    _, _, _, y_test_g = forger.test_data
    assert len(X_train_o) <= 12
    assert forger.training_metadata["split_method"] == "walk_forward"
    assert len(y_train_o) == len(y_train_g)
    assert len(forger.test_data[1]) == len(y_test_g)