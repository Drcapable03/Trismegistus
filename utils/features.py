import pandas as pd
from sqlalchemy import inspect, text

from utils.db import engine

TEAM_FORM_TABLE = "team_form"


def _apply_div_filter(matches: pd.DataFrame, div_filter: str | list[str] | None) -> pd.DataFrame:
    if not div_filter or "Div" not in matches.columns:
        return matches
    codes = [div_filter] if isinstance(div_filter, str) else list(div_filter)
    return matches[matches["Div"].isin(codes)]


def _ensure_team_form_schema() -> None:
    if not inspect(engine).has_table(TEAM_FORM_TABLE):
        pd.DataFrame(columns=["team", "Div", "avg_goals_scored", "avg_goals_conceded"]).to_sql(
            TEAM_FORM_TABLE, engine, index=False,
        )
        return
    cols = {c["name"] for c in inspect(engine).get_columns(TEAM_FORM_TABLE)}
    if "Div" not in cols:
        with engine.connect() as conn:
            conn.execute(text(f'ALTER TABLE {TEAM_FORM_TABLE} ADD COLUMN "Div" TEXT'))
            conn.execute(text(f'UPDATE {TEAM_FORM_TABLE} SET "Div" = \'ALL\' WHERE "Div" IS NULL'))
            conn.commit()


def _form_for_div(matches: pd.DataFrame, div: str) -> pd.DataFrame:
    home_stats = matches.groupby("HomeTeam").agg({
        "FTHG": "mean",
        "FTAG": "mean",
    }).rename(columns={"FTHG": "avg_goals_scored", "FTAG": "avg_goals_conceded"})

    away_stats = matches.groupby("AwayTeam").agg({
        "FTAG": "mean",
        "FTHG": "mean",
    }).rename(columns={"FTAG": "avg_goals_scored", "FTHG": "avg_goals_conceded"})

    form = pd.concat([home_stats, away_stats]).groupby(level=0).mean()
    form["team"] = form.index
    form["Div"] = div
    return form.reset_index(drop=True)


def calculate_team_form(div_filter: str | list[str] | None = None):
    """Persist league-scoped team form; upserts per Div without clobbering other competitions."""
    _ensure_team_form_schema()
    matches = pd.read_sql("SELECT * FROM matches", engine)
    matches = _apply_div_filter(matches, div_filter)
    if "FTR" in matches.columns:
        matches = matches[matches["FTR"].isin(["H", "D", "A"])]
    if matches.empty or "Div" not in matches.columns:
        print("No matches for team form calculation.")
        return

    divs = sorted(matches["Div"].dropna().unique())
    frames = [_form_for_div(matches[matches["Div"] == div], div) for div in divs]
    new_form = pd.concat(frames, ignore_index=True)

    placeholders = ", ".join(f":d{i}" for i in range(len(divs)))
    params = {f"d{i}": div for i, div in enumerate(divs)}
    with engine.connect() as conn:
        conn.execute(
            text(f'DELETE FROM {TEAM_FORM_TABLE} WHERE "Div" IN ({placeholders})'),
            params,
        )
        conn.commit()

    new_form.to_sql(TEAM_FORM_TABLE, engine, if_exists="append", index=False)
    scope = div_filter if div_filter else "all competitions"
    print(f"Team form calculated ({scope}, {len(divs)} divs) and saved!")