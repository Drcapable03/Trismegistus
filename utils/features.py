import pandas as pd

from utils.db import engine


def _apply_div_filter(matches: pd.DataFrame, div_filter: str | list[str] | None) -> pd.DataFrame:
    if not div_filter or "Div" not in matches.columns:
        return matches
    codes = [div_filter] if isinstance(div_filter, str) else list(div_filter)
    return matches[matches["Div"].isin(codes)]


def calculate_team_form(div_filter: str | list[str] | None = None):
    matches = pd.read_sql("SELECT * FROM matches", engine)
    matches = _apply_div_filter(matches, div_filter)
    if matches.empty:
        print("No matches for team form calculation.")
        return

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
    form.to_sql("team_form", engine, if_exists="replace", index=False)
    scope = div_filter if div_filter else "all competitions"
    print(f"Team form calculated ({scope}) and saved!")