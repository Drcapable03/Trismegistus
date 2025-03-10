import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:newpassword@localhost/Fooball_predictor")

def calculate_team_form():
    df = pd.read_sql("SELECT * FROM matches", engine)
    # Average goals scored last 5 games per team
    home_form = df.groupby("HomeTeam")["FTHG"].mean().reset_index()
    away_form = df.groupby("AwayTeam")["FTAG"].mean().reset_index()
    home_form.columns = ["team", "avg_goals_scored_home"]
    away_form.columns = ["team", "avg_goals_scored_away"]
    form = pd.merge(home_form, away_form, on="team", how="outer").fillna(0)
    form["avg_goals_scored"] = (form["avg_goals_scored_home"] + form["avg_goals_scored_away"]) / 2
    form.to_sql("team_form", engine, if_exists="replace", index=False)
    print("Team form calculated and saved!")

if __name__ == "__main__":
    calculate_team_form()