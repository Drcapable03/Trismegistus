import pandas as pd
from utils.db import engine

def calculate_team_form():
    matches = pd.read_sql("SELECT * FROM matches", engine)
    
    # Home team stats
    home_stats = matches.groupby("HomeTeam").agg({
        "FTHG": "mean",  # Avg goals scored at home
        "FTAG": "mean"   # Avg goals conceded at home
    }).rename(columns={"FTHG": "avg_goals_scored", "FTAG": "avg_goals_conceded"})
    
    # Away team stats
    away_stats = matches.groupby("AwayTeam").agg({
        "FTAG": "mean",  # Avg goals scored away
        "FTHG": "mean"   # Avg goals conceded away
    }).rename(columns={"FTAG": "avg_goals_scored", "FTHG": "avg_goals_conceded"})
    
    # Combine and average
    form = pd.concat([home_stats, away_stats]).groupby(level=0).mean()
    form["team"] = form.index
    form.to_sql("team_form", engine, if_exists="replace", index=False)
    print("Team form calculated and saved!")
    