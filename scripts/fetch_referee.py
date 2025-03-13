import numpy as np
import pandas as pd

def fetch_referee_bias(matches):
    # Mock referee data—replace with scrape (e.g., premierleague.com) or CSV later
    referee_data = {
        "referee": ["Mike Dean", "Anthony Taylor", "Martin Atkinson"],
        "home_win_pct": [0.55, 0.50, 0.48],  # Bias toward home wins
        "yellows_per_game": [3.8, 4.2, 3.5]
    }
    refs = pd.DataFrame(referee_data)
    
    # Randomly assign referees to matches (for now)
    matches["referee"] = np.random.choice(refs["referee"], size=len(matches))
    data = matches.merge(refs, on="referee")
    print("Referee bias assigned (mock data)")
    return data[["HomeTeam", "AwayTeam", "Date", "referee", "home_win_pct", "yellows_per_game"]]