"""Point-in-time (no leakage) team form and head-to-head features."""

import pandas as pd

H2H_DEFAULTS = {
    "h2h_home_win_pct": 0.46,
    "h2h_avg_home_goals": 1.35,
    "h2h_avg_away_goals": 1.15,
}
FORM_DEFAULTS = {
    "avg_goals_scored_home": 1.40,
    "avg_goals_conceded_home": 1.20,
    "avg_goals_scored_away": 1.20,
    "avg_goals_conceded_away": 1.40,
}


def _parse_dt(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")


def _team_form_prior(history: pd.DataFrame, team: str, before: pd.Timestamp) -> dict[str, float]:
    prior = history[history["_dt"] < before]
    home = prior[prior["HomeTeam"] == team]
    away = prior[prior["AwayTeam"] == team]
    scored_home = home["FTHG"].dropna()
    conceded_home = home["FTAG"].dropna()
    scored_away = away["FTAG"].dropna()
    conceded_away = away["FTHG"].dropna()

    def _mean(series: pd.Series, default: float) -> float:
        return float(series.mean()) if not series.empty else default

    return {
        "avg_goals_scored": _mean(
            pd.concat([scored_home, scored_away], ignore_index=True),
            FORM_DEFAULTS["avg_goals_scored_home"],
        ),
        "avg_goals_conceded": _mean(
            pd.concat([conceded_home, conceded_away], ignore_index=True),
            FORM_DEFAULTS["avg_goals_conceded_home"],
        ),
    }


def compute_pit_form_and_h2h(
    target: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach PIT form + h2h columns; history must be completed matches only."""
    out = target.copy()
    hist = history.copy()
    out["_dt"] = _parse_dt(out["Date"])
    hist["_dt"] = _parse_dt(hist["Date"])
    hist = hist.dropna(subset=["_dt"])

    form_home_scored, form_home_conceded = [], []
    form_away_scored, form_away_conceded = [], []
    h2h_win, h2h_hg, h2h_ag = [], [], []

    for _, row in out.iterrows():
        dt = row["_dt"]
        home, away = row["HomeTeam"], row["AwayTeam"]

        hf = _team_form_prior(hist, home, dt)
        af = _team_form_prior(hist, away, dt)
        form_home_scored.append(hf["avg_goals_scored"])
        form_home_conceded.append(hf["avg_goals_conceded"])
        form_away_scored.append(af["avg_goals_scored"])
        form_away_conceded.append(af["avg_goals_conceded"])

        prior_h2h = hist[
            (hist["HomeTeam"] == home)
            & (hist["AwayTeam"] == away)
            & (hist["_dt"] < dt)
        ]
        if prior_h2h.empty:
            h2h_win.append(H2H_DEFAULTS["h2h_home_win_pct"])
            h2h_hg.append(H2H_DEFAULTS["h2h_avg_home_goals"])
            h2h_ag.append(H2H_DEFAULTS["h2h_avg_away_goals"])
        else:
            h2h_win.append(float((prior_h2h["FTR"] == "H").mean()))
            h2h_hg.append(float(prior_h2h["FTHG"].mean()))
            h2h_ag.append(float(prior_h2h["FTAG"].mean()))

    out["avg_goals_scored_home"] = form_home_scored
    out["avg_goals_conceded_home"] = form_home_conceded
    out["avg_goals_scored_away"] = form_away_scored
    out["avg_goals_conceded_away"] = form_away_conceded
    out["h2h_home_win_pct"] = h2h_win
    out["h2h_avg_home_goals"] = h2h_hg
    out["h2h_avg_away_goals"] = h2h_ag
    return out.drop(columns=["_dt"])