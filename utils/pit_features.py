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
SHOT_DEFAULTS = {
    "avg_shots_on_target_home": 4.5,
    "avg_shots_on_target_away": 4.0,
    "avg_shots_home": 12.0,
    "avg_shots_away": 11.0,
}
SHOT_COLS = ("HS", "AS", "HST", "AST")


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


def _team_shot_prior(history: pd.DataFrame, team: str, before: pd.Timestamp) -> dict[str, float]:
    prior = history[history["_dt"] < before]
    home = prior[prior["HomeTeam"] == team]
    away = prior[prior["AwayTeam"] == team]

    def _mean_col(frames: list[pd.Series], default: float) -> float:
        parts = [s for s in frames if not s.empty]
        if not parts:
            return default
        return float(pd.concat(parts, ignore_index=True).mean())

    return {
        "avg_shots_on_target": _mean_col(
            [
                home["HST"] if "HST" in home.columns else pd.Series(dtype=float),
                away["AST"] if "AST" in away.columns else pd.Series(dtype=float),
            ],
            SHOT_DEFAULTS["avg_shots_on_target_home"],
        ),
        "avg_shots": _mean_col(
            [
                home["HS"] if "HS" in home.columns else pd.Series(dtype=float),
                away["AS"] if "AS" in away.columns else pd.Series(dtype=float),
            ],
            SHOT_DEFAULTS["avg_shots_home"],
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
    shots_home, shots_away = [], []
    shots_total_home, shots_total_away = [], []
    h2h_win, h2h_hg, h2h_ag = [], [], []
    has_shots = all(c in hist.columns for c in SHOT_COLS)

    for _, row in out.iterrows():
        dt = row["_dt"]
        home, away = row["HomeTeam"], row["AwayTeam"]

        hf = _team_form_prior(hist, home, dt)
        af = _team_form_prior(hist, away, dt)
        form_home_scored.append(hf["avg_goals_scored"])
        form_home_conceded.append(hf["avg_goals_conceded"])
        form_away_scored.append(af["avg_goals_scored"])
        form_away_conceded.append(af["avg_goals_conceded"])

        if has_shots:
            hs = _team_shot_prior(hist, home, dt)
            aws = _team_shot_prior(hist, away, dt)
            shots_home.append(hs["avg_shots_on_target"])
            shots_away.append(aws["avg_shots_on_target"])
            shots_total_home.append(hs["avg_shots"])
            shots_total_away.append(aws["avg_shots"])

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
    if has_shots:
        out["avg_shots_on_target_home"] = shots_home
        out["avg_shots_on_target_away"] = shots_away
        out["avg_shots_home"] = shots_total_home
        out["avg_shots_away"] = shots_total_away
    return out.drop(columns=["_dt"])