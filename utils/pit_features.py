"""Point-in-time (no leakage) team form and head-to-head features."""

import pandas as pd

from utils.elo_cache import ELO_DEFAULT, elo_on_date, load_elo_history
from utils.xg_cache import load_xg_matches

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
XG_ON_TARGET_WEIGHT = 0.35
XG_OFF_TARGET_WEIGHT = 0.08
XG_DEFAULTS = {
    "avg_xg_for": 1.35,
    "avg_xg_against": 1.25,
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


def _match_xg_proxy(hst: float, ast: float, hs: float, as_: float) -> tuple[float, float]:
    home_xg = XG_ON_TARGET_WEIGHT * hst + XG_OFF_TARGET_WEIGHT * max(0.0, hs - hst)
    away_xg = XG_ON_TARGET_WEIGHT * ast + XG_OFF_TARGET_WEIGHT * max(0.0, as_ - ast)
    return home_xg, away_xg


def _team_xg_prior_understat(xg_df: pd.DataFrame, team: str, before: pd.Timestamp) -> dict[str, float] | None:
    if xg_df.empty:
        return None
    xg_df = xg_df.copy()
    xg_df["_dt"] = _parse_dt(xg_df["MatchDate"])
    prior = xg_df[xg_df["_dt"] < before]
    xg_for, xg_against = [], []
    home = prior[prior["HomeTeam"] == team]
    away = prior[prior["AwayTeam"] == team]
    if not home.empty:
        xg_for.extend(home["xg_home"].astype(float).tolist())
        xg_against.extend(home["xg_away"].astype(float).tolist())
    if not away.empty:
        xg_for.extend(away["xg_away"].astype(float).tolist())
        xg_against.extend(away["xg_home"].astype(float).tolist())
    if not xg_for:
        return None

    def _mean(vals: list[float], default: float) -> float:
        return float(sum(vals) / len(vals)) if vals else default

    return {
        "avg_xg_for": _mean(xg_for, XG_DEFAULTS["avg_xg_for"]),
        "avg_xg_against": _mean(xg_against, XG_DEFAULTS["avg_xg_against"]),
    }


def _team_xg_prior(history: pd.DataFrame, team: str, before: pd.Timestamp) -> dict[str, float]:
    prior = history[history["_dt"] < before]
    xg_for, xg_against = [], []
    home = prior[prior["HomeTeam"] == team]
    away = prior[prior["AwayTeam"] == team]
    if not home.empty and all(c in home.columns for c in SHOT_COLS):
        for _, r in home.iterrows():
            hxg, axg = _match_xg_proxy(r["HST"], r["AST"], r["HS"], r["AS"])
            xg_for.append(hxg)
            xg_against.append(axg)
    if not away.empty and all(c in away.columns for c in SHOT_COLS):
        for _, r in away.iterrows():
            hxg, axg = _match_xg_proxy(r["HST"], r["AST"], r["HS"], r["AS"])
            xg_for.append(axg)
            xg_against.append(hxg)

    def _mean(vals: list[float], default: float) -> float:
        return float(sum(vals) / len(vals)) if vals else default

    return {
        "avg_xg_for": _mean(xg_for, XG_DEFAULTS["avg_xg_for"]),
        "avg_xg_against": _mean(xg_against, XG_DEFAULTS["avg_xg_against"]),
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
    for col in SHOT_COLS:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    form_home_scored, form_home_conceded = [], []
    form_away_scored, form_away_conceded = [], []
    shots_home, shots_away = [], []
    shots_total_home, shots_total_away = [], []
    xg_for_home, xg_against_home = [], []
    xg_for_away, xg_against_away = [], []
    h2h_win, h2h_hg, h2h_ag = [], [], []
    has_shots = all(c in hist.columns for c in SHOT_COLS)
    xg_cache = load_xg_matches()
    elo_hist = load_elo_history()
    elo_home, elo_away, elo_diff = [], [], []

    for _, row in out.iterrows():
        dt = row["_dt"]
        home, away = row["HomeTeam"], row["AwayTeam"]

        hf = _team_form_prior(hist, home, dt)
        af = _team_form_prior(hist, away, dt)
        form_home_scored.append(hf["avg_goals_scored"])
        form_home_conceded.append(hf["avg_goals_conceded"])
        form_away_scored.append(af["avg_goals_scored"])
        form_away_conceded.append(af["avg_goals_conceded"])

        eh = elo_on_date(home, dt, elo_hist)
        ea = elo_on_date(away, dt, elo_hist)
        elo_home.append(eh)
        elo_away.append(ea)
        elo_diff.append(eh - ea)

        hxg_u = _team_xg_prior_understat(xg_cache, home, dt)
        axg_u = _team_xg_prior_understat(xg_cache, away, dt)
        if hxg_u:
            hxg = hxg_u
        elif has_shots:
            hxg = _team_xg_prior(hist, home, dt)
        else:
            hxg = {"avg_xg_for": XG_DEFAULTS["avg_xg_for"], "avg_xg_against": XG_DEFAULTS["avg_xg_against"]}
        if axg_u:
            axg = axg_u
        elif has_shots:
            axg = _team_xg_prior(hist, away, dt)
        else:
            axg = {"avg_xg_for": XG_DEFAULTS["avg_xg_for"], "avg_xg_against": XG_DEFAULTS["avg_xg_against"]}
        xg_for_home.append(hxg["avg_xg_for"])
        xg_against_home.append(hxg["avg_xg_against"])
        xg_for_away.append(axg["avg_xg_for"])
        xg_against_away.append(axg["avg_xg_against"])

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
    out["elo_home"] = elo_home
    out["elo_away"] = elo_away
    out["elo_diff"] = elo_diff
    out["h2h_home_win_pct"] = h2h_win
    out["h2h_avg_home_goals"] = h2h_hg
    out["h2h_avg_away_goals"] = h2h_ag
    out["avg_xg_for_home"] = xg_for_home
    out["avg_xg_against_home"] = xg_against_home
    out["avg_xg_for_away"] = xg_for_away
    out["avg_xg_against_away"] = xg_against_away
    if has_shots:
        out["avg_shots_on_target_home"] = shots_home
        out["avg_shots_on_target_away"] = shots_away
        out["avg_shots_home"] = shots_total_home
        out["avg_shots_away"] = shots_total_away
    return out.drop(columns=["_dt"])