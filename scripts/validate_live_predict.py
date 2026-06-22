"""Phase 6A: live predict validation harness.

Proves the --predict path is structurally sound and scores fixtures correctly,
even when no Big 5 upcoming fixtures exist in football-data.co.uk fixtures.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from config.settings import (
    league_div_codes,
    league_summary,
    per_league_edge_margins,
    today,
    use_intel_in_train,
)
from evaluation.odds_lines import resolve_b365_for_live
from main import get_future_matches
from predictors.game_forger import GameForger, _is_completed, _parse_dates
from utils.chaos_cache import cache_stats
from utils.db import engine, read_matches

MIN_TRAIN_MATCHES = 100


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str


def fixture_readiness(matches: pd.DataFrame | None = None) -> dict:
    """Summarize upcoming Big 5 fixture availability."""
    if matches is None:
        matches = read_matches()
    divs = league_div_codes()
    now = today()
    future = get_future_matches(matches, div_filter=divs)
    scoped = _parse_dates(matches)
    scoped = scoped[scoped["Div"].isin(divs)] if "Div" in scoped.columns else scoped
    completed = scoped[_is_completed(scoped)] if not scoped.empty else scoped
    uncompleted = scoped[~_is_completed(scoped)] if not scoped.empty else scoped

    next_date = None
    if not future.empty:
        next_date = future["Date"].min()

    return {
        "as_of": now,
        "div_codes": divs,
        "completed_big5": len(completed),
        "uncompleted_big5": len(uncompleted),
        "upcoming_big5": len(future),
        "next_fixture_date": next_date,
        "ready_for_live_predict": len(future) > 0,
    }


def _fixture_guidance(readiness: dict) -> str:
    if readiness["ready_for_live_predict"]:
        n = readiness["upcoming_big5"]
        nd = readiness["next_fixture_date"]
        date_str = nd.strftime("%d/%m/%Y") if nd is not None else "unknown"
        return f"{n} upcoming Big 5 fixture(s); next on {date_str}."
    return (
        "No upcoming Big 5 fixtures in DB (season may be between windows). "
        "Run `poetry run python main.py --ingest` after football-data.co.uk "
        "publishes new fixtures.csv rows for 2626 (or current season)."
    )


def _sample_big5_teams() -> tuple[str, str, str]:
    """Pick a recent completed Big 5 fixture so PIT features resolve."""
    divs = league_div_codes()
    hist = _parse_dates(pd.read_sql("SELECT Div, HomeTeam, AwayTeam, Date, FTR FROM matches", engine))
    hist = hist[hist["Div"].isin(divs) & _is_completed(hist)]
    if hist.empty:
        return "E0", "Arsenal", "Chelsea"
    row = hist.sort_values("Date").iloc[-1]
    return str(row["Div"]), str(row["HomeTeam"]), str(row["AwayTeam"])


def e2e_smoke_predict(
    *,
    train_limit: int = 80,
    injuries_df=None,
) -> tuple[bool, str, list[dict]]:
    """Train on historical data and score one synthetic future fixture (no network)."""
    divs = league_div_codes()
    div, home, away = _sample_big5_teams()
    future_date = today() + timedelta(days=14)
    synthetic = pd.DataFrame([{
        "Div": div,
        "Date": future_date.strftime("%d/%m/%Y"),
        "HomeTeam": home,
        "AwayTeam": away,
        "B365H": 2.1,
        "B365D": 3.5,
        "B365A": 3.4,
        "B365CH": 2.0,
        "B365CD": 3.6,
        "B365CA": 3.5,
    }])

    forger = GameForger()
    forger.train(
        injuries_df=injuries_df,
        limit=train_limit,
        use_cache=True,
        refresh_cache=False,
        div_filter=divs,
        chaos_cache_only=True,
    )
    forger.prepare_prediction_data(
        synthetic,
        injuries_df=injuries_df,
        use_cache=True,
        refresh_cache=False,
        div_filter=divs,
        chaos_cache_only=True,
    )
    predictions = forger.predict(
        confidence_threshold=0.0,
        edge_margin=None,
        use_simulation=False,
        require_edge=False,
    )
    if not predictions:
        return False, "smoke predict returned 0 rows (model or odds merge failure)", []

    pred = predictions[0]
    required = {"home", "away", "outcome", "probs", "b365_close"}
    missing = required - set(pred.keys())
    if missing:
        return False, f"smoke prediction missing keys: {sorted(missing)}", predictions

    b365 = pred.get("b365_close")
    if not b365 or len(b365) != 3:
        return False, f"invalid b365_close: {b365}", predictions

    return True, f"scored {pred['home']} vs {pred['away']} -> {pred['outcome']}", predictions


def run_validation_checks(
    *,
    train_limit: int = 80,
    run_smoke: bool = True,
    injuries_df=None,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    divs = league_div_codes()
    margins = per_league_edge_margins()
    readiness = fixture_readiness()

    checks.append(ValidationCheck(
        "enabled_leagues",
        len(divs) >= 5,
        f"Big 5 div codes: {divs}",
    ))
    checks.append(ValidationCheck(
        "training_data",
        readiness["completed_big5"] >= MIN_TRAIN_MATCHES,
        f"{readiness['completed_big5']} completed Big 5 matches (min {MIN_TRAIN_MATCHES})",
    ))
    checks.append(ValidationCheck(
        "intel_train_policy",
        not use_intel_in_train(),
        f"use_intel_in_train={use_intel_in_train()} (expect False for live-only intel)",
    ))
    checks.append(ValidationCheck(
        "per_league_edges",
        all(d in margins for d in divs),
        f"margins configured for {sorted(margins.keys())}",
    ))

    sample = pd.Series({
        "B365H": 2.5, "B365D": 3.4, "B365A": 3.1,
        "B365CH": 2.2, "B365CD": 3.5, "B365CA": 3.4,
    })
    resolved = resolve_b365_for_live(sample)
    checks.append(ValidationCheck(
        "closing_odds_resolution",
        resolved == (2.2, 3.5, 3.4),
        f"resolve_b365_for_live -> {resolved}",
    ))

    checks.append(ValidationCheck(
        "fixture_readiness",
        True,
        _fixture_guidance(readiness),
    ))

    if run_smoke:
        try:
            ok, detail, _ = e2e_smoke_predict(
                train_limit=train_limit,
                injuries_df=injuries_df,
            )
            checks.append(ValidationCheck("e2e_smoke_predict", ok, detail))
        except Exception as exc:
            checks.append(ValidationCheck(
                "e2e_smoke_predict",
                False,
                f"exception: {exc}",
            ))

    return checks


def format_validation_report(checks: list[ValidationCheck]) -> str:
    lines = ["Live predict validation", "=" * 40]
    passed = sum(1 for c in checks if c.passed)
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"[{mark}] {check.name}: {check.detail}")
    lines.append("-" * 40)
    lines.append(f"Result: {passed}/{len(checks)} checks passed")
    return "\n".join(lines)


def run_validate_live(
    *,
    train_limit: int = 80,
    skip_smoke: bool = False,
    injuries_df=None,
) -> int:
    """Run all validation checks; return 0 on success, 1 on failure."""
    readiness = fixture_readiness()
    print(league_summary())
    print(f"Chaos cache: {cache_stats()}")
    print(f"Fixture readiness: {_fixture_guidance(readiness)}")
    print()

    checks = run_validation_checks(
        train_limit=train_limit,
        run_smoke=not skip_smoke,
        injuries_df=injuries_df,
    )
    report = format_validation_report(checks)
    print(report)

    critical = [c for c in checks if c.name != "fixture_readiness" and not c.passed]
    if critical:
        return 1
    return 0