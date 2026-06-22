"""Phase 6C: intel coverage and exploratory ROI report on holdout chaos cache."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.settings import league_div_codes
from predictors.game_forger import GameForger, _is_completed, _parse_dates
from utils.chaos_cache import INTEL_COLS, ensure_chaos_cache, intel_is_pit_safe
from utils.db import engine, read_matches


NEUTRAL = 0.5
# outcome codes: D=0, H=1, A=2 — b365 tuples are (H, D, A)
_ODDS_IDX = {0: 1, 1: 0, 2: 2}


def _holdout_matches(limit: int | None = None) -> pd.DataFrame:
    divs = league_div_codes()
    matches = _parse_dates(read_matches())
    matches = matches[matches["Div"].isin(divs)]
    completed = matches[_is_completed(matches)].sort_values("Date", ascending=False)
    if limit:
        completed = completed.head(limit)
    return completed.sort_values("Date")


def chaos_intel_coverage(holdout: pd.DataFrame) -> dict:
    ensure_chaos_cache()
    cache = pd.read_sql("SELECT * FROM chaos_cache", engine)
    if cache.empty or holdout.empty:
        return {
            "holdout_matches": len(holdout),
            "cached_rows": len(cache),
            "matched_rows": 0,
            "pit_safe_rows": 0,
            "coverage_pct": 0.0,
            "pit_safe_pct": 0.0,
        }

    matched = 0
    pit_safe = 0
    for _, row in holdout.iterrows():
        date_str = row["Date"].strftime("%d/%m/%Y") if hasattr(row["Date"], "strftime") else str(row["Date"])
        hit = cache[
            (cache["HomeTeam"] == row["HomeTeam"])
            & (cache["AwayTeam"] == row["AwayTeam"])
            & (cache["Date"].astype(str) == date_str)
        ]
        if hit.empty:
            continue
        matched += 1
        record = hit.iloc[0].to_dict()
        kickoff = row["Date"].to_pydatetime() if hasattr(row["Date"], "to_pydatetime") else datetime.now()
        if intel_is_pit_safe(record, kickoff):
            pit_safe += 1

    n = len(holdout)
    return {
        "holdout_matches": n,
        "cached_rows": len(cache),
        "matched_rows": matched,
        "pit_safe_rows": pit_safe,
        "coverage_pct": 100.0 * matched / n if n else 0.0,
        "pit_safe_pct": 100.0 * pit_safe / n if n else 0.0,
    }


def _mean_intel(cache_row: pd.Series, prefix: str) -> float:
    cols = [f"{prefix}_news_sentiment", f"{prefix}_reddit_sentiment", f"{prefix}_youtube_sentiment"]
    vals = [cache_row.get(c) for c in cols if c in cache_row.index]
    nums = [float(v) for v in vals if v is not None and not pd.isna(v) and float(v) > 0]
    return sum(nums) / len(nums) if nums else NEUTRAL


def intel_sentiment_summary(holdout: pd.DataFrame) -> dict:
    ensure_chaos_cache()
    cache = pd.read_sql("SELECT * FROM chaos_cache", engine)
    if cache.empty:
        return {"samples": 0}

    home_sentiments: list[float] = []
    away_sentiments: list[float] = []
    for _, row in holdout.iterrows():
        date_str = row["Date"].strftime("%d/%m/%Y") if hasattr(row["Date"], "strftime") else str(row["Date"])
        hit = cache[
            (cache["HomeTeam"] == row["HomeTeam"])
            & (cache["AwayTeam"] == row["AwayTeam"])
            & (cache["Date"].astype(str) == date_str)
        ]
        if hit.empty:
            continue
        rec = hit.iloc[0]
        home_sentiments.append(_mean_intel(rec, "home"))
        away_sentiments.append(_mean_intel(rec, "away"))

    if not home_sentiments:
        return {"samples": 0}

    return {
        "samples": len(home_sentiments),
        "home_sentiment_mean": sum(home_sentiments) / len(home_sentiments),
        "away_sentiment_mean": sum(away_sentiments) / len(away_sentiments),
    }


def run_intel_ablation_backtest(
    *,
    limit: int = 200,
    use_cache: bool = True,
) -> dict:
    """Compare selective holdout ROI: baseline (train-off intel) vs neutral intel overlay.

    Training always strips intel (use_intel_in_train=false). Ablation re-scores holdout
    with intel features pinned to neutral 0.5 — should match baseline when model
    was trained without intel signal.
    """
    divs = league_div_codes()
    forger = GameForger()
    forger.train(
        limit=limit,
        use_cache=use_cache,
        div_filter=divs,
        chaos_cache_only=True,
    )

    baseline_preds = forger.backtest_on_holdout(require_edge=True, edge_margin=None)
    baseline_roi = _flat_roi(baseline_preds)

    neutral_preds = forger.backtest_on_holdout(
        require_edge=True,
        edge_margin=None,
        intel_override=NEUTRAL,
    )
    neutral_roi = _flat_roi(neutral_preds)

    return {
        "baseline_picks": len(baseline_preds),
        "baseline_roi_pct": baseline_roi,
        "neutral_picks": len(neutral_preds),
        "neutral_roi_pct": neutral_roi,
        "delta_roi_pct": neutral_roi - baseline_roi,
    }


def _flat_roi(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    stakes = 0.0
    returns = 0.0
    for pred in predictions:
        odds = pred.get("b365") or pred.get("b365_close")
        if not odds:
            continue
        code = pred.get("outcome_code", -1)
        idx = _ODDS_IDX.get(code)
        price = odds[idx] if idx is not None and 0 <= idx < 3 else None
        if not price or price <= 0:
            continue
        stakes += 1.0
        if pred.get("actual_code") == code:
            returns += float(price)
    return 100.0 * (returns - stakes) / stakes if stakes else 0.0


def format_intel_roi_report(
    coverage: dict,
    sentiment: dict,
    ablation: dict | None = None,
) -> str:
    lines = ["Intel ROI / coverage report", "=" * 40]
    lines.append(
        f"Holdout matches: {coverage['holdout_matches']} | "
        f"chaos cache rows: {coverage['cached_rows']}"
    )
    lines.append(
        f"Cache match coverage: {coverage['matched_rows']} "
        f"({coverage['coverage_pct']:.1f}%)"
    )
    lines.append(
        f"PIT-safe intel rows: {coverage['pit_safe_rows']} "
        f"({coverage['pit_safe_pct']:.1f}%)"
    )
    if sentiment.get("samples", 0) > 0:
        lines.append(
            f"Cached sentiment samples: {sentiment['samples']} | "
            f"home mean={sentiment['home_sentiment_mean']:.3f}, "
            f"away mean={sentiment['away_sentiment_mean']:.3f}"
        )
    else:
        lines.append("Cached sentiment samples: 0 (run --predict --refresh-cache to populate)")

    if ablation:
        lines.append("-" * 40)
        lines.append(
            f"Selective holdout ROI — baseline: {ablation['baseline_roi_pct']:+.1f}% "
            f"({ablation['baseline_picks']} picks)"
        )
        lines.append(
            f"Selective holdout ROI — neutral intel overlay: {ablation['neutral_roi_pct']:+.1f}% "
            f"({ablation['neutral_picks']} picks)"
        )
        lines.append(f"Delta: {ablation['delta_roi_pct']:+.1f}%")
    return "\n".join(lines)


def run_intel_roi(*, limit: int = 200, use_cache: bool = True, ablation: bool = True) -> None:
    holdout = _holdout_matches(limit=limit)
    if holdout.empty:
        print("Intel ROI: no completed matches in DB — run --ingest first.")
        return

    coverage = chaos_intel_coverage(holdout)
    sentiment = intel_sentiment_summary(holdout)

    ablation_result = None
    if ablation:
        try:
            ablation_result = run_intel_ablation_backtest(limit=limit, use_cache=use_cache)
        except ValueError as exc:
            print(f"Ablation skipped: {exc}")

    print(format_intel_roi_report(coverage, sentiment, ablation_result))