"""Phase 6F: xG enrichment coverage and holdout ablation report."""

from __future__ import annotations

import pandas as pd

from config.settings import league_div_codes, xg_source_priority
from predictors.game_forger import GameForger, _is_completed, _parse_dates
from utils.db import read_matches
from utils.fbref_cache import fbref_cache_stats, load_fbref_xg
from utils.statsbomb_cache import load_statsbomb_xg, statsbomb_cache_stats
from utils.xg_cache import load_xg_matches, xg_cache_stats


def _holdout_matches(limit: int | None = None) -> pd.DataFrame:
    divs = league_div_codes()
    matches = _parse_dates(read_matches())
    matches = matches[matches["Div"].isin(divs)]
    completed = matches[_is_completed(matches)].sort_values("Date", ascending=False)
    if limit:
        completed = completed.head(limit)
    return completed.sort_values("Date")


def _match_in_cache(row: pd.Series, cache: pd.DataFrame) -> bool:
    if cache.empty:
        return False
    cache = cache.copy()
    cache["_dt"] = pd.to_datetime(cache["MatchDate"], format="%d/%m/%Y", errors="coerce")
    row_dt = row["Date"]
    hits = cache[
        (cache["HomeTeam"] == row["HomeTeam"])
        & (cache["AwayTeam"] == row["AwayTeam"])
        & (cache["_dt"] == row_dt)
    ]
    return not hits.empty


def enrichment_coverage(holdout: pd.DataFrame) -> dict[str, dict]:
    loaders = {
        "understat": load_xg_matches,
        "statsbomb": load_statsbomb_xg,
        "fbref": load_fbref_xg,
    }
    n = len(holdout)
    report: dict[str, dict] = {}
    for source in xg_source_priority():
        loader = loaders.get(source)
        if loader is None:
            continue
        cache = loader()
        matched = sum(1 for _, row in holdout.iterrows() if _match_in_cache(row, cache))
        report[source] = {
            "cache_rows": len(cache),
            "holdout_matched": matched,
            "coverage_pct": 100.0 * matched / n if n else 0.0,
        }
    return report


def run_enrichment_ablation(*, limit: int = 200, use_cache: bool = True) -> dict:
    div_codes = league_div_codes()
    results: dict[str, float] = {}

    for label, enrichment_xg in (("enriched", True), ("shots_only", False)):
        forger = GameForger()
        forger.train(
            limit=limit,
            use_cache=use_cache,
            div_filter=div_codes,
            chaos_cache_only=True,
            enrichment_xg=enrichment_xg,
        )
        preds = forger.backtest_on_holdout(require_edge=True)
        results[f"{label}_picks"] = float(len(preds))
        results[f"{label}_roi_pct"] = _flat_roi(preds)

    results["delta_roi_pct"] = results["enriched_roi_pct"] - results["shots_only_roi_pct"]
    return results


def _flat_roi(predictions: list[dict]) -> float:
    stakes = 0.0
    returns = 0.0
    for pred in predictions:
        odds = pred.get("b365") or pred.get("b365_close")
        if not odds:
            continue
        code = pred.get("outcome_code", -1)
        idx = {0: 1, 1: 0, 2: 2}.get(code)
        price = odds[idx] if idx is not None and 0 <= idx < 3 else None
        if not price or price <= 0:
            continue
        stakes += 1.0
        if pred.get("actual_code") == code:
            returns += float(price)
    return 100.0 * (returns - stakes) / stakes if stakes else 0.0


def format_enrichment_report(
    coverage: dict[str, dict],
    cache_stats: dict[str, int],
    ablation: dict | None = None,
) -> str:
    lines = ["Enrichment ROI / coverage report", "=" * 40]
    lines.append(f"xG source priority: {', '.join(xg_source_priority())}")
    lines.append(
        f"Cache totals — understat: {cache_stats.get('understat', 0)}, "
        f"statsbomb: {cache_stats.get('statsbomb', 0)}, "
        f"fbref: {cache_stats.get('fbref', 0)}"
    )
    for source, stats in coverage.items():
        lines.append(
            f"{source}: {stats['holdout_matched']} holdout hits "
            f"({stats['coverage_pct']:.1f}%) from {stats['cache_rows']} cached rows"
        )
    if ablation:
        lines.append("-" * 40)
        lines.append(
            f"Selective holdout ROI — enriched xG: {ablation['enriched_roi_pct']:+.1f}% "
            f"({int(ablation['enriched_picks'])} picks)"
        )
        lines.append(
            f"Selective holdout ROI — shots-only xG: {ablation['shots_only_roi_pct']:+.1f}% "
            f"({int(ablation['shots_only_picks'])} picks)"
        )
        lines.append(f"Delta: {ablation['delta_roi_pct']:+.1f}%")
    return "\n".join(lines)


def run_enrichment_roi(*, limit: int = 200, use_cache: bool = True, ablation: bool = True) -> None:
    holdout = _holdout_matches(limit=limit)
    if holdout.empty:
        print("Enrichment ROI: no completed matches in DB — run --ingest first.")
        return

    coverage = enrichment_coverage(holdout)
    cache_stats = {
        "understat": xg_cache_stats().get("understat_matches", 0),
        "statsbomb": statsbomb_cache_stats().get("statsbomb_matches", 0),
        "fbref": fbref_cache_stats().get("fbref_matches", 0),
    }

    ablation_result = None
    if ablation:
        try:
            ablation_result = run_enrichment_ablation(limit=limit, use_cache=use_cache)
        except ValueError as exc:
            print(f"Ablation skipped: {exc}")

    print(format_enrichment_report(coverage, cache_stats, ablation_result))


if __name__ == "__main__":
    run_enrichment_roi()