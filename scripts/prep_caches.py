"""Operational prep: populate xG, StatsBomb, and chaos weather caches before match weeks."""

from __future__ import annotations

from config.settings import historical_understat_seasons, league_div_codes, league_summary
from predictors.game_forger import _is_completed, _parse_dates
from utils.chaos_cache import cache_stats
from utils.db import read_matches
from utils.fbref_cache import fbref_cache_stats
from utils.statsbomb_cache import statsbomb_cache_stats
from utils.xg_cache import xg_cache_stats


def cache_snapshot() -> dict:
    return {
        "understat": xg_cache_stats().get("understat_matches", 0),
        "statsbomb": statsbomb_cache_stats().get("statsbomb_matches", 0),
        "fbref": fbref_cache_stats().get("fbref_matches", 0),
        "chaos": cache_stats().get("cached_matches", 0),
    }


def holdout_overlap_report() -> str:
    divs = league_div_codes()
    matches = _parse_dates(read_matches())
    completed = matches[matches["Div"].isin(divs) & _is_completed(matches)]
    n = len(completed)
    if n == 0:
        return "No completed Big 5 matches in DB."

    from scripts.enrichment_roi import enrichment_coverage

    coverage = enrichment_coverage(completed.tail(500))
    lines = [f"Holdout overlap (last 500 completed, n={min(500, n)}):"]
    for source, stats in coverage.items():
        lines.append(
            f"  {source}: {stats['holdout_matched']} hits ({stats['coverage_pct']:.1f}%)"
        )
    return "\n".join(lines)


def print_prep_report(label: str = "Cache snapshot") -> None:
    snap = cache_snapshot()
    print(f"\n{label}")
    print(f"  understat xG: {snap['understat']}")
    print(f"  statsbomb xG: {snap['statsbomb']}")
    print(f"  fbref xG:     {snap['fbref']}")
    print(f"  chaos cache:  {snap['chaos']}")
    print(holdout_overlap_report())


def prep_caches(
    *,
    fetch_understat: bool = True,
    fetch_statsbomb_data: bool = True,
    statsbomb_limit: int = 0,
    archive_weather: bool = True,
    archive_limit: int | None = None,
    archive_sleep_s: float = 0.3,
) -> dict:
    print(league_summary())
    print_prep_report("Before prep")

    if fetch_understat:
        from scripts.fetch_xg import fetch_xg
        print(f"\n--- Understat xG (seasons: {historical_understat_seasons()}) ---")
        fetch_xg()

    if fetch_statsbomb_data:
        from scripts.fetch_statsbomb import fetch_statsbomb
        print("\n--- StatsBomb open data xG ---")
        fetch_statsbomb(max_matches_per_season=statsbomb_limit)

    if archive_weather:
        from scripts.archive_chaos import archive_chaos
        print("\n--- Chaos weather archive (PIT-safe, no live intel) ---")
        archive_chaos(
            div_filter=league_div_codes(),
            limit=archive_limit,
            sleep_s=archive_sleep_s,
        )

    print_prep_report("After prep")
    print(
        "\nNote: live intel (news/Reddit/YouTube) fills on "
        "`poetry run python main.py --predict --refresh-cache` when fixtures exist."
    )
    return cache_snapshot()


if __name__ == "__main__":
    prep_caches(archive_limit=500)