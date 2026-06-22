"""Phase 6E: verify historical season URLs and report DB coverage."""

from __future__ import annotations

import re

import pandas as pd
import requests

from config.settings import (
    all_seasons,
    current_season,
    football_data_league_url,
    league_div_codes,
    league_summary,
    league_urls,
)
from predictors.game_forger import _is_completed, _parse_dates
from utils.db import read_matches


def season_from_league_label(label: str) -> str | None:
    match = re.search(r"\((\d{4})\)\s*$", label)
    return match.group(1) if match else None


def verify_season_urls(timeout: float = 15.0) -> dict[str, bool]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    results: dict[str, bool] = {}
    for label, url in league_urls().items():
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            ok = (
                response.status_code == 200
                and "text/csv" in response.headers.get("Content-Type", "").lower()
                and b"Div,Date" in response.content[:200]
            )
        except requests.RequestException:
            ok = False
        results[label] = ok
    return results


def season_coverage(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    df = read_matches() if matches is None else matches
    if df.empty:
        return pd.DataFrame(columns=["Season", "Div", "completed", "total"])

    df = _parse_dates(df.copy())
    codes = league_div_codes()
    scoped = df[df["Div"].isin(codes)] if "Div" in df.columns else df
    if scoped.empty:
        return pd.DataFrame(columns=["Season", "Div", "completed", "total"])

    if "Season" not in scoped.columns:
        scoped = scoped.copy()
        scoped["Season"] = "unknown"

    completed_mask = _is_completed(scoped)
    grouped = (
        scoped.assign(_completed=completed_mask.astype(int))
        .groupby(["Season", "Div"], dropna=False)
        .agg(total=("Div", "size"), completed=("_completed", "sum"))
        .reset_index()
        .sort_values(["Season", "Div"])
    )
    grouped["Season"] = grouped["Season"].astype(str)
    return grouped


def expand_history(verify_urls: bool = True) -> dict:
    print(league_summary())
    configured = all_seasons()
    print(f"Configured seasons ({len(configured)}): {', '.join(configured)}")

    url_health: dict[str, bool] = {}
    if verify_urls:
        print(f"\nVerifying {len(league_urls())} football-data CSV URLs...")
        url_health = verify_season_urls()
        ok = sum(url_health.values())
        print(f"Reachable CSVs: {ok}/{len(url_health)}")
        for label, healthy in sorted(url_health.items()):
            status = "ok" if healthy else "FAIL"
            print(f"  [{status}] {label}")

    matches = read_matches()
    coverage = season_coverage(matches)
    codes = league_div_codes()

    print("\nDB coverage (Big 5 completed matches):")
    if coverage.empty:
        print("  No matches in DB — run `poetry run python main.py --ingest`")
    else:
        for season in configured:
            season_rows = coverage[coverage["Season"] == season]
            if season_rows.empty:
                print(f"  {season}: 0 completed (not ingested)")
                continue
            completed = int(season_rows["completed"].sum())
            total = int(season_rows["total"].sum())
            per_div = ", ".join(
                f"{row.Div}:{int(row.completed)}"
                for row in season_rows.itertuples()
            )
            print(f"  {season}: {completed} completed / {total} rows ({per_div})")

        unknown = coverage[coverage["Season"] == "unknown"]
        if not unknown.empty:
            print(
                f"  unknown season tag: {int(unknown['completed'].sum())} completed "
                "(re-ingest with --reset to tag)"
            )

    completed_total = 0
    if not matches.empty and "Div" in matches.columns:
        scoped = matches[matches["Div"].isin(codes)]
        completed_total = int(_is_completed(scoped).sum())

    expected_urls = len(codes) * len(configured)
    return {
        "configured_seasons": configured,
        "current_season": current_season(),
        "url_health": url_health,
        "coverage": coverage,
        "completed_big5": completed_total,
        "expected_league_urls": expected_urls,
    }


if __name__ == "__main__":
    expand_history()