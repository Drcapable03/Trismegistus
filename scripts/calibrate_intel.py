"""Phase 6C: intel source calibration and pre-flight checks."""

from __future__ import annotations

import os
from dataclasses import dataclass

from agents.intel_agent import fetch_team_intel
from config.settings import (
    intel_calibration_probes,
    intel_config,
    league_div_codes,
    use_intel_in_train,
    youtube_channels_for_div,
)
from scrapers.reddit_scraper import reddit_credentials_configured, verify_reddit_connection


@dataclass
class IntelCheck:
    name: str
    passed: bool
    detail: str


def _scrapling_enabled() -> bool:
    return os.getenv("TRIS_USE_SCRAPLING", "true").lower() in {"1", "true", "yes"}


def run_static_checks() -> list[IntelCheck]:
    checks: list[IntelCheck] = []
    cfg = intel_config()

    checks.append(IntelCheck(
        "intel_train_policy",
        not use_intel_in_train(),
        f"use_intel_in_train={use_intel_in_train()} (live-only expected)",
    ))

    news_cfg = cfg.get("news") or {}
    checks.append(IntelCheck(
        "news_enabled",
        bool(news_cfg.get("enabled", True)),
        f"news enabled={news_cfg.get('enabled', True)}, scrapling={_scrapling_enabled()}",
    ))

    reddit_cfg = cfg.get("reddit") or {}
    reddit_on = bool(reddit_cfg.get("enabled", True))
    creds = reddit_credentials_configured()
    checks.append(IntelCheck(
        "reddit_credentials",
        not reddit_on or creds,
        "credentials set" if creds else "REDDIT_CLIENT_ID/SECRET missing (neutral 0.5 fallback)",
    ))

    if reddit_on and creds:
        ok, detail = verify_reddit_connection()
        checks.append(IntelCheck("reddit_api", ok, detail))

    youtube_cfg = cfg.get("youtube") or {}
    channels_ok = all(
        youtube_channels_for_div(div) for div in league_div_codes()
    )
    checks.append(IntelCheck(
        "youtube_channels",
        channels_ok,
        f"channels_by_div configured for Big 5: {channels_ok}",
    ))

    probes = intel_calibration_probes()
    checks.append(IntelCheck(
        "calibration_probes",
        len(probes) >= 1,
        f"{len(probes)} sample probe(s) in config/intel.yaml",
    ))

    return checks


def probe_live_intel(live_fetch: bool = False) -> list[dict]:
    """Fetch intel for calibration probes (mock-free when live_fetch=True)."""
    results = []
    for probe in intel_calibration_probes():
        team = probe["team"]
        opponent = probe.get("opponent")
        div = probe.get("div")
        if live_fetch and _scrapling_enabled():
            intel = fetch_team_intel(team, "2026-01-01", opponent=opponent, div_code=div)
        else:
            intel = {
                "news_attention": 0.0,
                "news_sentiment": 0.5,
                "reddit_sentiment": 0.5,
                "youtube_sentiment": 0.5,
            }
        results.append({
            "team": team,
            "opponent": opponent,
            "div": div,
            **intel,
        })
    return results


def format_intel_report(checks: list[IntelCheck], probes: list[dict] | None = None) -> str:
    lines = ["Intel calibration", "=" * 40]
    passed = sum(1 for c in checks if c.passed)
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"[{mark}] {check.name}: {check.detail}")

    if probes:
        lines.append("-" * 40)
        lines.append("Sample probe summary:")
        for row in probes:
            lines.append(
                f"  {row['team']} vs {row.get('opponent', '?')} [{row.get('div', '?')}]: "
                f"news={row['news_attention']:.2f}/{row['news_sentiment']:.2f}, "
                f"reddit={row['reddit_sentiment']:.2f}, youtube={row['youtube_sentiment']:.2f}"
            )

    lines.append("-" * 40)
    lines.append(f"Result: {passed}/{len(checks)} checks passed")
    return "\n".join(lines)


def run_calibrate_intel(*, live_fetch: bool = False) -> int:
    checks = run_static_checks()
    probes = probe_live_intel(live_fetch=live_fetch) if live_fetch else None
    print(format_intel_report(checks, probes))

    critical = [c for c in checks if c.name in {"intel_train_policy", "calibration_probes"} and not c.passed]
    optional_fail = [c for c in checks if c not in critical and not c.passed]
    if critical:
        return 1
    if optional_fail:
        print(f"Note: {len(optional_fail)} optional source(s) need attention (see above).")
    return 0