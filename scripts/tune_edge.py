"""Grid-search edge margin on selective holdout ROI (not raw accuracy)."""

from config.settings import edge_margin_min, league_div_codes, league_summary, set_edge_margin_min
from evaluation.metrics import summarize_predictions
from predictors.game_forger import GameForger


def tune_edge(
    limit: int = 500,
    use_cache: bool = True,
    step: float = 0.01,
    min_margin: float = 0.02,
    max_margin: float = 0.15,
    persist: bool = True,
) -> tuple[float, dict[float, dict]]:
    div_codes = league_div_codes()
    print(league_summary())
    print(f"Tuning edge margin on selective holdout ROI (limit={limit})")

    forger = GameForger()
    forger.train(
        limit=limit,
        use_cache=use_cache,
        div_filter=div_codes,
        chaos_cache_only=True,
    )

    margins = []
    m = min_margin
    while m <= max_margin + 1e-9:
        margins.append(round(m, 3))
        m += step

    results: dict[float, dict] = {}
    for margin in margins:
        preds = forger.backtest_on_holdout(
            confidence_threshold=0.0,
            edge_margin=margin,
            require_edge=True,
        )
        summary = summarize_predictions(preds, label=f"margin={margin:.0%}")
        results[margin] = summary

    def _roi_key(m: float) -> float:
        s = results[m]
        if s.get("roi_bets", 0) < 5:
            return -999.0
        return s.get("roi_pct", -999.0)

    best_margin = max(margins, key=_roi_key)
    current = edge_margin_min()

    print("\nEdge margin sweep (selective picks):")
    for margin in margins:
        s = results[margin]
        marker = " <-- best ROI" if margin == best_margin else ""
        if s.get("n", 0) == 0:
            print(f"  margin={margin:.0%}: no picks{marker}")
            continue
        print(
            f"  margin={margin:.0%}: {s['n']} picks, "
            f"acc {s['accuracy']:.1f}%, ROI {s['roi_pct']:+.1f}% "
            f"({s['roi_bets']} bets){marker}"
        )

    best = results[best_margin]
    print(f"\nCurrent edge_margin_min: {current:.3f}")
    if best.get("n", 0):
        print(
            f"Best ROI margin: {best_margin:.0%} — "
            f"ROI {best['roi_pct']:+.1f}% on {best['n']} picks"
        )
    else:
        print("No selective picks at any margin — config unchanged.")
        return current, results

    if persist and best_margin != current:
        set_edge_margin_min(best_margin)
        print(f"Saved edge_margin_min={best_margin:.3f} to config/leagues.yaml")
    elif persist:
        print("Config already at best ROI margin — no change.")

    return best_margin, results


if __name__ == "__main__":
    tune_edge()