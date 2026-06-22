"""Grid-search Dixon-Coles blend weight on walk-forward holdout (selective ROI)."""

from config.settings import (
    dixon_coles_blend_weight,
    edge_margin_min,
    league_div_codes,
    league_summary,
    set_dixon_coles_blend_weight,
)
from evaluation.metrics import summarize_predictions
from predictors.game_forger import GameForger


def tune_dc_blend(
    limit: int = 500,
    use_cache: bool = True,
    step: float = 0.05,
    max_weight: float = 0.30,
    persist: bool = True,
) -> tuple[float, dict[float, dict]]:
    div_codes = league_div_codes()
    margin = edge_margin_min()
    print(league_summary())
    print(
        f"Tuning Dixon-Coles blend on walk-forward holdout "
        f"(div={div_codes}, limit={limit})"
    )

    forger = GameForger()
    forger.train(
        limit=limit,
        use_cache=use_cache,
        div_filter=div_codes,
        chaos_cache_only=True,
    )

    original_dc = forger.dc_blend_weight
    forger.dc_blend_weight = 0.0
    no_dc = forger.holdout_blend_accuracy()
    forger.dc_blend_weight = original_dc
    with_dc = forger.holdout_blend_accuracy()
    print(f"Baseline — residual+bookie, DC w=0.0: {no_dc:.1f}%")
    print(f"Baseline — current DC w={original_dc:.2f}: {with_dc:.1f}%")

    weights = [round(i * step, 2) for i in range(int(max_weight / step) + 1)]
    acc_scores: dict[float, float] = {}
    results: dict[float, dict] = {}
    for w in weights:
        forger.dc_blend_weight = w
        acc_scores[w] = forger.holdout_blend_accuracy()
        preds = forger.backtest_on_holdout(
            confidence_threshold=0.0,
            edge_margin=margin,
            require_edge=True,
        )
        results[w] = summarize_predictions(preds, label=f"dc_w={w:.2f}")

    def _roi_key(w: float) -> float:
        s = results[w]
        if s.get("roi_bets", 0) < 5:
            return -999.0
        return s.get("roi_pct", -999.0)

    best_weight = max(weights, key=_roi_key)
    best_acc_weight = max(acc_scores, key=acc_scores.get)
    current = dixon_coles_blend_weight()

    print("\nDC blend weight sweep:")
    for w in sorted(weights):
        s = results[w]
        marker = " <-- best ROI" if w == best_weight else ""
        acc = acc_scores[w]
        if s.get("n", 0) == 0:
            print(f"  w={w:.2f}: acc {acc:.1f}%, no selective picks{marker}")
            continue
        print(
            f"  w={w:.2f}: acc {acc:.1f}%, "
            f"{s['n']} picks, ROI {s['roi_pct']:+.1f}% "
            f"({s['roi_bets']} bets){marker}"
        )

    best = results[best_weight]
    print(f"\nCurrent dixon_coles_blend_weight: {current:.3f}")
    print(
        f"Best accuracy weight: {best_acc_weight:.2f} "
        f"({acc_scores[best_acc_weight]:.1f}%)"
    )
    if best.get("roi_bets", 0) < 5:
        print("Insufficient selective bets at any weight — config unchanged.")
        return current, results

    print(
        f"Best ROI weight: {best_weight:.2f} — "
        f"ROI {best['roi_pct']:+.1f}% on {best['n']} picks"
    )

    if persist and best_weight != current:
        set_dixon_coles_blend_weight(best_weight)
        print(f"Saved dixon_coles_blend_weight={best_weight:.3f} to config/leagues.yaml")
    elif persist:
        print("Config already at best ROI weight — no change.")

    return best_weight, results


if __name__ == "__main__":
    tune_dc_blend()