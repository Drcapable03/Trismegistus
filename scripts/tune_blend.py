"""Grid-search bookie blend weight on walk-forward league holdout (residual model)."""

from config.settings import (
    bookie_blend_weight,
    edge_margin_min,
    league_div_codes,
    league_summary,
    set_bookie_blend_weight,
)
from evaluation.implied_odds import bookie_accuracy
from predictors.game_forger import GameForger
from utils.db import read_matches


def tune_blend(
    limit: int = 500,
    use_cache: bool = True,
    step: float = 0.05,
    persist: bool = True,
) -> tuple[float, dict[float, float]]:
    div_codes = league_div_codes()
    margin = edge_margin_min()
    print(league_summary())
    print(
        f"Tuning blend on walk-forward holdout "
        f"(residual model, div={div_codes}, limit={limit})"
    )

    forger = GameForger()
    forger.train(
        limit=limit,
        use_cache=use_cache,
        div_filter=div_codes,
        chaos_cache_only=True,
    )

    model_only = forger.holdout_blend_accuracy(blend_weight=0.0)
    bookie_only = forger.holdout_blend_accuracy(blend_weight=1.0)
    print(f"Baseline — residual model only (w=0.0): {model_only:.1f}%")
    print(f"Baseline — bookie only (w=1.0): {bookie_only:.1f}%")

    weights = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    scores: dict[float, float] = {}
    selective_scores: dict[float, float] = {}
    for w in weights:
        scores[w] = forger.holdout_blend_accuracy(blend_weight=w)
        selective_scores[w] = forger.holdout_blend_accuracy(
            blend_weight=w, require_edge=True,
        )

    best_weight = max(scores, key=scores.get)
    best_acc = scores[best_weight]
    best_sel_weight = max(selective_scores, key=selective_scores.get)
    current = bookie_blend_weight()

    print("\nBlend weight sweep (all holdout picks):")
    for w in sorted(scores):
        marker = " <-- best" if w == best_weight else ""
        sel = selective_scores[w]
        print(f"  w={w:.2f}: {scores[w]:.1f}% (selective ≥{margin:.0%}: {sel:.1f}%){marker}")

    matches = read_matches()
    completed = matches[matches["Div"].isin(div_codes) & matches["FTR"].isin(["H", "D", "A"])]
    if not completed.empty:
        bookie_acc = bookie_accuracy(completed.tail(limit))
        print(f"\nBookie accuracy (full sample, not holdout): {bookie_acc:.1f}%")

    print(f"\nCurrent config weight: {current:.3f}")
    print(f"Optimal walk-forward weight (all picks): {best_weight:.2f} ({best_acc:.1f}%)")
    print(
        f"Best selective weight (edge ≥{margin:.0%}): "
        f"{best_sel_weight:.2f} ({selective_scores[best_sel_weight]:.1f}%)"
    )

    if persist and best_weight != current:
        set_bookie_blend_weight(best_weight)
        print(f"Saved bookie_blend_weight={best_weight:.3f} to config/leagues.yaml")
    elif persist:
        print("Config already at optimal weight — no change.")

    return best_weight, scores


if __name__ == "__main__":
    tune_blend()