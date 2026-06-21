"""Per-league bookie blend tuning on walk-forward holdout."""

from config.settings import (
    league_div_codes,
    league_summary,
    per_league_blend_weights,
    set_per_league_blend_weights,
)
from predictors.game_forger import GameForger


def tune_leagues(
    limit: int = 500,
    use_cache: bool = True,
    step: float = 0.05,
    persist: bool = True,
) -> dict[str, float]:
    div_codes = league_div_codes()
    print(league_summary())
    print(f"Per-league blend tuning (holdout subsets, limit={limit})")

    forger = GameForger()
    forger.train(
        limit=limit,
        use_cache=use_cache,
        div_filter=div_codes,
        chaos_cache_only=True,
    )

    if forger.test_data is None or forger.context is None:
        raise ValueError("No holdout data after training")

    X_test_o, y_test_o, _, _ = forger.test_data
    weights = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    tuned: dict[str, float] = {}
    current = per_league_blend_weights()

    from predictors.game_forger import _b365_from_row

    for div in div_codes:
        positions = [
            i for i in range(len(forger.context))
            if forger.context.iloc[i].get("Div") == div
        ]
        if not positions:
            tuned[div] = current.get(div, forger.bookie_blend_weight)
            continue
        best_w, best_acc = forger.bookie_blend_weight, -1.0
        for w in weights:
            correct = 0
            for i in positions:
                row = forger.context.iloc[i]
                evaluated = forger._evaluate_row(
                    X_test_o.iloc[i],
                    _b365_from_row(row),
                    w,
                    0.0,
                    require_edge=False,
                    div=div,
                )
                if evaluated and evaluated[0] == int(y_test_o.iloc[i]):
                    correct += 1
            acc = correct / len(positions) * 100
            if acc > best_acc:
                best_acc, best_w = acc, w
        tuned[div] = best_w
        print(f"  {div}: w={best_w:.2f} ({best_acc:.1f}% on {len(positions)} holdout matches)")

    if persist:
        set_per_league_blend_weights(tuned)
        print("Saved per_league_blend to config/leagues.yaml")

    forger.per_league_blend = tuned
    return tuned


if __name__ == "__main__":
    tune_leagues()