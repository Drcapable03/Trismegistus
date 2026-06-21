"""Per-league edge margin tuning on selective holdout ROI."""

from config.settings import (
    edge_margin_for_div,
    league_div_codes,
    league_summary,
    per_league_edge_margins,
    set_per_league_edge_margins,
)
from evaluation.metrics import summarize_predictions
from predictors.game_forger import GameForger


def tune_edge_leagues(
    limit: int = 500,
    use_cache: bool = True,
    step: float = 0.01,
    min_margin: float = 0.03,
    max_margin: float = 0.15,
    persist: bool = True,
) -> dict[str, float]:
    div_codes = league_div_codes()
    print(league_summary())
    print(f"Per-league edge margin tuning (selective ROI, limit={limit})")

    forger = GameForger()
    forger.train(
        limit=limit,
        use_cache=use_cache,
        div_filter=div_codes,
        chaos_cache_only=True,
    )

    if forger.test_data is None or forger.context is None:
        raise ValueError("No holdout data after training")

    margins = []
    m = min_margin
    while m <= max_margin + 1e-9:
        margins.append(round(m, 3))
        m += step

    tuned: dict[str, float] = {}
    current = per_league_edge_margins()

    for div in div_codes:
        best_margin = edge_margin_for_div(div)
        best_roi = -999.0
        for margin in margins:
            preds = forger.backtest_on_holdout(
                confidence_threshold=0.0,
                edge_margin=margin,
                require_edge=True,
                div_filter=div,
            )
            summary = summarize_predictions(preds)
            if summary.get("roi_bets", 0) < 3:
                continue
            roi = summary.get("roi_pct", -999.0)
            if roi > best_roi:
                best_roi = roi
                best_margin = margin
        tuned[div] = best_margin
        n_preds = summarize_predictions(
            forger.backtest_on_holdout(
                confidence_threshold=0.0,
                edge_margin=best_margin,
                require_edge=True,
                div_filter=div,
            )
        ).get("n", 0)
        print(
            f"  {div}: margin={best_margin:.0%} "
            f"(ROI {best_roi:+.1f}%, {n_preds} picks)"
        )

    if persist:
        set_per_league_edge_margins(tuned)
        print("Saved per_league_edge_margin to config/leagues.yaml")

    forger.per_league_edge_margin = tuned
    return tuned


if __name__ == "__main__":
    tune_edge_leagues()