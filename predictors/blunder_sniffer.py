"""Find matches where the model disagreed with the bookie and was right."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluation.implied_odds import bookie_favorite

if TYPE_CHECKING:
    from predictors.game_forger import GameForger


class BlunderSniffer:
    """Phase 3: uses trained GameForger edge picks, not static implied rules."""

    def find_blunders(
        self,
        forger: GameForger,
        limit: int = 10,
        edge_margin: float | None = None,
    ) -> list[str]:
        margin = edge_margin if edge_margin is not None else forger.edge_margin
        preds = forger.backtest_on_holdout(
            confidence_threshold=0.0,
            edge_margin=margin,
            require_edge=True,
        )
        hits = []
        for p in preds:
            if p.get("bookie_code") is None:
                continue
            model_right = p["outcome_code"] == p["actual_code"]
            disagreed = p["outcome_code"] != p["bookie_code"]
            if model_right and disagreed:
                hits.append(p)

        hits.sort(key=lambda x: x.get("edge", 0), reverse=True)
        results = []
        for p in hits[:limit]:
            results.append(
                f"{p['home']} vs {p['away']} ({p['date']}) [{p.get('div', '?')}]: "
                f"model={p['outcome']} (edge {p.get('edge', 0):.1%}) beat "
                f"bookie={p.get('bookie_pick', '?')} (actual={p['actual_code']})"
            )
        return results

    def find_bookie_fav_failures(self, limit: int = 10) -> list[str]:
        """Legacy helper: heavy home favorites that lost (reference only)."""
        from utils.db import read_matches

        matches = read_matches()
        completed = matches[matches["FTR"].isin(["H", "D", "A"])].copy()
        if completed.empty or "B365H" not in completed.columns:
            return []

        completed["implied_prob_H"] = 1 / completed["B365H"].replace(0, float("nan"))
        blunders = completed[
            (completed["FTR"] != "H") & (completed["implied_prob_H"] > 0.7)
        ].head(limit)
        return [
            f"{r['HomeTeam']} vs {r['AwayTeam']} ({r['Date']}): "
            f"bookie favored home (>70%), actual={r['FTR']}"
            for _, r in blunders.iterrows()
        ]