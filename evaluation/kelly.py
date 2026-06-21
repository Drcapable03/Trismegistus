"""Fractional Kelly staking simulation for holdout evaluation."""

OUTCOME_TO_KEY = {0: "D", 1: "H", 2: "A"}


def kelly_stake_fraction(
    prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
    max_stake: float = 0.05,
) -> float:
    """Return bankroll fraction to stake (0 if no edge)."""
    if decimal_odds <= 1.0 or prob <= 0.0:
        return 0.0
    edge = prob * decimal_odds - 1.0
    if edge <= 0.0:
        return 0.0
    full_kelly = edge / (decimal_odds - 1.0)
    stake = max(0.0, fraction * full_kelly)
    return min(stake, max_stake)


def _pick_odds(prediction: dict, line: str = "close") -> tuple[float, int] | None:
    if line == "open":
        b365 = prediction.get("b365_open") or prediction.get("b365")
    else:
        b365 = prediction.get("b365_close") or prediction.get("b365")
    if b365 is None:
        return None
    code = prediction["outcome_code"]
    h, d, a = b365
    odds = {0: d, 1: h, 2: a}[code]
    if odds <= 0:
        return None
    return float(odds), code


def kelly_simulation(
    predictions: list[dict],
    initial_bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.05,
    odds_line: str = "close",
) -> dict:
    """Simulate fractional Kelly bankroll through chronological picks."""
    bankroll = float(initial_bankroll)
    peak = bankroll
    max_drawdown = 0.0
    bets = 0
    staked = 0.0

    for pred in predictions:
        probs = pred.get("probs") or {}
        pick = OUTCOME_TO_KEY[pred["outcome_code"]]
        prob = float(probs.get(pick, 0.0))
        odds_info = _pick_odds(pred, line=odds_line)
        if odds_info is None:
            continue
        odds, _ = odds_info
        stake_frac = kelly_stake_fraction(prob, odds, kelly_fraction, max_stake)
        if stake_frac <= 0.0:
            continue

        stake = bankroll * stake_frac
        staked += stake
        bets += 1
        if pred["outcome_code"] == pred["actual_code"]:
            bankroll += stake * (odds - 1.0)
        else:
            bankroll -= stake

        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

    profit = bankroll - initial_bankroll
    roi = (profit / initial_bankroll * 100) if initial_bankroll else 0.0
    return {
        "final_bankroll": bankroll,
        "profit_units": profit,
        "roi_pct": roi,
        "bets": bets,
        "total_staked": staked,
        "max_drawdown_pct": max_drawdown * 100,
        "odds_line": odds_line,
    }