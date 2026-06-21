from evaluation.kelly import kelly_simulation, kelly_stake_fraction


def test_kelly_stake_zero_without_edge():
    assert kelly_stake_fraction(0.4, 2.0) == 0.0


def test_kelly_stake_positive_with_edge():
    stake = kelly_stake_fraction(0.6, 2.0, fraction=0.25)
    assert stake > 0.0


def test_kelly_simulation_profit_on_winning_pick():
    preds = [{
        "outcome_code": 1,
        "actual_code": 1,
        "probs": {"H": 0.6, "D": 0.2, "A": 0.2},
        "b365_close": (1.8, 3.5, 4.0),
    }]
    sim = kelly_simulation(preds, initial_bankroll=100.0, kelly_fraction=0.25)
    assert sim["bets"] == 1
    assert sim["final_bankroll"] > 100.0