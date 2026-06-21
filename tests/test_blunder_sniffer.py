from predictors.blunder_sniffer import BlunderSniffer


class _StubForger:
    edge_margin = 0.05

    def backtest_on_holdout(self, confidence_threshold=0.0, edge_margin=None, require_edge=True):
        return [
            {
                "home": "Arsenal",
                "away": "Chelsea",
                "date": "01/01/2026",
                "div": "E0",
                "outcome": "Away Win",
                "outcome_code": 2,
                "actual_code": 2,
                "edge": 0.12,
                "bookie_code": 1,
                "bookie_pick": "Home Win",
            },
            {
                "home": "Liverpool",
                "away": "Everton",
                "date": "02/01/2026",
                "div": "E0",
                "outcome": "Home Win",
                "outcome_code": 1,
                "actual_code": 1,
                "edge": 0.08,
                "bookie_code": 1,
                "bookie_pick": "Home Win",
            },
        ]


def test_find_blunders_model_beats_bookie():
    sniffer = BlunderSniffer()
    results = sniffer.find_blunders(_StubForger(), limit=5)
    assert len(results) == 1
    assert "Arsenal" in results[0]
    assert "beat" in results[0]