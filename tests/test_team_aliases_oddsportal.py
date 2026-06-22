from utils.team_aliases import teams_match, to_oddsportal


def test_to_oddsportal_maps_football_data_names():
    assert to_oddsportal("Man United") == "Manchester Utd"
    assert to_oddsportal("Nott'm Forest") == "Nottingham"


def test_teams_match_alias_pairs():
    assert teams_match("Man United", "Manchester Utd")
    assert teams_match("Newcastle", "Newcastle Utd")
    assert teams_match("Nott'm Forest", "Nottingham")
    assert not teams_match("Arsenal", "Chelsea")