"""Fetch Understat league match xG into SQLite cache."""

from understatapi import UnderstatClient

from config.settings import historical_understat_seasons, league_div_codes, understat_league_map
from utils.xg_cache import parse_understat_match, save_match_rows, xg_cache_stats


def fetch_xg(seasons: list[str] | None = None, div_filter: list[str] | None = None) -> int:
    div_codes = div_filter or league_div_codes()
    season_list = seasons or historical_understat_seasons()
    league_map = understat_league_map()
    total = 0

    with UnderstatClient() as client:
        for div in div_codes:
            league = league_map.get(div)
            if not league:
                print(f"  {div}: no Understat mapping — skipped")
                continue
            for season in season_list:
                try:
                    matches = client.league(league).get_match_data(season=season)
                except Exception as e:
                    print(f"  {div}/{season}: fetch failed ({e})")
                    continue
                rows = []
                for m in matches:
                    parsed = parse_understat_match(m, div, season)
                    if parsed:
                        rows.append(parsed)
                saved = save_match_rows(rows)
                total += saved
                print(f"  {div}/{season}: cached {saved} matches ({league})")

    stats = xg_cache_stats()
    print(f"Understat xG cache: {stats['understat_matches']} matches total")
    return total


if __name__ == "__main__":
    fetch_xg()