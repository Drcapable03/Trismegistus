"""Phase 6F: cache FBref match xG via penaltyblog (may be blocked — degrades gracefully)."""

from config.settings import (
    all_seasons,
    fbref_competition_map,
    fbref_season_label,
    league_div_codes,
)
from utils.fbref_cache import fbref_cache_stats, parse_fbref_fixtures, save_match_rows


def fetch_fbref(
    seasons: list[str] | None = None,
    div_filter: list[str] | None = None,
) -> int:
    try:
        from penaltyblog.scrapers.fbref import FBRef
    except ImportError as exc:
        print(f"FBref fetch unavailable: {exc}")
        return 0

    div_codes = div_filter or league_div_codes()
    season_list = seasons or all_seasons()
    comp_map = fbref_competition_map()
    total = 0

    for div in div_codes:
        competition = comp_map.get(div)
        if not competition:
            print(f"  {div}: no FBref mapping — skipped")
            continue
        for season in season_list:
            label = fbref_season_label(season)
            try:
                scraper = FBRef(competition, label)
                fixtures = scraper.get_fixtures()
            except Exception as exc:
                print(f"  {div}/{season}: FBref blocked or failed ({exc})")
                continue
            rows = parse_fbref_fixtures(fixtures, div, season)
            saved = save_match_rows(rows)
            total += saved
            print(f"  {div}/{season}: cached {saved} matches ({competition})")

    stats = fbref_cache_stats()
    print(f"FBref xG cache: {stats['fbref_matches']} matches total")
    return total


if __name__ == "__main__":
    fetch_fbref()