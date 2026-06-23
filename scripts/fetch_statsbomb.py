"""Phase 6F: cache StatsBomb open-data match xG (aggregated from shot events)."""

from statsbombpy import sb

from config.settings import league_div_codes, statsbomb_competition_to_div
from utils.statsbomb_cache import (
    match_xg_from_events,
    parse_statsbomb_match,
    save_match_rows,
    statsbomb_cache_stats,
)


def open_big5_competitions() -> list[tuple[int, int, str, str]]:
    """Return (competition_id, season_id, competition_name, div) for open Big 5 seasons."""
    comps = sb.competitions()
    div_map = statsbomb_competition_to_div()
    targets: list[tuple[int, int, str, str]] = []
    for _, row in comps.iterrows():
        div = div_map.get(row["competition_name"])
        if div and div in league_div_codes():
            targets.append((
                int(row["competition_id"]),
                int(row["season_id"]),
                str(row["competition_name"]),
                div,
            ))
    return targets


def fetch_statsbomb(
    div_filter: list[str] | None = None,
    max_matches_per_season: int = 0,
) -> int:
    div_codes = set(div_filter or league_div_codes())
    total = 0

    for comp_id, season_id, comp_name, div in open_big5_competitions():
        if div not in div_codes:
            continue
        try:
            matches = sb.matches(comp_id, season_id)
        except Exception as exc:
            print(f"  {div} {comp_name}/{season_id}: matches fetch failed ({exc})")
            continue
        if matches.empty:
            print(f"  {div} {comp_name}/{season_id}: no matches")
            continue

        rows = []
        limit = len(matches) if max_matches_per_season <= 0 else min(
            len(matches), max_matches_per_season,
        )
        for i in range(limit):
            m = matches.iloc[i]
            try:
                events = sb.events(int(m["match_id"]))
                xg_home, xg_away = match_xg_from_events(
                    events, m["home_team"], m["away_team"],
                )
            except Exception as exc:
                print(f"    match {m['match_id']}: events failed ({exc})")
                continue
            rows.append(parse_statsbomb_match(m, xg_home, xg_away, div))

        saved = save_match_rows(rows)
        total += saved
        print(f"  {div} {comp_name}/{season_id}: cached {saved}/{limit} matches")

    stats = statsbomb_cache_stats()
    print(f"StatsBomb xG cache: {stats['statsbomb_matches']} matches total")
    return total


if __name__ == "__main__":
    fetch_statsbomb()